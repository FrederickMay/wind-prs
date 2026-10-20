"""Monte Carlo pricing and risk of the proxy revenue swap.

Contract (from the hedge provider's side, "the counterparty"):
  each month m it RECEIVES the proxy revenue  R_m = sum_h proxy_MWh_h * P_h
  and PAYS a fixed amount                     K_m = K * E[V_m]
where V_m is the proxy volume (MWh) and K is the fixed price in GBP per MWh
of *expected* volume. So the farm gets a fixed, seasonally shaped income.

Fair fixed price: K* = sum_m D_m E[R_m] / sum_m D_m E[V_m], i.e. the
discounted expected *capture price*. It sits below the baseload forward
because wind produces most when prices are low (cannibalisation).

The counterparty can hedge the price level by selling baseload forwards
for the expected volume. What is left after that is volume and shape risk,
which cannot be hedged with standard products. We report P&L both ways.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Contract:
    start: str = "2027-01-01"
    years: int = 10
    forward: float = 75.0            # baseload forward level, GBP/MWh
    discount_rate: float = 0.045     # flat annual rate
    hurdle_rate: float = 0.08        # cost of capital charged on risk capital
    capital_conf: float = 0.99       # ES level used as risk capital

    def index(self):
        start = pd.Timestamp(self.start, tz="UTC")
        end = start + pd.DateOffset(years=self.years)
        return pd.date_range(start, end, freq="h", inclusive="left")

    def months(self):
        return pd.date_range(self.start, periods=12 * self.years, freq="MS", tz="UTC")

    def discount_factors(self):
        m = self.months()
        t = (m - m[0]).days.values / 365.25 + 1 / 12   # pay at month end
        return (1 + self.discount_rate) ** (-t)


def month_starts(index):
    """Position of the first hour of each calendar month in `index`."""
    key = index.year * 12 + index.month
    return np.flatnonzero(np.r_[True, key[1:] != key[:-1]])


def simulate_monthly(wind_model, power_model, price_model, contract, n_paths,
                     seed=0, batch=100, e_shift=None, level_shift=None,
                     cannibal_mult=1.0):
    """Python reference Monte Carlo. Returns monthly arrays (n_paths, n_months):
    volume (MWh), revenue (GBP), baseload (mean hourly price, GBP/MWh).

    e_shift, level_shift, cannibal_mult are stress-test knobs (see stress()).
    """
    idx = contract.index()
    starts = month_starts(idx)
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("volume", "revenue", "baseload")}
    hours = np.diff(np.r_[starts, len(idx)])
    for b0 in range(0, n_paths, batch):
        nb = min(batch, n_paths - b0)
        ws, cf = wind_model.simulate(idx, nb, rng=rng, e_shift=e_shift)
        mw = power_model.predict_mw(ws)
        p = price_model.simulate(idx, cf, forward=contract.forward, rng=rng,
                                 level_shift=level_shift, cannibal_mult=cannibal_mult)
        out["volume"].append(np.add.reduceat(mw, starts, axis=1))
        out["revenue"].append(np.add.reduceat(mw * p, starts, axis=1))
        out["baseload"].append(np.add.reduceat(p, starts, axis=1) / hours)
        del ws, cf, mw, p
    return {k: np.vstack(v) for k, v in out.items()} | {"hours": hours}


# ----------------------------------------------------------------- pricing

def var_es(pnl, conf):
    """VaR and expected shortfall of a P&L sample, reported as positive losses."""
    q = np.quantile(pnl, 1 - conf)
    return -q, -pnl[pnl <= q].mean()


@dataclass
class Terms:
    """What is fixed when the deal is struck (from the base-case simulation)."""
    fixed_price: float              # fair K*, GBP/MWh of expected volume
    exp_volume: np.ndarray          # E[V_m]
    exp_revenue: np.ndarray         # E[R_m]
    hedge_mw: np.ndarray            # baseload MW sold forward each month
    hedge_price: np.ndarray         # forward price for each month = E[baseload_m]


def strike_terms(sim, contract):
    D = contract.discount_factors()
    ev, er = sim["volume"].mean(0), sim["revenue"].mean(0)
    return Terms(fixed_price=(D * er).sum() / (D * ev).sum(),
                 exp_volume=ev, exp_revenue=er,
                 hedge_mw=ev / sim["hours"], hedge_price=sim["baseload"].mean(0))


def counterparty_pnl(sim, terms, contract, fixed_price=None):
    """Monthly P&L (n_paths, n_months) for the hedge provider, unhedged and hedged.

    hedged adds a short baseload forward of hedge_mw at hedge_price: it gains
    when realised prices fall, offsetting the lower proxy revenue.
    """
    K = terms.fixed_price if fixed_price is None else fixed_price
    unhedged = sim["revenue"] - K * terms.exp_volume
    hedge = terms.hedge_mw * sim["hours"] * (terms.hedge_price - sim["baseload"])
    return {"unhedged": unhedged, "hedged": unhedged + hedge}


def risk_report(pnl_monthly, contract):
    """PV of total P&L and worst calendar-year P&L statistics, in GBP m."""
    D = contract.discount_factors()
    pv = (pnl_monthly * D).sum(1) / 1e6
    yearly = pnl_monthly.reshape(len(pnl_monthly), contract.years, 12).sum(2) / 1e6
    first_year = yearly[:, 0]
    rows = {}
    for name, x in [("PV lifetime", pv), ("Year 1", first_year)]:
        v95, es95 = var_es(x, 0.95)
        v99, es99 = var_es(x, 0.99)
        rows[name] = {"mean": x.mean(), "sd": x.std(), "VaR95": v95, "ES95": es95,
                      "VaR99": v99, "ES99": es99}
    return pd.DataFrame(rows).T


def risk_premium(sim, terms, contract):
    """Cost-of-capital risk premium, expressed as a discount on the fixed price.

    Capital = ES at capital_conf of the hedged 1-year P&L (economic capital a
    desk would hold against the deal), held for every year of the tenor and
    charged at the hurdle rate. Spread over expected discounted volume this
    gives GBP/MWh taken off the fair price.
    """
    pnl = counterparty_pnl(sim, terms, contract)["hedged"]
    yearly = pnl.reshape(len(pnl), contract.years, 12).sum(2)
    capital = np.mean([var_es(yearly[:, y], contract.capital_conf)[1]
                       for y in range(contract.years)])
    D = contract.discount_factors()
    annual_D = D.reshape(contract.years, 12)[:, -1]
    premium_pv = contract.hurdle_rate * capital * annual_D.sum()
    per_mwh = premium_pv / (D * terms.exp_volume).sum()
    return {"capital_gbp_m": capital / 1e6, "premium_pv_gbp_m": premium_pv / 1e6,
            "premium_gbp_per_mwh": per_mwh,
            "quoted_fixed_price": terms.fixed_price - per_mwh}


def summary(sim, terms, contract):
    """Headline numbers for the base case."""
    D = contract.discount_factors()
    ev = (D * terms.exp_volume).sum()
    base = (D * terms.hedge_price * terms.exp_volume).sum() / ev
    vol_yr = sim["volume"].reshape(len(sim["volume"]), contract.years, 12).sum(2)
    return {
        "fair_fixed_price_gbp_mwh": terms.fixed_price,
        "volume_weighted_baseload_gbp_mwh": base,
        "capture_rate": terms.fixed_price / base,
        "expected_annual_volume_gwh": vol_yr.mean() / 1000,
        "annual_volume_p90_gwh": np.quantile(vol_yr, 0.10) / 1000,
        "annual_volume_p10_gwh": np.quantile(vol_yr, 0.90) / 1000,
        "expected_capacity_factor": vol_yr.mean() / (228 * 8766),
    }


# ----------------------------------------------------------------- stress tests

def stress_inputs(contract, still_winter_shift=0.0, gas_shift=0.0):
    """Knobs for simulate_monthly() describing a year-1 stress.

    still_winter_shift  Gaussian-units shift applied to site AND national wind
                        in the year-1 winter months (Jan, Feb, Dec). Calibrated
                        from the calmest winter in the 1940-2025 ERA5 record.
    gas_shift           GBP/MWh added to every year-1 hour (a gas price shock
                        like 2022 that lifts the whole price level).
    """
    idx = contract.index()
    year1 = idx.year == idx[0].year
    winter = year1 & idx.month.isin([1, 2, 12])
    e_shift = np.where(winter, still_winter_shift, 0.0) if still_winter_shift else None
    level = np.where(year1, gas_shift, 0.0) if gas_shift else None
    return {"e_shift": e_shift, "level_shift": level}
