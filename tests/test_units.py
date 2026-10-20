"""Fast tests that need no downloaded data."""
import numpy as np
import pandas as pd

from prs.data import curtailment_flags, settlement_to_utc
from prs.power import CAPACITY_MW, PowerModel, farm_cf_curve, turbine_curve
from prs.price import fit_normal_mixture
from prs.pricing import Contract, month_starts, var_es


def test_settlement_periods_across_clock_changes():
    # 2024-03-31 has 46 periods (clocks forward), 2024-10-27 has 50 (back)
    spring = settlement_to_utc(pd.Series(["2024-03-31"] * 2), pd.Series([1, 46]))
    assert spring.iloc[0] == pd.Timestamp("2024-03-31 00:00", tz="UTC")
    assert spring.iloc[1] == pd.Timestamp("2024-03-31 22:30", tz="UTC")
    autumn = settlement_to_utc(pd.Series(["2024-10-27"] * 2), pd.Series([1, 50]))
    assert autumn.iloc[0] == pd.Timestamp("2024-10-26 23:00", tz="UTC")
    assert autumn.iloc[1] == pd.Timestamp("2024-10-27 23:30", tz="UTC")


def test_curtailment_flags_overlap():
    idx = pd.date_range("2024-01-01", periods=6, freq="30min", tz="UTC")
    boalf = pd.DataFrame({"timeFrom": [pd.Timestamp("2024-01-01 00:40", tz="UTC")],
                          "timeTo": [pd.Timestamp("2024-01-01 01:10", tz="UTC")]})
    flags = curtailment_flags(idx, boalf)
    assert flags.tolist() == [False, True, True, False, False, False]


def test_turbine_curve_physics():
    v = np.arange(0, 30, 0.5)
    p = turbine_curve(108, v)
    assert p.max() == 3000.0                     # capped at rating
    assert (p[v < 3] == 0).all() and (p[v > 25] == 0).all()
    ramp = p[(v >= 3) & (v <= 12)]
    assert (np.diff(ramp) >= 0).all()            # monotone in the ramp


def test_farm_curve_is_smoother_and_bounded():
    cf = farm_cf_curve(108)
    assert cf.min() >= 0 and cf.max() <= 1
    # smoothing moves output below cut-in and spreads it near rated
    v = np.arange(0, 40.01, 0.1)
    assert cf[np.isclose(v, 2.5)][0] > 0


def test_power_model_recovers_parameters():
    rng = np.random.default_rng(0)
    ws = rng.weibull(2.2, 20000) * 8
    truth = PowerModel(108, a=1.1, b=-0.8, eff=0.9)
    mw = truth.predict_mw(ws) + rng.normal(0, 5, ws.size)
    fit = PowerModel(108).fit(ws, mw)
    assert abs(fit.a - 1.1) < 0.02 and abs(fit.b + 0.8) < 0.15 and abs(fit.eff - 0.9) < 0.01
    # above cut-out the (smoothed) farm output falls well below rated
    assert truth.predict_mw(30.0) < 0.5 * truth.predict_mw(15.0) <= CAPACITY_MW


def test_mixture_em_recovers_spikes():
    rng = np.random.default_rng(1)
    n = 100_000
    spike = rng.random(n) < 0.04
    x = np.where(spike, rng.normal(2, 40, n), rng.normal(0, 8, n))
    w, mu, sd = fit_normal_mixture(x)
    assert abs(w - 0.04) < 0.01
    assert abs(sd[0] - 8) < 0.5 and abs(sd[1] - 40) < 4


def test_var_es_on_normal():
    x = np.random.default_rng(2).standard_normal(1_000_000)
    v, es = var_es(x, 0.99)
    assert abs(v - 2.326) < 0.02 and abs(es - 2.665) < 0.02


def test_contract_calendar():
    c = Contract()
    idx = c.index()
    assert len(c.months()) == 120
    assert len(month_starts(idx)) == 120
    assert idx[0] == pd.Timestamp("2027-01-01", tz="UTC")
    assert (np.diff(c.discount_factors()) < 0).all()
