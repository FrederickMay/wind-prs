"""Python front end for the C++ Monte Carlo engine (cpp/).

simulate_monthly_cpp() has the same inputs and outputs as the Python
reference pricing.simulate_monthly(), so the two are interchangeable.
Build the engine first:  powershell -ExecutionPolicy Bypass -File cpp\\build.ps1
"""
import numpy as np

from .power import CAPACITY_MW, GRID
from .pricing import month_starts
from .wind import _period_ids

try:
    from . import _engine
except ImportError:          # not built yet
    _engine = None


def available():
    return _engine is not None


def _calendar(index, price_model, e_shift=None, level_shift=None):
    mi, yi = _period_ids(index)
    T = len(index)
    return {
        "month_of_year": index.month.values - 1,
        "hour_of_day": index.hour.values,
        "month_id": mi, "year_id": yi,
        # deterministic part of the price: S(t) with the wind term left out
        "price_shape": price_model.shape(index, np.zeros(T)),
        "e_shift": np.zeros(0) if e_shift is None else np.asarray(e_shift, float),
        "level_shift": np.zeros(0) if level_shift is None else np.asarray(level_shift, float),
    }


def _params(wind_model, power_model, price_model, forward, cannibal_mult):
    w = wind_model
    wind = {"weibull_k": w.weibull_k, "weibull_c": w.weibull_c,
            "site_mu": w.site_mu, "site_sd": w.site_sd,
            "nat_mu": w.nat_mu, "nat_sd": w.nat_sd, "nat_q": w.nat_q,
            "var_coefs": w.var_coefs, "chol": np.linalg.cholesky(w.var_cov),
            "sigma_month": w.sigma_month, "sigma_year": w.sigma_year,
            "rho_shock": w.rho_shock}
    power = {"grid_start": GRID[0], "grid_step": GRID[1] - GRID[0],
             "table_cf": power_model.table_cf, "a": power_model.a,
             "b": power_model.b, "eff": power_model.eff, "capacity_mw": CAPACITY_MW}
    b1, b2 = price_model.cannibal
    p = price_model
    price = {"forward": forward, "b1": b1, "b2": b2, "cannibal_mult": cannibal_mult,
             "phi_hour": p.phi_hour, "spike_w": p.spike_w, "calm_mu": p.calm_mu,
             "calm_sd": p.calm_sd, "spike_mu": p.spike_mu, "spike_sd": p.spike_sd,
             "phi_month": p.phi_month, "sigma_month": p.sigma_month}
    return wind, power, price


def simulate_monthly_cpp(wind_model, power_model, price_model, contract, n_paths,
                         seed=0, n_threads=0, e_shift=None, level_shift=None,
                         cannibal_mult=1.0):
    """C++ version of pricing.simulate_monthly (n_threads=0: all cores)."""
    if _engine is None:
        raise ImportError("C++ engine not built: run cpp\\build.ps1")
    idx = contract.index()
    cal = _calendar(idx, price_model, e_shift, level_shift)
    wind, power, price = _params(wind_model, power_model, price_model,
                                 contract.forward, cannibal_mult)
    out = _engine.simulate(cal, wind, power, price, n_paths, seed, n_threads)
    out["hours"] = np.diff(np.r_[month_starts(idx), len(idx)])
    return out
