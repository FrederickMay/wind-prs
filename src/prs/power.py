"""Wind speed -> farm output (the "proxy generation" of the PRS).

Pipeline for one hour:
  ERA5 100 m speed --(bias correction a*ws + b)--> effective hub speed
  --(farm power curve)--> capacity factor --(x efficiency x 228 MW)--> MW

Turbine curve. Pen y Cymoedd has 76 Siemens 3.0 MW turbines with an
89.5 m hub; sources disagree on whether the rotor is 101 m or 108 m.
windpowerlib doesn't have an SWT-3.0 curve, so I build one from physics,
P = 1/2 rho A cp(v) v^3, borrowing cp(v) from the nearest Siemens model it
does have (SWT113/3200) and capping at 3.0 MW. Both rotor sizes fit the
metered data equally well, so I use 101 m to match the model name.

Farm curve. One turbine's curve is too sharp for a whole farm, because the
turbines see different winds (terrain, wakes) and ERA5 is a grid-box
average anyway. So the curve is smoothed with a Gaussian of width
sigma(v) = 0.6 + 0.2 v m/s (Staffell & Pfenninger 2016, "Using
bias-corrected reanalysis to simulate current and future wind power
output").
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erfc

N_TURBINES = 76
RATED_KW = 3000.0
CAPACITY_MW = N_TURBINES * RATED_KW / 1000   # 228 MW
HUB_HEIGHT = 89.5
CUT_OUT = 25.0
RHO = 1.225                                  # standard air density, kg/m3

# cp(v) for Siemens SWT113/3200 from the windpowerlib turbine library
_CP_V = np.arange(3.0, 26.0)
_CP = np.array([0.386, 0.43, 0.456, 0.47, 0.477, 0.479, 0.472, 0.442, 0.377,
                0.3, 0.237, 0.19, 0.154, 0.127, 0.106, 0.089, 0.076, 0.065,
                0.056, 0.049, 0.043, 0.038, 0.033])

GRID = np.arange(0.0, 40.01, 0.1)            # wind speed grid for curve tables


_CP_KNEE = 11.0   # above this the SWT113 cp falls only because it hits its own rating


CUT_OUT_SPREAD = 1.5   # m/s: turbines in a farm do not all cut out at once


def turbine_curve(rotor_diameter=101.0, v=GRID, cut_out=True):
    """Single-turbine power (kW) on wind speeds v.

    Above ~11 m/s the reference cp drops because the SWT113 is already
    pitching to hold *its* 3.2 MW rating. Our turbine has a different rotor,
    so we hold cp at its 11 m/s value and let the 3.0 MW cap do the work.
    """
    area = np.pi * (rotor_diameter / 2) ** 2
    cp = np.interp(np.minimum(v, _CP_KNEE), _CP_V, _CP, left=0.0)
    p_kw = 0.5 * RHO * area * cp * v ** 3 / 1000
    p_kw = np.minimum(p_kw, RATED_KW)
    p_kw[v < _CP_V[0]] = 0.0
    if cut_out:
        p_kw[v > CUT_OUT] = 0.0
    return p_kw


def farm_cf_curve(rotor_diameter=101.0, v=GRID):
    """Smoothed farm capacity factor (0..1) on wind speeds v.

    The ramp gets the widening Gaussian. Cut-out has its own fixed, narrow
    spread, because smoothing it with sigma ~5 m/s pulled output down from
    about 15 m/s, and the metered data stays flat there.
    """
    p = turbine_curve(rotor_diameter, GRID, cut_out=False) / RATED_KW
    out = np.empty_like(v, dtype=float)
    for i, vi in enumerate(v):
        sigma = 0.6 + 0.2 * vi
        w = np.exp(-0.5 * ((GRID - vi) / sigma) ** 2)
        out[i] = (w * p).sum() / w.sum()
    survive = 0.5 * erfc((v - CUT_OUT) / (CUT_OUT_SPREAD * np.sqrt(2)))
    return out * survive


class PowerModel:
    """Proxy generation model: MW = eff * 228 * farm_cf(a * ws + b).

    a, b   - linear bias correction of ERA5 speed (ERA5 smooths terrain, so
             speeds in hilly Wales tend to be biased); also absorbs the small
             100 m -> 89.5 m hub height difference.
    eff    - efficiency factor: wakes, electrical losses, typical availability.
             In a PRS this is the pre-agreed factor; the farm keeps the risk
             of doing worse than it.
    """

    def __init__(self, rotor_diameter=101.0, a=1.0, b=0.0, eff=1.0):
        self.rotor_diameter = rotor_diameter
        self.a, self.b, self.eff = a, b, eff
        self.table_cf = farm_cf_curve(rotor_diameter, GRID)

    def cf(self, ws):
        """Capacity factor after efficiency, 0..eff."""
        v = self.a * np.asarray(ws, dtype=float) + self.b
        return self.eff * np.interp(v, GRID, self.table_cf)

    def predict_mw(self, ws):
        return CAPACITY_MW * self.cf(ws)

    def fit(self, ws, mw):
        """Least-squares fit of (a, b, eff) on clean (non-curtailed) hours."""
        ws, mw = np.asarray(ws, float), np.asarray(mw, float)

        def resid(p):
            self.a, self.b, self.eff = p
            return self.predict_mw(ws) - mw

        res = least_squares(resid, x0=[1.0, 0.0, 0.9],
                            bounds=([0.5, -5.0, 0.3], [1.5, 5.0, 1.0]))
        self.a, self.b, self.eff = res.x
        return self

    def params(self):
        return {"rotor_diameter": self.rotor_diameter, "a": self.a,
                "b": self.b, "eff": self.eff}


def score(model, ws, mw):
    """Hourly R2, RMSE (MW) and bias (MW) of a fitted model."""
    pred = model.predict_mw(ws)
    err = pred - np.asarray(mw)
    r2 = 1 - (err ** 2).sum() / ((mw - np.mean(mw)) ** 2).sum()
    return {"r2": r2, "rmse_mw": np.sqrt((err ** 2).mean()), "bias_mw": err.mean()}


def monthly_energy(df, model):
    """Modelled vs metered monthly energy (GWh) over clean hours only."""
    d = df.assign(model_mw=model.predict_mw(df["ws100"]))
    m = d[["metered_mw", "model_mw"]].resample("MS").sum() / 1000
    m.columns = ["metered_gwh", "model_gwh"]
    m["error_pct"] = 100 * (m["model_gwh"] / m["metered_gwh"] - 1)
    return m
