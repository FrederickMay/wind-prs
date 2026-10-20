"""Fit every model once and cache the result, so notebooks stay short.

    from prs.pipeline import load_models
    m = load_models()            # dict: panel, power, wind, price, stress
"""
import pickle

import pandas as pd

from .data import PROCESSED, build_hourly_panel, load_era5_site
from .power import PowerModel
from .price import PriceModel
from .wind import WindModel

ERA5_METHOD = "interp"     # chosen in notebook 02 (best out-of-sample fit)
ROTOR = 101.0              # SWT-3.0-101; 108 m fits equally well (notebook 02)
FIT_YEARS = ("1991-01-01", "2025-12-31T23:00")


def clean_hours(panel):
    """Hours usable for fitting the power curve: metered, both half-hours
    present, no curtailment, ERA5 available."""
    return panel[(~panel["curtailed"]) & (panel["n_halfhours"] == 2)
                 & panel["metered_mw"].notna() & panel["ws100"].notna()]


def calmest_winter_shift(wind_model, start="1940-01-01"):
    """Mean Gaussian wind anomaly of the calmest Dec-Feb winter on record.

    Uses the full record back to 1940. Pre-1979 ERA5 is weaker, but this
    is only for picking an extreme winter, not for fitting anything.
    """
    ws = load_era5_site(ERA5_METHOD, start, FIT_YEARS[1])
    e = pd.Series(wind_model.site_to_e(ws.values, ws.index), ws.index)
    djf = e[e.index.month.isin([12, 1, 2])]
    winter_year = djf.index.year + (djf.index.month == 12)    # Dec counts with next year
    means = djf.groupby(winter_year).mean()
    means = means[means.index.isin(range(1941, 2026))]        # full winters only
    return float(means.min()), int(means.idxmin()), means


def fit_models():
    panel = build_hourly_panel(era5_method=ERA5_METHOD)
    clean = clean_hours(panel)
    power = PowerModel(ROTOR).fit(clean["ws100"], clean["metered_mw"])
    site = load_era5_site(ERA5_METHOD, *FIT_YEARS)
    wind = WindModel.fit(site, panel["national_cf"])
    price = PriceModel.fit(panel["price"], panel["national_cf"])

    shift, winter, _ = calmest_winter_shift(wind)
    post = panel["price"]["2023-07":"2025-12"].mean()
    gas = float(panel["price"]["2022"].mean() - post)
    stress = {"still_winter_shift": shift, "still_winter_year": winter,
              "gas_shift": gas}
    return {"panel": panel, "power": power, "wind": wind, "price": price,
            "stress": stress}


def load_models(refresh=False):
    path = PROCESSED / "models.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as f:
            return pickle.load(f)
    models = fit_models()
    with open(path, "wb") as f:
        pickle.dump(models, f)
    return models
