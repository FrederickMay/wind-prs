"""Check the fitted models reproduce the data they were fitted on.

These use the real data (skipped if it is not downloaded) and a shared
10-year simulation. Tolerances are loose on purpose: the aim is to catch a
broken model, not to test Monte Carlo noise.
"""
import numpy as np
import pandas as pd
from scipy import stats

from prs.pipeline import ERA5_METHOD, FIT_YEARS, clean_hours
from prs.power import score


def _acf(x, lag):
    x = x - x.mean()
    return (x[:-lag] * x[lag:]).sum() / (x * x).sum()


def test_power_curve_validates(models):
    clean = clean_hours(models["panel"])
    s = score(models["power"], clean["ws100"], clean["metered_mw"])
    assert s["r2"] > 0.8
    assert abs(s["bias_mw"]) < 2
    assert 0.8 < models["power"].eff < 1.0


def test_site_wind_moments(models, sim_10y):
    from prs.data import load_era5_site
    _, ws, _, _ = sim_10y
    hist = load_era5_site(ERA5_METHOD, *FIT_YEARS)
    assert abs(ws.mean() - hist.mean()) < 0.15
    assert abs(ws.std() - hist.std()) < 0.15
    # winter windier than summer, in both
    idx = sim_10y[0]
    sim_jan = ws[:, idx.month == 1].mean()
    sim_jul = ws[:, idx.month == 7].mean()
    assert sim_jan > sim_jul + 1.5


def test_site_wind_autocorrelation(models, sim_10y):
    from prs.data import load_era5_site
    _, ws, _, _ = sim_10y
    hist = load_era5_site(ERA5_METHOD, *FIT_YEARS).values
    for lag in (1, 6, 24, 72):
        a_hist = _acf(hist, lag)
        a_sim = np.mean([_acf(w, lag) for w in ws[:10]])
        assert abs(a_sim - a_hist) < 0.07, (lag, a_sim, a_hist)


def test_annual_wind_variability(models):
    """Year-to-year spread of annual mean wind matches 35 years of ERA5."""
    from prs.data import load_era5_site
    hist = load_era5_site(ERA5_METHOD, *FIT_YEARS)
    hist_sd = hist.groupby(hist.index.year).mean().std()
    idx = pd.date_range("2027-01-01", periods=8760 * 4, freq="h", tz="UTC")
    ws, _ = models["wind"].simulate(idx, 100, seed=7)
    years = ws.reshape(100, 4, 8760).mean(2).ravel()
    assert 0.6 * hist_sd < years.std() < 1.5 * hist_sd


def test_wind_price_correlation(models, sim_10y):
    panel = models["panel"].loc["2023-07":"2025-12"].dropna(subset=["price", "national_cf", "ws100"])
    hist_nat = stats.spearmanr(panel["national_cf"], panel["price"])[0]
    hist_site = stats.spearmanr(panel["ws100"], panel["price"])[0]
    _, ws, cf, p = sim_10y
    sim_nat = stats.spearmanr(cf[:5].ravel(), p[:5].ravel())[0]
    sim_site = stats.spearmanr(ws[:5].ravel(), p[:5].ravel())[0]
    assert hist_nat < -0.3 and sim_nat < -0.3
    assert abs(sim_nat - hist_nat) < 0.1
    assert abs(sim_site - hist_site) < 0.1


def test_price_moments(models, sim_10y):
    hist = models["panel"]["price"].loc["2023-07":"2025-12"].dropna()
    _, _, _, p = sim_10y
    assert abs(p.mean() - 75.0) < 2.0                  # centred on the forward
    assert abs(p.std() - hist.std()) < 0.15 * hist.std()
    assert 0.005 < (p < 0).mean() < 0.06                # negative prices happen
    assert abs(_acf(p[0], 1) - _acf(hist.values, 1)) < 0.05


def test_monthly_wind_variability(models, sim_10y):
    """Spread of monthly mean wind (the settlement period) matches ERA5."""
    from prs.data import load_era5_site
    hist = load_era5_site(ERA5_METHOD, *FIT_YEARS).resample("MS").mean()
    hist_sd = (hist - hist.groupby(hist.index.month).transform("mean")).std()
    idx, ws, _, _ = sim_10y
    sims = []
    for w in ws:
        s = pd.Series(w, idx).resample("MS").mean()
        sims.append(s - s.groupby(s.index.month).transform("mean"))
    sim_sd = np.std(np.concatenate(sims))
    assert abs(sim_sd - hist_sd) < 0.15 * hist_sd, (sim_sd, hist_sd)
