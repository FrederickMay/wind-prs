"""Data fetchers and loaders.

Everything is cached under data/raw (gitignored). Fetchers loop over months,
save one parquet file per month, and skip files that already exist, so
re-running is cheap and only fills gaps.

All timestamps returned by the loaders are UTC, labelled by the *start* of
the period (an hourly value at 10:00 covers 10:00-11:00).
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

ELEXON = "https://data.elexon.co.uk/bmrs/api/v1"
BMU = "T_PNYCW-1"
SITE_LAT, SITE_LON = 51.71, -3.56


# ---------------------------------------------------------------- ERA5

def download_era5_site(out_name="era5_pyc_100m_1940_2026.nc"):
    """ERA5 100 m wind, 3x3 grid around Pen y Cymoedd (T_PNYCW-1)."""
    import cdsapi  # imported here so the rest of the module works without it

    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / out_name
    if out_path.exists():  # already cached, don't re-request 86 years
        return out_path
    dataset = "reanalysis-era5-single-levels-timeseries"
    request = {
        "variable": [
            "100m_u_component_of_wind",
            "100m_v_component_of_wind"
        ],
        "date": ["1940-01-01/2026-10-01"],
        "data_format": "netcdf",
        "area": [52, -3.75, 51.5, -3.25]
    }
    cdsapi.Client().retrieve(dataset, request).download(str(out_path))
    return out_path


def load_era5_site(method="interp", start=None, end=None):
    """Hourly 100 m wind speed (m/s) at the site from the 3x3 ERA5 grid.

    method:
      "nearest" - the single grid point closest to the site centre
      "mean"    - average of all 9 points
      "interp"  - bilinear interpolation of u and v to the site centre

    ERA5 values are instantaneous at each hour. To match hourly *averages*
    of metered output we average consecutive instants, so the value labelled
    10:00 is the mean of the 10:00 and 11:00 instants.
    """
    import xarray as xr

    ds = xr.open_dataset(RAW / "era5_pyc_100m_1940_2026.nc")
    ds = ds.sel(valid_time=slice(start, end))
    if method == "nearest":
        p = ds.sel(latitude=SITE_LAT, longitude=SITE_LON, method="nearest")
        ws = np.hypot(p["u100"], p["v100"])
    elif method == "mean":
        ws = np.hypot(ds["u100"], ds["v100"]).mean(["latitude", "longitude"])
    elif method == "interp":
        # interpolate the vector components, then take the speed
        p = ds.interp(latitude=SITE_LAT, longitude=SITE_LON)
        ws = np.hypot(p["u100"], p["v100"])
    else:
        raise ValueError(f"unknown method {method!r}")

    s = ws.to_series().astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_localize("UTC")
    s = (s + s.shift(-1)) / 2           # instant -> hour-beginning average
    return s.dropna().rename("ws100")


# ---------------------------------------------------------------- Elexon

def _get_json(url, params, retries=4):
    """GET with simple retry/backoff; the Elexon API occasionally times out."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            return js["data"] if isinstance(js, dict) and "data" in js else js
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _months(start, end):
    return pd.date_range(pd.Timestamp(start).to_period("M").to_timestamp(),
                         pd.Timestamp(end), freq="MS")


def _fetch_monthly(name, start, end, fetch_month):
    """Run fetch_month(month_start, month_end) for each month and cache it.

    Files for months that finished more than ~2 months ago are treated as
    final and never re-downloaded. Recent months are refetched because
    settlement data is revised (B1610 runs) and the current month is partial.
    """
    folder = RAW / "elexon" / name
    folder.mkdir(parents=True, exist_ok=True)
    recent = pd.Timestamp.now().to_period("M").to_timestamp() - pd.DateOffset(months=2)
    frames = []
    for m0 in _months(start, end):
        path = folder / f"{m0:%Y-%m}.parquet"
        if not path.exists() or m0 >= recent:
            m1 = m0 + pd.DateOffset(months=1)
            df = pd.DataFrame(fetch_month(m0, m1))
            df.to_parquet(path)
            print(f"  {name} {m0:%Y-%m}: {len(df)} rows")
        frames.append(pd.read_parquet(path))
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def settlement_to_utc(settlement_date, settlement_period):
    """GB settlement date + period (1..46/48/50) -> UTC start time.

    Period 1 starts at *local* midnight. Converting local midnight to UTC and
    adding 30 min steps is correct even on clock-change days (46 or 50 periods).
    """
    midnight = pd.to_datetime(settlement_date).dt.tz_localize("Europe/London")
    return midnight.dt.tz_convert("UTC") + pd.to_timedelta(
        (settlement_period - 1) * 30, unit="min")


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%MZ")


def fetch_b1610(start="2017-01-01", end=None, bmu=BMU):
    """Half-hourly metered output (MWh per half-hour) for one BM unit.

    The API only holds B1610 for T_PNYCW-1 from Feb 2019, earlier months
    come back empty (and are cached as empty files).
    """
    end = end or pd.Timestamp.now()

    def month(m0, m1):
        return _get_json(f"{ELEXON}/datasets/B1610/stream",
                         {"from": _iso(m0), "to": _iso(m1), "bmUnit": bmu})

    df = _fetch_monthly("B1610", start, end, month)
    df = df.drop_duplicates(["settlementDate", "settlementPeriod"], keep="last")
    df["time"] = settlement_to_utc(df["settlementDate"], df["settlementPeriod"])
    # MWh in a half-hour -> average MW
    return (df.set_index("time")["quantity"].astype(float) * 2).sort_index().rename("metered_mw")


def fetch_mid(start="2017-01-01", end=None):
    """Half-hourly Market Index price (GBP/MWh), APX provider.

    The N2EX series is almost always zero volume, so we keep APXMIDP only.
    """
    end = end or pd.Timestamp.now()

    def month(m0, m1):
        return _get_json(f"{ELEXON}/datasets/MID/stream",
                         {"from": _iso(m0), "to": _iso(m1)})

    df = _fetch_monthly("MID", start, end, month)
    df = df[df["dataProvider"] == "APXMIDP"]
    df["time"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.drop_duplicates("time", keep="last").set_index("time").sort_index()
    # zero volume means no trades: the 0.00 price is a placeholder, not a price
    price = df["price"].astype(float).where(df["volume"].astype(float) > 0)
    return price.rename("price")


def fetch_fuelhh_wind(start="2017-01-01", end=None):
    """Half-hourly transmission-connected GB wind generation (MW)."""
    end = end or pd.Timestamp.now()

    def month(m0, m1):
        last = (m1 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return _get_json(f"{ELEXON}/datasets/FUELHH/stream",
                         {"settlementDateFrom": m0.strftime("%Y-%m-%d"),
                          "settlementDateTo": last, "fuelType": "WIND"})

    df = _fetch_monthly("FUELHH_WIND", start, end, month)
    df["time"] = pd.to_datetime(df["startTime"], utc=True)
    df = df.sort_values("publishTime").drop_duplicates("time", keep="last")
    return df.set_index("time")["generation"].astype(float).sort_index().rename("national_wind_mw")


def fetch_boalf(start="2017-01-01", end=None, bmu=BMU):
    """Bid-offer acceptances for the BM unit (one row per acceptance segment)."""
    end = end or pd.Timestamp.now()

    def month(m0, m1):
        return _get_json(f"{ELEXON}/datasets/BOALF/stream",
                         {"from": _iso(m0), "to": _iso(m1), "bmUnit": bmu})

    df = _fetch_monthly("BOALF", start, end, month)
    if df.empty:
        return df
    df["timeFrom"] = pd.to_datetime(df["timeFrom"], utc=True)
    df["timeTo"] = pd.to_datetime(df["timeTo"], utc=True)
    return df.drop_duplicates(["acceptanceNumber", "timeFrom", "timeTo"])


def curtailment_flags(index, boalf):
    """True for each half-hour in `index` that overlaps an accepted BM action.

    For a wind farm almost every acceptance is a bid (turn down), so any
    overlap means output was not weather-limited and should be excluded
    when fitting the power curve.
    """
    flag = pd.Series(False, index=index)
    if boalf.empty:
        return flag
    starts = index.values
    ends = (index + pd.Timedelta(minutes=30)).values
    for t0, t1 in zip(boalf["timeFrom"].values, boalf["timeTo"].values):
        # half-hour [s, e) overlaps acceptance [t0, t1] if s < t1 and e > t0
        lo = np.searchsorted(ends, t0, side="right")
        hi = np.searchsorted(starts, t1, side="left")
        flag.iloc[lo:hi] = True
    return flag


# ---------------------------------------------------------------- national wind

def load_national_wind(source="elexon", start="2017-01-01", end=None):
    """Hourly GB national wind *capacity factor* (0..1), UTC hour-beginning.

    This is the single entry point for the national wind factor, so the
    source can be swapped without touching downstream code.

    "elexon": FUELHH transmission wind divided by an estimate of installed
      capacity. Capacity grows over time and FUELHH has no capacity column,
      so we use a rolling 12-month max of output as the capacity proxy (the
      fleet rarely runs above ~90% at once, so the CF scale is slightly
      high, but it is consistent over time, which is what matters).
    "era5": not implemented yet (optional Phase 5: gridded ERA5 over GB).
    """
    if source == "era5":
        raise NotImplementedError("ERA5 national wind factor is a Phase 5 extension")
    if source != "elexon":
        raise ValueError(f"unknown source {source!r}")

    mw = fetch_fuelhh_wind(start, end)
    hourly = mw.resample("h").mean()
    monthly_max = hourly.resample("MS").max()
    cap = monthly_max.rolling(12, min_periods=1).max()       # never decreases
    cap = cap.reindex(hourly.index, method="ffill")
    return (hourly / cap).rename("national_cf")


# ---------------------------------------------------------------- hourly panel

def build_hourly_panel(start="2017-01-01", end="2026-01-01", era5_method="interp",
                       refresh=False):
    """Join everything onto one hourly UTC table and cache it.

    Columns: ws100 (ERA5 site wind), metered_mw (site), curtailed (bool),
    price (MID, GBP/MWh), national_cf.
    An hour counts as curtailed if either half-hour overlaps an acceptance.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"hourly_panel_{era5_method}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    metered = fetch_b1610(start, end)
    boalf = fetch_boalf(start, end)
    curt = curtailment_flags(metered.index, boalf)
    price = fetch_mid(start, end)

    df = pd.DataFrame({
        "metered_mw": metered.resample("h").mean(),
        "n_halfhours": metered.resample("h").count(),
        "curtailed": curt.resample("h").max().astype(float),
        "price": price.resample("h").mean(),
    })
    df["national_cf"] = load_national_wind("elexon", start, end)
    df["ws100"] = load_era5_site(era5_method)
    df = df.loc[(df.index >= pd.Timestamp(start, tz="UTC")) &
                (df.index < pd.Timestamp(end, tz="UTC"))]
    df["curtailed"] = df["curtailed"].fillna(0).astype(bool)
    df.to_parquet(path)
    return df
