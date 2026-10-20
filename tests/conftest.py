import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prs.data import RAW  # noqa: E402

HAVE_DATA = (RAW / "era5_pyc_100m_1940_2026.nc").exists() and (RAW / "elexon").exists()


@pytest.fixture(scope="session")
def models():
    if not HAVE_DATA:
        pytest.skip("raw data not downloaded")
    from prs.pipeline import load_models
    return load_models()


@pytest.fixture(scope="session")
def sim_10y(models):
    """One shared simulation of 10 years x 40 paths for the moment tests."""
    import pandas as pd
    idx = pd.date_range("2027-01-01", periods=8760 * 10, freq="h", tz="UTC")
    ws, cf = models["wind"].simulate(idx, 40, seed=123)
    price = models["price"].simulate(idx, cf, forward=75.0, seed=456)
    return idx, ws, cf, price
