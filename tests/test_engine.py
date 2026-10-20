"""The C++ engine must agree with the Python reference (statistically: the
random numbers differ) and be reproducible whatever the thread count."""
import numpy as np
import pytest

from prs import engine
from prs.pricing import Contract, simulate_monthly

pytestmark = pytest.mark.skipif(not engine.available(), reason="C++ engine not built")


@pytest.fixture(scope="module")
def both(models):
    c = Contract(years=3)
    args = (models["wind"], models["power"], models["price"], c)
    return (simulate_monthly(*args, 120, seed=1),
            engine.simulate_monthly_cpp(*args, 1200, seed=2), args)


def _close(a, b, n_sd=4):
    """Means of two independent samples agree within n_sd standard errors."""
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return abs(a.mean() - b.mean()) < n_sd * se


@pytest.mark.parametrize("key", ["volume", "revenue", "baseload"])
def test_annual_means_agree(both, key):
    py, cpp, _ = both
    a = py[key].reshape(len(py[key]), 3, 12).sum(2)[:, 0]
    b = cpp[key].reshape(len(cpp[key]), 3, 12).sum(2)[:, 0]
    assert _close(a, b)


@pytest.mark.parametrize("key", ["volume", "revenue"])
def test_monthly_spread_agrees(both, key):
    py, cpp, _ = both
    assert abs(py[key].std() / cpp[key].std() - 1) < 0.1


def test_thread_count_does_not_change_results(both):
    *_, args = both
    a = engine.simulate_monthly_cpp(*args, 12, seed=9, n_threads=1)
    b = engine.simulate_monthly_cpp(*args, 12, seed=9, n_threads=6)
    assert np.array_equal(a["revenue"], b["revenue"])
