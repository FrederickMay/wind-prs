"""Hourly GB power price model. Additive, not log-based, because GB prices
can go negative.

  P_t = F + Y_m(t) + S(t) + g(cf_t) + X_t

  F      baseload forward level (GBP/MWh). This is an input, not fitted:
         the 2021-23 gas crisis dominates the history, so the level is set
         by hand and only the shape and dynamics come from data.
  Y_m    monthly drift in the level, AR(1) from month to month. Picks up
         slow changes (fuel, demand) the hourly part can't.
  S(t)   seasonal shape: month of year plus hour of day, split weekday and
         weekend. Uses UK local time, since demand follows the clock.
  g(cf)  cannibalisation: g = b1*cf + b2*cf^2, so prices fall as national
         wind rises, and fall faster once the system is flooded with wind.
         This is the term that links price to wind.
  X_t    hourly AR(1) noise. Its shocks come from two normals, a calm one
         and a rare, much wider spike one, which gives GB prices their
         fat tails.

Shape and hourly dynamics are fitted on Jul 2023 - Dec 2025, the period
most like the market going forward. The monthly AR(1) uses both non-crisis
windows (2017 - Jun 2021 and Jul 2023 - 2025), each with its own mean
removed.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SHAPE_START, SHAPE_END = "2023-07-01", "2026-01-01"
LEVEL_WINDOWS = [("2017-01-01", "2021-07-01"), ("2023-07-01", "2026-01-01")]


def design_matrix(index, cf):
    """Regressors: 12 month dummies, 47 hour x daytype dummies, cf, cf^2."""
    local = index.tz_convert("Europe/London")
    n = len(index)
    month = local.month.values
    slot = local.hour.values + 24 * (local.dayofweek.values >= 5)   # 0..47
    X = np.zeros((n, 12 + 47 + 2))
    X[np.arange(n), month - 1] = 1.0
    has_slot = slot > 0                                             # slot 0 is the base
    X[np.where(has_slot)[0], 12 + slot[has_slot] - 1] = 1.0
    X[:, -2] = cf
    X[:, -1] = cf ** 2
    return X


def fit_normal_mixture(x, n_iter=200):
    """EM fit of a two-component normal mixture (calm + spike).

    Returns weight of the spike component, (mu, sd) for calm and spike.
    """
    x = np.asarray(x)
    w, mu, sd = 0.05, np.array([0.0, 0.0]), np.array([x.std() * 0.7, x.std() * 3])
    for _ in range(n_iter):
        p_calm = (1 - w) * np.exp(-0.5 * ((x - mu[0]) / sd[0]) ** 2) / sd[0]
        p_spike = w * np.exp(-0.5 * ((x - mu[1]) / sd[1]) ** 2) / sd[1]
        g = p_spike / (p_calm + p_spike)                 # P(spike | x)
        w = g.mean()
        mu = np.array([np.average(x, weights=1 - g), np.average(x, weights=g)])
        sd = np.sqrt([np.average((x - mu[0]) ** 2, weights=1 - g),
                      np.average((x - mu[1]) ** 2, weights=g)])
    if sd[0] > sd[1]:                                     # label spike = wider
        w, mu, sd = 1 - w, mu[::-1], sd[::-1]
    return w, mu, sd


@dataclass
class PriceModel:
    beta: np.ndarray = None          # regression coefficients (61,)
    shape_offset: float = 0.0        # makes mean(S + g) = 0 historically
    phi_hour: float = 0.0            # AR(1) of X
    spike_w: float = 0.0
    calm_mu: float = 0.0
    calm_sd: float = 0.0
    spike_mu: float = 0.0
    spike_sd: float = 0.0
    phi_month: float = 0.0           # AR(1) of Y
    sigma_month: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def cannibal(self):
        return self.beta[-2], self.beta[-1]

    def shape(self, index, cf, cannibal_mult=1.0):
        """S(t) + g(cf) - offset, broadcast over paths if cf is 2-D."""
        X0 = design_matrix(index, np.zeros(len(index)))
        s = X0 @ self.beta - self.shape_offset
        b1, b2 = self.cannibal
        return s + cannibal_mult * (b1 * cf + b2 * cf ** 2)

    @classmethod
    def fit(cls, price, cf):
        self = cls()
        df = pd.DataFrame({"p": price, "cf": cf}).dropna()

        # --- shape + cannibalisation on post-crisis data (OLS)
        post = df.loc[SHAPE_START:SHAPE_END]
        X = design_matrix(post.index, post["cf"].values)
        y = post["p"].values
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ self.beta
        self.info["shape_r2"] = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # average by calendar month first: the sample (Jul 2023 - Dec 2025)
        # has more winter months than summer ones, which would bias the level
        local_month = post.index.tz_convert("Europe/London").month
        self.shape_offset = pd.Series(fitted).groupby(local_month.values).mean().mean()
        self.info["post_mean_price"] = y.mean()

        # residual = level deviation + hourly noise
        resid_all = df["p"].values - (design_matrix(df.index, df["cf"].values) @ self.beta)
        resid_all = pd.Series(resid_all, df.index)

        # --- hourly AR(1) + mixture innovations (post-crisis)
        r = resid_all.loc[SHAPE_START:SHAPE_END]
        Ym = r.groupby([r.index.year, r.index.month]).transform("mean")
        x = (r - Ym).asfreq("h")
        x0, x1 = x.shift(1).values, x.values
        ok = ~np.isnan(x0) & ~np.isnan(x1)
        self.phi_hour = float((x0[ok] * x1[ok]).sum() / (x0[ok] ** 2).sum())
        innov = x1[ok] - self.phi_hour * x0[ok]
        w, mu, sd = fit_normal_mixture(innov)
        self.spike_w, self.calm_mu, self.spike_mu = float(w), float(mu[0]), float(mu[1])
        self.calm_sd, self.spike_sd = float(sd[0]), float(sd[1])

        # --- monthly level AR(1), pooled over the non-crisis windows
        pairs = []
        for a, b in LEVEL_WINDOWS:
            m = resid_all.loc[a:b].resample("MS").mean().dropna()
            m = m - m.mean()
            pairs.append(np.c_[m.values[:-1], m.values[1:]])
        pairs = np.vstack(pairs)
        self.phi_month = float((pairs[:, 0] * pairs[:, 1]).sum() / (pairs[:, 0] ** 2).sum())
        self.sigma_month = float(np.std(pairs[:, 1] - self.phi_month * pairs[:, 0], ddof=1))
        self.info["monthly_residual_levels"] = resid_all.resample("MS").mean()
        return self

    def simulate(self, index, cf, forward=75.0, rng=None, seed=None,
                 level_shift=None, cannibal_mult=1.0):
        """Simulate hourly prices given national CF paths (n_paths, T).

        forward       baseload level F (scalar or array over hours).
        level_shift   optional extra GBP/MWh added per hour (stress tests).
        """
        rng = rng or np.random.default_rng(seed)
        n_paths, T = cf.shape

        # monthly level Y: AR(1) started from its stationary distribution
        months = index.year.values * 12 + index.month.values
        um, mi = np.unique(months, return_inverse=True)
        Y = np.zeros((n_paths, len(um)))
        sd_stat = self.sigma_month / np.sqrt(1 - self.phi_month ** 2)
        Y[:, 0] = sd_stat * rng.standard_normal(n_paths)
        for k in range(1, len(um)):
            Y[:, k] = self.phi_month * Y[:, k - 1] + self.sigma_month * rng.standard_normal(n_paths)

        # hourly AR(1) X with two-regime innovations
        spike = rng.random((T, n_paths)) < self.spike_w
        z = rng.standard_normal((T, n_paths))
        innov = np.where(spike, self.spike_mu + self.spike_sd * z,
                         self.calm_mu + self.calm_sd * z)
        X = np.empty((T, n_paths))
        X[0] = innov[0] / np.sqrt(1 - self.phi_hour ** 2)
        for t in range(1, T):
            X[t] = self.phi_hour * X[t - 1] + innov[t]

        p = forward + Y[:, mi] + self.shape(index, cf, cannibal_mult) + X.T
        if level_shift is not None:
            p = p + level_shift
        return p

    def to_dict(self):
        return {"beta": self.beta, "shape_offset": self.shape_offset,
                "phi_hour": self.phi_hour, "spike_w": self.spike_w,
                "calm_mu": self.calm_mu, "calm_sd": self.calm_sd,
                "spike_mu": self.spike_mu, "spike_sd": self.spike_sd,
                "phi_month": self.phi_month, "sigma_month": self.sigma_month}
