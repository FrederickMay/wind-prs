"""Stochastic model of hourly site wind jointly with the GB national wind factor.

Idea: hourly wind speed is skewed, seasonal and strongly autocorrelated.
We map it to a standard normal variable, remove what is predictable
(season, time of day), and model what is left as a linear Gaussian process.

Site wind (ERA5, fitted 1991-2025):
  1. Marginal: a Weibull distribution for each calendar month (k_m, c_m).
  2. Gaussianise: z = Phi^-1( F_Weibull,m(ws) ), so z ~ N(0,1) per month.
  3. Remove the diurnal cycle: e = (z - mu[month,hour]) / sd[month,hour].

National wind (Elexon capacity factor, fitted on the recent fleet):
  1. Marginal: empirical quantiles of CF for each calendar month.
  2. Same Gaussianise / deseasonalise steps.

Dynamics, in Gaussian units, for site (s) and national (n):
  e_t = r_t + month_shock + year_shock

  r_t     bivariate VAR(24) with correlated innovations, fitted on hours
          after removing each calendar month's mean. Hour-to-day weather.
          Correlated innovations + cross-lags = a Gaussian copula linking
          site and national wind.
  month_shock  one draw per month: "windy month / calm month" weather
          regimes. An hourly VAR forgets within days, so on its own it makes
          monthly totals too similar (20% too little spread). Its variance
          is whatever the VAR leaves unexplained in 35 years of ERA5
          monthly means.
  year_shock   same idea for whole years, sized to match the spread of
          annual means. Matters for a 10-year contract.
  The national shocks are correlated with the site shocks (rho fitted on
  2017-2025 monthly means): weather regimes cover all of GB, but not
  uniformly.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

EPS = 1e-6
N_Q = 201                       # quantile grid size for the national CF marginal
Q_LEVELS = np.linspace(0.0, 1.0, N_Q)
VAR_ORDER = 24                  # one day of lags: matches the 24 h autocorrelation


def _deseason_fit(z, index):
    """Mean and sd of z for each (month, hour) as 12x24 arrays."""
    df = pd.DataFrame({"z": z, "m": index.month, "h": index.hour})
    g = df.groupby(["m", "h"])["z"]
    return g.mean().unstack().values, g.std().unstack().values


def _period_ids(index):
    """Integer month and year ids for each hour (0, 1, 2, ...)."""
    _, mi = np.unique(index.year.values * 12 + index.month.values, return_inverse=True)
    _, yi = np.unique(index.year.values, return_inverse=True)
    return mi, yi


@dataclass
class WindModel:
    weibull_k: np.ndarray = None      # (12,)
    weibull_c: np.ndarray = None      # (12,)
    site_mu: np.ndarray = None        # (12, 24)
    site_sd: np.ndarray = None
    nat_q: np.ndarray = None          # (12, N_Q) CF quantiles per month
    nat_mu: np.ndarray = None
    nat_sd: np.ndarray = None
    var_coefs: np.ndarray = None      # (p, 2, 2) VAR lag matrices
    var_cov: np.ndarray = None        # (2, 2) innovation covariance
    sigma_month: float = 0.0          # sd of monthly shock (Gaussian units)
    sigma_year: float = 0.0           # sd of annual shock
    rho_shock: float = 1.0            # site-national correlation of the shocks
    info: dict = field(default_factory=dict)

    # ------------------------------------------------ transforms
    def site_to_e(self, ws, index):
        m, h = index.month.values, index.hour.values
        u = stats.weibull_min.cdf(ws, self.weibull_k[m - 1], scale=self.weibull_c[m - 1])
        z = stats.norm.ppf(np.clip(u, EPS, 1 - EPS))
        return (z - self.site_mu[m - 1, h]) / self.site_sd[m - 1, h]

    def e_to_site(self, e, month, hour):
        z = e * self.site_sd[month - 1, hour] + self.site_mu[month - 1, hour]
        u = stats.norm.cdf(z)
        # Weibull inverse CDF in closed form: c * (-ln(1-u))^(1/k)
        k, c = self.weibull_k[month - 1], self.weibull_c[month - 1]
        return c * (-np.log1p(-np.clip(u, 0, 1 - 1e-12))) ** (1 / k)

    def nat_to_e(self, cf, index):
        m, h = index.month.values, index.hour.values
        u = np.empty(len(cf))
        for mm in range(1, 13):
            sel = m == mm
            u[sel] = np.interp(cf[sel], self.nat_q[mm - 1], Q_LEVELS)
        z = stats.norm.ppf(np.clip(u, EPS, 1 - EPS))
        return (z - self.nat_mu[m - 1, h]) / self.nat_sd[m - 1, h]

    def e_to_nat(self, e, month, hour):
        z = e * self.nat_sd[month - 1, hour] + self.nat_mu[month - 1, hour]
        u = stats.norm.cdf(z)
        out = np.empty_like(u)
        for mm in np.unique(month):
            sel = month == mm
            out[sel] = np.interp(u[sel], Q_LEVELS, self.nat_q[mm - 1])
        return out

    # ------------------------------------------------ fitting
    @classmethod
    def fit(cls, site_ws, national_cf, var_order=VAR_ORDER,
            nat_marginal_start="2022-01-01"):
        """site_ws: long hourly ERA5 series (e.g. 1991-2025).
        national_cf: hourly national capacity factor (2017+).
        """
        self = cls()
        site_ws = site_ws.dropna()

        # site marginal: Weibull per calendar month (location fixed at 0)
        k, c = np.zeros(12), np.zeros(12)
        for mm in range(1, 13):
            x = site_ws[site_ws.index.month == mm].values
            k[mm - 1], _, c[mm - 1] = stats.weibull_min.fit(np.maximum(x, 0.05), floc=0)
        self.weibull_k, self.weibull_c = k, c

        # Gaussianise and remove (month, hour) means/sds
        m = site_ws.index.month.values
        u = stats.weibull_min.cdf(site_ws.values, k[m - 1], scale=c[m - 1])
        z = stats.norm.ppf(np.clip(u, EPS, 1 - EPS))
        self.site_mu, self.site_sd = _deseason_fit(z, site_ws.index)
        e_site = pd.Series(self.site_to_e(site_ws.values, site_ws.index), site_ws.index)

        # national marginal: empirical quantiles per month on the recent fleet
        nat = national_cf.dropna()
        recent = nat[nat.index >= pd.Timestamp(nat_marginal_start, tz="UTC")]
        self.nat_q = np.array([np.quantile(recent[recent.index.month == mm], Q_LEVELS)
                               for mm in range(1, 13)])
        mn = nat.index.month.values
        u_n = np.empty(len(nat))
        for mm in range(1, 13):
            sel = mn == mm
            u_n[sel] = np.interp(nat.values[sel], self.nat_q[mm - 1], Q_LEVELS)
        z_n = stats.norm.ppf(np.clip(u_n, EPS, 1 - EPS))
        self.nat_mu, self.nat_sd = _deseason_fit(z_n, nat.index)
        e_nat = pd.Series(self.nat_to_e(nat.values, nat.index), nat.index)

        # VAR on overlapping hours, after removing each calendar month's mean
        joint = pd.DataFrame({"site": e_site, "nat": e_nat}).dropna()
        ym = [joint.index.year, joint.index.month]
        month_means = joint.groupby(ym).mean()
        joint = joint - joint.groupby(ym).transform("mean")
        joint = joint.asfreq("h").interpolate(limit=3).dropna()
        res = VAR(joint.values).fit(var_order, trend="n")
        self.var_coefs = res.coefs
        self.var_cov = np.asarray(res.sigma_u)
        self.info["var_order"] = var_order
        self.info["hourly_corr"] = joint.corr().iloc[0, 1]
        self.rho_shock = float(month_means.corr().iloc[0, 1])

        # size the slow shocks from what the VAR alone cannot explain
        vm, vy = self._var_only_spread(n_years=200)
        obs_m = e_site.groupby([e_site.index.year, e_site.index.month]).mean().var()
        full = e_site.groupby(e_site.index.year).count() >= 8700
        obs_y = e_site.groupby(e_site.index.year).mean()[full].var()
        self.sigma_month = float(np.sqrt(max(obs_m - vm ** 2, 0.0)))
        # iid monthly shocks already add sigma_month^2 / 12 to the annual variance
        self.sigma_year = float(np.sqrt(max(obs_y - vy ** 2 - self.sigma_month ** 2 / 12, 0.0)))
        self.info.update(obs_month_sd=float(np.sqrt(obs_m)), var_month_sd=float(vm),
                         obs_year_sd=float(np.sqrt(obs_y)), var_year_sd=float(vy))
        return self

    def _var_only_spread(self, n_years, seed=0):
        """sd of monthly and annual means of r_site from the VAR alone."""
        rng = np.random.default_rng(seed)
        r = self._simulate_var(rng, n_paths=n_years, n_hours=8760)[:, :, 0]
        monthly = r[:, :8760 // 730 * 730].reshape(n_years, 12, 730).mean(2)
        return monthly.std(), r.mean(axis=1).std()

    # ------------------------------------------------ simulation
    def _simulate_var(self, rng, n_paths, n_hours):
        """VAR residuals r, shape (n_paths, n_hours, 2), burned in 1000 hours."""
        p = self.var_coefs.shape[0]
        chol = np.linalg.cholesky(self.var_cov)
        burn = 1000
        total = n_hours + burn
        eps = rng.standard_normal((total, n_paths, 2)) @ chol.T
        # stack lag matrices so one matmul applies all p lags: (p*2, 2)
        A = np.concatenate([self.var_coefs[i].T for i in range(p)], axis=0)
        r = np.zeros((total, n_paths, 2))
        for t in range(p, total):
            hist = r[t - p:t][::-1].transpose(1, 0, 2).reshape(n_paths, 2 * p)
            r[t] = eps[t] + hist @ A
        return r[burn:].transpose(1, 0, 2)

    def draw_shocks(self, rng, n_paths, n_months, n_years):
        """Monthly + annual shocks for site and national, Gaussian units.
        Returns (site_month, nat_month, site_year, nat_year)."""
        rho, s = self.rho_shock, np.sqrt(1 - self.rho_shock ** 2)
        a, b = rng.standard_normal((2, n_paths, n_months))
        c, d = rng.standard_normal((2, n_paths, n_years))
        return (self.sigma_month * a, self.sigma_month * (rho * a + s * b),
                self.sigma_year * c, self.sigma_year * (rho * c + s * d))

    def simulate(self, index, n_paths, seed=None, rng=None, e_shift=None):
        """Simulate site wind (m/s) and national CF on an hourly UTC index.

        Returns two arrays of shape (n_paths, len(index)).
        e_shift: optional (len(index),) shift in Gaussian units added to both
        site and national wind, e.g. a negative shift over one winter for the
        "still winter" stress test.
        """
        rng = rng or np.random.default_rng(seed)
        r = self._simulate_var(rng, n_paths, len(index))
        mi, yi = _period_ids(index)
        sm, nm, sy, ny = self.draw_shocks(rng, n_paths, mi.max() + 1, yi.max() + 1)
        e_s = r[:, :, 0] + sm[:, mi] + sy[:, yi]
        e_n = r[:, :, 1] + nm[:, mi] + ny[:, yi]
        if e_shift is not None:
            e_s, e_n = e_s + e_shift, e_n + e_shift
        month = np.broadcast_to(index.month.values, e_s.shape)
        hour = np.broadcast_to(index.hour.values, e_s.shape)
        return self.e_to_site(e_s, month, hour), self.e_to_nat(e_n, month, hour)

    # ------------------------------------------------ export
    def to_dict(self):
        """Plain arrays, used to pass the fitted model to the C++ engine."""
        return {"weibull_k": self.weibull_k, "weibull_c": self.weibull_c,
                "site_mu": self.site_mu, "site_sd": self.site_sd,
                "nat_q": self.nat_q, "nat_mu": self.nat_mu, "nat_sd": self.nat_sd,
                "var_coefs": self.var_coefs, "var_cov": self.var_cov,
                "sigma_month": self.sigma_month, "sigma_year": self.sigma_year,
                "rho_shock": self.rho_shock}
