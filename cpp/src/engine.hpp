// Monte Carlo engine for the wind proxy revenue swap.
//
// Mirrors the Python reference (src/prs/wind.py, price.py, pricing.py)
// step by step. Everything deterministic (calendar, intraday price shape,
// stress shifts) is precomputed in Python and passed in as arrays, so this
// file only contains the random part of the model.
//
// Paths are independent, so they are split across threads. Each path has
// its own random generator seeded from (seed, path index), which makes
// results identical whatever the number of threads.
#pragma once

#include <cstdint>
#include <vector>

namespace prs {

struct Calendar {
    int n_hours = 0;
    int n_months = 0;
    int n_years = 0;
    std::vector<int> month_of_year;   // 0..11 (UTC, for the wind model)
    std::vector<int> hour_of_day;     // 0..23 (UTC)
    std::vector<int> month_id;        // 0..n_months-1, contract month
    std::vector<int> year_id;         // 0..n_years-1
    std::vector<double> price_shape;  // S(t) - offset, GBP/MWh
    std::vector<double> e_shift;      // stress: Gaussian-units wind shift
    std::vector<double> level_shift;  // stress: GBP/MWh added to price
};

struct WindParams {
    double weibull_k[12], weibull_c[12];
    double site_mu[12][24], site_sd[12][24];
    double nat_mu[12][24], nat_sd[12][24];
    std::vector<double> nat_q;        // 12 x n_q, row-major
    int n_q = 0;
    int var_order = 0;
    std::vector<double> var_coefs;    // p x 2 x 2, row-major (A_lag[i][j])
    double chol[2][2];                // Cholesky factor of innovation cov
    double sigma_month = 0, sigma_year = 0, rho_shock = 1;
};

struct PowerParams {
    double grid_start = 0, grid_step = 0.1;
    std::vector<double> table_cf;     // farm curve on the grid
    double a = 1, b = 0, eff = 1, capacity_mw = 228;
};

struct PriceParams {
    double forward = 75;
    double b1 = 0, b2 = 0, cannibal_mult = 1;
    double phi_hour = 0;
    double spike_w = 0, calm_mu = 0, calm_sd = 1, spike_mu = 0, spike_sd = 1;
    double phi_month = 0, sigma_month = 0;
};

struct Result {
    int n_paths = 0, n_months = 0;
    std::vector<double> volume;       // n_paths x n_months, MWh
    std::vector<double> revenue;      // GBP
    std::vector<double> baseload;     // mean hourly price, GBP/MWh
};

Result simulate(const Calendar& cal, const WindParams& wind,
                const PowerParams& power, const PriceParams& price,
                int n_paths, std::uint64_t seed, int n_threads);

}  // namespace prs
