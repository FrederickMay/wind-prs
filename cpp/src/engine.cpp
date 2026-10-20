#include "engine.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <random>
#include <thread>

namespace prs {
namespace {

constexpr int kBurnIn = 1000;           // hours discarded to reach stationarity
constexpr double kEps = 1e-12;

// Standard normal CDF via the complementary error function.
inline double norm_cdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

// Linear interpolation on a uniform grid, clamped at both ends (= np.interp).
inline double interp_uniform(double x, double x0, double dx, const double* y, int n) {
    double pos = (x - x0) / dx;
    if (pos <= 0) return y[0];
    if (pos >= n - 1) return y[n - 1];
    int i = static_cast<int>(pos);
    double w = pos - i;
    return y[i] * (1 - w) + y[i + 1] * w;
}

// Seed each path independently so results don't depend on thread count.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

void simulate_path(int path, const Calendar& cal, const WindParams& W,
                   const PowerParams& P, const PriceParams& Q,
                   std::uint64_t seed, Result& out) {
    std::mt19937_64 rng(splitmix64(seed ^ splitmix64(static_cast<std::uint64_t>(path))));
    std::normal_distribution<double> N(0.0, 1.0);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const int p = W.var_order;
    std::vector<Eigen::Matrix2d> A(p);
    for (int l = 0; l < p; ++l)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) A[l](i, j) = W.var_coefs[(l * 2 + i) * 2 + j];
    Eigen::Matrix2d L;
    L << W.chol[0][0], W.chol[0][1], W.chol[1][0], W.chol[1][1];

    // ring buffer of the last p VAR states; hist[(t - k) mod p] = r_{t-k}
    std::vector<Eigen::Vector2d> hist(p, Eigen::Vector2d::Zero());
    int head = 0;                                    // slot of the next state
    auto var_step = [&]() {
        Eigen::Vector2d z(N(rng), N(rng));
        Eigen::Vector2d r = L * z;
        for (int k = 1; k <= p; ++k) r += A[k - 1] * hist[(head - k + p) % p];
        hist[head] = r;
        head = (head + 1) % p;
        return r;
    };
    for (int t = 0; t < kBurnIn; ++t) var_step();

    // slow shocks: monthly and annual, site and national (correlated)
    const double rho = W.rho_shock, s = std::sqrt(1 - rho * rho);
    std::vector<double> sm(cal.n_months), nm(cal.n_months), sy(cal.n_years), ny(cal.n_years);
    for (int m = 0; m < cal.n_months; ++m) {
        double a = N(rng), b = N(rng);
        sm[m] = W.sigma_month * a;
        nm[m] = W.sigma_month * (rho * a + s * b);
    }
    for (int y = 0; y < cal.n_years; ++y) {
        double c = N(rng), d = N(rng);
        sy[y] = W.sigma_year * c;
        ny[y] = W.sigma_year * (rho * c + s * d);
    }

    // price level Y (monthly AR(1)) and hourly AR(1) X
    std::vector<double> Y(cal.n_months);
    Y[0] = Q.sigma_month / std::sqrt(1 - Q.phi_month * Q.phi_month) * N(rng);
    for (int m = 1; m < cal.n_months; ++m) Y[m] = Q.phi_month * Y[m - 1] + Q.sigma_month * N(rng);
    double X = 0.0;
    bool first = true;

    const int n_tab = static_cast<int>(P.table_cf.size());
    const double q_step = 1.0 / (W.n_q - 1);
    double* vol = &out.volume[static_cast<size_t>(path) * cal.n_months];
    double* rev = &out.revenue[static_cast<size_t>(path) * cal.n_months];
    double* base = &out.baseload[static_cast<size_t>(path) * cal.n_months];
    std::vector<int> hours_in_month(cal.n_months, 0);

    for (int t = 0; t < cal.n_hours; ++t) {
        const int mo = cal.month_of_year[t], h = cal.hour_of_day[t];
        const int mi = cal.month_id[t], yi = cal.year_id[t];
        Eigen::Vector2d r = var_step();
        const double shift = cal.e_shift.empty() ? 0.0 : cal.e_shift[t];

        // site wind: Gaussian -> uniform -> Weibull quantile (closed form)
        double e_s = r(0) + sm[mi] + sy[yi] + shift;
        double u = norm_cdf(e_s * W.site_sd[mo][h] + W.site_mu[mo][h]);
        u = std::min(u, 1.0 - kEps);
        double ws = W.weibull_c[mo] * std::pow(-std::log1p(-u), 1.0 / W.weibull_k[mo]);

        // national CF: Gaussian -> uniform -> empirical quantile table
        double e_n = r(1) + nm[mi] + ny[yi] + shift;
        double un = norm_cdf(e_n * W.nat_sd[mo][h] + W.nat_mu[mo][h]);
        double cf = interp_uniform(un, 0.0, q_step, &W.nat_q[static_cast<size_t>(mo) * W.n_q], W.n_q);

        // proxy generation
        double v = P.a * ws + P.b;
        double mw = P.capacity_mw * P.eff * interp_uniform(v, P.grid_start, P.grid_step, P.table_cf.data(), n_tab);

        // price: two-regime innovations, AR(1)
        bool spike = U(rng) < Q.spike_w;
        double z = N(rng);
        double innov = spike ? Q.spike_mu + Q.spike_sd * z : Q.calm_mu + Q.calm_sd * z;
        X = first ? innov / std::sqrt(1 - Q.phi_hour * Q.phi_hour) : Q.phi_hour * X + innov;
        first = false;
        double price = Q.forward + Y[mi] + cal.price_shape[t]
                     + Q.cannibal_mult * (Q.b1 * cf + Q.b2 * cf * cf) + X;
        if (!cal.level_shift.empty()) price += cal.level_shift[t];

        vol[mi] += mw;
        rev[mi] += mw * price;
        base[mi] += price;
        ++hours_in_month[mi];
    }
    for (int m = 0; m < cal.n_months; ++m) base[m] /= hours_in_month[m];
}

}  // namespace

Result simulate(const Calendar& cal, const WindParams& wind, const PowerParams& power,
                const PriceParams& price, int n_paths, std::uint64_t seed, int n_threads) {
    Result out;
    out.n_paths = n_paths;
    out.n_months = cal.n_months;
    const size_t n = static_cast<size_t>(n_paths) * cal.n_months;
    out.volume.assign(n, 0.0);
    out.revenue.assign(n, 0.0);
    out.baseload.assign(n, 0.0);

    if (n_threads <= 0) n_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    n_threads = std::min(n_threads, n_paths);

    // each thread takes every n_threads-th path; paths write to disjoint rows
    auto worker = [&](int tid) {
        for (int path = tid; path < n_paths; path += n_threads)
            simulate_path(path, cal, wind, power, price, seed, out);
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();
    return out;
}

}  // namespace prs
