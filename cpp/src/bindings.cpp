// pybind11 bindings: takes plain dicts / numpy arrays from Python
// (see src/prs/engine.py) and returns numpy arrays.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

#include "engine.hpp"

namespace py = pybind11;
using prs::Calendar;
using prs::PowerParams;
using prs::PriceParams;
using prs::WindParams;

namespace {

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int, py::array::c_style | py::array::forcecast>;

std::vector<double> to_vec(const py::handle& h) {
    Arr a = py::cast<Arr>(h);
    return std::vector<double>(a.data(), a.data() + a.size());
}
std::vector<int> to_ivec(const py::handle& h) {
    IArr a = py::cast<IArr>(h);
    return std::vector<int>(a.data(), a.data() + a.size());
}
void copy_to(const py::handle& h, double* dst, size_t n) {
    auto v = to_vec(h);
    if (v.size() != n) throw std::runtime_error("array has wrong size");
    std::copy(v.begin(), v.end(), dst);
}

py::array_t<double> as_matrix(std::vector<double>&& v, int rows, int cols) {
    py::array_t<double> out({rows, cols});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

}  // namespace

py::dict simulate(py::dict cal_d, py::dict wind_d, py::dict power_d, py::dict price_d,
                  int n_paths, std::uint64_t seed, int n_threads) {
    Calendar cal;
    cal.month_of_year = to_ivec(cal_d["month_of_year"]);
    cal.hour_of_day = to_ivec(cal_d["hour_of_day"]);
    cal.month_id = to_ivec(cal_d["month_id"]);
    cal.year_id = to_ivec(cal_d["year_id"]);
    cal.price_shape = to_vec(cal_d["price_shape"]);
    cal.e_shift = to_vec(cal_d["e_shift"]);
    cal.level_shift = to_vec(cal_d["level_shift"]);
    cal.n_hours = static_cast<int>(cal.month_of_year.size());
    cal.n_months = cal.month_id.back() + 1;
    cal.n_years = cal.year_id.back() + 1;

    WindParams w;
    copy_to(wind_d["weibull_k"], w.weibull_k, 12);
    copy_to(wind_d["weibull_c"], w.weibull_c, 12);
    copy_to(wind_d["site_mu"], &w.site_mu[0][0], 288);
    copy_to(wind_d["site_sd"], &w.site_sd[0][0], 288);
    copy_to(wind_d["nat_mu"], &w.nat_mu[0][0], 288);
    copy_to(wind_d["nat_sd"], &w.nat_sd[0][0], 288);
    w.nat_q = to_vec(wind_d["nat_q"]);
    w.n_q = static_cast<int>(w.nat_q.size() / 12);
    w.var_coefs = to_vec(wind_d["var_coefs"]);
    w.var_order = static_cast<int>(w.var_coefs.size() / 4);
    copy_to(wind_d["chol"], &w.chol[0][0], 4);
    w.sigma_month = wind_d["sigma_month"].cast<double>();
    w.sigma_year = wind_d["sigma_year"].cast<double>();
    w.rho_shock = wind_d["rho_shock"].cast<double>();

    PowerParams pw;
    pw.grid_start = power_d["grid_start"].cast<double>();
    pw.grid_step = power_d["grid_step"].cast<double>();
    pw.table_cf = to_vec(power_d["table_cf"]);
    pw.a = power_d["a"].cast<double>();
    pw.b = power_d["b"].cast<double>();
    pw.eff = power_d["eff"].cast<double>();
    pw.capacity_mw = power_d["capacity_mw"].cast<double>();

    PriceParams pr;
    for (auto [key, field] : std::initializer_list<std::pair<const char*, double*>>{
             {"forward", &pr.forward}, {"b1", &pr.b1}, {"b2", &pr.b2},
             {"cannibal_mult", &pr.cannibal_mult}, {"phi_hour", &pr.phi_hour},
             {"spike_w", &pr.spike_w}, {"calm_mu", &pr.calm_mu}, {"calm_sd", &pr.calm_sd},
             {"spike_mu", &pr.spike_mu}, {"spike_sd", &pr.spike_sd},
             {"phi_month", &pr.phi_month}, {"sigma_month", &pr.sigma_month}})
        *field = price_d[key].cast<double>();

    prs::Result r;
    {
        py::gil_scoped_release release;   // let other Python threads run
        r = prs::simulate(cal, w, pw, pr, n_paths, seed, n_threads);
    }
    py::dict out;
    out["volume"] = as_matrix(std::move(r.volume), r.n_paths, r.n_months);
    out["revenue"] = as_matrix(std::move(r.revenue), r.n_paths, r.n_months);
    out["baseload"] = as_matrix(std::move(r.baseload), r.n_paths, r.n_months);
    return out;
}

PYBIND11_MODULE(_engine, m) {
    m.doc() = "C++ Monte Carlo engine for the wind proxy revenue swap";
    m.def("simulate", &simulate, py::arg("calendar"), py::arg("wind"), py::arg("power"),
          py::arg("price"), py::arg("n_paths"), py::arg("seed") = 0, py::arg("n_threads") = 0);
}
