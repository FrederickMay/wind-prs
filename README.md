# wind-prs

Pricing a **wind farm proxy revenue swap (PRS)** for Pen y Cymoedd, a 228 MW
onshore wind farm in South Wales, with correlated stochastic models of wind
and GB power prices, a Python reference Monte Carlo and a multithreaded C++
engine.

A PRS swaps the farm's uncertain *proxy revenue* (what the turbines should
earn given the measured wind and an agreed efficiency, times the hourly
power price) for a fixed payment from a hedge provider. The farm keeps
operational risk; the hedge provider takes weather, price and shape risk.
The questions: **what fixed price is fair, and how much risk does the hedge
provider carry?**

## Results (10-year deal from Jan 2027, monthly settlement, GBP 75/MWh baseload forward)

| | |
|---|---|
| Expected proxy volume | 630 GWh/yr (capacity factor 31.5%), P90 574 GWh |
| Volume-weighted baseload price | GBP 76.7/MWh |
| **Fair fixed price** | **GBP 67.9/MWh** (capture rate 88.6%) |
| Risk premium (8% hurdle on 99% ES capital) | GBP 1.1/MWh, quoted price ~GBP 66.8/MWh |
| Hedge provider lifetime PV risk, 99% ES | GBP 51.5m unhedged, GBP 23.3m with baseload forward hedge |
| Worst stress (calm winter + gas shock, hedged) | -GBP 13.1m mean year-1 P&L |
| C++ engine | 10,000 ten-year hourly paths in ~27 s; ~5x Python on 1 thread, ~25x on 16 |

The fair price sits ~GBP 9/MWh below baseload because of **cannibalisation**:
the farm generates most when the whole GB wind fleet does, which is when
prices are lowest. The most dangerous scenario for a hedged provider is a
still winter during a gas price spike: it sold forward the power it expected
the farm to make, the farm doesn't make it, and it buys back at crisis prices.

## Method

1. **Data.** ERA5 100 m wind on a 3x3 grid around the site (1940-2026,
   calibrated on 1991-2025); Elexon metered output (B1610, BMU `T_PNYCW-1`,
   from Feb 2019), bid acceptances (BOALF, to remove curtailed hours),
   Market Index prices (MID) and national wind generation (FUELHH).
2. **Power curve** (`power.py`). Turbine curve from P = 1/2 rho A cp(v) v^3,
   smoothed to farm level, with a linear ERA5 bias correction and an
   efficiency factor. Out of sample (2023-25): hourly R2 0.85, monthly R2
   0.95, mean monthly error 6%. [notebook 02]
3. **Wind** (`wind.py`). Monthly Weibull marginals mapped to Gaussian,
   seasonal/diurnal cycle removed, bivariate VAR(24) of site and national
   wind with correlated innovations (a Gaussian copula), plus monthly
   weather-regime shocks sized so monthly and annual variability match 35
   years of ERA5. [notebook 03]
4. **Price** (`price.py`). Arithmetic (GB prices go negative):
   forward level + monthly AR(1) level + month/hour/weekday shape +
   quadratic in national wind (cannibalisation) + hourly AR(1) with a
   two-regime (calm/spike) innovation mixture. [notebook 04]
5. **Monte Carlo** (`pricing.py`, `cpp/`). Python reference and a C++17
   engine (Eigen, `<random>`, `std::thread`, pybind11, CMake). Per-path
   seeding makes results identical for any thread count. [notebook 06]
6. **Pricing and risk** (`pricing.py`). Fair fixed price, hedged and
   unhedged VaR/ES, cost-of-capital risk premium, stress tests (still
   winter calibrated on the 1963/64 winter, 2022-style gas shock, stronger
   cannibalisation), and a policy discussion of GB farms coming off
   Renewables Obligation support. [notebook 05]

## Repository

```
src/prs/        data.py (fetchers, loaders), power.py, wind.py, price.py,
                pricing.py, engine.py (C++ front end), pipeline.py (fit + cache), plotting.py
cpp/            C++ engine: src/engine.{hpp,cpp}, src/bindings.cpp, CMakeLists.txt, build.ps1
notebooks/      01 explore, 02 power curve, 03 wind, 04 price, 05 pricing & risk, 06 C++ benchmark
tests/          pytest: unit tests, model-vs-data checks, C++ vs Python agreement
```

## Reproduce

```powershell
conda env create -f environment.yml          # or: conda env update -f environment.yml --prune
conda activate prs

# 1. ERA5 needs a CDS API key in ~/.cdsapirc; Elexon needs no key
python -c "import sys; sys.path.insert(0,'src'); from prs.data import download_era5_site, build_hourly_panel; download_era5_site(); build_hourly_panel()"

# 2. C++ engine (needs MSVC Build Tools with the C++ workload)
powershell -ExecutionPolicy Bypass -File cpp\build.ps1

# 3. Tests, then run the notebooks in order
python -m pytest -q
```

Data are cached under `data/` (not committed). Fitted models are cached in
`data/processed/models.pkl`; delete it (or call `load_models(refresh=True)`)
after changing a model.

## Limitations

See the end of notebook 05. The main ones: price shape and cannibalisation
fitted on 2.5 post-crisis years with a fixed slope; no stochastic multi-year
gas price factor (handled as a stress test); FUELHH excludes embedded wind;
extreme (>GBP 300/MWh) price spikes are under-represented.

## Data sources and attribution

No raw data is stored in this repository; the code downloads it from the
original providers. Figures and tables in the notebooks are derived from:

- **Elexon BMRS** (metered output B1610, bid-offer acceptances BOALF, Market
  Index prices MID, generation by fuel type FUELHH), via the Elexon Insights
  API. Contains BMRS data © Elexon Limited copyright and database right 2026,
  used under the [BMRS data licence](https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/).
  Elexon does not endorse this project.
- **ERA5** hourly reanalysis (Hersbach et al., 2020, *Q. J. R. Meteorol. Soc.*
  146, 1999-2049), from the Copernicus Climate Data Store. Generated using
  Copernicus Climate Change Service information 2026. The ERA5 data have been
  modified (interpolated, transformed and bias-corrected). Neither the
  European Commission nor ECMWF is responsible for any use that may be made
  of the Copernicus information or data it contains.
- **windpowerlib** turbine library (OpenEnergy Platform): the power
  coefficient curve of the Siemens SWT113/3200 in `src/prs/power.py` is taken
  from it.
- Farm details (capacity, turbine type, hub height) from public sources
  including Wikipedia and The Wind Power database.

## Disclaimer

This is an independent student project built from public data. It is not
affiliated with or endorsed by Vattenfall, Pen y Cymoedd Wind Farm Ltd,
Elexon or ECMWF. Prices, risk figures and stress results are illustrative
model outputs, not valuations, offers or investment advice.

## Licence

Code: MIT (see `LICENSE`). The MIT licence covers this repository's code
only; third-party data remain under their providers' licences above.
