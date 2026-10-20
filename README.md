# wind-prs

How much should a hedge provider pay for the output of a Welsh wind farm,
and how much risk is it taking on?

This project prices a proxy revenue swap (PRS) for Pen y Cymoedd, a 228 MW
onshore wind farm near Neath. In a PRS the farm gives up its *proxy
revenue* in exchange for a fixed payment. Proxy revenue is what the turbines
should have earned given the measured wind and an agreed efficiency,
multiplied by the hourly power price. The farm keeps its operational risk
(breakdowns, curtailment); the hedge provider takes the weather and price
risk.

To get there I built a stochastic model of site and national wind, a GB
power price model linked to national wind, a Monte Carlo in Python, and a
multithreaded C++ version of the same Monte Carlo.

## Results

Base case: a 10-year deal starting January 2027, settled monthly, with a
GBP 75/MWh baseload forward.

| | |
|---|---|
| Expected proxy volume | 630 GWh/yr (31.5% capacity factor); P90 is 574 GWh |
| Volume-weighted baseload price | GBP 76.7/MWh |
| Fair fixed price | GBP 67.9/MWh, i.e. a capture rate of 88.6% |
| Risk premium (8% hurdle rate on 99% ES capital) | GBP 1.1/MWh, so a quote of about GBP 66.8/MWh |
| Hedge provider's lifetime risk, 99% ES | GBP 51.5m unhedged, GBP 23.3m with a baseload forward hedge |
| Worst stress (calm winter during a gas shock, hedged) | about -GBP 13m in year 1 |
| C++ engine | 10,000 ten-year hourly paths in about 27 s; 5x Python on one thread, about 25x on 16 |

The fair price comes in about GBP 9/MWh below baseload. That's
cannibalisation: the farm generates most when the rest of the GB wind fleet
does, and those are the hours when power is cheapest.

The stress tests surprised me a little. A gas price shock on its own barely
touches a hedged provider. A calm winter on its own costs a few million. Put
them together, though, and the provider has sold forward power the farm
never generates, then has to buy it back at crisis prices. That combination
loses more than twice as much as the calm winter alone.

## How it works

Data. ERA5 100 m wind on a 3x3 grid around the site, 1940-2026 (models are
fitted on 1991-2025). From Elexon: metered output for BMU `T_PNYCW-1`
(B1610, only available from February 2019), bid acceptances (BOALF, used to
remove curtailed hours), Market Index prices (MID) and national wind
generation (FUELHH).

Power curve, `power.py` (notebook 02). The turbine curve comes from
P = 1/2 rho A cp(v) v^3, smoothed to farm level, with a linear correction
for ERA5's bias and an efficiency factor. On 2023-2025 data the fit never
saw, it gets hourly R2 0.85, monthly R2 0.95 and a typical monthly error of
about 6%.

Wind, `wind.py` (notebook 03). Monthly Weibull distributions are mapped to
a standard normal and the seasonal and daily cycles are taken out. Site and
national wind then follow a bivariate VAR(24) with correlated shocks
(effectively a Gaussian copula). On top of that sits a monthly "weather
regime" shock, sized so monthly and annual variability match 35 years of
ERA5.

Price, `price.py` (notebook 04). Additive rather than log-based, since GB
prices go negative. It's the forward level, plus a monthly AR(1) drift, plus
a month/hour/weekday shape, plus a quadratic in national wind (the
cannibalisation term), plus an hourly AR(1) whose shocks switch between a
calm regime and a spike regime.

Monte Carlo, `pricing.py` and `cpp/` (notebook 06). A readable Python
version and a C++17 engine (Eigen, `<random>`, `std::thread`, pybind11,
CMake). Every path seeds its own random generator, so the C++ results don't
change with the number of threads.

Pricing and risk, `pricing.py` (notebook 05). The fair fixed price, hedged
and unhedged VaR and ES, a cost-of-capital risk premium, and stress tests: a
still winter calibrated on 1963/64, a 2022-style gas shock and steeper
cannibalisation. The notebook ends with a section on GB farms reaching the
end of Renewables Obligation support.

## Repository layout

```
src/prs/        data.py (fetchers, loaders), power.py, wind.py, price.py,
                pricing.py, engine.py (C++ front end), pipeline.py (fit + cache), plotting.py
cpp/            C++ engine: src/engine.{hpp,cpp}, src/bindings.cpp, CMakeLists.txt, build.ps1
notebooks/      01 explore, 02 power curve, 03 wind, 04 price, 05 pricing & risk, 06 C++ benchmark
tests/          pytest: unit tests, model-vs-data checks, C++ vs Python agreement
```

## Running it yourself

```powershell
conda env create -f environment.yml          # or: conda env update -f environment.yml --prune
conda activate prs

# 1. Download data. ERA5 needs a CDS API key in ~/.cdsapirc; Elexon needs none
python -c "import sys; sys.path.insert(0,'src'); from prs.data import download_era5_site, build_hourly_panel; download_era5_site(); build_hourly_panel()"

# 2. Build the C++ engine (needs MSVC Build Tools with the C++ workload)
powershell -ExecutionPolicy Bypass -File cpp\build.ps1

# 3. Run the tests, then the notebooks in order
python -m pytest -q
```

Downloaded data are cached under `data/`, which isn't committed. Fitted
models are cached in `data/processed/models.pkl`. If you change a model,
delete that file or call `load_models(refresh=True)`.

## Limitations

The full list is at the end of notebook 05. The main ones:
- The price shape and wind slope are fitted on just 2.5 post-crisis years,
  and the slope is held fixed.
- There's no random long-run gas price factor; gas only enters through a
  stress test.
- FUELHH leaves out embedded wind.
- Extreme price spikes above GBP 300/MWh are under-represented.

## Data sources and attribution

This repository contains no raw data; the code downloads everything from
the original providers. The notebook figures and tables are derived from:

- Elexon BMRS (B1610 metered output, BOALF bid-offer acceptances, MID
  Market Index prices, FUELHH generation by fuel type), through the Elexon
  Insights API. Contains BMRS data © Elexon Limited copyright and database
  right 2026, used under the [BMRS data licence](https://www.elexon.co.uk/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/).
  Elexon does not endorse this project.
- ERA5 hourly reanalysis (Hersbach et al., 2020, *Q. J. R. Meteorol. Soc.*
  146, 1999-2049), from the Copernicus Climate Data Store. Generated using
  Copernicus Climate Change Service information 2026. The ERA5 data have
  been modified (interpolated, transformed and bias-corrected). Neither the
  European Commission nor ECMWF is responsible for any use that may be made
  of the Copernicus information or data it contains.
- The windpowerlib turbine library (OpenEnergy Platform), for the power
  coefficient curve of the Siemens SWT113/3200 used in `src/prs/power.py`.
- Public sources, including Wikipedia and The Wind Power database, for farm
  details (capacity, turbine type, hub height).

## Disclaimer

This is an independent student project built from public data. It is not
affiliated with or endorsed by Vattenfall, Pen y Cymoedd Wind Farm Ltd,
Elexon or ECMWF. The prices, risk figures and stress results are
illustrative model outputs, not valuations, offers or investment advice.

## Licence

The code is MIT licensed (see `LICENSE`). That covers the code in this
repository only; the data remain under their providers' licences above.
