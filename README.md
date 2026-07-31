# Forecasting India's Minimum Support Prices

Ten forecasting models trained on five years of official MSP data for 28
agricultural commodity-variety series, compared under one identical
rolling-origin protocol — then **tested against the prices the Government of
India actually announced** for the following two seasons.

*Course project — Fundamentals of AI using Agriculture Data Set, ANNAM.AI Centre
of Excellence, Ministry of Education, IIT Ropar.*

---

## Headline result

| | Backtest MAPE | Hold-out MAPE | Hold-out MASE | Within ±5% |
|---|---|---|---|---|
| **RidgePoly2** | 9.79% | **2.77%** | **0.39** | 79% |
| **Drift** | 7.15% | 2.92% | 0.38 | 79% |
| Ensemble | 8.59% | 3.78% | 0.50 | 81% |
| ARIMA(1,1,0) | **2.90%** | 5.05% | 0.59 | 57% |
| Naive baseline | 5.47% | 8.43% | 1.00 | 26% |
| RandomForest | 7.63% | 9.76% | 1.16 | 12% |

**The two error columns disagree, and that is the finding.** ARIMA(1,1,0) wins
the internal backtest and finishes mid-table against real announcements: with
three or four training points per fold it fits the revision noise rather than
the trend. Trend-following models generalise; on 42 published MSP values that no
model ever saw, the best lands under 3% mean error.

Three further results worth stating:

- **Tree ensembles are the wrong tool here.** Random Forest and Gradient Boosting
  predict a constant outside their training range, so they cannot extrapolate a
  trend. Both match or lose to "assume no change" and under-forecast by
  ₹430-500/quintal.
- **Per-crop model selection did not pay.** Picking each series' backtest winner
  (`Selected`, 5.02% MAPE) scored *worse* than applying one robust model
  everywhere. With this little data, selection mostly captures noise.
- **The year labels needed correcting first.** The CSV labels columns by crop
  year, but Rabi MSP is announced per marketing season one year later — the
  column marked "2021-22" for Wheat holds the RMS 2022-23 price (₹2,015).
  Fixing that offset was worth more accuracy than any model choice.

## Deliverables

| Path | What it is |
|---|---|
| `notebooks/MSP_Forecasting_Analysis.ipynb` | Full executed analysis with outputs and figures |
| `outputs/dashboard.html` | Self-contained interactive dashboard (open in any browser) |
| `app/streamlit_app.py` | Interactive explorer — pick any crop, model and horizon |
| `report/` | Project report on the official course template |
| `outputs/figures/` | Seven publication-quality figures |
| `outputs/tables/` | Every result as CSV |

## Quick start

```bash
pip install -r requirements.txt

python run_pipeline.py                  # full pipeline -> outputs/
pytest -q                               # 25 tests
streamlit run app/streamlit_app.py      # interactive explorer
python tools/build_notebook.py          # regenerate + execute the notebook
```

`run_pipeline.py` takes about a minute and writes every figure, table, the JSON
summary and the dashboard.

## Data

**Training** — Minimum Support Prices for agricultural crops, Rajya Sabha
Session 255, Unstarred Question AU-1460, published on
[data.gov.in](https://www.data.gov.in) under the Open Government Data licence.
28 commodity-variety series (17 Kharif, 7 Rabi, 4 other), crop years 2017-18 to
2021-22, ₹ per quintal.

**Hold-out** — `data/external/msp_actuals_holdout.csv`: 42 MSP values announced
by the Cabinet for 2022-23 and 2023-24, each row carrying its PIB citation.
These were never used for fitting or model selection. Series with no
independently verifiable announcement (Jute, De-Husked Coconut, Toria) are
excluded from scoring rather than guessed at.

## How it is put together

```
src/
  config.py        paths, year columns, season-offset semantics
  data_loader.py   ingest, audit, clean, tidy long format, price matrix
  models.py        the ten models behind one shared callable interface
  backtest.py      rolling-origin CV, MAE/RMSE/MAPE/bias/MASE
  forecast.py      final forecasts, hold-out scoring, growth statistics
  visualize.py     seven matplotlib figures
  dashboard.py     self-contained HTML dashboard
run_pipeline.py    orchestrates all of the above
tools/             notebook generator
tests/             pytest suite
```

### Evaluation protocol

Every model is scored by the same expanding-window backtest — no model gets a
gentler protocol than another:

```
train 2017-2019  ->  predict 2020, 2021    (horizons 1 and 2)
train 2017-2020  ->  predict 2021          (horizon 1)
```

Then all five years are used to forecast 2022-23 through 2024-25, and the first
two of those are scored against the published announcements.

MASE (model MAE ÷ naive MAE over identical folds) is reported throughout,
because a raw MAE of ₹170/quintal means nothing until you know that carrying
last season's price forward costs ₹434.

### Models

| Model | Family | Idea |
|---|---|---|
| Naive | baseline | carry last season's MSP forward |
| Drift | baseline | last value + average historical step |
| Linear | parametric | OLS on the year index |
| RidgePoly2 | parametric | degree-2 polynomial with L2 shrinkage |
| CAGR | parametric | constant compound growth (log-linear) |
| Holt | time series | double exponential smoothing with trend |
| ARIMA(1,1,0) | time series | one AR lag on the first difference |
| RandomForest | ensemble | 300 trees on the year index |
| GradBoost | ensemble | boosted stumps |
| Ensemble | combination | median of Drift, Linear, CAGR, Holt |

## Limitations

Five observations per series is the binding constraint and no method escapes it.
The dataset carries no input-cost inflation, no global commodity prices and no
procurement volumes — all of which feed the CACP's actual recommendation. MSP is
set by a committee, so a price-only model can track the policy's momentum but
never anticipate its decisions: the largest hold-out misses (Copra, Nigerseed)
are exactly the commodities that were deliberately re-based toward the "1.5×
cost of production" commitment.

## References

1. Ministry of Agriculture & Farmers Welfare, Government of India — MSP dataset, data.gov.in
2. Commission for Agricultural Costs and Prices (CACP), Price Policy Reports 2017-18 to 2023-24
3. Hyndman, R. J. & Koehler, A. B. (2006). *Another look at measures of forecast accuracy.* International Journal of Forecasting, 22(4), 679-688.
4. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed.
5. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825-2830.
6. Breiman, L. (2001). *Random Forests.* Machine Learning 45(1), 5-32.
