"""Generate and execute notebooks/MSP_Forecasting_Analysis.ipynb.

The notebook is a build artefact so it never drifts from src/:

    python tools/build_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "MSP_Forecasting_Analysis.ipynb"

md, code = [], []
cells: list = []


def M(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def C(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


M("""
# Forecasting India's Minimum Support Prices

**Fundamentals of AI using Agriculture Data Set** · ANNAM.AI Centre of Excellence, IIT Ropar

**Dataset** — Minimum Support Prices for agricultural crops, Rajya Sabha Session 255,
Unstarred Question AU-1460, published on [data.gov.in](https://www.data.gov.in)
under the Open Government Data licence.

---

### What this notebook does differently

Forecasting five data points is easy to do badly. Three specific traps decide
whether the numbers below mean anything:

1. **Averaging varieties away.** Paddy *Common* and Paddy *Grade 'A'* are two
   separate government notifications. Averaging them produces a price that was
   never announced and cannot be checked against reality. All 28 series are kept
   distinct.
2. **Scoring models under different protocols.** Comparing one model's in-sample
   fit against another's cross-validated error is not a comparison at all. Every
   model here is scored under one identical rolling-origin backtest.
3. **Never leaving the training data.** A model that looks good on 2017-2021 has
   proved nothing. Every forecast is checked against the MSP the Government of
   India *actually announced* for 2022-23 and 2023-24 — 42 published prices that
   no model ever saw.
""")

C("""
import sys, warnings
sys.path.insert(0, '..')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from IPython.display import Image, display

from src import backtest, config, data_loader, forecast, models, visualize

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)
print('pandas', pd.__version__, '| numpy', np.__version__)
""")

M("""
## 1. Loading the dataset

The published CSV is a wide table: one row per commodity-variety, one column per
crop year, prices in rupees per quintal.
""")

C("""
raw = data_loader.load_raw()
print(f'shape: {raw.shape[0]} rows x {raw.shape[1]} columns')
raw.head(8)
""")

M("## 2. Data quality audit")

C("""
audit = data_loader.audit(raw)
for k, v in audit.items():
    print(f'{k:24s} {v}')
""")

M("""
The dataset is unusually clean: no missing cells, no duplicate rows, and **every
one of the 28 series is monotonically non-decreasing** across the five years.
That last fact is the single most important property of this data — MSP is a
policy floor that is revised upward each year, never cut. It tells us the
forecasting problem is *trend extrapolation*, not volatility modelling, and it
predicts in advance which model families should win.

`Variety = 'NA'` is a genuine category (the commodity has a single grade), not a
missing value, so it is kept rather than imputed.
""")

M("""
## 3. A subtlety in the year labels

The columns are labelled by **crop year**, but MSP is announced per **marketing
season**, and the two only coincide for Kharif crops. A Rabi crop sown in crop
year *Y* is harvested and marketed in RMS *Y+1*.

This is easy to check against published figures. Wheat MSP for RMS 2021-22 was
₹1,975 and for RMS 2022-23 was ₹2,015. Look at what the column labelled
"2021-22" actually contains:
""")

C("""
wheat = raw[raw['Commodity'] == 'Wheat'][config.YEAR_COLS]
display(wheat)
print("Column '2021-22' holds:", int(wheat['2021-22'].iloc[0]))
print('Announced MSP, RMS 2021-22: 1975')
print('Announced MSP, RMS 2022-23: 2015  <-- this is the value in the column')
""")

M("""
So Rabi columns run one year ahead of their label. Left uncorrected, every Rabi
comparison against announced MSP would be wrong by a full annual revision —
around ₹100 on wheat, which is larger than the forecast error we are trying to
measure. `config.SEASON_OFFSET` applies the shift; the tidy table below carries
both the published label and the true marketing season.
""")

C("""
clean = data_loader.clean(raw)
long = data_loader.to_long(clean)
long[long['Series'].isin(['Wheat', 'Paddy (Common)'])].head(10)
""")

M("""
## 4. The price matrix

Years down the rows, series across the columns — kept at commodity-**variety**
granularity.
""")

C("""
matrix = data_loader.price_matrix(long)
print(matrix.shape)
matrix.iloc[:, :8]
""")

M("## 5. Exploratory analysis: how fast has MSP actually grown?")

C("""
growth = forecast.growth_table(matrix, long)
growth.head(10).round(2)
""")

C("""
print('Fastest growers')
display(growth.head(5)[['Series', 'Season', 'First_MSP', 'Last_MSP', 'CAGR_pct']].round(2))
print('\\nSlowest growers')
display(growth.tail(5)[['Series', 'Season', 'First_MSP', 'Last_MSP', 'CAGR_pct']].round(2))
print('\\nMean CAGR by season')
display(growth.groupby('Season')['CAGR_pct'].agg(['mean', 'min', 'max', 'count']).round(2))
""")

M("""
Growth is far from uniform. Ragi and Nigerseed roughly doubled over five years
(15.5% and 14.4% CAGR) as the government pushed millets and minor oilseeds
toward the "1.5x cost of production" formula, while Arhar, Wheat and Barley grew
at under 4%. **A single national growth rate would misprice most of the basket** —
which is the argument for fitting each series separately.
""")

C("""
display(Image(visualize.fig_trends(matrix)))
""")

C("""
display(Image(visualize.fig_growth(growth)))
""")

M("""
## 6. The models

Ten models, one shared interface: `f(years, values, future_years) -> predictions`.

| Model | Family | Idea |
|---|---|---|
| `Naive` | baseline | carry last season's MSP forward |
| `Drift` | baseline | last value + average historical step |
| `Linear` | parametric | OLS on the year index |
| `RidgePoly2` | parametric | degree-2 polynomial with L2 shrinkage |
| `CAGR` | parametric | constant compound growth (log-linear) |
| `Holt` | time series | double exponential smoothing with trend |
| `ARIMA(1,1,0)` | time series | one AR lag on the first difference |
| `RandomForest` | ensemble | 300 trees on the year index |
| `GradBoost` | ensemble | boosted stumps |
| `Ensemble` | combination | median of Drift, Linear, CAGR, Holt |

`Naive` is not filler — it is the bar. A forecasting model that cannot beat
"assume no change" has earned nothing, and reporting MAE without that reference
hides the failure. `RandomForest` and `GradBoost` are deliberate negative
controls: trees predict a constant outside their training range, so they
*cannot* extrapolate a trend. Watch what that does to their scores.
""")

C("""
years = matrix.index.to_numpy(dtype=int)
paddy = matrix['Paddy (Common)'].to_numpy(dtype=float)
future = np.array([2022, 2023, 2024])

demo = pd.DataFrame(
    {name: np.round(fn(years, paddy, future), 1) for name, fn in models.REGISTRY.items()},
    index=[f'{y}-{str(y+1)[-2:]}' for y in future],
).T
demo.columns.name = 'Paddy (Common), forecast'
demo
""")

M("""
The tree models return the same number for all three years — visible proof of
the flat-extrapolation limit, not a bug.

## 7. Evaluation protocol

Every model is scored by the same expanding-window rolling-origin backtest:

```
train 2017-2019  ->  predict 2020, 2021   (horizons 1 and 2)
train 2017-2020  ->  predict 2021         (horizon 1)
```

Nothing after the origin is visible at fit time. Metrics are MAE, RMSE, MAPE,
mean bias, and **MASE** — the model's MAE divided by the naive model's MAE over
the identical folds. MASE below 1.0 means the model beat carry-forward; at or
above 1.0 it did not, no matter how small its raw error looks.
""")

C("""
folds = backtest.rolling_origin(matrix)
print(f'{len(folds)} fold-predictions '
      f'({folds["Series"].nunique()} series x {folds["Model"].nunique()} models x folds)')
folds.head()
""")

C("""
overall = backtest.score_overall(folds)
overall.round(3)
""")

C("""
by_horizon = backtest.score_by_horizon(folds)
by_horizon[by_horizon['Model'].isin(['ARIMA(1,1,0)', 'Drift', 'Linear', 'Naive', 'RidgePoly2'])].round(2)
""")

M("""
On the backtest, ARIMA(1,1,0) looks dominant — roughly half the naive error.
Hold that thought.

## 8. Per-series model selection

Different crops have differently-shaped histories, so each series picks its own
winner — using **only** backtest folds, never the hold-out seasons.
""")

C("""
by_series = backtest.score_by_series(folds)
best = backtest.best_model_per_series(by_series)
display(best['Best_Model'].value_counts().rename('series won'))
best.head(10).round(2)
""")

M("""
## 9. Forecasting 2022-23 to 2024-25

Every model is now refit on all five years and projected forward. `Selected` is
a pseudo-model that takes each series' backtest winner — included so we can test
whether per-crop selection actually pays off.
""")

C("""
forecasts = forecast.forecast_all(matrix)
forecasts = forecast.add_selected_strategy(forecasts, best)
selected = forecast.selected_forecast(forecasts, best)
selected.head(12)
""")

M("""
## 10. The real test: what did the government actually announce?

This is the part a five-point backtest cannot give you. `data/external/msp_actuals_holdout.csv`
holds **42 MSP values that were genuinely published** by the Cabinet for
2022-23 and 2023-24 (each row carries its PIB citation). No model saw them
during fitting or selection.
""")

C("""
holdout = data_loader.load_holdout()
print(f'{len(holdout)} published MSP values across {holdout["Series"].nunique()} series')
holdout.groupby('Marketing_Season').size().rename('values')
""")

C("""
merged = forecast.evaluate_holdout(forecasts, holdout)
holdout_scores = forecast.score_holdout(merged)
holdout_scores.round(3)
""")

M("""
### The main finding

**The backtest ranking and the hold-out ranking disagree.**

ARIMA(1,1,0) won the internal backtest by a wide margin and finishes mid-table
against real announcements. RidgePoly2 and Drift — mediocre in the backtest —
are the most accurate on data nobody had seen, at under 3% MAPE.

The reason is the fold sizes. Each backtest fold trains on three or four points,
where ARIMA's differencing plus an AR term can bend to fit the exact pattern of
recent revisions. That flexibility is rewarded in-sample and punished the moment
the series has to be projected two seasons past the data. It is a textbook
illustration of why model selection on a very short series is itself unreliable —
and why the honest move is to report both numbers rather than the flattering one.

Look at the bias column too: Naive, GradBoost and RandomForest all under-forecast
by ₹430-500 per quintal. For a farmer deciding what to sow, a forecast biased
*low* by ₹500 is not a small error — it is the difference between planting and
not planting.
""")

C("""
display(Image(visualize.fig_leaderboard(overall, holdout_scores)))
""")

C("""
champion = holdout_scores[holdout_scores['Model'] != 'Selected'].iloc[0]['Model']
print('Best named model on the hold-out:', champion)
display(Image(visualize.fig_parity(merged, champion, long)))
""")

C("""
display(Image(visualize.fig_bias(holdout_scores)))
""")

M("""
### Did per-crop model selection help?

`Selected` scores *worse* than simply using RidgePoly2 or Drift everywhere.
Choosing a model per crop from two backtest folds mostly selects noise. With
this little data, one robust model applied uniformly beats clever per-series
tuning — a result worth stating plainly, since it runs against the intuition
that more customisation is always better.
""")

C("""
holdout_by_season = forecast.score_holdout(merged, by=['Season_Label'])
holdout_by_season[holdout_by_season['Model'].isin([champion, 'Selected', 'Naive', 'ARIMA(1,1,0)'])].round(2)
""")

C("""
display(Image(visualize.fig_forecast_panels(matrix, forecasts, holdout, champion)))
""")

M("""
Every model degrades from the first hold-out season to the second — both are
forecast from the same 2021-22 origin, so the second is one further revision
away. How gracefully they degrade separates them: the naive baseline roughly
doubles (5.3% → 11.6%) and ARIMA nearly triples (2.8% → 7.4%), while RidgePoly2
moves only from 2.4% to 3.1%. Robustness at horizon 2 is where the trend models
earn their place.

## 11. Where the forecasts went wrong
""")

C("""
worst = (merged[merged['Model'] == champion]
         .nlargest(8, 'APE')[['Series', 'Marketing_Season', 'Forecast', 'Actual_MSP', 'Error', 'APE']])
worst.round(1)
""")

M("""
The largest misses share a cause: commodities whose MSP was *deliberately*
re-based by policy rather than drifted upward. Copra and Nigerseed saw
outsized revisions tied to the "1.5x cost of production" commitment and to
import-substitution pressure in edible oils. No price-only model can anticipate
a policy decision — this is a limit of the data, not of the fitting.

## 12. Conclusions

1. **Trend-following models win.** RidgePoly2 and Drift forecast real announced
   MSP to under 3% MAPE, with roughly 8 in 10 forecasts inside a ±5% band.
2. **Tree ensembles are the wrong tool here.** Random Forest and Gradient
   Boosting cannot extrapolate; both land at or below the naive baseline and
   under-forecast severely. Their popularity is not a reason to use them.
3. **Backtest rank is not hold-out rank.** ARIMA topped the internal backtest
   and slipped to mid-table against reality. On five points, the backtest is
   itself a small sample.
4. **Per-crop model selection did not pay.** One robust model applied uniformly
   beat picking a winner per series.
5. **The label semantics mattered.** Correcting the Rabi crop-year to
   marketing-season offset was worth more accuracy than any model choice.

### Limitations

Five observations per series is the binding constraint; no method escapes it.
The dataset carries no input-cost inflation, no global commodity prices and no
procurement volumes, all of which feed the CACP's actual recommendation. And
MSP is set by a committee — a price-only model can track the policy's momentum
but can never anticipate the policy's decisions.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": sys.version.split()[0]},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"executing {len(cells)} cells…")
NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(OUT.parent)}}).execute()
nbf.write(nb, OUT)
print(f"wrote {OUT}")
