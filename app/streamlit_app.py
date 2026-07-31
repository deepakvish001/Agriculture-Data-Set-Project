"""Interactive MSP forecasting explorer.

    streamlit run app/streamlit_app.py

Lets a user pick any commodity, choose a model, and see the forecast against
the MSP that was actually announced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backtest, config, data_loader, forecast, models  # noqa: E402

st.set_page_config(page_title="MSP Forecasting Explorer", page_icon="🌾", layout="wide")


@st.cache_data(show_spinner="Loading dataset and running the backtest…")
def load():
    data = data_loader.build()
    folds = backtest.rolling_origin(data["matrix"])
    by_series = backtest.score_by_series(folds)
    best = backtest.best_model_per_series(by_series)
    fc = forecast.forecast_all(data["matrix"])
    fc = forecast.add_selected_strategy(fc, best)
    merged = forecast.evaluate_holdout(fc, data["holdout"])
    return {
        "data": data,
        "overall": backtest.score_overall(folds),
        "by_series": by_series,
        "best": best,
        "forecasts": fc,
        "holdout_scores": forecast.score_holdout(merged),
        "merged": merged,
        "growth": forecast.growth_table(data["matrix"], data["long"]),
    }


S = load()
matrix = S["data"]["matrix"]
holdout = S["data"]["holdout"]

st.title("🌾 Minimum Support Price Forecasting")
st.caption(
    "Rajya Sabha Session 255, AU-1460 (data.gov.in) · "
    "28 commodity-variety series · 10 models · validated against announced MSP"
)

top = S["holdout_scores"].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Series modelled", matrix.shape[1])
c2.metric("Hold-out forecasts scored", int(S["merged"]["Model"].value_counts().iloc[0]))
c3.metric(f"Best model ({top['Model']})", f"{top['MAPE']:.2f}% MAPE")
c4.metric("Beats naive baseline by", f"{(1 - top['MASE']) * 100:.0f}%")

tab_crop, tab_models, tab_table, tab_growth = st.tabs(
    ["Per-crop forecast", "Model comparison", "All forecasts", "Growth analysis"]
)

with tab_crop:
    left, right = st.columns([1, 3])
    with left:
        series = st.selectbox("Commodity", sorted(matrix.columns), index=list(sorted(matrix.columns)).index("Wheat"))
        auto = S["best"].set_index("Series").loc[series, "Best_Model"]
        model = st.selectbox(
            "Model", list(models.REGISTRY),
            index=list(models.REGISTRY).index(auto),
            help=f"Backtest picked {auto} for this crop.",
        )
        horizon = st.slider("Seasons to forecast", 1, 5, 3)

    years = matrix.index.to_numpy(dtype=int)
    values = matrix[series].to_numpy(dtype=float)
    future = np.arange(years[-1] + 1, years[-1] + 1 + horizon)
    preds = np.asarray(models.REGISTRY[model](years, values, future), dtype=float)

    hist = pd.DataFrame({"Year": years, "Observed": values}).set_index("Year")
    fut = pd.DataFrame({"Year": future, f"Forecast ({model})": preds}).set_index("Year")
    act = holdout[holdout["Series"] == series].set_index("Year")["Actual_MSP"]
    chart = pd.concat([hist, fut, act.rename("Announced MSP")], axis=1)

    with right:
        st.line_chart(chart, height=380)

    if len(act):
        rows = []
        for year, actual in act.items():
            if year in fut.index:
                pred = fut.loc[year].iloc[0]
                rows.append({
                    "Season": f"{year}-{str(year + 1)[-2:]}",
                    "Forecast (₹)": round(pred),
                    "Announced (₹)": int(actual),
                    "Error (₹)": round(pred - actual),
                    "Error (%)": round(abs(pred - actual) / actual * 100, 2),
                })
        st.subheader("Scored against the MSP actually announced")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("No published hold-out MSP for this series — forecast shown without scoring.")

with tab_models:
    st.subheader("Backtest rank vs hold-out rank")
    st.markdown(
        "The internal walk-forward backtest and the real hold-out disagree. "
        "With only three or four training points per fold, flexible models fit "
        "the revision noise; the trend-following models generalise better to "
        "seasons nobody has seen."
    )
    comp = (
        S["overall"][["Model", "MAPE", "MASE"]].rename(columns={"MAPE": "Backtest MAPE %", "MASE": "Backtest MASE"})
        .merge(
            S["holdout_scores"][["Model", "MAPE", "MAE", "Bias", "MASE", "Within_5pct"]].rename(
                columns={"MAPE": "Hold-out MAPE %", "MAE": "Hold-out MAE ₹",
                         "Bias": "Bias ₹", "MASE": "Hold-out MASE", "Within_5pct": "Within ±5% (%)"}
            ),
            on="Model", how="right",
        )
        .sort_values("Hold-out MAPE %")
    )
    st.dataframe(comp.round(2), hide_index=True, use_container_width=True)
    st.bar_chart(comp.set_index("Model")["Hold-out MAPE %"], height=340)

with tab_table:
    st.subheader("Forecasts for every commodity")
    sel = forecast.selected_forecast(S["forecasts"], S["best"])
    st.dataframe(sel.round(0), hide_index=True, use_container_width=True)
    st.download_button(
        "Download forecasts (CSV)", sel.to_csv(index=False).encode(),
        "msp_forecasts.csv", "text/csv",
    )

with tab_growth:
    st.subheader("Five-year growth, 2017-18 → 2021-22")
    g = S["growth"]
    st.bar_chart(g.set_index("Series")["CAGR_pct"], height=520)
    st.dataframe(g.round(2), hide_index=True, use_container_width=True)

st.caption(
    "Hold-out actuals sourced from PIB Cabinet MSP announcements; per-value "
    "citations in data/external/msp_actuals_holdout.csv"
)
