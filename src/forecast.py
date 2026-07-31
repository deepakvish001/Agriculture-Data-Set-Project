"""Final forecasts and the out-of-sample hold-out evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import _metrics
from .models import REGISTRY


def forecast_all(matrix: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """Refit every model on the full history and project the future seasons."""
    labels = labels or config.FORECAST_LABELS
    future_years = np.array([int(l.split("-")[0]) for l in labels])
    years = matrix.index.to_numpy(dtype=int)

    rows = []
    for series in matrix.columns:
        values = matrix[series].to_numpy(dtype=float)
        if np.isnan(values).any():
            continue
        for name, fn in REGISTRY.items():
            preds = np.asarray(fn(years, values, future_years), dtype=float)
            for label, year, pred in zip(labels, future_years, preds):
                rows.append(
                    {
                        "Series": series,
                        "Model": name,
                        "Season_Label": label,
                        "Year": int(year),
                        "Forecast": round(float(pred), 1),
                    }
                )
    return pd.DataFrame(rows)


def selected_forecast(forecasts: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Keep only each series' backtest-selected model, wide by season."""
    merged = forecasts.merge(best[["Series", "Best_Model"]], on="Series")
    picked = merged[merged["Model"] == merged["Best_Model"]]
    wide = picked.pivot_table(
        index=["Series", "Best_Model"], columns="Season_Label", values="Forecast"
    ).reset_index()
    wide.columns.name = None
    return wide


def add_selected_strategy(forecasts: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Append a pseudo-model, "Selected", holding each series' backtest winner.

    Scoring this alongside the fixed global models answers a question the
    single-model comparison cannot: does picking a different model per crop
    actually pay off, or does it just overfit the backtest folds?
    """
    lookup = dict(zip(best["Series"], best["Best_Model"]))
    picked = forecasts[
        forecasts.apply(lambda r: lookup.get(r["Series"]) == r["Model"], axis=1)
    ].copy()
    picked["Chosen_Model"] = picked["Model"]
    picked["Model"] = "Selected"
    return pd.concat([forecasts, picked], ignore_index=True)


def evaluate_holdout(forecasts: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts to the MSP the government actually announced.

    These seasons were never seen during fitting or model selection, so this
    is a true out-of-sample test rather than a re-description of the training
    data.
    """
    merged = forecasts.merge(
        holdout[["Series", "Season_Label", "Marketing_Season", "Actual_MSP", "Source"]],
        on=["Series", "Season_Label"],
        how="inner",
    )
    merged["Error"] = merged["Forecast"] - merged["Actual_MSP"]
    merged["AbsError"] = merged["Error"].abs()
    merged["APE"] = (merged["AbsError"] / merged["Actual_MSP"]) * 100
    return merged


def score_holdout(merged: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate hold-out accuracy, with MASE against the naive baseline."""
    keys = ["Model"] + (by or [])
    scores = (
        merged.groupby(keys)
        .apply(
            lambda g: pd.Series(
                _metrics(g["Actual_MSP"].to_numpy(dtype=float), g["Forecast"].to_numpy(dtype=float))
            ),
            include_groups=False,
        )
        .reset_index()
    )

    if by:
        base = (
            merged[merged["Model"] == "Naive"].groupby(by)["AbsError"].mean().rename("nb")
        )
        scores = scores.merge(base, left_on=by, right_index=True, how="left")
        scores["MASE"] = scores["MAE"] / scores["nb"]
        scores = scores.drop(columns=["nb"])
    else:
        scores["MASE"] = scores["MAE"] / merged.loc[merged["Model"] == "Naive", "AbsError"].mean()

    scores["N"] = merged.groupby(keys).size().to_numpy()
    scores["Within_5pct"] = (
        merged.assign(hit=merged["APE"] <= 5).groupby(keys)["hit"].mean().to_numpy() * 100
    )
    return scores.sort_values(["MAPE"] + (by or [])).reset_index(drop=True)


def growth_table(matrix: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    """Descriptive growth statistics used in the EDA section."""
    years = matrix.index.to_numpy(dtype=int)
    span = years[-1] - years[0]
    meta = long.drop_duplicates("Series").set_index("Series")[["Season", "Commodity", "Variety"]]

    rows = []
    for series in matrix.columns:
        v = matrix[series].to_numpy(dtype=float)
        yoy = np.diff(v) / v[:-1] * 100
        rows.append(
            {
                "Series": series,
                "Season": meta.loc[series, "Season"],
                "First_MSP": v[0],
                "Last_MSP": v[-1],
                "Absolute_Increase": v[-1] - v[0],
                "Total_Growth_pct": (v[-1] / v[0] - 1) * 100,
                "CAGR_pct": ((v[-1] / v[0]) ** (1 / span) - 1) * 100,
                "Mean_YoY_pct": yoy.mean(),
                "Volatility_YoY_pct": yoy.std(ddof=1),
                "Max_YoY_pct": yoy.max(),
                "Min_YoY_pct": yoy.min(),
            }
        )
    return pd.DataFrame(rows).sort_values("CAGR_pct", ascending=False).reset_index(drop=True)
