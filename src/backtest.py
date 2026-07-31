"""Rolling-origin backtesting.

Every model is scored under one identical protocol. This is the main
methodological fix over the naive approach of comparing an in-sample fit for
one model against cross-validated residuals for another: those numbers are not
on the same scale and the resulting "best model" ranking is meaningless.

Protocol (expanding window, origin walks forward):

    train 2017-2019 -> predict 2020, 2021    (h = 1, 2)
    train 2017-2020 -> predict 2021          (h = 1)

Nothing after the origin is visible to the model at fit time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_TRAIN_POINTS
from .models import REGISTRY


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - actual
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err / actual)) * 100),
        "Bias": float(np.mean(err)),
    }


def rolling_origin(matrix: pd.DataFrame, max_horizon: int = 2) -> pd.DataFrame:
    """Per-fold predictions for every series x model x horizon."""
    years = matrix.index.to_numpy(dtype=int)
    rows = []

    for series in matrix.columns:
        values = matrix[series].to_numpy(dtype=float)
        if np.isnan(values).any():
            continue

        for origin in range(MIN_TRAIN_POINTS, len(years)):
            train_years, train_values = years[:origin], values[:origin]
            test_idx = np.arange(origin, min(origin + max_horizon, len(years)))
            if test_idx.size == 0:
                continue
            test_years = years[test_idx]

            for name, fn in REGISTRY.items():
                preds = np.asarray(fn(train_years, train_values, test_years), dtype=float)
                for k, ti in enumerate(test_idx):
                    rows.append(
                        {
                            "Series": series,
                            "Model": name,
                            "Origin": int(years[origin - 1]),
                            "Target_Year": int(years[ti]),
                            "Horizon": int(years[ti] - years[origin - 1]),
                            "Actual": float(values[ti]),
                            "Predicted": float(preds[k]),
                        }
                    )

    folds = pd.DataFrame(rows)
    folds["Error"] = folds["Predicted"] - folds["Actual"]
    folds["AbsError"] = folds["Error"].abs()
    folds["APE"] = (folds["AbsError"] / folds["Actual"]) * 100
    return folds


def _attach_mase(scores: pd.DataFrame, folds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """MASE = model MAE / naive MAE over the same folds.

    Below 1.0 the model beats "carry last year's MSP forward"; at or above
    1.0 it does not, whatever its raw MAE looks like.
    """
    base = (
        folds[folds["Model"] == "Naive"]
        .groupby(keys)["AbsError"]
        .mean()
        .rename("Naive_MAE")
    )
    out = scores.merge(base, left_on=keys, right_index=True, how="left")
    out["MASE"] = out["MAE"] / out["Naive_MAE"]
    return out.drop(columns=["Naive_MAE"])


def score_overall(folds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy per model across all series and horizons."""
    grouped = folds.groupby("Model")
    scores = grouped.apply(
        lambda g: pd.Series(_metrics(g["Actual"].to_numpy(), g["Predicted"].to_numpy())),
        include_groups=False,
    ).reset_index()

    naive_mae = folds.loc[folds["Model"] == "Naive", "AbsError"].mean()
    scores["MASE"] = scores["MAE"] / naive_mae
    scores["Folds"] = grouped.size().to_numpy()
    return scores.sort_values("MAE").reset_index(drop=True)


def score_by_horizon(folds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per model split by forecast horizon."""
    scores = (
        folds.groupby(["Model", "Horizon"])
        .apply(
            lambda g: pd.Series(_metrics(g["Actual"].to_numpy(), g["Predicted"].to_numpy())),
            include_groups=False,
        )
        .reset_index()
    )
    return _attach_mase(scores, folds, ["Horizon"]).sort_values(["Horizon", "MAE"])


def score_by_series(folds: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per model for each individual commodity-variety series."""
    scores = (
        folds.groupby(["Series", "Model"])
        .apply(
            lambda g: pd.Series(_metrics(g["Actual"].to_numpy(), g["Predicted"].to_numpy())),
            include_groups=False,
        )
        .reset_index()
    )
    return _attach_mase(scores, folds, ["Series"])


def best_model_per_series(series_scores: pd.DataFrame) -> pd.DataFrame:
    """Pick each series' winner on backtest MAPE, tie-broken by MAE.

    Selection uses only backtest folds, never the hold-out seasons, so the
    hold-out remains a genuinely untouched test set.
    """
    ranked = series_scores.sort_values(["Series", "MAPE", "MAE"])
    best = ranked.groupby("Series", as_index=False).first()
    return best.rename(columns={"Model": "Best_Model"})[
        ["Series", "Best_Model", "MAE", "RMSE", "MAPE", "MASE"]
    ]
