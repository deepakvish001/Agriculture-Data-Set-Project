"""Regression tests for the MSP forecasting pipeline.

Run with: pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import backtest, config, data_loader, forecast, models


@pytest.fixture(scope="module")
def data():
    return data_loader.build()


@pytest.fixture(scope="module")
def matrix(data):
    return data["matrix"]


# --- data integrity ---------------------------------------------------------

def test_dataset_shape(data):
    assert data["audit"]["rows"] == 28
    assert data["audit"]["missing_cells"] == 0
    assert data["audit"]["duplicate_rows"] == 0


def test_matrix_is_variety_level(matrix):
    """Varieties must stay separate — averaging them invents a price."""
    assert matrix.shape == (5, 28)
    assert "Paddy (Common)" in matrix.columns
    assert "Paddy (Grade 'A')" in matrix.columns
    assert "Paddy" not in matrix.columns


def test_no_series_ever_fell(matrix):
    assert (matrix.diff().dropna() >= 0).all().all()


def test_rabi_season_offset_applied(data):
    """A Rabi crop year Y is marketed in RMS Y+1; Kharif is not shifted."""
    long = data["long"]
    wheat = long[(long["Series"] == "Wheat") & (long["Season_Label"] == "2021-22")]
    assert wheat["Marketing_Season"].iloc[0] == "2022-23"
    assert wheat["MSP"].iloc[0] == 2015

    paddy = long[(long["Series"] == "Paddy (Common)") & (long["Season_Label"] == "2021-22")]
    assert paddy["Marketing_Season"].iloc[0] == "2021-22"


def test_holdout_series_all_exist_in_training(data):
    unknown = set(data["holdout"]["Series"]) - set(data["matrix"].columns)
    assert not unknown, f"hold-out references unknown series: {unknown}"


# --- model contract ---------------------------------------------------------

@pytest.mark.parametrize("name", list(models.REGISTRY))
def test_model_signature_and_shape(name):
    years = np.array([2017, 2018, 2019, 2020, 2021])
    values = np.array([1550.0, 1750, 1815, 1868, 1940])
    future = np.array([2022, 2023, 2024])
    out = np.asarray(models.REGISTRY[name](years, values, future), dtype=float)
    assert out.shape == (3,)
    assert np.isfinite(out).all()
    assert (out > 0).all()


def test_naive_is_flat():
    years = np.array([2017, 2018, 2019, 2020, 2021])
    values = np.array([100.0, 200, 300, 400, 500])
    out = models.naive(years, values, np.array([2022, 2023]))
    assert np.allclose(out, 500)


def test_drift_extends_the_average_step():
    years = np.array([2017, 2018, 2019, 2020, 2021])
    values = np.array([100.0, 200, 300, 400, 500])
    out = models.drift(years, values, np.array([2022, 2023]))
    assert np.allclose(out, [600, 700])


def test_trees_cannot_extrapolate():
    """The documented weakness of RF/GBM on a trending series."""
    years = np.array([2017, 2018, 2019, 2020, 2021])
    values = np.array([100.0, 200, 300, 400, 500])
    for fn in (models.random_forest, models.gradient_boosting):
        out = fn(years, values, np.array([2022, 2030]))
        assert out[0] == pytest.approx(out[1], rel=1e-6)
        assert out[0] < values[-1] + 1


# --- evaluation protocol ----------------------------------------------------

def test_backtest_never_leaks_the_future(matrix):
    folds = backtest.rolling_origin(matrix)
    assert (folds["Target_Year"] > folds["Origin"]).all()
    assert folds["Horizon"].between(1, 2).all()
    assert set(folds["Model"]) == set(models.REGISTRY)


def test_every_series_scored(matrix):
    folds = backtest.rolling_origin(matrix)
    assert folds["Series"].nunique() == matrix.shape[1]


def test_mase_is_one_for_the_naive_baseline(matrix):
    scores = backtest.score_overall(backtest.rolling_origin(matrix))
    naive = scores[scores["Model"] == "Naive"].iloc[0]
    assert naive["MASE"] == pytest.approx(1.0)


def test_holdout_is_disjoint_from_training(data):
    """The hold-out seasons must lie strictly after the training window."""
    assert data["holdout"]["Year"].min() > data["matrix"].index.max()


def test_champion_beats_the_naive_baseline(data, matrix):
    fc = forecast.forecast_all(matrix)
    merged = forecast.evaluate_holdout(fc, data["holdout"])
    scores = forecast.score_holdout(merged)
    assert scores.iloc[0]["MASE"] < 1.0
    assert scores.iloc[0]["MAPE"] < 5.0


def test_forecasts_cover_every_series_and_season(matrix):
    fc = forecast.forecast_all(matrix)
    expected = matrix.shape[1] * len(models.REGISTRY) * len(config.FORECAST_LABELS)
    assert len(fc) == expected
    assert set(fc["Season_Label"]) == set(config.FORECAST_LABELS)


def test_growth_table_matches_raw_values(matrix):
    growth = forecast.growth_table(matrix, data_loader.build()["long"])
    row = growth[growth["Series"] == "Wheat"].iloc[0]
    assert row["First_MSP"] == 1735
    assert row["Last_MSP"] == 2015
    assert row["CAGR_pct"] == pytest.approx(
        ((2015 / 1735) ** 0.25 - 1) * 100, rel=1e-6
    )
