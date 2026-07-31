"""Forecasting models.

Every model is a plain callable with the same signature::

    f(years: np.ndarray, values: np.ndarray, future_years: np.ndarray) -> np.ndarray

so the backtester, the forecaster and the app can all treat them
interchangeably and no model gets an unfair evaluation protocol.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .config import RANDOM_STATE

warnings.filterwarnings("ignore")


def _col(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1, 1)


# --- baselines --------------------------------------------------------------

def naive(years, values, future_years):
    """Last observed MSP carried forward. The bar every model must clear."""
    return np.repeat(float(values[-1]), len(future_years))


def drift(years, values, future_years):
    """Random walk with drift: last value + average historical step."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    step = (values[-1] - values[0]) / (n - 1) if n > 1 else 0.0
    h = np.asarray(future_years, dtype=float) - float(years[-1])
    return values[-1] + step * h


# --- parametric trend models ------------------------------------------------

def linear(years, values, future_years):
    """Ordinary least squares on the year index."""
    model = LinearRegression().fit(_col(years), np.asarray(values, dtype=float))
    return model.predict(_col(future_years))


def ridge_poly2(years, values, future_years):
    """Degree-2 polynomial with L2 shrinkage to curb the curvature.

    Captures the mild acceleration in MSP revisions without the wild
    extrapolation an unregularised quadratic would produce on five points.
    """
    y0 = float(years[0])
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    ).fit(_col(np.asarray(years) - y0), np.asarray(values, dtype=float))
    return model.predict(_col(np.asarray(future_years) - y0))


def cagr(years, values, future_years):
    """Log-linear (constant compound growth) model.

    MSP revisions are announced as percentages of the previous year, so a
    constant-growth-rate model matches how the policy is actually set.
    """
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        return linear(years, values, future_years)
    model = LinearRegression().fit(_col(years), np.log(values))
    return np.exp(model.predict(_col(future_years)))


# --- time-series models -----------------------------------------------------

def holt(years, values, future_years):
    """Holt's linear (double) exponential smoothing."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    values = np.asarray(values, dtype=float)
    horizon = int(max(future_years) - years[-1])
    try:
        fit = ExponentialSmoothing(values, trend="add", initialization_method="estimated").fit()
        path = np.asarray(fit.forecast(horizon), dtype=float)
    except Exception:
        return drift(years, values, future_years)
    idx = (np.asarray(future_years) - years[-1]).astype(int) - 1
    return path[idx]


def arima110(years, values, future_years):
    """ARIMA(1,1,0): one AR lag on the first difference.

    First-order differencing handles the non-stationary upward trend; the AR
    term carries momentum from the previous revision.
    """
    from statsmodels.tsa.arima.model import ARIMA

    values = np.asarray(values, dtype=float)
    horizon = int(max(future_years) - years[-1])
    try:
        fit = ARIMA(values, order=(1, 1, 0)).fit()
        path = np.asarray(fit.forecast(steps=horizon), dtype=float)
    except Exception:
        return drift(years, values, future_years)
    idx = (np.asarray(future_years) - years[-1]).astype(int) - 1
    return path[idx]


# --- tree ensembles ---------------------------------------------------------

def random_forest(years, values, future_years):
    """Random Forest on the year index.

    Included as a deliberate negative control: trees predict a constant
    outside the training range, so this cannot extrapolate a trend by design.
    """
    model = RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE
    ).fit(_col(years), np.asarray(values, dtype=float))
    return model.predict(_col(future_years))


def gradient_boosting(years, values, future_years):
    """Gradient boosted trees. Shares the flat-extrapolation limit of RF."""
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=2, learning_rate=0.1, random_state=RANDOM_STATE
    ).fit(_col(years), np.asarray(values, dtype=float))
    return model.predict(_col(future_years))


# --- combination ------------------------------------------------------------

ENSEMBLE_MEMBERS = ("Drift", "Linear", "CAGR", "Holt")


def ensemble(years, values, future_years):
    """Median of the four trend-following models.

    Combining forecasts is the cheapest known variance reduction in
    forecasting; the median also blunts any single model that extrapolates
    badly on a short series.
    """
    preds = np.vstack(
        [REGISTRY[name](years, values, future_years) for name in ENSEMBLE_MEMBERS]
    )
    return np.median(preds, axis=0)


REGISTRY = {
    "Naive": naive,
    "Drift": drift,
    "Linear": linear,
    "RidgePoly2": ridge_poly2,
    "CAGR": cagr,
    "Holt": holt,
    "ARIMA(1,1,0)": arima110,
    "RandomForest": random_forest,
    "GradBoost": gradient_boosting,
    "Ensemble": ensemble,
}

#: Models that can, in principle, project a trend beyond the training range.
EXTRAPOLATING = [m for m in REGISTRY if m not in ("Naive", "RandomForest", "GradBoost")]
