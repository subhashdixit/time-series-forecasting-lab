"""Bounded univariate forecasting and chronological evaluation."""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

MODELS = ["seasonal_naive", "exponential_smoothing", "arima", "sarima"]


def validate_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Require finite chronological observations with explicit dates and values."""
    if not isinstance(frame, pd.DataFrame) or not {"date", "value"} <= set(frame):
        raise ValueError("CSV requires date and value columns.")
    if not 12 <= len(frame) <= 1200:
        raise ValueError("Provide 12 to 1200 observations.")
    result = frame[["date", "value"]].copy()
    try:
        result["date"] = pd.to_datetime(
            result["date"], errors="raise", format="ISO8601"
        )
        result["value"] = pd.to_numeric(result["value"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Dates must be parseable and values must be numeric.") from exc
    if result["date"].isna().any() or not np.isfinite(result["value"]).all():
        raise ValueError("Dates and values cannot be missing or infinite.")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("Dates must be unique and strictly increasing.")
    return result.reset_index(drop=True)


def forecast(values: object, model: str, horizon: int, season: int) -> np.ndarray:
    """Fit one fixed algorithm without looking at future observations."""
    if model not in MODELS:
        raise ValueError("Unknown forecasting model.")
    if (
        not isinstance(horizon, int)
        or isinstance(horizon, bool)
        or not 1 <= horizon <= 240
    ):
        raise ValueError("Horizon must be an integer between 1 and 240.")
    if not isinstance(season, int) or isinstance(season, bool) or not 1 <= season <= 24:
        raise ValueError("Season length must be an integer between 1 and 24.")
    array = np.asarray(values, dtype=float)
    if (
        array.ndim != 1
        or len(array) < max(6, 2 * season)
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            "Need finite training values and at least two seasonal cycles."
        )
    if model == "seasonal_naive":
        return np.resize(array[-season:], horizon)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        if model == "exponential_smoothing":
            fitted = ExponentialSmoothing(
                array,
                trend="add",
                seasonal="add" if season > 1 else None,
                seasonal_periods=season if season > 1 else None,
                initialization_method="estimated",
            ).fit()
        elif model == "arima":
            fitted = ARIMA(array, order=(1, 1, 1)).fit(method_kwargs={"maxiter": 80})
        else:
            fitted = SARIMAX(
                array,
                order=(1, 1, 0),
                seasonal_order=(1, 0, 0, season) if season > 1 else (0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=80)
        predicted = np.asarray(fitted.forecast(horizon))
    if not np.isfinite(predicted).all():
        raise ValueError("The model produced non-finite forecasts.")
    return predicted


def errors(actual: object, predicted: object) -> dict[str, float]:
    """Compute interpretable scale-dependent errors; no divide-by-zero MAPE."""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or len(actual) == 0:
        raise ValueError(
            "Actual and forecast arrays must have matching nonzero length."
        )
    return {
        "mae": float(np.abs(actual - predicted).mean()),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }


def evaluate(frame: pd.DataFrame, season: int) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Select on middle validation window; refit on development before final test."""
    data = validate_series(frame)
    count = len(data)
    first, second = int(count * 0.6), int(count * 0.8)
    values = data["value"].to_numpy()
    candidates, failures = {}, {}
    for model in MODELS:
        try:
            predicted = forecast(values[:first], model, second - first, season)
            candidates[model] = errors(values[first:second], predicted)
        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
            ConvergenceWarning,
        ) as exc:
            failures[model] = str(exc)
    if not candidates:
        raise ValueError(
            "No candidate can be fitted; provide more history or a shorter season."
        )
    selected = min(candidates, key=lambda model: candidates[model]["mae"])
    predicted = forecast(values[:second], selected, count - second, season)
    future = forecast(values, selected, 24, season)
    report = {
        "selected_model": selected,
        "season": season,
        "rows": count,
        "split": {"train": first, "validation": second - first, "test": count - second},
        "validation_candidates": candidates,
        "failed_candidates": failures,
        "test": errors(values[second:], predicted),
        "test_start": str(data.iloc[second]["date"].date()),
        "last_observation": str(data.iloc[-1]["date"].date()),
    }
    test_frame = data.iloc[second:].copy()
    test_frame["forecast"] = predicted
    return report, test_frame, future
