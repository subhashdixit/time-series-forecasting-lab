"""Chronological evaluation, bounded forecasts and app behavior checks."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from forecasting.core import errors, evaluate, forecast, validate_series

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(name="series")
def fixture_series():
    """A deterministic seasonal example with enough history."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=60, freq="MS"),
            "value": np.tile([10.0, 20.0, 30.0, 40.0], 15),
        }
    )


def test_naive_pattern():
    """A seasonal baseline repeats the last cycle, not the whole history."""
    assert forecast([1, 2, 3, 4, 5, 6, 7, 8], "seasonal_naive", 6, 4).tolist() == [
        5,
        6,
        7,
        8,
        5,
        6,
    ]


@pytest.mark.parametrize(
    "model,horizon,season",
    [
        ("bad", 1, 1),
        ("arima", 0, 1),
        ("arima", 241, 1),
        ("arima", 2, 0),
        ("arima", 2, 25),
        ("arima", True, 1),
    ],
)
def test_forecast_bounds(model, horizon, season):
    """Reject unsupported algorithms and unreasonable work sizes."""
    with pytest.raises(ValueError):
        forecast(list(range(60)), model, horizon, season)


@pytest.mark.parametrize(
    "case",
    [
        "missing_column",
        "short",
        "long",
        "bad_date",
        "duplicate",
        "unordered",
        "nan",
        "infinite",
    ],
)
def test_invalid_series(series, case):
    """Never silently sort, fill future observations or accept invalid values."""
    if case == "missing_column":
        series = series.drop(columns="date")
    elif case == "short":
        series = series.head(5)
    elif case == "long":
        series = pd.concat([series] * 21)
    elif case == "bad_date":
        series["date"] = series["date"].astype(object)
        series.loc[0, "date"] = "not-a-date"
    elif case == "duplicate":
        series.loc[1, "date"] = series.loc[0, "date"]
    elif case == "unordered":
        series = series.iloc[::-1]
    elif case == "nan":
        series.loc[0, "value"] = np.nan
    else:
        series.loc[0, "value"] = np.inf
    with pytest.raises(ValueError):
        validate_series(series)


def test_errors():
    """MAE and RMSE follow their documented definitions."""
    result = errors([1, 3], [2, 1])
    assert result["mae"] == 1.5
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError):
        errors([1], [1, 2])


def test_test_window_does_not_select_model(series):
    """Changing only final test observations cannot alter validation selection."""
    report, _, _ = evaluate(series, 4)
    changed = series.copy()
    changed.loc[48:, "value"] += 100
    other, _, _ = evaluate(changed, 4)
    assert report["validation_candidates"] == other["validation_candidates"]
    assert report["selected_model"] == other["selected_model"]
    assert report["split"] == {"train": 36, "validation": 12, "test": 12}
    assert other["test"]["mae"] > report["test"]["mae"]


def test_app_examples_and_upload_error():
    """Both prepared examples render and missing-upload errors are visible."""
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Forecasting Lab"
    app.sidebar.selectbox[0].select("Quarterly news demand").run()
    assert not app.exception
    app.button[0].click().run()
    assert app.error and "Choose a CSV" in app.error[0].value
