"""Interactive forecasting laboratory with precomputed historical evidence."""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from forecasting.core import MODELS, forecast, validate_series

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Render measured examples and a bounded CSV forecasting exercise."""
    st.set_page_config(page_title="Forecasting Lab", page_icon="📈", layout="wide")
    st.caption("TIME SERIES · SUBHASH DIXIT")
    st.title("Forecasting Lab")
    st.write(
        (
            "Learn how a forecast is built, compared and checked "
            "against later observations."
        )
    )
    st.info(
        (
            "Historical learning examples. Forecasts are anchored to "
            "each dataset's last observation, not today's date. No "
            "trading or demand-planning guarantee is implied."
        )
    )
    case = st.sidebar.selectbox(
        "Historical example", ["Candy production", "Quarterly news demand"]
    )
    name = "candy" if case == "Candy production" else "news"
    report = json.loads((ROOT / f"reports/{name}.json").read_text())
    history = pd.read_csv(ROOT / f"reports/{name}-history.csv", parse_dates=["date"])
    test = pd.read_csv(ROOT / f"reports/{name}-test.csv", parse_dates=["date"])
    future = pd.read_csv(ROOT / f"reports/{name}-forecast.csv")
    horizon = st.sidebar.slider("Future observation steps", 1, 24, 12)
    tabs = st.tabs(
        ["History and forecast", "Evaluation", "Try your CSV", "Understand the flow"]
    )
    with tabs[0]:
        st.subheader(case)
        st.caption(
            (
                f"Last source observation: {report['last_observation']}. "
                f"Units are those of the supplied source series."
            )
        )
        st.line_chart(history.set_index("date")["value"], use_container_width=True)
        st.write(
            (
                f"Selected model: "
                f"**{report['selected_model'].replace('_',' ')}** · season "
                f"length: **{report['season']} observations**."
            )
        )
        st.subheader("Forecast after the last source observation")
        st.line_chart(
            future.head(horizon).set_index("future_step"), use_container_width=True
        )
        st.caption(
            (
                "The x-axis is observation steps, not a fabricated "
                "calendar. Negative forecasts are not clipped or silently "
                "corrected. No calibrated prediction interval is claimed."
            )
        )
        st.download_button(
            "Download forecast",
            future.head(horizon).to_csv(index=False),
            f"{name}-forecast.csv",
            "text/csv",
        )
    with tabs[1]:
        left, right = st.columns(2)
        left.metric("Held-out MAE", f"{report['test']['mae']:.3f}")
        right.metric("Held-out RMSE", f"{report['test']['rmse']:.3f}")
        st.write(
            (
                "Validation chooses the model; the selected model is "
                "refitted on train + validation before its final test. The "
                "final future forecast is then refitted using all "
                "historical observations."
            )
        )
        st.dataframe(
            pd.DataFrame(report["validation_candidates"]).T, use_container_width=True
        )
        if report["failed_candidates"]:
            st.warning(
                (
                    "Some candidate fits failed or did not converge; they were "
                    "excluded rather than reported as successful."
                )
            )
            st.json(report["failed_candidates"])
        st.line_chart(
            test.set_index("date")[["value", "forecast"]], use_container_width=True
        )
        st.json(report["split"])
        if name == "news":
            st.warning(
                (
                    "Only 16 quarterly observations are available. The final "
                    "test is tiny: treat comparisons as a learning exercise, "
                    "not reliable evidence for business planning."
                )
            )
    with tabs[2]:
        st.subheader("Forecast your own series, including stock-price observations")
        st.write(
            (
                "Upload UTF-8 CSV with date,value columns: 12–1200 "
                "chronological unique observations, no missing values, "
                "maximum 1 MiB. Irregular trading dates are treated as "
                "ordered observation steps; no missing trading days are "
                "invented."
            )
        )
        st.download_button(
            "Download input example",
            history.rename(columns={"value": "value"}).to_csv(index=False),
            "series-example.csv",
            "text/csv",
        )
        uploaded = st.file_uploader("Your series", type=["csv"])
        model = st.selectbox("Exercise model", MODELS)
        season = st.number_input(
            "Season length in observations", min_value=1, max_value=24, value=1, step=1
        )
        if st.button("Fit exercise forecast"):
            if uploaded is None:
                st.error("Choose a CSV before fitting.")
            else:
                try:
                    if uploaded.size > 1024 * 1024:
                        raise ValueError("CSV exceeds 1 MiB.")
                    data = validate_series(
                        pd.read_csv(io.BytesIO(uploaded.getvalue()), nrows=1201)
                    )
                    with st.spinner("Fitting a bounded model..."):
                        predicted = forecast(data["value"], model, horizon, int(season))
                    output = pd.DataFrame(
                        {"future_step": range(1, horizon + 1), "forecast": predicted}
                    )
                    st.line_chart(
                        output.set_index("future_step"), use_container_width=True
                    )
                    st.download_button(
                        "Download exercise forecast",
                        output.to_csv(index=False),
                        "exercise.csv",
                        "text/csv",
                    )
                    st.caption(
                        (
                            "This custom fit has no independent test score. Compare it "
                            "against a held-out later period before judging its "
                            "quality. Inputs are processed in memory, not written by "
                            "the app."
                        )
                    )
                except (
                    ValueError,
                    UnicodeError,
                    RuntimeError,
                    np.linalg.LinAlgError,
                    ConvergenceWarning,
                ) as exc:
                    st.error(str(exc))
    with tabs[3]:
        st.markdown(
            """### A forecast follows time
1. Inspect the history and its observation frequency.
2. Keep the last observations out of training.
3. Compare a seasonal-naive baseline, exponential smoothing, ARIMA and
SARIMA on validation.
4. Freeze the choice and test on a later window.
5. Refit the chosen algorithm on all known history to forecast beyond it.

**Seasonal naive** repeats the last seasonal cycle. **Exponential
smoothing** combines level, trend and seasonality. **ARIMA** models
differenced dynamics; **SARIMA** adds seasonal dynamics. Complexity does not
guarantee a better forecast.

MAE is average absolute error in source units. RMSE penalizes large mistakes
more strongly. Neither compares scale fairly across unrelated datasets.
Stock prices need a random-walk baseline and evaluation of costs before any
trading interpretation; this app makes no trading claim."""
        )


if __name__ == "__main__":
    main()
