"""Prepare measured historical examples and precomputed future-step forecasts."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from forecasting.core import evaluate

ROOT = Path(__file__).resolve().parents[1]


def prepare() -> None:
    """Build each source example without requiring runtime retraining."""
    candy_path = (
        ROOT / "Candy_Production/1597402668_datasets_3004_5028_candy_production.csv"
    )
    candy = pd.read_csv(candy_path).rename(
        columns={"observation_date": "date", "IPG3113N": "value"}
    )
    news_path = ROOT / "NEWS_Demand_Dataset/NEWS_Dataset_Quarterly Demand.csv"
    news = pd.read_csv(news_path)
    news = pd.DataFrame(
        {
            "date": pd.PeriodIndex(
                news["year"].astype(str) + news["quarter"], freq="Q"
            ).to_timestamp(),
            "value": news["demand"],
        }
    )
    destination = ROOT / "reports"
    destination.mkdir(exist_ok=True)
    for name, frame, season, path in [
        ("candy", candy, 12, candy_path),
        ("news", news, 4, news_path),
    ]:
        report, test, future = evaluate(frame, season)
        report["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (destination / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        frame.to_csv(destination / f"{name}-history.csv", index=False)
        test.to_csv(destination / f"{name}-test.csv", index=False)
        pd.DataFrame({"future_step": range(1, 25), "forecast": future}).to_csv(
            destination / f"{name}-forecast.csv", index=False
        )
        print(name, json.dumps(report), flush=True)


if __name__ == "__main__":
    prepare()
