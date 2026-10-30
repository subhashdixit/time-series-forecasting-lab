"""Create static scientific figures from measured forecast artifacts."""

import json
from pathlib import Path

from matplotlib.figure import Figure
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def plot_results() -> None:
    """Plot actual/test forecasts and validation comparison in source units."""
    destination = ROOT / "docs/images"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["candy", "news"]:
        report = json.loads((ROOT / f"reports/{name}.json").read_text())
        test = pd.read_csv(ROOT / f"reports/{name}-test.csv", parse_dates=["date"])
        figure = Figure(figsize=(10, 6), layout="constrained", facecolor="#f4f6f5")
        top, bottom = figure.subplots(2, 1)
        top.plot(test["date"], test["value"], label="Observed", color="#263b56")
        top.plot(
            test["date"],
            test["forecast"],
            label=report["selected_model"],
            color="#26704e",
        )
        top.set(
            title=f"{name.capitalize()}: later held-out observations",
            ylabel="Source units",
        )
        top.legend()
        comparison = report["validation_candidates"]
        bottom.barh(
            list(comparison),
            [value["mae"] for value in comparison.values()],
            color="#26704e",
        )
        bottom.set(
            xlabel="Validation MAE (lower is better)",
            title="Model selected before final test",
        )
        for axis in [top, bottom]:
            axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(destination / f"{name}-evaluation.png", dpi=150)


if __name__ == "__main__":
    plot_results()
