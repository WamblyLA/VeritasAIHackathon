import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .data import ROOT
from .models import LABELS

COLORS = {
    "ordinary": "#a5a5a5",
    "temporal_mdae": "#888cb5",
    "saits": "#78a8cb",
    "ensemble": "#9ac7b2",
}


def style(axes):
    for axis in axes:
        axis.set_facecolor("#090909")
        axis.tick_params(colors="#cfcfcf", labelsize=9)
        axis.xaxis.label.set_color("#e8e8e8")
        axis.yaxis.label.set_color("#e8e8e8")
        for spine in axis.spines.values():
            spine.set_color("#383838")


def main():
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    overall = metrics.query(
        "scope == 'overall' and method in ['ridge','ordinary','extreme','saits','temporal_mdae','ensemble']"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#090909")
    style(axes)
    names = [LABELS[name] for name in overall.method]
    colors = [COLORS.get(name, "#777777") for name in overall.method]
    for axis, column, label in zip(axes, ["mae", "f1"], ["MAE", "F1"]):
        axis.barh(names, overall[column], color=colors)
        axis.invert_yaxis()
        axis.set_xlabel(label)
    figure.tight_layout()
    figure.savefig(
        ROOT / "results/model_comparison.png", dpi=160, facecolor=figure.get_facecolor()
    )
    plt.close(figure)
    for filename, source, x, xlabel in [
        (
            "continuous_failures.png",
            pd.read_csv(ROOT / "results/continuous.csv").query(
                "scope == 'endpoint' and k == 0"
            ),
            "duration_hours",
            "Failure hours",
        ),
        (
            "available_stations.png",
            metrics.query("scope == 'by_k'"),
            "k",
            "Operational stations",
        ),
    ]:
        figure, axis = plt.subplots(figsize=(8, 4), facecolor="#090909")
        style([axis])
        for method, color in COLORS.items():
            frame = source[source.method == method].sort_values(x)
            axis.plot(
                frame[x], frame.mae, marker="o", color=color, label=LABELS[method]
            )
        if x == "duration_hours":
            axis.set_xscale("log")
            axis.set_xticks([1, 6, 24, 72, 168], labels=[1, 6, 24, 72, 168])
        axis.set_xlabel(xlabel)
        axis.set_ylabel("MAE")
        axis.legend(frameon=False, labelcolor="#eeeeee", fontsize=8)
        figure.tight_layout()
        figure.savefig(
            ROOT / "results" / filename, dpi=160, facecolor=figure.get_facecolor()
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
