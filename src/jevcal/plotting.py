from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RATE_COLORS = {0.20: "#4C78A8", 0.10: "#F58518", 0.05: "#54A24B", 0.02: "#B279A2"}


def plot_primary(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = list(dict.fromkeys(results["edge"]))
    labels = []
    rows = []
    for edge in edges:
        for rate in sorted(results.loc[results["edge"] == edge, "rate"], reverse=True):
            row = results[(results["edge"] == edge) & (results["rate"] == rate)].iloc[0]
            rows.append(row)
            labels.append(f"{edge.replace('->', ' → ')} · {rate:.0%}")
    figure, axes = plt.subplots(1, 2, figsize=(11, max(4.5, len(rows) * 0.48)), sharey=True)
    y = np.arange(len(rows))[::-1]
    for axis, metric, title in [
        (axes[0], "conditional_signed_error", "Signed error (overprediction +)"),
        (axes[1], "conditional_brier", "Brier loss (higher = worse)"),
    ]:
        for position, row in zip(y, rows, strict=True):
            value = float(row[metric])
            low = float(row[f"{metric}_ci_low"])
            high = float(row[f"{metric}_ci_high"])
            color = RATE_COLORS.get(round(float(row["rate"]), 2), "#4C78A8")
            axis.hlines(position, low, high, color=color, linewidth=1.5)
            axis.plot(value, position, "o", color=color)
        axis.axvline(0, color="black", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Routed − non-routed, within score bins")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_matrix(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = list(dict.fromkeys(results["edge"]))
    figure, axis = plt.subplots(figsize=(9, max(4.5, len(edges) * 0.62)))
    offsets = {0.10: 0.10, 0.05: -0.10}
    markers = {0.10: "o", 0.05: "s"}
    for edge_index, edge in enumerate(edges):
        subset = results[results["edge"] == edge]
        for row in subset.itertuples(index=False):
            rate = round(float(row.rate), 2)
            value = float(row.conditional_signed_error)
            low = float(row.conditional_signed_error_ci_low)
            high = float(row.conditional_signed_error_ci_high)
            significant = float(row.q_signed_error) < 0.05
            color = "#2F80B7" if value >= 0 else "#E07A27"
            position = edge_index + offsets.get(rate, 0)
            axis.hlines(position, low, high, color=color, linewidth=1.5)
            axis.plot(
                value,
                position,
                marker=markers.get(rate, "o"),
                markerfacecolor=color if significant else "white",
                markeredgecolor=color,
                color=color,
                linestyle="none",
                label=f"{rate:.0%} routed" if edge_index == 0 else None,
            )
    axis.axvline(0, color="black", linestyle=":", linewidth=1)
    axis.set_yticks(np.arange(len(edges)), [value.replace("->", " → ") for value in edges])
    axis.invert_yaxis()
    axis.set_xlabel("Conditional signed-error shift (routed − non-routed)")
    axis.set_title("Civil Comments routing matrix")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
