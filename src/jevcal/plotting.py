from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RATE_COLORS = {0.20: "#4C78A8", 0.10: "#F58518", 0.05: "#54A24B", 0.02: "#B279A2"}
GROUP_COLORS = {"routed": "#D95F02", "non_routed": "#1B9E77"}


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


def plot_absolute_calibration(points: pd.DataFrame, output_path: str | Path) -> None:
    """Overlay routed/non-routed reliability curves on common pooled bins."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = points[["edge", "rate"]].drop_duplicates().reset_index(drop=True)
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(10, 4.3 * nrows), squeeze=False)
    for axis, panel in zip(axes.ravel(), panels.itertuples(index=False), strict=False):
        subset = points[(points["edge"] == panel.edge) & (points["rate"] == panel.rate)]
        for group in ("non_routed", "routed"):
            group_rows = subset[subset["group"] == group].sort_values("mean_probability")
            axis.plot(
                group_rows["mean_probability"],
                group_rows["outcome_rate"],
                marker="o",
                linewidth=1.5,
                color=GROUP_COLORS[group],
                label=group.replace("_", " "),
            )
        axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
        axis.set_title(f"{panel.edge.replace('->', ' → ')} · {panel.rate:.0%}")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed outcome rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.18)
    for axis in axes.ravel()[len(panels) :]:
        axis.set_visible(False)
    if len(panels):
        axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_binning_sensitivity(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = list(dict.fromkeys(results["edge"]))
    figure, axes = plt.subplots(1, len(edges), figsize=(6 * len(edges), 4.6), squeeze=False)
    for axis, edge in zip(axes.ravel(), edges, strict=True):
        subset = results[results["edge"] == edge]
        for rate in sorted(subset["rate"].unique(), reverse=True):
            rows = subset[subset["rate"] == rate].sort_values("n_bins")
            axis.errorbar(
                rows["n_bins"],
                rows["conditional_signed_error"],
                yerr=np.vstack(
                    [
                        rows["conditional_signed_error"]
                        - rows["conditional_signed_error_ci_low"],
                        rows["conditional_signed_error_ci_high"]
                        - rows["conditional_signed_error"],
                    ]
                ),
                marker="o",
                capsize=3,
                label=f"{rate:.0%} routed",
                color=RATE_COLORS.get(round(float(rate), 2)),
            )
        axis.axhline(0, color="black", linestyle=":", linewidth=1)
        axis.set_title(edge.replace("->", " → "))
        axis.set_xlabel("Pooled quantile bins")
        axis.set_ylabel("Conditional signed-error shift")
        axis.set_xticks(sorted(subset["n_bins"].unique()))
        axis.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_spline_calibration(curves: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = curves[["edge", "rate"]].drop_duplicates().reset_index(drop=True)
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(10, 4.3 * nrows), squeeze=False)
    for axis, panel in zip(axes.ravel(), panels.itertuples(index=False), strict=False):
        subset = curves[(curves["edge"] == panel.edge) & (curves["rate"] == panel.rate)]
        for group in ("non_routed", "routed"):
            rows = subset[subset["group"] == group].sort_values("probability")
            color = GROUP_COLORS[group]
            axis.plot(rows["probability"], rows["estimated_outcome"], color=color, label=group.replace("_", " "))
            axis.fill_between(
                rows["probability"], rows["ci_low"], rows["ci_high"], color=color, alpha=0.16
            )
        axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
        axis.set_title(f"{panel.edge.replace('->', ' → ')} · {panel.rate:.0%}")
        axis.set_xlabel("Downstream probability")
        axis.set_ylabel("Spline-estimated outcome rate")
        axis.set_xlim(float(subset["probability"].min()), float(subset["probability"].max()))
        axis.set_ylim(0, min(1.0, max(0.1, float(subset["ci_high"].max()) * 1.08)))
        axis.grid(alpha=0.18)
    for axis in axes.ravel()[len(panels) :]:
        axis.set_visible(False)
    if len(panels):
        axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_prediction_set_comparison(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = list(dict.fromkeys(results["edge"]))
    figure, axes = plt.subplots(1, len(edges), figsize=(6.2 * len(edges), 4.8), squeeze=False)
    for axis, edge in zip(axes.ravel(), edges, strict=True):
        subset = results[results["edge"] == edge]
        sets = list(dict.fromkeys(subset["prediction_set"]))
        offsets = np.linspace(-0.18, 0.18, max(1, len(sets)))
        rates = sorted(subset["rate"].unique(), reverse=True)
        for offset, name in zip(offsets, sets, strict=True):
            rows = subset[subset["prediction_set"] == name].set_index("rate").loc[rates]
            y = np.arange(len(rates)) + offset
            axis.errorbar(
                rows["conditional_signed_error"],
                y,
                xerr=np.vstack(
                    [
                        rows["conditional_signed_error"]
                        - rows["conditional_signed_error_ci_low"],
                        rows["conditional_signed_error_ci_high"]
                        - rows["conditional_signed_error"],
                    ]
                ),
                marker="o",
                linestyle="none",
                capsize=2,
                label=name,
            )
        axis.axvline(0, color="black", linestyle=":", linewidth=1)
        axis.set_yticks(np.arange(len(rates)), [f"{rate:.0%}" for rate in rates])
        axis.set_xlabel("Conditional signed-error shift")
        axis.set_ylabel("Routed fraction")
        axis.set_title(edge.replace("->", " → "))
        axis.grid(axis="x", alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
