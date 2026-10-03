from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd


def regenerate_reports(config: dict[str, Any]) -> dict[str, str]:
    """Rebuild every figure and LaTeX table without rerunning inference/statistics."""
    from .plotting import (
        plot_absolute_calibration,
        plot_binning_sensitivity,
        plot_matrix,
        plot_prediction_set_comparison,
        plot_primary,
        plot_spline_calibration,
    )

    results_dir = Path(config["paths"]["results_dir"])
    outputs = generate_publication_tables(config)
    core_tables = results_dir / "tables"
    core_figures = results_dir / "figures"
    core_figures.mkdir(parents=True, exist_ok=True)
    primary = core_tables / "civil_primary.csv"
    matrix = core_tables / "civil_matrix.csv"
    if primary.exists():
        target = core_figures / "figure1_primary.png"
        plot_primary(pd.read_csv(primary), target)
        outputs["figure1"] = str(target)
    if matrix.exists():
        target = core_figures / "figure2_matrix.png"
        plot_matrix(pd.read_csv(matrix), target)
        outputs["figure2"] = str(target)

    robustness_tables = results_dir / "robustness" / "tables"
    robustness_figures = results_dir / "robustness" / "figures"
    robustness_figures.mkdir(parents=True, exist_ok=True)
    reliability = robustness_tables / "reliability_points.csv"
    binning = robustness_tables / "binning_sensitivity.csv"
    curves = robustness_tables / "continuous_spline_curves.csv"
    prediction_sets = robustness_tables / "prediction_set_comparison.csv"
    display_rates = {
        float(value) for value in config.get("robustness", {}).get("display_rates", [0.10, 0.05])
    }
    if reliability.exists():
        target = robustness_figures / "figure3_absolute_calibration.png"
        frame = pd.read_csv(reliability)
        plot_absolute_calibration(frame[frame["rate"].isin(display_rates)], target)
        outputs["figure_absolute_calibration"] = str(target)
    if binning.exists():
        target = robustness_figures / "figure4_binning_sensitivity.png"
        plot_binning_sensitivity(pd.read_csv(binning), target)
        outputs["figure_binning_sensitivity"] = str(target)
    if curves.exists():
        target = robustness_figures / "figure5_continuous_spline.png"
        frame = pd.read_csv(curves)
        plot_spline_calibration(frame[frame["rate"].isin(display_rates)], target)
        outputs["figure_continuous_spline"] = str(target)
    if prediction_sets.exists():
        target = robustness_figures / "figure6_prediction_set_comparison.png"
        plot_prediction_set_comparison(pd.read_csv(prediction_sets), target)
        outputs["figure_prediction_sets"] = str(target)
    return outputs


def generate_publication_tables(config: dict[str, Any]) -> dict[str, str]:
    results_dir = Path(config["paths"]["results_dir"])
    source_dir = results_dir / "tables"
    robustness_dir = results_dir / "robustness" / "tables"
    output_dir = results_dir / "publication_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    primary_path = source_dir / "civil_primary.csv"
    if primary_path.exists():
        frame = pd.read_csv(primary_path)
        body = _effect_table(
            frame,
            caption="Primary Civil Comments routing-conditional calibration results.",
            label="tab:primary",
        )
        outputs["publication_table_primary"] = _write_text(
            output_dir / "table_primary.tex", body
        )

    matrix_path = source_dir / "civil_matrix.csv"
    if matrix_path.exists():
        frame = pd.read_csv(matrix_path)
        body = _effect_table(
            frame,
            caption="Extended Civil Comments routing matrix.",
            label="tab:matrix",
        )
        outputs["publication_table_matrix"] = _write_text(
            output_dir / "table_matrix.tex", body
        )

    absolute_path = robustness_dir / "absolute_calibration.csv"
    if absolute_path.exists():
        frame = pd.read_csv(absolute_path)
        body = _latex_table(
            frame,
            columns=[
                ("edge", "Edge", str),
                ("rate", "Route", _percent),
                ("group", "Group", lambda value: str(value).replace("_", " ")),
                ("n", "$n$", _integer),
                ("prevalence", "Prev.", _number),
                ("mean_probability", "Mean $p$", _number),
                ("brier", "Brier", _number),
                ("ici", "ICI", _number),
                ("calibration_intercept", "Intercept", _number),
                ("calibration_slope", "Slope", _number),
            ],
            caption="Absolute calibration diagnostics by routing group.",
            label="tab:absolute-calibration",
        )
        outputs["publication_table_absolute"] = _write_text(
            output_dir / "table_absolute_calibration.tex", body
        )

    binning_path = robustness_dir / "binning_sensitivity.csv"
    if binning_path.exists():
        frame = pd.read_csv(binning_path)
        frame["signed_error_ci"] = frame.apply(
            lambda row: _estimate_ci(
                row["conditional_signed_error"],
                row["conditional_signed_error_ci_low"],
                row["conditional_signed_error_ci_high"],
            ),
            axis=1,
        )
        body = _latex_table(
            frame,
            columns=[
                ("edge", "Edge", str),
                ("rate", "Route", _percent),
                ("n_bins", "Bins", _integer),
                ("signed_error_ci", "$\\Delta$ signed error [95\\% CI]", str),
                ("common_support", "Support", _percent),
                ("supported_bins", "Used", _integer),
            ],
            caption="Sensitivity to the number of pooled downstream-score bins.",
            label="tab:binning-sensitivity",
        )
        outputs["publication_table_binning"] = _write_text(
            output_dir / "table_binning_sensitivity.tex", body
        )

    spline_path = robustness_dir / "continuous_spline.csv"
    permutation_path = robustness_dir / "within_bin_permutation.csv"
    if spline_path.exists() and permutation_path.exists():
        spline = pd.read_csv(spline_path)
        permutation = pd.read_csv(permutation_path)
        if "permutation_q_signed_error" not in permutation:
            from .statistics import benjamini_hochberg

            permutation["permutation_q_signed_error"] = benjamini_hochberg(
                permutation["permutation_p_signed_error"].fillna(1.0).to_numpy()
            )
            permutation["permutation_q_brier"] = benjamini_hochberg(
                permutation["permutation_p_brier"].fillna(1.0).to_numpy()
            )
        frame = spline.merge(
            permutation[
                [
                    "edge",
                    "rate",
                    "permutation_p_signed_error",
                    "permutation_q_signed_error",
                    "permutation_p_brier",
                    "permutation_q_brier",
                ]
            ],
            on=["edge", "rate"],
            validate="one_to_one",
        )
        frame["continuous_ci"] = frame.apply(
            lambda row: _estimate_ci(
                row["continuous_signed_error_shift"],
                row["continuous_signed_error_ci_low"],
                row["continuous_signed_error_ci_high"],
            ),
            axis=1,
        )
        body = _latex_table(
            frame,
            columns=[
                ("edge", "Edge", str),
                ("rate", "Route", _percent),
                ("continuous_ci", "Spline $\\Delta$SE [95\\% CI]", str),
                ("interaction_p", "$p_{\\mathrm{interaction}}$", _pvalue),
                ("permutation_p_signed_error", "$p_{\\mathrm{perm}}$", _pvalue),
                ("permutation_q_signed_error", "$q_{\\mathrm{perm}}$", _pvalue),
                ("routed_common_support", "Support", _percent),
            ],
            caption="Continuous calibration-curve and within-bin permutation robustness.",
            label="tab:continuous-robustness",
        )
        outputs["publication_table_continuous"] = _write_text(
            output_dir / "table_continuous_robustness.tex", body
        )

    prediction_path = robustness_dir / "prediction_set_comparison.csv"
    if prediction_path.exists():
        frame = pd.read_csv(prediction_path)
        frame["signed_error_ci"] = frame.apply(
            lambda row: _estimate_ci(
                row["conditional_signed_error"],
                row["conditional_signed_error_ci_low"],
                row["conditional_signed_error_ci_high"],
            ),
            axis=1,
        )
        body = _latex_table(
            frame,
            columns=[
                ("prediction_set", "Prediction set", str),
                ("model", "Model", str),
                ("prompt_set", "Prompt", str),
                ("edge", "Edge", str),
                ("rate", "Route", _percent),
                ("signed_error_ci", "$\\Delta$ signed error [95\\% CI]", str),
                ("common_support", "Support", _percent),
            ],
            caption="Prompt and model robustness on the primary Civil Comments edges.",
            label="tab:prediction-set-robustness",
        )
        outputs["publication_table_prediction_sets"] = _write_text(
            output_dir / "table_prediction_sets.tex", body
        )
    return outputs


def _effect_table(frame: pd.DataFrame, *, caption: str, label: str) -> str:
    working = frame.copy()
    working["signed_error_ci"] = working.apply(
        lambda row: _estimate_ci(
            row["conditional_signed_error"],
            row["conditional_signed_error_ci_low"],
            row["conditional_signed_error_ci_high"],
        ),
        axis=1,
    )
    working["brier_ci"] = working.apply(
        lambda row: _estimate_ci(
            row["conditional_brier"],
            row["conditional_brier_ci_low"],
            row["conditional_brier_ci_high"],
        ),
        axis=1,
    )
    return _latex_table(
        working,
        columns=[
            ("edge", "Edge", str),
            ("rate", "Route", _percent),
            ("signed_error_ci", "$\\Delta$ signed error [95\\% CI]", str),
            ("q_signed_error", "$q_{SE}$", _pvalue),
            ("brier_ci", "$\\Delta$ Brier [95\\% CI]", str),
            ("q_brier", "$q_{Brier}$", _pvalue),
            ("common_support", "Support", _percent),
            ("supported_bins", "Bins", _integer),
        ],
        caption=caption,
        label=label,
    )


def _latex_table(
    frame: pd.DataFrame,
    *,
    columns: list[tuple[str, str, Callable[[Any], str]]],
    caption: str,
    label: str,
) -> str:
    alignment = "ll" + "r" * max(0, len(columns) - 2)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(header for _, header, _ in columns) + " \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        values = row._asdict()
        rendered = []
        for key, _, formatter in columns:
            value = values[key]
            text = formatter(value)
            rendered.append(text if "$" in text or "\\" in text else _latex_escape(text))
        lines.append(" & ".join(rendered) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _estimate_ci(value: Any, low: Any, high: Any) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.4f} [{float(low):.4f}, {float(high):.4f}]"


def _number(value: Any) -> str:
    return "--" if pd.isna(value) else f"{float(value):.4f}"


def _pvalue(value: Any) -> str:
    if pd.isna(value):
        return "--"
    number = float(value)
    return "$<0.001$" if number < 0.001 else f"{number:.4f}"


def _percent(value: Any) -> str:
    return "--" if pd.isna(value) else f"{100.0 * float(value):.1f}\\%"


def _integer(value: Any) -> str:
    return "--" if pd.isna(value) else f"{round(float(value)):,}"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)
