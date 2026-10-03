from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2

from .config import config_digest
from .io import atomic_write_json, atomic_write_parquet, sha256_file
from .plotting import (
    plot_absolute_calibration,
    plot_binning_sensitivity,
    plot_prediction_set_comparison,
    plot_spline_calibration,
)
from .reporting import generate_publication_tables
from .statistics import (
    Experiment,
    analyze_family,
    benjamini_hochberg,
    build_experiments,
    conditional_shift,
    pooled_quantile_bins,
    top_fraction_mask,
)


@dataclass(frozen=True)
class SplineSpec:
    center: float
    scale: float
    knots: tuple[float, ...]


def run_robustness(
    config: dict[str, Any], *, quick: bool = False, strict_prediction_sets: bool = False
) -> dict[str, Any]:
    paths = config["paths"]
    settings = dict(config.get("robustness", {}))
    results_dir = Path(paths["results_dir"])
    output_dir = results_dir / "robustness"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    bootstrap_dir = output_dir / "bootstrap"
    for directory in (tables_dir, figures_dir, bootstrap_dir):
        directory.mkdir(parents=True, exist_ok=True)

    civil_path = Path(paths["predictions_dir"]) / "civil_comments.parquet"
    civil = pd.read_parquet(civil_path)
    analysis = dict(config["analysis"])
    resamples = int(settings.get("bootstrap_resamples", analysis["bootstrap_resamples"]))
    permutations = int(settings.get("permutation_resamples", 2000))
    spline_draws = int(settings.get("spline_draws", 2000))
    if quick:
        resamples = min(50, resamples)
        permutations = min(100, permutations)
        spline_draws = min(100, spline_draws)
    analysis["bootstrap_resamples"] = resamples
    seed = int(config["seed"])
    experiments = build_experiments(
        config["experiments"]["civil_primary"], analysis["primary_rates"]
    )

    outputs: dict[str, str] = {}
    bin_tables: list[pd.DataFrame] = []
    bin_draws: dict[str, np.ndarray] = {}
    for offset, n_bins in enumerate(settings.get("bin_counts", [5, 10, 20])):
        variant = dict(analysis)
        variant["n_bins"] = int(n_bins)
        table, draws = analyze_family(civil, experiments, variant, seed + 1000 + offset)
        table.insert(4, "n_bins", int(n_bins))
        table.insert(5, "min_per_group", int(variant["min_per_group_per_bin"]))
        bin_tables.append(table)
        for metric, values in draws.items():
            bin_draws[f"bins_{int(n_bins)}_{metric}"] = values
    bin_sensitivity = pd.concat(bin_tables, ignore_index=True)
    _write_table(bin_sensitivity, tables_dir / "binning_sensitivity", outputs)
    bin_bootstrap_path = bootstrap_dir / "binning_sensitivity.npz"
    np.savez_compressed(bin_bootstrap_path, **bin_draws)
    outputs["binning_sensitivity_bootstrap"] = str(bin_bootstrap_path)

    absolute, reliability = absolute_calibration_tables(
        civil,
        experiments,
        n_bins=int(settings.get("absolute_bins", analysis["n_bins"])),
        ridge=float(settings.get("spline_ridge", 1e-6)),
    )
    display_rates = {float(value) for value in settings.get("display_rates", [0.10, 0.05])}
    reliability_display = reliability[reliability["rate"].isin(display_rates)].copy()
    _write_table(absolute, tables_dir / "absolute_calibration", outputs)
    _write_table(reliability, tables_dir / "reliability_points", outputs)

    permutation = permutation_tests(
        civil,
        experiments,
        analysis,
        resamples=permutations,
        seed=seed + 2000,
    )
    _write_table(permutation, tables_dir / "within_bin_permutation", outputs)

    spline_rows: list[dict[str, Any]] = []
    curve_tables: list[pd.DataFrame] = []
    for index, experiment in enumerate(experiments):
        summary, curves = routing_spline_analysis(
            civil,
            experiment,
            draws=spline_draws,
            seed=seed + 3000 + index,
            trim_quantile=float(settings.get("spline_trim_quantile", 0.01)),
            ridge=float(settings.get("spline_ridge", 1e-6)),
        )
        spline_rows.append(summary)
        curve_tables.append(curves)
    spline = pd.DataFrame(spline_rows)
    spline_curves = pd.concat(curve_tables, ignore_index=True)
    _write_table(spline, tables_dir / "continuous_spline", outputs)
    _write_table(spline_curves, tables_dir / "continuous_spline_curves", outputs)

    label_sensitivity = label_robustness(
        civil, experiments, analysis, seed=seed + 4000
    )
    _write_table(label_sensitivity, tables_dir / "label_sensitivity", outputs)

    prediction_sets, missing_sets = prediction_set_robustness(
        config,
        experiments,
        analysis,
        seed=seed + 5000,
        strict=strict_prediction_sets,
    )
    if len(prediction_sets):
        _write_table(prediction_sets, tables_dir / "prediction_set_comparison", outputs)

    figure_paths = {
        "figure_absolute_calibration": figures_dir / "figure3_absolute_calibration.png",
        "figure_binning_sensitivity": figures_dir / "figure4_binning_sensitivity.png",
        "figure_continuous_spline": figures_dir / "figure5_continuous_spline.png",
    }
    plot_absolute_calibration(reliability_display, figure_paths["figure_absolute_calibration"])
    plot_binning_sensitivity(bin_sensitivity, figure_paths["figure_binning_sensitivity"])
    plot_spline_calibration(
        spline_curves[spline_curves["rate"].isin(display_rates)],
        figure_paths["figure_continuous_spline"],
    )
    if len(prediction_sets):
        figure_paths["figure_prediction_sets"] = (
            figures_dir / "figure6_prediction_set_comparison.png"
        )
        plot_prediction_set_comparison(
            prediction_sets, figure_paths["figure_prediction_sets"]
        )
    outputs.update({key: str(value) for key, value in figure_paths.items()})

    publication_outputs = generate_publication_tables(config)
    outputs.update(publication_outputs)
    manifest = {
        "config_sha256": config_digest(config),
        "quick_mode": quick,
        "input": {"path": str(civil_path), "sha256": sha256_file(civil_path)},
        "protocol": {
            "bin_counts": [int(value) for value in settings.get("bin_counts", [5, 10, 20])],
            "bootstrap_resamples": resamples,
            "permutation_resamples": permutations,
            "spline_draws": spline_draws,
            "spline_basis": "restricted_cubic_spline_with_five_quantile_knots",
            "spline_trim_quantile": float(settings.get("spline_trim_quantile", 0.01)),
            "soft_label_brier": "p^2 - 2*p*y + y",
        },
        "missing_prediction_sets": missing_sets,
        "outputs": outputs,
    }
    atomic_write_json(manifest, output_dir / "manifest.json")
    return manifest


def absolute_calibration_tables(
    frame: pd.DataFrame,
    experiments: list[Experiment],
    *,
    n_bins: int,
    ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    tie_id = np.arange(len(frame), dtype=np.int64)
    for experiment in experiments:
        p_up = frame[f"p_{experiment.upstream}"].to_numpy(float)
        probability = frame[f"p_{experiment.downstream}"].to_numpy(float)
        outcome = frame[f"y_{experiment.downstream}"].to_numpy(float)
        routed = top_fraction_mask(p_up, experiment.rate, tie_id)
        bins = pooled_quantile_bins(probability, n_bins)
        for group, mask in (("routed", routed), ("non_routed", ~routed)):
            group_p = probability[mask]
            group_y = outcome[mask]
            brier = group_p**2 - 2.0 * group_p * group_y + group_y
            intercept, slope = calibration_intercept_slope(group_p, group_y, ridge=ridge)
            smooth = fit_probability_spline(group_p, group_y, ridge=ridge)
            fitted = predict_probability_spline(group_p, smooth)
            ece_numerator = 0.0
            for bin_id in np.unique(bins):
                selected = mask & (bins == bin_id)
                count = int(selected.sum())
                if not count:
                    continue
                mean_p = float(probability[selected].mean())
                mean_y = float(outcome[selected].mean())
                ece_numerator += count * abs(mean_p - mean_y)
                point_rows.append(
                    {
                        "edge": f"{experiment.upstream}->{experiment.downstream}",
                        "upstream": experiment.upstream,
                        "downstream": experiment.downstream,
                        "rate": experiment.rate,
                        "group": group,
                        "bin": int(bin_id),
                        "n": count,
                        "mean_probability": mean_p,
                        "outcome_rate": mean_y,
                        "signed_error": mean_p - mean_y,
                    }
                )
            metric_rows.append(
                {
                    "edge": f"{experiment.upstream}->{experiment.downstream}",
                    "upstream": experiment.upstream,
                    "downstream": experiment.downstream,
                    "rate": experiment.rate,
                    "group": group,
                    "n": int(mask.sum()),
                    "prevalence": float(group_y.mean()),
                    "mean_probability": float(group_p.mean()),
                    "signed_error": float((group_p - group_y).mean()),
                    "brier": float(brier.mean()),
                    "ece": ece_numerator / int(mask.sum()),
                    "ici": float(np.mean(np.abs(group_p - fitted))),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(point_rows)


def calibration_intercept_slope(
    probability: np.ndarray, outcome: np.ndarray, *, ridge: float
) -> tuple[float, float]:
    logits = _logit(probability)
    design = np.column_stack([np.ones(len(logits)), logits])
    beta, _, _, _ = _fit_binomial(design, outcome, np.ones(len(outcome)), ridge=ridge)
    return float(beta[0]), float(beta[1])


def fit_probability_spline(
    probability: np.ndarray, outcome: np.ndarray, *, ridge: float
) -> tuple[np.ndarray, np.ndarray, SplineSpec]:
    aggregated = (
        pd.DataFrame({"p": probability, "y": outcome})
        .groupby("p", sort=True, as_index=False)
        .agg(successes=("y", "sum"), trials=("y", "size"))
    )
    spec = _spline_spec(aggregated["p"].to_numpy(float))
    design = _spline_basis(aggregated["p"].to_numpy(float), spec)
    beta, covariance, _, _ = _fit_binomial(
        design,
        aggregated["successes"].to_numpy(float),
        aggregated["trials"].to_numpy(float),
        ridge=ridge,
    )
    return beta, covariance, spec


def predict_probability_spline(
    probability: np.ndarray, model: tuple[np.ndarray, np.ndarray, SplineSpec]
) -> np.ndarray:
    beta, _, spec = model
    return expit(_spline_basis(probability, spec) @ beta)


def permutation_tests(
    frame: pd.DataFrame,
    experiments: list[Experiment],
    analysis: dict[str, Any],
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tie_id = np.arange(len(frame), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        upstream = frame[f"p_{experiment.upstream}"].to_numpy(float)
        downstream = frame[f"p_{experiment.downstream}"].to_numpy(float)
        outcome = frame[f"y_{experiment.downstream}"].to_numpy(float)
        routed = top_fraction_mask(upstream, experiment.rate, tie_id)
        bins = pooled_quantile_bins(downstream, int(analysis["n_bins"]))
        signed_error = downstream - outcome
        brier = downstream**2 - 2.0 * downstream * outcome + outcome
        observed_se, support, supported_bins = conditional_shift(
            signed_error, routed, bins, int(analysis["min_per_group_per_bin"])
        )
        observed_brier, _, _ = conditional_shift(
            brier, routed, bins, int(analysis["min_per_group_per_bin"])
        )
        draws_se = np.full(resamples, np.nan)
        draws_brier = np.full(resamples, np.nan)
        bin_indices = [np.flatnonzero(bins == value) for value in np.unique(bins)]
        routed_counts = [int(routed[index].sum()) for index in bin_indices]
        for draw in range(resamples):
            permuted = np.zeros(len(frame), dtype=bool)
            for index, count in zip(bin_indices, routed_counts, strict=True):
                if count:
                    permuted[rng.choice(index, size=count, replace=False)] = True
            draws_se[draw] = conditional_shift(
                signed_error,
                permuted,
                bins,
                int(analysis["min_per_group_per_bin"]),
            )[0]
            draws_brier[draw] = conditional_shift(
                brier,
                permuted,
                bins,
                int(analysis["min_per_group_per_bin"]),
            )[0]
        rows.append(
            {
                "edge": f"{experiment.upstream}->{experiment.downstream}",
                "upstream": experiment.upstream,
                "downstream": experiment.downstream,
                "rate": experiment.rate,
                "conditional_signed_error": observed_se,
                "permutation_p_signed_error": _randomization_pvalue(draws_se, observed_se),
                "conditional_brier": observed_brier,
                "permutation_p_brier": _randomization_pvalue(draws_brier, observed_brier),
                "common_support": support,
                "supported_bins": supported_bins,
                "permutation_resamples": resamples,
            }
        )
    result = pd.DataFrame(rows)
    result["permutation_q_signed_error"] = benjamini_hochberg(
        result["permutation_p_signed_error"].fillna(1.0).to_numpy()
    )
    result["permutation_q_brier"] = benjamini_hochberg(
        result["permutation_p_brier"].fillna(1.0).to_numpy()
    )
    return result


def routing_spline_analysis(
    frame: pd.DataFrame,
    experiment: Experiment,
    *,
    draws: int,
    seed: int,
    trim_quantile: float,
    ridge: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    upstream = frame[f"p_{experiment.upstream}"].to_numpy(float)
    probability = frame[f"p_{experiment.downstream}"].to_numpy(float)
    outcome = frame[f"y_{experiment.downstream}"].to_numpy(float)
    routed = top_fraction_mask(upstream, experiment.rate, np.arange(len(frame)))
    low = max(
        float(np.quantile(probability[routed], trim_quantile)),
        float(np.quantile(probability[~routed], trim_quantile)),
    )
    high = min(
        float(np.quantile(probability[routed], 1.0 - trim_quantile)),
        float(np.quantile(probability[~routed], 1.0 - trim_quantile)),
    )
    if not high > low:
        raise ValueError(
            f"No continuous common score support for {experiment.key}: [{low}, {high}]"
        )
    common = (probability >= low) & (probability <= high)
    aggregated = (
        pd.DataFrame(
            {"p": probability[common], "routed": routed[common].astype(int), "y": outcome[common]}
        )
        .groupby(["p", "routed"], sort=True, as_index=False)
        .agg(successes=("y", "sum"), trials=("y", "size"))
    )
    p_agg = aggregated["p"].to_numpy(float)
    r_agg = aggregated["routed"].to_numpy(float)
    spec = _spline_spec(p_agg)
    base = _spline_basis(p_agg, spec)
    full_design = np.column_stack([base, r_agg[:, None] * base])
    successes = aggregated["successes"].to_numpy(float)
    trials = aggregated["trials"].to_numpy(float)
    beta, covariance, log_likelihood, converged = _fit_binomial(
        full_design, successes, trials, ridge=ridge
    )
    _, _, reduced_log_likelihood, _ = _fit_binomial(
        base, successes, trials, ridge=ridge
    )
    statistic = max(0.0, 2.0 * (log_likelihood - reduced_log_likelihood))
    interaction_p = float(chi2.sf(statistic, base.shape[1]))

    evaluation, evaluation_counts = np.unique(
        probability[routed & common], return_counts=True
    )
    evaluation_weights = evaluation_counts / evaluation_counts.sum()
    eval_base = _spline_basis(evaluation, spec)
    eval_zero = np.column_stack([eval_base, np.zeros_like(eval_base)])
    eval_one = np.column_stack([eval_base, eval_base])
    q_zero = expit(eval_zero @ beta)
    q_one = expit(eval_one @ beta)
    signed_gap = float(np.sum(evaluation_weights * (q_zero - q_one)))
    brier_gap = float(
        np.sum(evaluation_weights * (1.0 - 2.0 * evaluation) * (q_one - q_zero))
    )

    rng = np.random.default_rng(seed)
    coefficient_draws = rng.multivariate_normal(beta, _nearest_psd(covariance), size=draws)
    signed_draws = np.empty(draws)
    brier_draws = np.empty(draws)
    for index, draw in enumerate(coefficient_draws):
        draw_zero = expit(eval_zero @ draw)
        draw_one = expit(eval_one @ draw)
        signed_draws[index] = np.sum(evaluation_weights * (draw_zero - draw_one))
        brier_draws[index] = np.sum(
            evaluation_weights * (1.0 - 2.0 * evaluation) * (draw_one - draw_zero)
        )

    grid = np.linspace(low, high, 101)
    grid_base = _spline_basis(grid, spec)
    grid_zero = np.column_stack([grid_base, np.zeros_like(grid_base)])
    grid_one = np.column_stack([grid_base, grid_base])
    curve_rows: list[dict[str, Any]] = []
    for group, design in (("non_routed", grid_zero), ("routed", grid_one)):
        estimates = expit(design @ beta)
        draw_estimates = expit(design @ coefficient_draws.T)
        low_curve, high_curve = np.quantile(draw_estimates, [0.025, 0.975], axis=1)
        for p_value, estimate, ci_low, ci_high in zip(
            grid, estimates, low_curve, high_curve, strict=True
        ):
            curve_rows.append(
                {
                    "edge": f"{experiment.upstream}->{experiment.downstream}",
                    "rate": experiment.rate,
                    "group": group,
                    "probability": p_value,
                    "estimated_outcome": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    summary = {
        "edge": f"{experiment.upstream}->{experiment.downstream}",
        "upstream": experiment.upstream,
        "downstream": experiment.downstream,
        "rate": experiment.rate,
        "continuous_signed_error_shift": signed_gap,
        "continuous_signed_error_ci_low": float(np.quantile(signed_draws, 0.025)),
        "continuous_signed_error_ci_high": float(np.quantile(signed_draws, 0.975)),
        "continuous_brier_shift": brier_gap,
        "continuous_brier_ci_low": float(np.quantile(brier_draws, 0.025)),
        "continuous_brier_ci_high": float(np.quantile(brier_draws, 0.975)),
        "interaction_lrt": statistic,
        "interaction_df": base.shape[1],
        "interaction_p": interaction_p,
        "common_low": low,
        "common_high": high,
        "routed_common_support": float((routed & common).sum() / routed.sum()),
        "fit_converged": converged,
    }
    return summary, pd.DataFrame(curve_rows)


def label_robustness(
    frame: pd.DataFrame,
    experiments: list[Experiment],
    analysis: dict[str, Any],
    *,
    seed: int,
) -> pd.DataFrame:
    hard, _ = analyze_family(frame, experiments, analysis, seed)
    hard.insert(4, "label_mode", "hard_threshold_0.5")
    soft = frame.copy()
    downstreams = {experiment.downstream for experiment in experiments}
    missing = [name for name in downstreams if f"soft_{name}" not in frame]
    if missing:
        raise ValueError(f"Soft-label robustness requires soft targets for: {missing}")
    for downstream in downstreams:
        soft[f"y_{downstream}"] = soft[f"soft_{downstream}"].astype(float)
    soft_results, _ = analyze_family(soft, experiments, analysis, seed + 1)
    soft_results.insert(4, "label_mode", "soft_annotation")
    return pd.concat([hard, soft_results], ignore_index=True)


def prediction_set_robustness(
    config: dict[str, Any],
    experiments: list[Experiment],
    analysis: dict[str, Any],
    *,
    seed: int,
    strict: bool,
) -> tuple[pd.DataFrame, list[str]]:
    specifications = config.get("robustness", {}).get("prediction_sets", [])
    tables: list[pd.DataFrame] = []
    missing: list[str] = []
    for index, specification in enumerate(specifications):
        path = Path(specification["path"])
        if not path.exists():
            missing.append(f"{specification['name']}: {path}")
            continue
        table, _ = analyze_family(pd.read_parquet(path), experiments, analysis, seed + index)
        table.insert(0, "prediction_set", str(specification["name"]))
        table.insert(1, "model", str(specification.get("model", "unknown")))
        table.insert(2, "prompt_set", str(specification.get("prompt_set", "unknown")))
        tables.append(table)
    if strict and missing:
        raise FileNotFoundError("Missing configured prediction sets: " + "; ".join(missing))
    if not tables:
        return pd.DataFrame(), missing
    return pd.concat(tables, ignore_index=True), missing


def _fit_binomial(
    design: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    design = np.asarray(design, dtype=float)
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    penalty = np.full(design.shape[1], ridge, dtype=float)
    penalty[0] = 0.0
    initial = np.zeros(design.shape[1], dtype=float)
    mean = float(np.clip(successes.sum() / trials.sum(), 1e-6, 1 - 1e-6))
    initial[0] = math.log(mean / (1.0 - mean))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ beta
        value = float(np.sum(trials * np.logaddexp(0.0, eta) - successes * eta))
        value += 0.5 * float(np.sum(penalty * beta**2))
        gradient = design.T @ (trials * expit(eta) - successes) + penalty * beta
        return value, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-11},
    )
    beta = np.asarray(result.x, dtype=float)
    fitted = expit(design @ beta)
    weights = trials * fitted * (1.0 - fitted)
    hessian = design.T @ (weights[:, None] * design) + np.diag(penalty)
    covariance = np.linalg.pinv(hessian, hermitian=True)
    eta = design @ beta
    log_likelihood = float(np.sum(successes * eta - trials * np.logaddexp(0.0, eta)))
    return beta, covariance, log_likelihood, bool(result.success)


def _spline_spec(probability: np.ndarray) -> SplineSpec:
    logits = _logit(probability)
    center = float(np.mean(logits))
    scale = float(np.std(logits))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    standardized = (logits - center) / scale
    knots_array = np.quantile(standardized, [0.05, 0.275, 0.5, 0.725, 0.95])
    if len(np.unique(knots_array)) < 5:
        knots_array = np.linspace(float(standardized.min()), float(standardized.max()), 5)
    knots = tuple(float(value) for value in knots_array)
    return SplineSpec(center=center, scale=scale, knots=knots)


def _spline_basis(probability: np.ndarray, spec: SplineSpec) -> np.ndarray:
    z = (_logit(probability) - spec.center) / spec.scale
    knots = np.asarray(spec.knots, dtype=float)
    if len(knots) < 3 or not knots[-1] > knots[-2]:
        raise ValueError("Restricted cubic spline requires at least three distinct knots")
    scale = max((knots[-1] - knots[0]) ** 2, 1e-8)
    columns = [np.ones(len(z)), z]
    for knot in knots[:-2]:
        term = np.maximum(z - knot, 0.0) ** 3
        term -= np.maximum(z - knots[-2], 0.0) ** 3 * (
            (knots[-1] - knot) / (knots[-1] - knots[-2])
        )
        term += np.maximum(z - knots[-1], 0.0) ** 3 * (
            (knots[-2] - knot) / (knots[-1] - knots[-2])
        )
        columns.append(term / scale)
    return np.column_stack(columns)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    values = np.clip(values, 1e-12, None)
    return (vectors * values) @ vectors.T


def _randomization_pvalue(draws: np.ndarray, observed: float) -> float:
    finite = draws[np.isfinite(draws)]
    if not len(finite) or not np.isfinite(observed):
        return np.nan
    return float((np.count_nonzero(np.abs(finite) >= abs(observed)) + 1) / (len(finite) + 1))


def _write_table(frame: pd.DataFrame, base_path: Path, outputs: dict[str, str]) -> None:
    csv_path = base_path.with_suffix(".csv")
    parquet_path = base_path.with_suffix(".parquet")
    frame.to_csv(csv_path, index=False)
    atomic_write_parquet(frame, parquet_path)
    outputs[f"{base_path.name}_csv"] = str(csv_path)
    outputs[f"{base_path.name}_parquet"] = str(parquet_path)
