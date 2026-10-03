from __future__ import annotations

import math
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

METRICS = ("raw_brier", "raw_signed_error", "conditional_brier", "conditional_signed_error")


@dataclass(frozen=True)
class Experiment:
    upstream: str
    downstream: str
    rate: float

    @property
    def key(self) -> str:
        return f"{self.upstream}->{self.downstream}@{self.rate:.4f}"


def build_experiments(
    edges: Iterable[Iterable[str]], rates: Iterable[float]
) -> list[Experiment]:
    return [Experiment(str(up), str(down), float(rate)) for up, down in edges for rate in rates]


def analyze_family(
    frame: pd.DataFrame,
    experiments: list[Experiment],
    analysis: dict[str, Any],
    seed: int,
    workers: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    arrays = _extract_arrays(frame, experiments)
    observed = [
        _compute_experiment(arrays, experiment, analysis, np.arange(len(frame), dtype=np.int64))
        for experiment in experiments
    ]
    bootstrap_resamples = int(analysis["bootstrap_resamples"])
    worker_count = _worker_count(workers if workers is not None else analysis["bootstrap_workers"])
    bootstrap = bootstrap_family(
        arrays, experiments, analysis, seed, bootstrap_resamples, worker_count
    )

    alpha = 1.0 - float(analysis["confidence_level"])
    rows: list[dict[str, Any]] = []
    for index, (experiment, point) in enumerate(zip(experiments, observed, strict=True)):
        conditional_supported = bool(
            point["supported_bins"] > 0
            and np.isfinite(point["conditional_signed_error"])
            and np.isfinite(point["conditional_brier"])
        )
        row: dict[str, Any] = {
            "edge": f"{experiment.upstream}->{experiment.downstream}",
            "upstream": experiment.upstream,
            "downstream": experiment.downstream,
            "rate": experiment.rate,
            "conditional_supported": conditional_supported,
            **point,
        }
        for metric in METRICS:
            samples = bootstrap[metric][:, index]
            finite = samples[np.isfinite(samples)]
            # A bootstrap resample can accidentally have common support even when
            # the observed comparison does not. Such draws must not manufacture a
            # confidence interval for an undefined observed estimand.
            if metric.startswith("conditional_") and not conditional_supported:
                finite = np.empty(0, dtype=float)
            row[f"{metric}_ci_low"] = (
                float(np.quantile(finite, alpha / 2)) if len(finite) else np.nan
            )
            row[f"{metric}_ci_high"] = (
                float(np.quantile(finite, 1 - alpha / 2)) if len(finite) else np.nan
            )
            row[f"{metric}_bootstrap_valid"] = len(finite)
        row["p_signed_error"] = (
            bootstrap_sign_pvalue(bootstrap["conditional_signed_error"][:, index])
            if conditional_supported
            else np.nan
        )
        row["p_brier"] = (
            bootstrap_sign_pvalue(bootstrap["conditional_brier"][:, index])
            if conditional_supported
            else np.nan
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    # Unsupported planned comparisons stay in the multiplicity family with p=1.
    result["q_signed_error"] = benjamini_hochberg(result["p_signed_error"].fillna(1).to_numpy())
    result["q_brier"] = benjamini_hochberg(result["p_brier"].fillna(1).to_numpy())
    return result, bootstrap


def bootstrap_family(
    arrays: dict[str, np.ndarray],
    experiments: list[Experiment],
    analysis: dict[str, Any],
    seed: int,
    resamples: int,
    workers: int,
) -> dict[str, np.ndarray]:
    output = {metric: np.full((resamples, len(experiments)), np.nan) for metric in METRICS}
    seeds = np.random.SeedSequence(seed).spawn(resamples)
    seed_values = [int(value.generate_state(1, dtype=np.uint64)[0]) for value in seeds]
    if workers == 1:
        iterator = (_bootstrap_one(value, arrays, experiments, analysis) for value in seed_values)
        results = tqdm(iterator, total=resamples, desc="Bootstrap", unit="sample")
        for row_index, values in enumerate(results):
            _assign_bootstrap_row(output, row_index, values)
        return output

    chunksize = max(1, resamples // (workers * 20))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(arrays, experiments, analysis),
    ) as pool:
        iterator = pool.map(_bootstrap_worker, seed_values, chunksize=chunksize)
        for row_index, values in enumerate(
            tqdm(iterator, total=resamples, desc="Bootstrap", unit="sample")
        ):
            _assign_bootstrap_row(output, row_index, values)
    return output


_WORKER_ARRAYS: dict[str, np.ndarray] | None = None
_WORKER_EXPERIMENTS: list[Experiment] | None = None
_WORKER_ANALYSIS: dict[str, Any] | None = None


def _init_worker(
    arrays: dict[str, np.ndarray], experiments: list[Experiment], analysis: dict[str, Any]
) -> None:
    global _WORKER_ARRAYS, _WORKER_EXPERIMENTS, _WORKER_ANALYSIS
    _WORKER_ARRAYS = arrays
    _WORKER_EXPERIMENTS = experiments
    _WORKER_ANALYSIS = analysis


def _bootstrap_worker(seed: int) -> list[dict[str, float]]:
    if _WORKER_ARRAYS is None or _WORKER_EXPERIMENTS is None or _WORKER_ANALYSIS is None:
        raise RuntimeError("Bootstrap worker was not initialized")
    return _bootstrap_one(seed, _WORKER_ARRAYS, _WORKER_EXPERIMENTS, _WORKER_ANALYSIS)


def _bootstrap_one(
    seed: int,
    arrays: dict[str, np.ndarray],
    experiments: list[Experiment],
    analysis: dict[str, Any],
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    size = len(arrays["tie_id"])
    indices = rng.integers(0, size, size=size, dtype=np.int64)
    return [_compute_experiment(arrays, experiment, analysis, indices) for experiment in experiments]


def _assign_bootstrap_row(
    output: dict[str, np.ndarray], row_index: int, values: list[dict[str, float]]
) -> None:
    for column_index, value in enumerate(values):
        for metric in METRICS:
            output[metric][row_index, column_index] = value[metric]


def _compute_experiment(
    arrays: dict[str, np.ndarray],
    experiment: Experiment,
    analysis: dict[str, Any],
    indices: np.ndarray,
) -> dict[str, float]:
    upstream = arrays[f"p_{experiment.upstream}"][indices]
    downstream = arrays[f"p_{experiment.downstream}"][indices]
    labels = arrays[f"y_{experiment.downstream}"][indices]
    tie_id = arrays["tie_id"][indices]
    routed = top_fraction_mask(upstream, experiment.rate, tie_id)
    signed_error = downstream - labels
    # Expected Bernoulli Brier loss. This equals (p-y)^2 for binary labels and
    # remains the proper expected score when y is a soft annotation in [0, 1].
    brier = downstream**2 - 2.0 * downstream * labels + labels
    raw_signed_error = float(signed_error[routed].mean() - signed_error.mean())
    raw_brier = float(brier[routed].mean() - brier.mean())
    bins = pooled_quantile_bins(
        downstream,
        int(analysis["n_bins"]),
        method=str(analysis.get("bin_quantile_method", "linear")),
    )
    conditional_signed_error, support, supported_bins = conditional_shift(
        signed_error,
        routed,
        bins,
        int(analysis["min_per_group_per_bin"]),
    )
    conditional_brier, _, _ = conditional_shift(
        brier,
        routed,
        bins,
        int(analysis["min_per_group_per_bin"]),
    )
    return {
        "raw_brier": raw_brier,
        "raw_signed_error": raw_signed_error,
        "conditional_brier": conditional_brier,
        "conditional_signed_error": conditional_signed_error,
        "common_support": support,
        "supported_bins": float(supported_bins),
        "routed_n": float(routed.sum()),
    }


def top_fraction_mask(scores: np.ndarray, rate: float, tie_id: np.ndarray) -> np.ndarray:
    if not 0 < rate < 1:
        raise ValueError(f"Routing rate must be in (0, 1): {rate}")
    size = len(scores)
    routed_n = math.ceil(rate * size)
    # The final positional key makes ties deterministic even in a bootstrap sample
    # where the same source example can occur more than once.
    position = np.arange(size, dtype=np.int64)
    order = np.lexsort((position, tie_id, -scores))
    routed = np.zeros(size, dtype=bool)
    routed[order[:routed_n]] = True
    return routed


def pooled_quantile_bins(scores: np.ndarray, n_bins: int, method: str = "linear") -> np.ndarray:
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(scores, quantiles, method=method)
    internal = np.unique(edges[1:-1])
    # Identical scores are never split across bins. Rounding by the OpenJev helper
    # can therefore reduce the number of effective bins.
    return np.searchsorted(internal, scores, side="right").astype(np.int16)


def conditional_shift(
    values: np.ndarray,
    routed: np.ndarray,
    bins: np.ndarray,
    min_per_group: int,
) -> tuple[float, float, int]:
    weighted_sum = 0.0
    routed_supported = 0
    supported_bins = 0
    for bin_id in np.unique(bins):
        in_bin = bins == bin_id
        routed_in_bin = in_bin & routed
        non_routed_in_bin = in_bin & ~routed
        routed_n = int(routed_in_bin.sum())
        non_routed_n = int(non_routed_in_bin.sum())
        if routed_n < min_per_group or non_routed_n < min_per_group:
            continue
        difference = float(values[routed_in_bin].mean() - values[non_routed_in_bin].mean())
        weighted_sum += routed_n * difference
        routed_supported += routed_n
        supported_bins += 1
    total_routed = int(routed.sum())
    support = routed_supported / total_routed if total_routed else np.nan
    shift = weighted_sum / routed_supported if routed_supported else np.nan
    return float(shift), float(support), supported_bins


def bootstrap_sign_pvalue(samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return np.nan
    lower = int(np.count_nonzero(finite <= 0))
    upper = int(np.count_nonzero(finite >= 0))
    return min(1.0, 2.0 * (min(lower, upper) + 1) / (len(finite) + 1))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
        raise ValueError("BH p-values must be finite and in [0, 1]")
    size = len(values)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * size / np.arange(1, size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(size, dtype=float)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def aggregate_controls(
    frame: pd.DataFrame,
    downstreams: Iterable[str],
    rates: Iterable[float],
    subsets: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    size = len(frame)
    oracle_scores = frame["soft_toxicity"].to_numpy(float)
    tie_id = np.arange(size, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for downstream in downstreams:
        probability = frame[f"p_{downstream}"].to_numpy(float)
        labels = frame[f"y_{downstream}"].to_numpy(float)
        signed_error = probability - labels
        brier = signed_error**2
        for rate in rates:
            routed_n = math.ceil(rate * size)
            random_brier = np.empty(subsets)
            random_se = np.empty(subsets)
            for index in range(subsets):
                selected = rng.choice(size, size=routed_n, replace=False)
                random_brier[index] = brier[selected].mean() - brier.mean()
                random_se[index] = signed_error[selected].mean() - signed_error.mean()
            oracle = top_fraction_mask(oracle_scores, float(rate), tie_id)
            rows.append(
                {
                    "edge": f"toxicity->{downstream}",
                    "rate": float(rate),
                    "random_brier": float(random_brier.mean()),
                    "random_brier_low": float(np.quantile(random_brier, 0.025)),
                    "random_brier_high": float(np.quantile(random_brier, 0.975)),
                    "random_signed_error": float(random_se.mean()),
                    "random_signed_error_low": float(np.quantile(random_se, 0.025)),
                    "random_signed_error_high": float(np.quantile(random_se, 0.975)),
                    "oracle_brier": float(brier[oracle].mean() - brier.mean()),
                    "oracle_signed_error": float(
                        signed_error[oracle].mean() - signed_error.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _extract_arrays(
    frame: pd.DataFrame, experiments: list[Experiment]
) -> dict[str, np.ndarray]:
    names = {experiment.upstream for experiment in experiments} | {
        experiment.downstream for experiment in experiments
    }
    required = {f"p_{name}" for name in names} | {
        f"y_{experiment.downstream}" for experiment in experiments
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    arrays = {name: frame[name].to_numpy(float) for name in sorted(required)}
    arrays["tie_id"] = np.arange(len(frame), dtype=np.int64)
    return arrays


def _worker_count(value: int | None) -> int:
    if value is None or int(value) <= 0:
        return max(1, min(16, (os.cpu_count() or 2) - 1))
    return int(value)
