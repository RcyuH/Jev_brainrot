import numpy as np
import pandas as pd

from jevcal.statistics import (
    Experiment,
    analyze_family,
    benjamini_hochberg,
    bootstrap_sign_pvalue,
    conditional_shift,
    pooled_quantile_bins,
    top_fraction_mask,
)


def test_top_fraction_uses_ceil_and_stable_ties():
    scores = np.array([0.9, 0.8, 0.8, 0.1])
    tie_id = np.array([3, 2, 1, 0])
    routed = top_fraction_mask(scores, 0.5, tie_id)
    assert routed.tolist() == [True, False, True, False]
    assert top_fraction_mask(scores, 0.26, tie_id).sum() == 2


def test_quantile_bins_never_split_tied_scores():
    scores = np.array([0.1, 0.1, 0.1, 0.4, 0.5, 0.9])
    bins = pooled_quantile_bins(scores, 4)
    assert len(set(bins[:3])) == 1


def test_conditional_shift_weights_by_routed_count():
    values = np.array([3.0, 1.0, 9.0, 1.0, 1.0, 1.0])
    routed = np.array([True, False, True, True, False, False])
    bins = np.array([0, 0, 1, 1, 1, 1])
    # Bin shifts are 2 and 4, weighted by routed counts 1 and 2.
    shift, support, count = conditional_shift(values, routed, bins, min_per_group=1)
    assert np.isclose(shift, 10 / 3)
    assert support == 1.0
    assert count == 2


def test_bootstrap_sign_pvalue_and_bh():
    samples = np.ones(2000)
    assert np.isclose(bootstrap_sign_pvalue(samples), 2 / 2001)
    adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 1.0]))
    assert np.allclose(adjusted, [0.04, 0.0533333333, 0.0533333333, 1.0])


def test_small_family_analysis_runs():
    size = 80
    probability = np.linspace(0.01, 0.99, size)
    frame = pd.DataFrame(
        {
            "p_up": probability[::-1],
            "p_down": probability,
            "y_down": (probability > 0.6).astype(int),
        }
    )
    analysis = {
        "n_bins": 4,
        "min_per_group_per_bin": 1,
        "bin_quantile_method": "linear",
        "bootstrap_resamples": 10,
        "bootstrap_workers": 1,
        "confidence_level": 0.95,
    }
    result, draws = analyze_family(
        frame, [Experiment("up", "down", 0.25)], analysis, seed=7, workers=1
    )
    assert len(result) == 1
    assert draws["conditional_signed_error"].shape == (10, 1)
    assert result.loc[0, "routed_n"] == 20


def test_small_family_analysis_multiprocessing():
    size = 40
    probability = np.linspace(0.01, 0.99, size)
    frame = pd.DataFrame(
        {
            "p_up": probability[::-1],
            "p_down": probability,
            "y_down": (probability > 0.5).astype(int),
        }
    )
    analysis = {
        "n_bins": 2,
        "min_per_group_per_bin": 1,
        "bin_quantile_method": "linear",
        "bootstrap_resamples": 4,
        "bootstrap_workers": 2,
        "confidence_level": 0.95,
    }
    result, draws = analyze_family(
        frame, [Experiment("up", "down", 0.25)], analysis, seed=9, workers=2
    )
    assert len(result) == 1
    assert draws["raw_brier"].shape == (4, 1)


def test_unsupported_observed_comparison_has_no_conditional_ci_or_pvalue():
    size = 40
    frame = pd.DataFrame(
        {
            "p_up": np.linspace(0.0, 1.0, size),
            "p_down": np.linspace(0.05, 0.95, size),
            "y_down": np.zeros(size),
        }
    )
    analysis = {
        "n_bins": 4,
        "min_per_group_per_bin": 50,
        "bin_quantile_method": "linear",
        "bootstrap_resamples": 5,
        "bootstrap_workers": 1,
        "confidence_level": 0.95,
    }
    result, _ = analyze_family(
        frame, [Experiment("up", "down", 0.25)], analysis, seed=11, workers=1
    )
    row = result.iloc[0]
    assert not row["conditional_supported"]
    assert np.isnan(row["conditional_signed_error_ci_low"])
    assert row["conditional_signed_error_bootstrap_valid"] == 0
    assert np.isnan(row["p_signed_error"])
    assert row["q_signed_error"] == 1.0


def test_soft_target_uses_expected_bernoulli_brier():
    frame = pd.DataFrame(
        {
            "p_up": [0.9, 0.8, 0.2, 0.1],
            "p_down": [0.8, 0.6, 0.4, 0.2],
            "y_down": [0.75, 0.50, 0.25, 0.10],
        }
    )
    analysis = {
        "n_bins": 2,
        "min_per_group_per_bin": 1,
        "bin_quantile_method": "linear",
        "bootstrap_resamples": 2,
        "bootstrap_workers": 1,
        "confidence_level": 0.95,
    }
    result, _ = analyze_family(
        frame, [Experiment("up", "down", 0.5)], analysis, seed=13, workers=1
    )
    p = frame["p_down"].to_numpy()
    y = frame["y_down"].to_numpy()
    expected = p**2 - 2 * p * y + y
    manual = expected[:2].mean() - expected.mean()
    assert np.isclose(result.loc[0, "raw_brier"], manual)
