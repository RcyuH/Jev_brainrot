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
