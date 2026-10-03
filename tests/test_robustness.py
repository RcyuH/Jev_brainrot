import numpy as np
import pandas as pd

from jevcal.robustness import (
    absolute_calibration_tables,
    routing_spline_analysis,
)
from jevcal.statistics import Experiment


def _frame(size: int = 500) -> pd.DataFrame:
    probability = np.linspace(0.02, 0.98, size)
    rng = np.random.default_rng(5)
    routed_score = rng.uniform(size=size)
    outcome_probability = np.clip(probability - 0.12 * (routed_score > 0.8), 0.01, 0.99)
    return pd.DataFrame(
        {
            "p_up": routed_score,
            "p_down": probability,
            "y_down": rng.binomial(1, outcome_probability),
        }
    )


def test_absolute_calibration_outputs_both_groups_and_common_bins():
    metrics, points = absolute_calibration_tables(
        _frame(), [Experiment("up", "down", 0.2)], n_bins=5, ridge=1e-6
    )
    assert set(metrics["group"]) == {"routed", "non_routed"}
    assert set(points["group"]) == {"routed", "non_routed"}
    assert metrics["calibration_slope"].notna().all()


def test_continuous_spline_returns_finite_effect_and_curves():
    summary, curves = routing_spline_analysis(
        _frame(),
        Experiment("up", "down", 0.2),
        draws=20,
        seed=17,
        trim_quantile=0.01,
        ridge=1e-5,
    )
    assert np.isfinite(summary["continuous_signed_error_shift"])
    assert summary["routed_common_support"] > 0
    assert set(curves["group"]) == {"routed", "non_routed"}
    assert len(curves) == 202
