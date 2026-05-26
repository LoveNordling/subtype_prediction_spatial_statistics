import numpy as np
import pandas as pd

import analyze_feature_discriminativity as mod


def test_compute_feature_discriminativity_returns_expected_auc_fields() -> None:
    df = pd.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1],
            "feat": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        }
    )

    stats = mod.compute_feature_discriminativity(df, "feat")

    assert stats["n_total"] == 6
    assert stats["n_label0"] == 3
    assert stats["n_label1"] == 3
    assert np.isclose(stats["delta_mean"], 0.7)
    assert np.isclose(stats["auc"], 1.0)
    assert np.isclose(stats["auc_strength"], 1.0)
    assert np.isclose(stats["signed_auc_delta"], 0.5)
    assert np.isclose(stats["abs_auc_delta"], 0.5)
    assert stats["cohen_d"] > 0


def test_compute_feature_discriminativity_handles_zero_variance() -> None:
    df = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "feat": [1.0, 1.0, 1.0, 1.0],
        }
    )

    stats = mod.compute_feature_discriminativity(df, "feat")

    assert np.isnan(stats["auc"])
    assert np.isnan(stats["auc_strength"])
    assert np.isnan(stats["signed_auc_delta"])
    assert np.isnan(stats["abs_auc_delta"])
    assert np.isnan(stats["cohen_d"])


def test_add_transfer_columns_marks_direction_and_signal_loss() -> None:
    df = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "source_signed_auc_delta": [0.2, 0.2],
            "target_signed_auc_delta": [0.1, -0.1],
            "source_abs_auc_delta": [0.2, 0.2],
            "target_abs_auc_delta": [0.1, 0.15],
            "source_cohen_d": [1.2, -0.8],
            "target_cohen_d": [0.6, -0.2],
        }
    )

    out = mod.add_transfer_columns(df)

    assert bool(out.loc[out["feature"] == "a", "direction_match"].iloc[0]) is True
    assert bool(out.loc[out["feature"] == "b", "direction_match"].iloc[0]) is False
    assert np.isclose(out.loc[out["feature"] == "a", "signal_loss"].iloc[0], 0.1)
    assert np.isclose(out.loc[out["feature"] == "a", "abs_cohen_d_loss"].iloc[0], 0.6)
