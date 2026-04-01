import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def matched_module(load_script_module, stub_spatial_utils_module):
    return load_script_module(
        "make_cellpose_matched_to_external_v2.py",
        "make_cellpose_matched_to_external_v2_under_test",
        stub_modules={"spatial_utils": stub_spatial_utils_module},
    )


def test_sample_name_and_suffix_helpers(matched_module):
    filename = "BOMI2_TIL_1_Core[1,10,A]_[5091,35249]_component_data.tif"

    assert matched_module.extract_sample_name_from_filename(filename) == "BOMI2_TIL_1_[10,A]"
    assert matched_module.insert_suffix_before_core(filename, "__thin1") == (
        "BOMI2_TIL_1__thin1_Core[1,10,A]_[5091,35249]_component_data.tif"
    )
    assert matched_module.insert_suffix_in_canonical_sample_name("BOMI2_TIL_1_[10,A]", "__thin1") == (
        "BOMI2_TIL_1__thin1_[10,A]"
    )
    assert matched_module.canonicalize_samples_csv_name("BOMI2_TIL_1_Core[1,10,A]_1") == "BOMI2_TIL_1_[10,A]"


def test_compute_sample_median_nnd_for_all_and_stroma(matched_module):
    df = pd.DataFrame(
        {
            "sample_name": ["s1"] * 6 + ["s2"] * 6,
            "x": [0, 1, 2, 3, 4, 5, 0, 0, 10, 10, 20, 20],
            "y": [0, 0, 0, 0, 0, 0, 0, 1, 10, 11, 20, 21],
            "CK": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        }
    )

    all_stats = matched_module.compute_sample_median_nnd(df, min_cells=5, compartment="all")
    stroma_stats = matched_module.compute_sample_median_nnd(df, min_cells=4, compartment="stroma")

    assert all_stats["sample_name"].tolist() == ["s1", "s2"]
    assert np.allclose(all_stats["median_nnd_all"].to_numpy(), [1.0, 1.0])
    assert all_stats["n_cells"].tolist() == [6, 6]

    assert stroma_stats["sample_name"].tolist() == ["s1", "s2"]
    assert np.allclose(stroma_stats["median_nnd_stroma"].to_numpy(), [1.0, 1.0])
    assert stroma_stats["n_cells_stroma"].tolist() == [6, 4]

    with pytest.raises(ValueError, match="Unknown compartment"):
        matched_module.compute_sample_median_nnd(df, min_cells=4, compartment="tumor")


def test_binary_search_keep_mask_returns_boolean_mask_with_minimum_size(matched_module):
    coords = np.array([[float(i), 0.0] for i in range(20)])

    keep_mask = matched_module._binary_search_keep_mask(
        coords,
        target_nnd=2.0,
        min_cells=5,
        rng=np.random.default_rng(0),
        p_init=0.5,
        n_binsrch=8,
        tol_rel=0.2,
    )

    assert keep_mask is not None
    assert keep_mask.dtype == bool
    assert keep_mask.shape == (20,)
    assert keep_mask.sum() >= 5


def test_select_feature_panel_caps_redundant_feature_families(matched_module):
    shift_summary = pd.DataFrame(
        {
            "feature": [
                "cancer_ripley_L_20.0",
                "cancer_ripley_L_40.0",
                "stroma_ripley_L_20.0",
                "tumor_to_stroma_mean_dist",
                "degree_centrality_ratio",
            ],
            "family": [
                "cancer_ripley",
                "cancer_ripley",
                "stroma_ripley",
                "interface_distance",
                "centrality_ratio",
            ],
            "cohens_d": [1.5, 1.4, 1.3, 1.2, 1.1],
            "abs_cohens_d": [1.5, 1.4, 1.3, 1.2, 1.1],
            "mean_internal": [0.0] * 5,
            "mean_external": [0.0] * 5,
            "std_external": [1.0] * 5,
        }
    )

    panel = matched_module.select_feature_panel(shift_summary, panel_size=4, max_per_family=1)

    assert panel["feature"].tolist() == [
        "cancer_ripley_L_20.0",
        "stroma_ripley_L_20.0",
        "tumor_to_stroma_mean_dist",
        "degree_centrality_ratio",
    ]


def test_build_feature_targets_and_scoring_follow_quantile_matching(matched_module):
    internal = pd.DataFrame(
        {
            "sample_name": ["s1", "s2", "s3"],
            "tumor_to_stroma_mean_dist": [10.0, 20.0, 30.0],
            "degree_centrality_ratio": [1.0, 2.0, 3.0],
        }
    )
    external = pd.DataFrame(
        {
            "sample_name": ["e1", "e2", "e3"],
            "tumor_to_stroma_mean_dist": [40.0, 50.0, 60.0],
            "degree_centrality_ratio": [4.0, 5.0, 6.0],
        }
    )
    panel = pd.DataFrame(
        {
            "feature": ["tumor_to_stroma_mean_dist", "degree_centrality_ratio"],
            "family": ["interface_distance", "centrality_ratio"],
            "cohens_d": [1.5, 1.0],
            "abs_cohens_d": [1.5, 1.0],
            "mean_internal": [20.0, 2.0],
            "mean_external": [50.0, 5.0],
            "std_external": [10.0, 1.0],
        }
    )

    targets, scales, weights = matched_module.build_feature_targets(internal, external, panel)

    assert np.isclose(targets["s1"]["tumor_to_stroma_mean_dist"], 43.333333333333336)
    assert np.isclose(targets["s2"]["tumor_to_stroma_mean_dist"], 50.0)
    assert np.isclose(targets["s3"]["tumor_to_stroma_mean_dist"], 56.666666666666664)
    assert np.isclose(scales["degree_centrality_ratio"], 1.0)
    assert np.isclose(weights["tumor_to_stroma_mean_dist"] + weights["degree_centrality_ratio"], 1.0)

    perfect_score = matched_module.score_feature_candidate(
        {
            "tumor_to_stroma_mean_dist": targets["s1"]["tumor_to_stroma_mean_dist"],
            "degree_centrality_ratio": targets["s1"]["degree_centrality_ratio"],
        },
        targets["s1"],
        scales,
        weights,
    )
    worse_score = matched_module.score_feature_candidate(
        {
            "tumor_to_stroma_mean_dist": 55.0,
            "degree_centrality_ratio": 6.0,
        },
        targets["s1"],
        scales,
        weights,
    )

    assert np.isclose(perfect_score, 0.0)
    assert worse_score > perfect_score


def test_predict_keep_adjustments_responds_to_feature_direction(matched_module):
    baseline = {
        "tumor_to_stroma_mean_dist": 30.0,
        "stroma_ripley_L_20.0": 14.0,
        "cancer_ripley_L_20.0": 10.0,
        "degree_centrality_ratio": 0.8,
    }
    target = {
        "tumor_to_stroma_mean_dist": 50.0,
        "stroma_ripley_L_20.0": 10.0,
        "cancer_ripley_L_20.0": 16.0,
        "degree_centrality_ratio": 1.2,
    }
    scales = {k: 1.0 for k in baseline}
    weights = {
        "tumor_to_stroma_mean_dist": 0.4,
        "stroma_ripley_L_20.0": 0.3,
        "cancer_ripley_L_20.0": 0.2,
        "degree_centrality_ratio": 0.1,
    }

    keep_c_mult, keep_s_mult = matched_module.predict_keep_adjustments(baseline, target, scales, weights)

    assert keep_s_mult < 1.0
    assert keep_c_mult > 1.0


def test_build_separate_paired_targets_uses_single_coherent_external_pair(matched_module):
    src_stats_c = pd.DataFrame(
        {
            "sample_name": ["s1", "s2"],
            "median_nnd_all": [10.0, 30.0],
            "n_cells": [80, 20],
        }
    )
    src_stats_s = pd.DataFrame(
        {
            "sample_name": ["s1", "s2"],
            "median_nnd_stroma": [50.0, 20.0],
            "n_cells_stroma": [20, 80],
        }
    )
    ref_stats_c = pd.DataFrame(
        {
            "sample_name": ["e1", "e2", "e3"],
            "median_nnd_all": [11.0, 31.0, 29.0],
            "n_cells": [85, 25, 70],
        }
    )
    ref_stats_s = pd.DataFrame(
        {
            "sample_name": ["e1", "e2", "e3"],
            "median_nnd_stroma": [49.0, 19.0, 48.0],
            "n_cells_stroma": [15, 75, 30],
        }
    )

    targets = matched_module.build_separate_paired_targets(src_stats_c, src_stats_s, ref_stats_c, ref_stats_s)
    targets = targets.set_index("sample_name")

    assert targets.loc["s1", "matched_external_sample_name"] == "e1"
    assert np.isclose(targets.loc["s1", "target_nnd_cancer"], 11.0)
    assert np.isclose(targets.loc["s1", "target_nnd_stroma"], 49.0)

    assert targets.loc["s2", "matched_external_sample_name"] == "e2"
    assert np.isclose(targets.loc["s2", "target_nnd_cancer"], 31.0)
    assert np.isclose(targets.loc["s2", "target_nnd_stroma"], 19.0)


def test_build_separate_paired_targets_fraction_aware_changes_match_when_ratios_differ(matched_module):
    src_stats_c = pd.DataFrame(
        {
            "sample_name": ["s1", "s2"],
            "median_nnd_all": [10.0, 30.0],
            "n_cells": [90, 50],
        }
    )
    src_stats_s = pd.DataFrame(
        {
            "sample_name": ["s1", "s2"],
            "median_nnd_stroma": [20.0, 40.0],
            "n_cells_stroma": [10, 50],
        }
    )
    ref_stats_c = pd.DataFrame(
        {
            "sample_name": ["e1", "e2", "e3"],
            "median_nnd_all": [10.2, 10.8, 30.0],
            "n_cells": [10, 90, 50],
        }
    )
    ref_stats_s = pd.DataFrame(
        {
            "sample_name": ["e1", "e2", "e3"],
            "median_nnd_stroma": [20.2, 20.8, 40.0],
            "n_cells_stroma": [90, 10, 50],
        }
    )

    plain = matched_module.build_separate_paired_targets(src_stats_c, src_stats_s, ref_stats_c, ref_stats_s)
    frac = matched_module.build_separate_paired_targets(
        src_stats_c,
        src_stats_s,
        ref_stats_c,
        ref_stats_s,
        include_fraction=True,
    )

    plain = plain.set_index("sample_name")
    frac = frac.set_index("sample_name")

    assert plain.loc["s1", "matched_external_sample_name"] == "e1"
    assert frac.loc["s1", "matched_external_sample_name"] == "e2"
    assert np.isclose(frac.loc["s1", "src_tumor_fraction"], 0.9)
    assert np.isclose(frac.loc["s1", "matched_external_tumor_fraction"], 0.9)
