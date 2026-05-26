import pandas as pd
import pytest


def test_get_metric_columns_excludes_meta_and_counts_by_default(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test",
    )

    frames = [
        pd.DataFrame(
            {
                "ID": ["p1"],
                "label": [1],
                "sample_name": ["s1"],
                "cell_count": [10],
                "feat_a": [1.0],
                "feat_b": [2.0],
            }
        ),
        pd.DataFrame(
            {
                "ID": ["p1"],
                "label": [1],
                "sample_name": ["s1"],
                "cell_count": [11],
                "feat_a": [3.0],
                "feat_b": [4.0],
                "feat_c": [5.0],
            }
        ),
    ]

    assert mod.get_metric_columns(frames, include_counts=False) == ["feat_a", "feat_b"]
    assert mod.get_metric_columns(frames, include_counts=True) == ["cell_count", "feat_a", "feat_b"]


def test_build_pair_table_and_correlations(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_build_pair",
    )

    left = pd.DataFrame({"ID": ["p1", "p2", "p3"], "metric_x": [1.0, 2.0, None]})
    right = pd.DataFrame({"ID": ["p1", "p2", "p4"], "metric_x": [2.0, 4.0, 10.0]})
    pair = mod.build_pair_table(left, right, "metric_x", "inform", "cellpose", "ID")

    assert pair.to_dict(orient="records") == [
        {"ID": "p1", "metric_x__inform": 1.0, "metric_x__cellpose": 2.0},
        {"ID": "p2", "metric_x__inform": 2.0, "metric_x__cellpose": 4.0},
    ]

    pearson_r, spearman_rho = mod.compute_pairwise_correlations(
        pair["metric_x__inform"],
        pair["metric_x__cellpose"],
    )
    assert pearson_r == pytest.approx(1.0)
    assert spearman_rho == pytest.approx(1.0)


def test_aggregate_metrics_patient_uses_weighted_patient_helper(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_aggregate",
    )

    df = pd.DataFrame(
        {
            "ID": ["p1", "p1", "p2"],
            "label": [1, 1, 0],
            "sample_name": ["s1", "s2", "s3"],
            "cell_count": [10, 30, 5],
            "cancer_count": [4, 8, 2],
            "stroma_count": [6, 22, 3],
            "feat_a": [1.0, 3.0, 7.0],
        }
    )

    patient = mod.aggregate_metrics(df, "patient")
    assert patient.to_dict(orient="records") == [
        {
            "ID": "p1",
            "label": 1,
            "cell_count": 40,
            "cancer_count": 12,
            "stroma_count": 28,
            "feat_a": 2.5,
        },
        {
            "ID": "p2",
            "label": 0,
            "cell_count": 5,
            "cancer_count": 2,
            "stroma_count": 3,
            "feat_a": 7.0,
        },
    ]


def test_metric_family_groups_related_features(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_family",
    )

    assert mod.metric_family("cancer_ripley_L_20.0") == "Ripley"
    assert mod.metric_family("stroma_to_tumor_mean_dist") == "Nearest Distance"
    assert mod.metric_family("cross_k_auc") == "Cross-Type Correlation"
    assert mod.metric_family("cancer_in_cancer_enrichment") == "Neighborhood Enrichment"
    assert mod.metric_family("newmans_assortativity") == "Cluster / Mixing"
    assert mod.metric_family("betweenness_centrality_ratio") == "Centrality"
