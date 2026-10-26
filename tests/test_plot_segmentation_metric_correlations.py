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


def test_get_behanova_metric_columns_uses_active_panel(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_behanova",
    )

    frames = [
        pd.DataFrame(
            {
                "ID": ["p1"],
                "label": [1],
                "sample_name": ["s1"],
                "cell_count": [10],
                "cancer_ripley_L_auc": [1.0],
                "stroma_ripley_L_auc": [2.0],
                "legacy_metric": [3.0],
            }
        ),
        pd.DataFrame(
            {
                "ID": ["p1"],
                "label": [1],
                "sample_name": ["s1"],
                "cell_count": [11],
                "cancer_ripley_L_auc": [1.5],
                "legacy_metric": [3.5],
            }
        ),
    ]

    assert mod.get_behanova_metric_columns(frames, include_counts=False) == ["cancer_ripley_L_auc"]
    assert mod.get_behanova_metric_columns(frames, include_counts=True) == ["cell_count", "cancer_ripley_L_auc"]


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

    assert mod.metric_family("cancer_ripley_L_auc") == "Ripley"
    assert mod.metric_family("newmans_assortativity") == "Newman's Assortativity"
    assert mod.metric_family("cancer_degree_centrality") == "Graph Centrality / Clustering"
    assert mod.metric_family("stroma_average_clustering") == "Graph Centrality / Clustering"
    assert mod.metric_family("cancer_stroma_cooccurrence_auc") == "Cell-Type Co-Occurrence Ratio"
    assert mod.metric_family("cancer_stroma_neighborhood_enrichment_zscore") == "Neighborhood Enrichment"
    assert mod.metric_family("cancer_stroma_object_object_pearson_correlation") == "Object-Object Correlation"


def test_pretty_metric_names_use_precise_estimator_labels(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_labels",
    )

    assert mod.pretty_metric_name("cancer_ripley_L_auc") == "Cancer Ripley's L AUC"
    assert mod.pretty_metric_name("stroma_ripley_L_auc") == "Stroma Ripley's L AUC"
    assert mod.pretty_metric_name("cancer_degree_centrality") == "Cancer group degree centrality"
    assert mod.pretty_metric_name("stroma_degree_centrality") == "Stroma group degree centrality"
    assert mod.pretty_metric_name("cancer_closeness_centrality") == "Cancer group closeness centrality"
    assert mod.pretty_metric_name("stroma_closeness_centrality") == "Stroma group closeness centrality"
    assert mod.pretty_metric_name("cancer_stroma_cooccurrence_auc") == (
        "Cancer-stroma co-occurrence ratio AUC"
    )

    assert mod.pretty_metric_name("newmans_assortativity") == "Newman's assortativity"
    assert mod.pretty_metric_name("cancer_stroma_object_object_pearson_correlation") == (
        "Cancer-stroma object-object Pearson correlation"
    )


def test_select_family_representative_metrics_uses_family_median(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_representatives",
    )

    summary = pd.DataFrame(
        {
            "family": ["Ripley", "Ripley", "Ripley", "Graph Centrality / Clustering"],
            "metric": ["ripley_low", "ripley_mid", "ripley_high", "degree"],
            "spearman_rho": [0.8, 0.9, 1.0, 0.7],
        }
    )

    assert mod.select_family_representative_metrics(summary) == ["ripley_mid", "degree"]


def test_supplementary_scatter_groups_cover_expected_families(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_supplementary_groups",
    )

    grouped_families = [family for _, _, families in mod.SUPPLEMENTARY_SCATTER_GROUPS for family in families]

    assert grouped_families == mod.FAMILY_ORDER
    assert len(grouped_families) == len(set(grouped_families))


def test_percentile_rank_handles_ties_and_missing(load_script_module):
    mod = load_script_module(
        "plot_segmentation_metric_correlations.py",
        "plot_segmentation_metric_correlations_under_test_percentiles",
    )

    ranks = mod._percentile_rank(pd.Series([10.0, 20.0, 20.0, None]))

    assert ranks.iloc[0] == pytest.approx(0.0)
    assert ranks.iloc[1] == pytest.approx(0.75)
    assert ranks.iloc[2] == pytest.approx(0.75)
    assert pd.isna(ranks.iloc[3])
