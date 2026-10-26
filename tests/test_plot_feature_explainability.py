from plot_feature_explainability import FEATURE_FAMILIES, FEATURE_LABELS


def test_explainability_labels_describe_the_implemented_estimators():
    assert FEATURE_LABELS["cancer_stroma_cooccurrence_auc"] == (
        "Cancer-stroma co-occurrence ratio AUC"
    )
    assert FEATURE_LABELS["cancer_stroma_cooccurrence_deviation_auc"] == (
        "Cancer-stroma co-occurrence deviation"
    )
    assert FEATURE_FAMILIES["cancer_stroma_cooccurrence_deviation_auc"] == (
        "Deviation from random mixing"
    )

    assert FEATURE_LABELS["cancer_degree_centrality"].endswith("degree centrality")
    assert FEATURE_LABELS["stroma_closeness_centrality"].endswith("closeness centrality")
    assert FEATURE_LABELS["cancer_ripley_L_auc"].endswith("L AUC")


def test_cooccurrence_deviation_is_not_labeled_as_segregation():
    assert "segregation" not in FEATURE_LABELS["cancer_stroma_cooccurrence_deviation_auc"].lower()
    assert FEATURE_FAMILIES["cancer_stroma_cooccurrence_deviation_auc"] == (
        "Deviation from random mixing"
    )


def test_established_concise_labels_are_preserved():
    assert FEATURE_LABELS["newmans_assortativity"] == "Cancer/stroma assortativity"
    assert FEATURE_LABELS["cancer_average_clustering"] == "Cancer local clustering"
    assert FEATURE_LABELS["cancer_cancer_neighborhood_enrichment_zscore"] == (
        "Cancer-cancer enrichment"
    )
    assert FEATURE_LABELS["cancer_stroma_object_object_pearson_correlation"] == (
        "Cancer-stroma local correlation"
    )
