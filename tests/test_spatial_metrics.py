import networkx as nx
import numpy as np
import pytest

from spatial_metrics import (
    BEHANOVA_FEATURE_COLUMNS,
    calculate_behanova_spatial_metrics,
    calculate_cell_type_cooccurrence_ratio,
    calculate_cell_type_cooccurrence_ratio_curve,
    calculate_newmans_assortativity,
)


def test_behanova_metric_panel_has_only_declared_features():
    coords = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [4.0, 4.0],
            [8.0, 0.0],
            [8.0, 4.0],
            [80.0, 80.0],
            [84.0, 80.0],
            [80.0, 84.0],
            [84.0, 84.0],
            [88.0, 80.0],
            [88.0, 84.0],
        ],
        dtype=float,
    )
    cell_types = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=int)

    metrics = calculate_behanova_spatial_metrics(coords, cell_types)

    assert tuple(metrics) == BEHANOVA_FEATURE_COLUMNS
    assert len(metrics) == 15
    assert np.isfinite(list(metrics.values())).all()
    forbidden = (
        "cross_k",
        "mean_dist",
        "mixed_cluster",
        "betweenness",
        "centrality_ratio",
        "correlation_strength",
    )
    assert not any(any(token in key for token in forbidden) for key in metrics)


def test_newmans_assortativity_matches_networkx_on_same_radius_graph():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    cell_types = np.array([0, 0, 1, 1], dtype=int)
    radius = 1.1

    graph = nx.Graph()
    for idx, cell_type in enumerate(cell_types):
        graph.add_node(idx, cell_type=int(cell_type))
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            if np.linalg.norm(coords[left] - coords[right]) <= radius:
                graph.add_edge(left, right)

    expected = nx.attribute_assortativity_coefficient(graph, "cell_type")

    assert calculate_newmans_assortativity(coords, cell_types, radius=radius) == pytest.approx(expected)


def test_cell_type_cooccurrence_ratio_uses_symmetric_cross_type_curve_summary():
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]], dtype=float)
    cell_types = np.array([1, 0, 1, 0], dtype=int)

    curve = calculate_cell_type_cooccurrence_ratio_curve(coords, cell_types, radii=(2.0,))
    metrics = calculate_cell_type_cooccurrence_ratio(coords, cell_types, radii=(2.0,))

    assert curve["cancer_stroma_cooccurrence_ratio_2.0"] == pytest.approx(2.0)
    assert metrics["cancer_stroma_cooccurrence_auc"] == pytest.approx(2.0)
    assert metrics["cancer_stroma_cooccurrence_deviation_auc"] == pytest.approx(1.0)
