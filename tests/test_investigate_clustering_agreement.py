import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from investigate_clustering_agreement import local_clustering


def test_sparse_clustering_matches_networkx_with_isolates():
    coords = np.random.default_rng(7).uniform(0, 100, size=(120, 2))
    coords[-1] = [1000, 1000]
    graph = nx.Graph()
    graph.add_nodes_from(range(len(coords)))
    graph.add_edges_from(cKDTree(coords).query_pairs(20))
    clustering, degree = local_clustering(coords, 20)
    np.testing.assert_allclose(clustering, list(nx.clustering(graph).values()))
    np.testing.assert_array_equal(degree, [d for _, d in graph.degree])


def test_sparse_clustering_empty_edges():
    clustering, degree = local_clustering(np.array([[0., 0.], [100., 100.]]), 1)
    np.testing.assert_array_equal(clustering, [0, 0])
    np.testing.assert_array_equal(degree, [0, 0])
