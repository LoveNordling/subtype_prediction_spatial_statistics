"""spatial_metrics.py

Spatial statistics for binary cancer/stroma point-clouds.

This file is intended as a drop-in replacement for the old spatial_metrics.py
used by spatial_data_pipeline.py / spatial_statistics5.py. It keeps the same
public function names, but implements the named statistics in a way that more
closely matches the definitions summarized in:

    Behanova, Klemm & Wahlby. Spatial Statistics for Understanding Tissue
    Organization. Frontiers in Physiology, 2022.

Conventions
-----------
- coords: numpy array with shape (n_cells, 2), columns x and y.
- cell_types: binary array with 0 = stroma/non-cancer, 1 = cancer.
- Distances are in the same units as coords.
- Returned values are deterministic for fixed inputs.

Important manuscript note
-------------------------
The compatibility key ``cross_k_auc`` is kept to avoid breaking old scripts,
but it now stores the CytoMAP-style object-object Pearson correlation summary.
For new manuscripts/tables, use the key ``object_object_pearson_correlation``
instead of calling this a cross-K AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import pearsonr


RANDOM_STATE = 42
DEFAULT_RADII: Tuple[float, ...] = (20.0, 40.0, 60.0, 80.0, 100.0)
TYPE_NAMES = {0: "stroma", 1: "cancer"}


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for spatial neighborhood graphs."""

    radius: float = 50.0
    k: int = 10
    mode: str = "radius"  # "radius" or "knn"


def _validate_inputs(coords: np.ndarray, cell_types: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    cell_types = np.asarray(cell_types, dtype=int)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n_cells, 2), got {coords.shape}.")
    if cell_types.ndim != 1:
        raise ValueError(f"cell_types must be 1D, got shape {cell_types.shape}.")
    if coords.shape[0] != cell_types.shape[0]:
        raise ValueError(
            f"coords and cell_types length mismatch: {coords.shape[0]} vs {cell_types.shape[0]}."
        )
    if coords.shape[0] == 0:
        raise ValueError("coords/cell_types contain no cells.")
    if not np.isfinite(coords).all():
        bad = np.argwhere(~np.isfinite(coords))[:5]
        raise ValueError(f"coords contain non-finite values. First bad indices: {bad.tolist()}.")

    unique = set(np.unique(cell_types).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"cell_types must be binary with 0=stroma and 1=cancer, got {sorted(unique)}.")

    return coords, cell_types


def _zero_dict(keys: Iterable[str]) -> Dict[str, float]:
    return {key: 0.0 for key in keys}


def _finite(value: float, default: float = 0.0) -> float:
    value = float(value)
    return value if np.isfinite(value) else float(default)


def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    if den == 0 or not np.isfinite(den):
        return float(default)
    value = float(num) / float(den)
    return _finite(value, default=default)


def _radius_key(radius: float) -> str:
    return f"{float(radius):.1f}"


def _window_area(coords: np.ndarray) -> float:
    """Estimate the 2D observation-window area from all observed points.

    We use the convex hull of all cells as the default observation window, with
    a bounding-box fallback for degenerate/collinear point clouds. This avoids
    estimating a separate area for each cell type, which would inflate cell-type
    specific Ripley's statistics.
    """

    if coords.shape[0] >= 3:
        try:
            area = float(ConvexHull(coords).volume)  # in 2D scipy calls area "volume"
            if np.isfinite(area) and area > 0:
                return area
        except QhullError:
            pass

    x_span = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    y_span = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    area = x_span * y_span
    if np.isfinite(area) and area > 0:
        return float(area)

    # Degenerate but valid input, e.g. all points have the same coordinate.
    return 1.0


def _build_radius_graph(coords: np.ndarray, cell_types: np.ndarray, radius: float) -> nx.Graph:
    G = nx.Graph()
    for i, ctype in enumerate(cell_types):
        G.add_node(i, cell_type=TYPE_NAMES[int(ctype)], cell_type_int=int(ctype))

    if coords.shape[0] < 2:
        return G

    pairs = cKDTree(coords).query_pairs(float(radius), output_type="ndarray")
    for i, j in pairs:
        dist = float(np.linalg.norm(coords[int(i)] - coords[int(j)]))
        G.add_edge(int(i), int(j), distance=dist)
    return G


def _build_knn_graph(coords: np.ndarray, cell_types: np.ndarray, k: int) -> nx.Graph:
    G = nx.Graph()
    for i, ctype in enumerate(cell_types):
        G.add_node(i, cell_type=TYPE_NAMES[int(ctype)], cell_type_int=int(ctype))

    n = coords.shape[0]
    if n < 2:
        return G

    k_eff = min(int(k), n - 1)
    if k_eff <= 0:
        return G

    distances, indices = cKDTree(coords).query(coords, k=k_eff + 1)
    for i in range(n):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            G.add_edge(int(i), int(j), distance=float(dist))
    return G


def _build_graph(coords: np.ndarray, cell_types: np.ndarray, config: GraphBuildConfig) -> nx.Graph:
    if config.mode == "radius":
        return _build_radius_graph(coords, cell_types, radius=config.radius)
    if config.mode == "knn":
        return _build_knn_graph(coords, cell_types, k=config.k)
    raise ValueError(f"Unknown graph mode: {config.mode!r}. Expected 'radius' or 'knn'.")


def calculate_ripley_l(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Dict[str, float]:
    """Calculate univariate Ripley's L for cancer and stroma separately.

    For a cell type with n points and observation-window area A, this uses

        K(r) = A * ordered_pair_count(d_ij <= r) / (n * (n - 1))
        L(r) = sqrt(K(r) / pi)

    No edge correction is applied. The output key format intentionally matches
    the old pipeline, e.g. ``cancer_ripley_L_20.0``.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    area = _window_area(coords)

    results: Dict[str, float] = {}
    for label_int, label_name in [(1, "cancer"), (0, "stroma")]:
        points = coords[cell_types == label_int]
        n = points.shape[0]
        keys = [f"{label_name}_ripley_L_{_radius_key(r)}" for r in radii]
        if n < 2:
            results.update(_zero_dict(keys))
            continue

        tree = cKDTree(points)
        for r, key in zip(radii, keys):
            unordered_pairs = len(tree.query_pairs(float(r)))
            ordered_pairs = 2 * unordered_pairs
            k_r = area * ordered_pairs / (n * (n - 1))
            l_r = np.sqrt(max(k_r, 0.0) / np.pi)
            results[key] = _finite(l_r)

    return results


def calculate_bidirectional_min_distance(coords: np.ndarray, cell_types: np.ndarray) -> Dict[str, float]:
    """Nearest-neighbor cancer-stroma distances in both directions.

    This is not one of the Behanova et al. review statistics, but is a useful
    interpretable interface-distance descriptor. The old mean-distance keys are
    preserved and median/combined summaries are added.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    cancer = coords[cell_types == 1]
    stroma = coords[cell_types == 0]

    keys = [
        "stroma_to_tumor_mean_dist",
        "tumor_to_stroma_mean_dist",
        "stroma_to_tumor_median_dist",
        "tumor_to_stroma_median_dist",
        "cancer_stroma_mean_min_dist",
    ]
    if len(cancer) == 0 or len(stroma) == 0:
        return _zero_dict(keys)

    stroma_to_tumor, _ = cKDTree(cancer).query(stroma, k=1)
    tumor_to_stroma, _ = cKDTree(stroma).query(cancer, k=1)
    both = np.concatenate([stroma_to_tumor, tumor_to_stroma])

    return {
        "stroma_to_tumor_mean_dist": _finite(np.mean(stroma_to_tumor)),
        "tumor_to_stroma_mean_dist": _finite(np.mean(tumor_to_stroma)),
        "stroma_to_tumor_median_dist": _finite(np.median(stroma_to_tumor)),
        "tumor_to_stroma_median_dist": _finite(np.median(tumor_to_stroma)),
        "cancer_stroma_mean_min_dist": _finite(np.mean(both)),
    }


def calculate_newmans_assortativity(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radius: float = 50.0,
) -> float:
    """Newman's categorical assortativity on a spatial radius graph.

    Nodes are cells and edges connect cells within ``radius``. The node attribute
    is the binary cancer/stroma label. The returned value is NetworkX's
    categorical attribute assortativity coefficient.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    if len(np.unique(cell_types)) < 2 or coords.shape[0] < 3:
        return 0.0

    G = _build_radius_graph(coords, cell_types, radius=radius)
    if G.number_of_edges() == 0:
        return 0.0

    value = nx.attribute_assortativity_coefficient(G, "cell_type")
    return _finite(value)


def _mean_node_centrality_ratio(G: nx.Graph, cell_types: np.ndarray, centrality: Dict[int, float]) -> float:
    cancer_vals = [centrality[i] for i in range(len(cell_types)) if cell_types[i] == 1]
    stroma_vals = [centrality[i] for i in range(len(cell_types)) if cell_types[i] == 0]
    cancer_mean = float(np.mean(cancer_vals)) if cancer_vals else 0.0
    stroma_mean = float(np.mean(stroma_vals)) if stroma_vals else 0.0
    return _safe_ratio(cancer_mean, stroma_mean)


def _average_clustering_by_group_ratio(G: nx.Graph, cell_types: np.ndarray) -> float:
    clustering = nx.clustering(G)
    return _mean_node_centrality_ratio(G, cell_types, clustering)


def _group_betweenness_ratio(
    G: nx.Graph,
    cancer_nodes: set[int],
    stroma_nodes: set[int],
    *,
    exact_max_nodes: int,
    approx_k: int,
) -> float:
    # Exact group betweenness is expensive. Use it when feasible and use the
    # standard NetworkX sampled node-betweenness approximation on larger graphs.
    # The approximation preserves the old scalar feature while keeping the full
    # pipeline tractable on dense cores.
    n = G.number_of_nodes()
    if n <= exact_max_nodes and len(cancer_nodes) <= n - 2 and len(stroma_nodes) <= n - 2:
        cancer_gb = nx.group_betweenness_centrality(
            G, cancer_nodes, normalized=True, weight="distance"
        )
        stroma_gb = nx.group_betweenness_centrality(
            G, stroma_nodes, normalized=True, weight="distance"
        )
        return _safe_ratio(cancer_gb, stroma_gb)

    k_eff = min(int(approx_k), n)
    if k_eff >= n:
        node_b = nx.betweenness_centrality(G, normalized=True, weight="distance")
    else:
        node_b = nx.betweenness_centrality(
            G, k=k_eff, normalized=True, weight="distance", seed=RANDOM_STATE
        )
    cell_types_arr = np.array([G.nodes[i]["cell_type_int"] for i in range(n)], dtype=int)
    return _mean_node_centrality_ratio(G, cell_types_arr, node_b)


def calculate_centrality_scores(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radius: float = 50.0,
    *,
    exact_betweenness_max_nodes: int = 750,
    approximate_betweenness_k: int = 256,
) -> Dict[str, float]:
    """Graph centrality summaries for cancer versus stroma.

    A spatial radius graph is built first. The main outputs are ratios of the
    cancer-group value divided by the stroma-group value:

    - degree centrality uses NetworkX group degree centrality, i.e. the fraction
      of non-group nodes connected to the group.
    - closeness centrality uses NetworkX group closeness centrality with edge
      distances.
    - betweenness centrality uses exact NetworkX group betweenness centrality
      for graphs up to ``exact_betweenness_max_nodes``. For larger graphs, a
      deterministic sampled node-betweenness approximation is used for runtime
      reasons.
    - average_clustering_ratio is the ratio of mean local clustering
      coefficients in cancer versus stroma.

    The three historical output keys are preserved, and
    ``average_clustering_ratio`` is added because this is one of the centrality
    summaries described in the review/Squidpy centrality workflow.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    keys = [
        "degree_centrality_ratio",
        "closeness_centrality_ratio",
        "betweenness_centrality_ratio",
        "average_clustering_ratio",
    ]
    if coords.shape[0] < 3 or len(np.unique(cell_types)) < 2:
        return _zero_dict(keys)

    G = _build_radius_graph(coords, cell_types, radius=radius)
    if G.number_of_edges() == 0:
        return _zero_dict(keys)

    cancer_nodes = {int(i) for i in np.where(cell_types == 1)[0]}
    stroma_nodes = {int(i) for i in np.where(cell_types == 0)[0]}

    cancer_degree = nx.group_degree_centrality(G, cancer_nodes)
    stroma_degree = nx.group_degree_centrality(G, stroma_nodes)

    cancer_closeness = nx.group_closeness_centrality(G, cancer_nodes, weight="distance")
    stroma_closeness = nx.group_closeness_centrality(G, stroma_nodes, weight="distance")

    betweenness_ratio = _group_betweenness_ratio(
        G,
        cancer_nodes,
        stroma_nodes,
        exact_max_nodes=int(exact_betweenness_max_nodes),
        approx_k=int(approximate_betweenness_k),
    )

    return {
        "degree_centrality_ratio": _safe_ratio(cancer_degree, stroma_degree),
        "closeness_centrality_ratio": _safe_ratio(cancer_closeness, stroma_closeness),
        "betweenness_centrality_ratio": betweenness_ratio,
        "average_clustering_ratio": _average_clustering_by_group_ratio(G, cell_types),
    }


def _cooccurrence_ratio_at_radius(
    coords: np.ndarray,
    cell_types: np.ndarray,
    *,
    cond_type: int,
    exp_type: int,
    radius: float,
) -> float:
    """p(exp_type near cond_type at radius) / p(exp_type)."""

    cond_indices = np.where(cell_types == cond_type)[0]
    if len(cond_indices) == 0:
        return 0.0

    p_exp = np.mean(cell_types == exp_type)
    if p_exp == 0:
        return 0.0

    tree = cKDTree(coords)
    exp_count = 0
    total_count = 0
    for idx in cond_indices:
        neigh = tree.query_ball_point(coords[idx], r=float(radius))
        # Exclude the conditioning cell itself.
        neigh = [j for j in neigh if j != idx]
        if not neigh:
            continue
        labels = cell_types[neigh]
        exp_count += int(np.sum(labels == exp_type))
        total_count += int(len(labels))

    if total_count == 0:
        return 0.0
    return _safe_ratio(exp_count / total_count, p_exp)


def calculate_cluster_cooccurrence_ratio(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
    k: int = 5,
) -> Dict[str, float]:
    """Squidpy/Tosti-style cluster co-occurrence summary for binary labels.

    In this method, ``cluster`` means the cell annotation category, not a
    spatial DBSCAN aggregate. For each radius r, the co-occurrence score is

        p(expected_type within r | conditioned_type) / p(expected_type).

    The paper describes the score as a curve over distances. For tabular ML we
    return directional means across ``radii`` and a backwards-compatible scalar
    ``cluster_cooccurrence_ratio`` equal to the average of the two directional
    cross-type means.

    ``mixed_cluster_proportion`` is retained for compatibility, but it is better
    described as a mixed-neighborhood proportion: the fraction of cells whose
    k-nearest neighbors include both cancer and stroma.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    keys = [
        "cluster_cooccurrence_ratio",
        "stroma_given_cancer_cooccurrence_mean",
        "cancer_given_stroma_cooccurrence_mean",
        "mixed_cluster_proportion",
    ]
    if coords.shape[0] < 3 or len(np.unique(cell_types)) < 2:
        return _zero_dict(keys)

    stroma_given_cancer_curve = np.array(
        [
            _cooccurrence_ratio_at_radius(
                coords, cell_types, cond_type=1, exp_type=0, radius=float(r)
            )
            for r in radii
        ],
        dtype=float,
    )
    cancer_given_stroma_curve = np.array(
        [
            _cooccurrence_ratio_at_radius(
                coords, cell_types, cond_type=0, exp_type=1, radius=float(r)
            )
            for r in radii
        ],
        dtype=float,
    )

    stroma_given_cancer = _finite(np.mean(stroma_given_cancer_curve))
    cancer_given_stroma = _finite(np.mean(cancer_given_stroma_curve))
    cross_mean = _finite(np.mean([stroma_given_cancer, cancer_given_stroma]))

    # Compatibility/local-mixing scalar.
    n = coords.shape[0]
    k_eff = min(int(k), n - 1)
    if k_eff <= 0:
        mixed_prop = 0.0
    else:
        _, indices = cKDTree(coords).query(coords, k=k_eff + 1)
        mixed = 0
        for i in range(n):
            neigh_labels = cell_types[indices[i, 1:]]
            mixed += int((neigh_labels == 0).any() and (neigh_labels == 1).any())
        mixed_prop = mixed / n

    return {
        "cluster_cooccurrence_ratio": cross_mean,
        "stroma_given_cancer_cooccurrence_mean": stroma_given_cancer,
        "cancer_given_stroma_cooccurrence_mean": cancer_given_stroma,
        "mixed_cluster_proportion": _finite(mixed_prop),
    }


def calculate_neighborhood_enrichment_test(
    coords: np.ndarray,
    cell_types: np.ndarray,
    k: int = 10,
    n_perms: int = 100,
    seed: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Permutation-based neighborhood enrichment z-scores.

    A kNN spatial graph is constructed. Observed edge counts between cell-type
    pairs are compared with a null distribution generated by shuffling the cell
    labels over fixed cell coordinates/graph topology. Returned values are
    z-scores: positive means over-representation relative to the permutation
    null; negative means depletion.

    The old keys are retained:
    - cancer_in_cancer_enrichment: cancer-cancer edge z-score.
    - non_cancer_in_cancer_enrichment: cancer-stroma edge z-score.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    keys = ["cancer_in_cancer_enrichment", "non_cancer_in_cancer_enrichment"]
    if coords.shape[0] <= k or len(np.unique(cell_types)) < 2:
        return _zero_dict(keys)

    G = _build_knn_graph(coords, cell_types, k=k)
    if G.number_of_edges() == 0:
        return _zero_dict(keys)

    edges = np.array(list(G.edges()), dtype=int)
    if edges.size == 0:
        return _zero_dict(keys)

    def _counts(labels: np.ndarray) -> tuple[int, int]:
        a = labels[edges[:, 0]]
        b = labels[edges[:, 1]]
        cancer_cancer = int(np.sum((a == 1) & (b == 1)))
        cancer_stroma = int(np.sum(a != b))
        return cancer_cancer, cancer_stroma

    obs_cc, obs_cs = _counts(cell_types)
    rng = np.random.default_rng(seed)
    perm_cc = np.empty(int(n_perms), dtype=float)
    perm_cs = np.empty(int(n_perms), dtype=float)
    for p in range(int(n_perms)):
        labels_perm = rng.permutation(cell_types)
        perm_cc[p], perm_cs[p] = _counts(labels_perm)

    def _z(obs: float, null: np.ndarray) -> float:
        sd = float(np.std(null, ddof=1)) if null.size > 1 else 0.0
        if sd == 0.0 or not np.isfinite(sd):
            return 0.0
        return _finite((float(obs) - float(np.mean(null))) / sd)

    return {
        "cancer_in_cancer_enrichment": _z(obs_cc, perm_cc),
        "non_cancer_in_cancer_enrichment": _z(obs_cs, perm_cs),
    }


def calculate_objectobject_correlation(
    coords: np.ndarray,
    cell_types: np.ndarray,
    neighborhood_radius: float = 50.0,
    grid_spacing: float | None = None,
) -> Dict[str, float]:
    """CytoMAP-style object-object correlation analysis.

    Circular neighborhoods are placed on a uniform grid over the tissue window.
    For each neighborhood, cancer and stroma counts are computed. The statistic
    is the Pearson correlation coefficient between those two count vectors over
    all non-empty neighborhoods.

    ``correlation_strength`` and ``cross_k_auc`` are retained as compatibility
    aliases. For new analyses, prefer ``object_object_pearson_correlation``.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    if coords.shape[0] < 4 or len(np.unique(cell_types)) < 2:
        return {
            "object_object_pearson_correlation": 0.0,
            "correlation_strength": 0.0,
            "cross_k_auc": 0.0,
        }

    if grid_spacing is None:
        grid_spacing = float(neighborhood_radius)
    if grid_spacing <= 0 or neighborhood_radius <= 0:
        raise ValueError("grid_spacing and neighborhood_radius must be positive.")

    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    xs = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    ys = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    if len(xs) == 0 or len(ys) == 0:
        return {
            "object_object_pearson_correlation": 0.0,
            "correlation_strength": 0.0,
            "cross_k_auc": 0.0,
        }

    centers = np.array([(x, y) for x in xs for y in ys], dtype=float)
    tree = cKDTree(coords)
    cancer_counts = []
    stroma_counts = []
    for center in centers:
        idx = tree.query_ball_point(center, r=float(neighborhood_radius))
        if not idx:
            continue
        labels = cell_types[idx]
        cancer_counts.append(int(np.sum(labels == 1)))
        stroma_counts.append(int(np.sum(labels == 0)))

    if len(cancer_counts) < 2:
        r = 0.0
    else:
        cancer_arr = np.asarray(cancer_counts, dtype=float)
        stroma_arr = np.asarray(stroma_counts, dtype=float)
        if np.std(cancer_arr) == 0 or np.std(stroma_arr) == 0:
            r = 0.0
        else:
            r = _finite(pearsonr(cancer_arr, stroma_arr).statistic)

    return {
        "object_object_pearson_correlation": r,
        "correlation_strength": r,
        "cross_k_auc": r,
    }


# Optional helper functions retained from the old module. These are not called
# by spatial_data_pipeline.py, but keeping them avoids breaking ad hoc scripts.


def calculate_mixing_score(coords: np.ndarray, cell_types: np.ndarray, k: int = 10) -> float:
    """Mean binary Shannon entropy of kNN neighborhood composition."""

    from scipy.stats import entropy as shannon_entropy

    coords, cell_types = _validate_inputs(coords, cell_types)
    n = coords.shape[0]
    if n <= k:
        return 0.0

    _, indices = cKDTree(coords).query(coords, k=min(k + 1, n))
    entropies = []
    for i in range(n):
        labels = cell_types[indices[i, 1:]]
        counts = np.bincount(labels, minlength=2)
        entropies.append(float(shannon_entropy(counts, base=2)))
    return _finite(np.mean(entropies))


def calculate_clustering_metric(coords: np.ndarray, cell_types: np.ndarray, metric_type: str = "silhouette") -> float:
    """Classical cluster-separation score using coordinates and binary labels."""

    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    coords, cell_types = _validate_inputs(coords, cell_types)
    if coords.shape[0] < 4 or len(np.unique(cell_types)) < 2:
        return 0.0

    if metric_type == "silhouette":
        return _finite(silhouette_score(coords, cell_types))
    if metric_type == "calinski_harabasz":
        return _finite(calinski_harabasz_score(coords, cell_types))
    if metric_type == "davies_bouldin":
        return _finite(davies_bouldin_score(coords, cell_types))
    raise ValueError(f"Unknown metric_type: {metric_type!r}.")


def calculate_distance_to_nearest_diff(coords: np.ndarray, cell_types: np.ndarray) -> float:
    """Mean nearest-neighbor distance from each cell to the nearest cell of the other type."""

    out = calculate_bidirectional_min_distance(coords, cell_types)
    return out["cancer_stroma_mean_min_dist"]
