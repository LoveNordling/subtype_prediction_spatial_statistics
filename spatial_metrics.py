"""Behanova-aligned spatial statistics for binary cancer/stroma point clouds.

The active manuscript feature set is restricted to methods reviewed in:

    Behanova, Klemm & Wahlby. Spatial Statistics for Understanding Tissue
    Organization. Frontiers in Physiology, 2022.

Only single-object and two-object methods are implemented here:

- Ripley's L function
- Newman's assortativity
- Centrality scores
- Cell-type co-occurrence ratio
- Neighborhood enrichment test
- Object-object correlation analysis

Counts are intentionally not defined here; they are non-spatial QC fields added
by ``spatial_data_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import pearsonr


RANDOM_STATE = 42
DEFAULT_RADII: Tuple[float, ...] = (20.0, 40.0, 60.0, 80.0, 100.0)
DEFAULT_GRAPH_RADIUS = 50.0
DEFAULT_K_NEIGHBORS = 10
DEFAULT_N_PERMUTATIONS = 100
TYPE_NAMES = {0: "stroma", 1: "cancer"}


def _radius_key(radius: float) -> str:
    return f"{float(radius):.1f}"


RIPLEY_CURVE_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    f"{cell_type}_ripley_L_{_radius_key(radius)}"
    for cell_type in ("cancer", "stroma")
    for radius in DEFAULT_RADII
)
RIPLEY_FEATURE_COLUMNS: Tuple[str, ...] = (
    "cancer_ripley_L_auc",
    "stroma_ripley_L_auc",
)
ASSORTATIVITY_FEATURE_COLUMNS: Tuple[str, ...] = ("newmans_assortativity",)
CENTRALITY_FEATURE_COLUMNS: Tuple[str, ...] = (
    "cancer_degree_centrality",
    "stroma_degree_centrality",
    "cancer_closeness_centrality",
    "stroma_closeness_centrality",
    "cancer_average_clustering",
    "stroma_average_clustering",
)
COOCCURRENCE_CURVE_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    f"cancer_stroma_cooccurrence_ratio_{_radius_key(radius)}"
    for radius in DEFAULT_RADII
)
COOCCURRENCE_FEATURE_COLUMNS: Tuple[str, ...] = (
    "cancer_stroma_cooccurrence_auc",
    "cancer_stroma_cooccurrence_deviation_auc",
)
NEIGHBORHOOD_ENRICHMENT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "cancer_cancer_neighborhood_enrichment_zscore",
    "cancer_stroma_neighborhood_enrichment_zscore",
    "stroma_stroma_neighborhood_enrichment_zscore",
)
OBJECT_OBJECT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "cancer_stroma_object_object_pearson_correlation",
)
BEHANOVA_FEATURE_COLUMNS: Tuple[str, ...] = (
    RIPLEY_FEATURE_COLUMNS
    + ASSORTATIVITY_FEATURE_COLUMNS
    + CENTRALITY_FEATURE_COLUMNS
    + COOCCURRENCE_FEATURE_COLUMNS
    + NEIGHBORHOOD_ENRICHMENT_FEATURE_COLUMNS
    + OBJECT_OBJECT_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class SpatialGraphConfig:
    """Parameters for spatial graph methods."""

    radius: float = DEFAULT_GRAPH_RADIUS
    k: int = DEFAULT_K_NEIGHBORS


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


def _curve_auc(values: Sequence[float], radii: Sequence[float]) -> float:
    """Area under a fixed-radius curve, normalized by the radius range."""

    vals = np.asarray(list(values), dtype=float)
    radii_arr = np.asarray(list(radii), dtype=float)
    if vals.size == 0:
        return 0.0
    vals = np.where(np.isfinite(vals), vals, 0.0)
    if vals.size == 1 or radii_arr.size == 1:
        return _finite(vals[0])
    order = np.argsort(radii_arr)
    vals = vals[order]
    radii_arr = radii_arr[order]
    span = float(radii_arr[-1] - radii_arr[0])
    if not np.isfinite(span) or span <= 0:
        return _finite(np.mean(vals))
    auc = float(np.sum(0.5 * (vals[:-1] + vals[1:]) * np.diff(radii_arr)) / span)
    return _finite(auc)


def _window_area(coords: np.ndarray) -> float:
    """Estimate the observation-window area from all observed cells.

    Ripley's K/L require an observation window. We use the convex hull of all
    cells as a practical tissue-window estimate, with a bounding-box fallback for
    degenerate point clouds. No edge correction is applied.
    """

    if coords.shape[0] >= 3:
        try:
            area = float(ConvexHull(coords).volume)
            if np.isfinite(area) and area > 0:
                return area
        except QhullError:
            pass

    x_span = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    y_span = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    area = x_span * y_span
    return float(area) if np.isfinite(area) and area > 0 else 1.0


def _radius_edges(coords: np.ndarray, radius: float) -> np.ndarray:
    if coords.shape[0] < 2:
        return np.empty((0, 2), dtype=int)
    return cKDTree(coords).query_pairs(float(radius), output_type="ndarray").astype(int, copy=False)


def _knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    k_eff = min(int(k), n - 1)
    if n < 2 or k_eff <= 0:
        return np.empty((0, 2), dtype=int)

    _, indices = cKDTree(coords).query(coords, k=k_eff + 1)
    src = np.repeat(np.arange(n, dtype=int), k_eff)
    dst = indices[:, 1:].reshape(-1).astype(int, copy=False)
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    edges = np.column_stack([a, b])
    return np.unique(edges, axis=0)


def _build_graph_from_edges(cell_types: np.ndarray, edges: np.ndarray) -> nx.Graph:
    graph = nx.Graph()
    for idx, ctype in enumerate(cell_types):
        graph.add_node(idx, cell_type=TYPE_NAMES[int(ctype)], cell_type_int=int(ctype))
    graph.add_edges_from((int(i), int(j)) for i, j in edges)
    return graph


def calculate_ripley_l_curve(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Dict[str, float]:
    """Calculate univariate Ripley's L curves for cancer and stroma separately.

    The estimator is:

        K(r) = A * ordered_pair_count(d_ij <= r) / (n * (n - 1))
        L(r) = sqrt(K(r) / pi)

    where ``A`` is the shared observation-window area estimated from all cells.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    radii_arr = np.asarray(list(radii), dtype=float)
    area = _window_area(coords)

    results: Dict[str, float] = {}
    for label_int, label_name in ((1, "cancer"), (0, "stroma")):
        points = coords[cell_types == label_int]
        n = points.shape[0]
        keys = [f"{label_name}_ripley_L_{_radius_key(radius)}" for radius in radii_arr]
        if n < 2:
            results.update(_zero_dict(keys))
            continue

        tree = cKDTree(points)
        ordered_counts = np.asarray(tree.count_neighbors(tree, radii_arr, cumulative=True), dtype=float) - n
        denom = float(n * (n - 1))
        for key, ordered_count in zip(keys, ordered_counts):
            k_r = area * max(float(ordered_count), 0.0) / denom
            results[key] = _finite(np.sqrt(max(k_r, 0.0) / np.pi))

    return results


def calculate_ripley_l(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Dict[str, float]:
    """Summarize compartment-specific Ripley's L curves by normalized AUC."""

    radii_arr = np.asarray(list(radii), dtype=float)
    curve = calculate_ripley_l_curve(coords, cell_types, radii_arr)
    out: Dict[str, float] = {}
    for label_name in ("cancer", "stroma"):
        vals = [
            curve.get(f"{label_name}_ripley_L_{_radius_key(radius)}", 0.0)
            for radius in radii_arr
        ]
        out[f"{label_name}_ripley_L_auc"] = _curve_auc(vals, radii_arr)
    return out


def calculate_newmans_assortativity(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radius: float = DEFAULT_GRAPH_RADIUS,
) -> float:
    """Calculate Newman's categorical assortativity on a spatial radius graph."""

    coords, cell_types = _validate_inputs(coords, cell_types)
    if coords.shape[0] < 3 or len(np.unique(cell_types)) < 2:
        return 0.0

    edges = _radius_edges(coords, radius)
    if edges.size == 0:
        return 0.0

    mixing = np.zeros((2, 2), dtype=float)
    a = cell_types[edges[:, 0]]
    b = cell_types[edges[:, 1]]
    for left, right in zip(a, b):
        mixing[int(left), int(right)] += 1.0
        mixing[int(right), int(left)] += 1.0

    total = float(mixing.sum())
    if total == 0:
        return 0.0
    e = mixing / total
    expected = float(np.dot(e.sum(axis=0), e.sum(axis=1)))
    denom = 1.0 - expected
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return _finite((float(np.trace(e)) - expected) / denom)


def calculate_centrality_scores(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radius: float = DEFAULT_GRAPH_RADIUS,
) -> Dict[str, float]:
    """Calculate Squidpy-style centrality scores per cell type.

    Behanova et al. list centrality scores as a single-object method. Squidpy's
    implementation reports degree centrality, closeness centrality, and average
    clustering for each cell-type group; this function mirrors that feature
    surface for the binary cancer/stroma setting.
    """

    coords, cell_types = _validate_inputs(coords, cell_types)
    keys = CENTRALITY_FEATURE_COLUMNS
    if coords.shape[0] < 3 or len(np.unique(cell_types)) < 2:
        return _zero_dict(keys)

    edges = _radius_edges(coords, radius)
    if edges.size == 0:
        return _zero_dict(keys)

    graph = _build_graph_from_edges(cell_types, edges)
    out: Dict[str, float] = {}
    for label_int, label_name in ((1, "cancer"), (0, "stroma")):
        nodes = {int(i) for i in np.where(cell_types == label_int)[0]}
        if not nodes or len(nodes) >= graph.number_of_nodes():
            out[f"{label_name}_degree_centrality"] = 0.0
            out[f"{label_name}_closeness_centrality"] = 0.0
            out[f"{label_name}_average_clustering"] = 0.0
            continue

        out[f"{label_name}_degree_centrality"] = _finite(nx.group_degree_centrality(graph, nodes))
        out[f"{label_name}_closeness_centrality"] = _finite(nx.group_closeness_centrality(graph, nodes))
        out[f"{label_name}_average_clustering"] = _finite(nx.average_clustering(graph, nodes=nodes))

    return out


def _symmetric_cooccurrence_curve(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords, cell_types = _validate_inputs(coords, cell_types)
    radii_arr = np.asarray(list(radii), dtype=float)
    ratio_curve = np.zeros(len(radii_arr), dtype=float)
    total_counts = np.zeros(len(radii_arr), dtype=float)
    if coords.shape[0] < 3 or len(np.unique(cell_types)) < 2 or len(radii_arr) == 0:
        return radii_arr, ratio_curve, total_counts

    max_radius = float(np.max(radii_arr))
    pairs = _radius_edges(coords, max_radius)
    if pairs.size == 0:
        return radii_arr, ratio_curve, total_counts

    dists = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    first_radius = np.searchsorted(radii_arr, dists, side="left")
    bin_counts = np.zeros((len(radii_arr), 2, 2), dtype=float)
    valid = first_radius < len(radii_arr)
    for idx, i, j in zip(first_radius[valid], pairs[valid, 0], pairs[valid, 1]):
        left = int(cell_types[i])
        right = int(cell_types[j])
        bin_counts[int(idx), left, right] += 1.0
        bin_counts[int(idx), right, left] += 1.0

    cumulative = np.cumsum(bin_counts, axis=0)
    for idx, counts in enumerate(cumulative):
        total = float(counts.sum())
        total_counts[idx] = total
        if total == 0:
            continue

        exp_prob = counts.sum(axis=0) / total

        cancer_row = float(counts[1].sum())
        stroma_given_cancer = (
            (counts[1, 0] / cancer_row) / exp_prob[0]
            if cancer_row > 0 and exp_prob[0] > 0
            else 0.0
        )

        stroma_row = float(counts[0].sum())
        cancer_given_stroma = (
            (counts[0, 1] / stroma_row) / exp_prob[1]
            if stroma_row > 0 and exp_prob[1] > 0
            else 0.0
        )

        ratio_curve[idx] = _finite(np.mean([stroma_given_cancer, cancer_given_stroma]))

    return radii_arr, ratio_curve, total_counts


def calculate_cell_type_cooccurrence_ratio_curve(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Dict[str, float]:
    """Calculate a symmetric cancer-stroma co-occurrence ratio curve.

    The symmetric curve is the mean of the two directional binary cross-type
    ratios at each radius. It represents one cancer-stroma association surface
    while avoiding duplicated directional features in a two-label dataset.
    """

    radii_arr, ratio_curve, _ = _symmetric_cooccurrence_curve(coords, cell_types, radii)
    return {
        f"cancer_stroma_cooccurrence_ratio_{_radius_key(radius)}": _finite(value)
        for radius, value in zip(radii_arr, ratio_curve)
    }


def calculate_cell_type_cooccurrence_ratio(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Dict[str, float]:
    """Summarize cancer-stroma cell-type co-occurrence over fixed radii.

    ``cancer_stroma_cooccurrence_auc`` is the normalized AUC of the symmetric
    cross-type ratio curve. Values above 1 indicate cross-type enrichment,
    values below 1 indicate segregation/depletion. The deviation AUC captures
    the magnitude of departure from the random baseline of 1.
    """

    radii_arr, ratio_curve, total_counts = _symmetric_cooccurrence_curve(coords, cell_types, radii)
    if not np.any(total_counts > 0):
        return _zero_dict(COOCCURRENCE_FEATURE_COLUMNS)

    return {
        "cancer_stroma_cooccurrence_auc": _curve_auc(ratio_curve, radii_arr),
        "cancer_stroma_cooccurrence_deviation_auc": _curve_auc(np.abs(ratio_curve - 1.0), radii_arr),
    }


def calculate_cluster_cooccurrence_ratio_curve(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Dict[str, float]:
    """Backward-compatible alias for cell-type co-occurrence curve features."""

    return calculate_cell_type_cooccurrence_ratio_curve(coords, cell_types, radii)


def calculate_cluster_cooccurrence_ratio(
    coords: np.ndarray,
    cell_types: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> Dict[str, float]:
    """Backward-compatible alias for cell-type co-occurrence summary features."""

    return calculate_cell_type_cooccurrence_ratio(coords, cell_types, radii)


def calculate_neighborhood_enrichment_test(
    coords: np.ndarray,
    cell_types: np.ndarray,
    k: int = DEFAULT_K_NEIGHBORS,
    n_perms: int = DEFAULT_N_PERMUTATIONS,
    seed: int = RANDOM_STATE,
) -> Dict[str, float]:
    """Calculate neighborhood enrichment z-scores on a kNN graph."""

    coords, cell_types = _validate_inputs(coords, cell_types)
    keys = NEIGHBORHOOD_ENRICHMENT_FEATURE_COLUMNS
    if coords.shape[0] <= k or len(np.unique(cell_types)) < 2:
        return _zero_dict(keys)

    edges = _knn_edges(coords, k)
    if edges.size == 0:
        return _zero_dict(keys)

    def _counts(labels: np.ndarray) -> np.ndarray:
        left = labels[edges[:, 0]]
        right = labels[edges[:, 1]]
        cc = float(np.sum((left == 1) & (right == 1)))
        cs = float(np.sum(left != right))
        ss = float(np.sum((left == 0) & (right == 0)))
        return np.array([cc, cs, ss], dtype=float)

    observed = _counts(cell_types)
    rng = np.random.default_rng(seed)
    null = np.empty((int(n_perms), 3), dtype=float)
    for idx in range(int(n_perms)):
        null[idx] = _counts(rng.permutation(cell_types))

    sd = np.std(null, axis=0, ddof=1) if len(null) > 1 else np.zeros(3, dtype=float)
    mean = np.mean(null, axis=0) if len(null) else np.zeros(3, dtype=float)
    z = np.divide(observed - mean, sd, out=np.zeros(3, dtype=float), where=np.isfinite(sd) & (sd > 0))

    return {
        "cancer_cancer_neighborhood_enrichment_zscore": _finite(z[0]),
        "cancer_stroma_neighborhood_enrichment_zscore": _finite(z[1]),
        "stroma_stroma_neighborhood_enrichment_zscore": _finite(z[2]),
    }


def calculate_object_object_correlation(
    coords: np.ndarray,
    cell_types: np.ndarray,
    neighborhood_radius: float = DEFAULT_GRAPH_RADIUS,
    grid_spacing: float | None = None,
) -> Dict[str, float]:
    """Calculate CytoMAP-style object-object Pearson correlation."""

    coords, cell_types = _validate_inputs(coords, cell_types)
    key = "cancer_stroma_object_object_pearson_correlation"
    if coords.shape[0] < 4 or len(np.unique(cell_types)) < 2:
        return {key: 0.0}

    if grid_spacing is None:
        grid_spacing = float(neighborhood_radius)
    if grid_spacing <= 0 or neighborhood_radius <= 0:
        raise ValueError("grid_spacing and neighborhood_radius must be positive.")

    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    xs = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    ys = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    if len(xs) == 0 or len(ys) == 0:
        return {key: 0.0}

    centers = np.array([(x, y) for x in xs for y in ys], dtype=float)
    neighborhoods = cKDTree(coords).query_ball_point(centers, r=float(neighborhood_radius))
    cancer_counts = []
    stroma_counts = []
    for idx in neighborhoods:
        if not idx:
            continue
        labels = cell_types[idx]
        cancer_counts.append(int(np.sum(labels == 1)))
        stroma_counts.append(int(np.sum(labels == 0)))

    if len(cancer_counts) < 2:
        return {key: 0.0}
    cancer_arr = np.asarray(cancer_counts, dtype=float)
    stroma_arr = np.asarray(stroma_counts, dtype=float)
    if np.std(cancer_arr) == 0 or np.std(stroma_arr) == 0:
        return {key: 0.0}

    return {key: _finite(pearsonr(cancer_arr, stroma_arr).statistic)}


def calculate_behanova_spatial_metrics(coords: np.ndarray, cell_types: np.ndarray) -> Dict[str, float]:
    """Calculate the complete Behanova-aligned binary feature panel."""

    coords, cell_types = _validate_inputs(coords, cell_types)
    metrics: Dict[str, float] = {}
    metrics.update(calculate_ripley_l(coords, cell_types))
    metrics["newmans_assortativity"] = calculate_newmans_assortativity(coords, cell_types)
    metrics.update(calculate_centrality_scores(coords, cell_types))
    metrics.update(calculate_cell_type_cooccurrence_ratio(coords, cell_types))
    metrics.update(calculate_neighborhood_enrichment_test(coords, cell_types))
    metrics.update(calculate_object_object_correlation(coords, cell_types))
    return {key: float(metrics.get(key, 0.0)) for key in BEHANOVA_FEATURE_COLUMNS}
