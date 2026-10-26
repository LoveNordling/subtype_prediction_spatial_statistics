#!/usr/bin/env python3
"""
Create a matched BOMI2 internal dataset whose per-core nearest-neighbor distance (NND)
distribution matches the external BOMI1 cohort.

Supported internal sources:
- cellpose
- inform

Supported matching compartments:
- all:      thin all cells to match the external all-cell NND distribution
- stroma:   keep cancer cells unchanged and thin only stroma cells
- separate: thin cancer and stroma independently against the corresponding
            external cancer and stroma NND distributions

The script writes:
- a new internal cells CSV with replicated / thinned cores
- a duplicated samples.csv aligned to those new cores
- QC histograms showing the original vs matched NND distributions
"""
from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from spatial_metrics import BEHANOVA_FEATURE_COLUMNS
from spatial_utils import preprocess_external_cells


_CORE_PAT = re.compile(r"(.+?)_Core\[1,(\d+,[A-Z])\]")
MODEL_FEATURE_COLUMNS = list(BEHANOVA_FEATURE_COLUMNS)
DEFAULT_FEATURE_PANEL = [
    "stroma_ripley_L_auc",
    "cancer_ripley_L_auc",
    "cancer_degree_centrality",
    "cancer_stroma_cooccurrence_auc",
    "cancer_stroma_neighborhood_enrichment_zscore",
    "cancer_stroma_object_object_pearson_correlation",
]
FEATURE_SEARCH_MULTIPLIERS = (0.85, 1.0, 1.15)
FEATURE_KEEP_ADJUSTMENT_MAX = 0.35


def extract_sample_name_from_filename(filename: str) -> str:
    match = _CORE_PAT.match(str(filename))
    if not match:
        raise ValueError(f"Could not parse sample name from filename: {filename}")
    return f"{match.group(1)}_[{match.group(2)}]"


def insert_suffix_before_core(value: str, suffix: str) -> str:
    text = str(value)
    if "_Core[1," not in text:
        raise ValueError(f"Expected '_Core[1,' in value, got: {text}")
    return text.replace("_Core[1,", f"{suffix}_Core[1,", 1)


def insert_suffix_in_canonical_sample_name(sample_name: str, suffix: str) -> str:
    text = str(sample_name)
    token = "_["
    if token not in text:
        raise ValueError(f"Expected canonical sample like '<prefix>_[<pos>]', got: {text}")
    prefix, rest = text.split(token, 1)
    return f"{prefix}{suffix}{token}{rest}"


def canonicalize_samples_csv_name(value: str) -> str:
    text = str(value)
    if "_" not in text:
        return text
    text = text[: text.rfind("_")]
    return text.replace("Core[1,", "[")


def remove_isolated_points(coords: np.ndarray, k: int = 5, max_percentile: float = 99.0) -> np.ndarray:
    if coords.shape[0] <= k:
        return np.ones(coords.shape[0], dtype=bool)
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k + 1)
    kth = dists[:, k]
    return kth <= np.percentile(kth, max_percentile)


def _median_nnd(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return float("nan")
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return float(np.median(dists[:, 1]))


def compute_sample_median_nnd(
    df: pd.DataFrame,
    *,
    min_cells: int,
    compartment: str,
) -> pd.DataFrame:
    rows = []
    for sample_name, group in df.groupby("sample_name", sort=False):
        if compartment == "all":
            selected = group
            metric_col = "median_nnd_all"
            count_col = "n_cells"
        elif compartment == "stroma":
            selected = group[group["CK"] == 0]
            metric_col = "median_nnd_stroma"
            count_col = "n_cells_stroma"
        else:
            raise ValueError(f"Unknown compartment: {compartment}")

        if len(selected) < min_cells:
            continue

        coords = selected[["x", "y"]].to_numpy(float, copy=False)
        keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
        coords_f = coords[keep]
        if coords_f.shape[0] < min_cells:
            continue

        rows.append(
            {
                "sample_name": sample_name,
                metric_col: _median_nnd(coords_f),
                count_col: int(coords_f.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _merge_separate_sample_stats(
    cancer_stats: pd.DataFrame,
    stroma_stats: pd.DataFrame,
) -> pd.DataFrame:
    if cancer_stats.empty or stroma_stats.empty:
        return pd.DataFrame(
            columns=[
                "sample_name",
                "median_nnd_cancer",
                "median_nnd_stroma",
                "n_cells_cancer",
                "n_cells_stroma",
                "tumor_fraction",
            ]
        )

    merged = cancer_stats.rename(
        columns={
            "median_nnd_all": "median_nnd_cancer",
            "n_cells": "n_cells_cancer",
        }
    ).merge(
        stroma_stats.rename(
            columns={
                "median_nnd_stroma": "median_nnd_stroma",
                "n_cells_stroma": "n_cells_stroma",
            }
        ),
        on="sample_name",
        how="inner",
    )
    total = pd.to_numeric(merged["n_cells_cancer"], errors="coerce").fillna(0) + pd.to_numeric(merged["n_cells_stroma"], errors="coerce").fillna(0)
    merged["tumor_fraction"] = np.where(total > 0, pd.to_numeric(merged["n_cells_cancer"], errors="coerce").fillna(0) / total, np.nan)
    return merged


def build_separate_paired_targets(
    src_stats_c: pd.DataFrame,
    src_stats_s: pd.DataFrame,
    ref_stats_c: pd.DataFrame,
    ref_stats_s: pd.DataFrame,
    *,
    include_fraction: bool = False,
) -> pd.DataFrame:
    src_pairs = _merge_separate_sample_stats(src_stats_c, src_stats_s)
    ref_pairs = _merge_separate_sample_stats(ref_stats_c, ref_stats_s)
    if src_pairs.empty or ref_pairs.empty:
        raise ValueError("No valid paired cancer/stroma per-sample stats available for separate paired matching.")

    src_c_sorted = np.sort(src_pairs["median_nnd_cancer"].dropna().to_numpy(float))
    src_s_sorted = np.sort(src_pairs["median_nnd_stroma"].dropna().to_numpy(float))
    ref_c_vals = ref_pairs["median_nnd_cancer"].dropna().to_numpy(float)
    ref_s_vals = ref_pairs["median_nnd_stroma"].dropna().to_numpy(float)
    if min(len(src_c_sorted), len(src_s_sorted), len(ref_c_vals), len(ref_s_vals)) == 0:
        raise ValueError("Missing finite cancer/stroma NND values for separate paired matching.")

    scale_c = float(np.std(ref_c_vals, ddof=1)) if len(ref_c_vals) > 1 else 1.0
    scale_s = float(np.std(ref_s_vals, ddof=1)) if len(ref_s_vals) > 1 else 1.0
    if not np.isfinite(scale_c) or scale_c <= 0:
        scale_c = 1.0
    if not np.isfinite(scale_s) or scale_s <= 0:
        scale_s = 1.0
    ref_frac_vals = ref_pairs["tumor_fraction"].dropna().to_numpy(float)
    scale_frac = float(np.std(ref_frac_vals, ddof=1)) if len(ref_frac_vals) > 1 else 1.0
    if not np.isfinite(scale_frac) or scale_frac <= 0:
        scale_frac = 1.0

    rows = []
    ref_pairs = ref_pairs.copy()
    for row in src_pairs.itertuples(index=False):
        nnd_src_c = float(row.median_nnd_cancer)
        nnd_src_s = float(row.median_nnd_stroma)
        frac_src = float(row.tumor_fraction)
        rank_c = np.searchsorted(src_c_sorted, nnd_src_c, side="left")
        rank_s = np.searchsorted(src_s_sorted, nnd_src_s, side="left")
        quantile_c = (rank_c + 0.5) / len(src_c_sorted)
        quantile_s = (rank_s + 0.5) / len(src_s_sorted)
        nnd_tgt_c = float(np.quantile(ref_c_vals, quantile_c))
        nnd_tgt_s = float(np.quantile(ref_s_vals, quantile_s))

        dist_sq = (
            ((pd.to_numeric(ref_pairs["median_nnd_cancer"], errors="coerce") - nnd_tgt_c) / scale_c) ** 2
            + ((pd.to_numeric(ref_pairs["median_nnd_stroma"], errors="coerce") - nnd_tgt_s) / scale_s) ** 2
        )
        if include_fraction:
            dist_sq = dist_sq + (
                (pd.to_numeric(ref_pairs["tumor_fraction"], errors="coerce") - frac_src) / scale_frac
            ) ** 2
        ref_pairs["pair_distance"] = np.sqrt(dist_sq)
        best = ref_pairs.sort_values(["pair_distance", "sample_name"], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "sample_name": row.sample_name,
                "src_nnd_cancer": nnd_src_c,
                "src_nnd_stroma": nnd_src_s,
                "src_tumor_fraction": frac_src,
                "quantile_cancer": quantile_c,
                "quantile_stroma": quantile_s,
                "target_nnd_cancer": float(best["median_nnd_cancer"]),
                "target_nnd_stroma": float(best["median_nnd_stroma"]),
                "matched_external_sample_name": str(best["sample_name"]),
                "matched_external_pair_distance": float(best["pair_distance"]),
                "matched_external_tumor_fraction": float(best["tumor_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    denom = x.size + y.size - 2
    if denom <= 0:
        return float("nan")
    sp = np.sqrt(((x.size - 1) * vx + (y.size - 1) * vy) / denom)
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


def feature_family(feature: str) -> str:
    if feature.startswith("cancer_ripley_L_"):
        return "cancer_ripley"
    if feature.startswith("stroma_ripley_L_"):
        return "stroma_ripley"
    if feature == "newmans_assortativity":
        return "newmans_assortativity"
    if feature.endswith("_degree_centrality") or feature.endswith("_closeness_centrality") or feature.endswith("_average_clustering"):
        return "centrality_score"
    if feature.startswith("cancer_stroma_cooccurrence_") or "_cooccurrence_ratio_" in feature:
        return "cell_type_cooccurrence_ratio"
    if feature.endswith("_neighborhood_enrichment_zscore"):
        return "neighborhood_enrichment"
    if feature.endswith("_object_object_pearson_correlation"):
        return "object_object_correlation"
    return feature


def summarize_feature_shift_table(
    df_internal_metrics: pd.DataFrame,
    df_external_metrics: pd.DataFrame,
    features: Iterable[str],
) -> pd.DataFrame:
    rows = []
    for feature in features:
        a = pd.to_numeric(df_internal_metrics[feature], errors="coerce").to_numpy(float)
        b = pd.to_numeric(df_external_metrics[feature], errors="coerce").to_numpy(float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        rows.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "mean_internal": float(np.mean(a)) if a.size else float("nan"),
                "mean_external": float(np.mean(b)) if b.size else float("nan"),
                "std_external": float(np.std(b, ddof=1)) if b.size > 1 else float("nan"),
                "cohens_d": cohens_d(a, b),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_cohens_d"] = out["cohens_d"].abs()
    return out.sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)


def select_feature_panel(
    shift_summary: pd.DataFrame,
    *,
    panel_size: int,
    max_per_family: int,
    manual_features: list[str] | None = None,
) -> pd.DataFrame:
    if manual_features:
        order = {feature: i for i, feature in enumerate(manual_features)}
        selected = shift_summary[shift_summary["feature"].isin(order)].copy()
        selected["manual_order"] = selected["feature"].map(order)
        selected = selected.sort_values("manual_order").drop(columns=["manual_order"])
        return selected.reset_index(drop=True)

    family_counts: Dict[str, int] = {}
    chosen = []
    for row in shift_summary.itertuples(index=False):
        family = str(row.family)
        if family_counts.get(family, 0) >= max_per_family:
            continue
        chosen.append(row.feature)
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(chosen) >= panel_size:
            break
    return shift_summary[shift_summary["feature"].isin(chosen)].reset_index(drop=True)


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_spatial_metric_functions():
    from spatial_metrics import calculate_behanova_spatial_metrics

    return {"behanova": calculate_behanova_spatial_metrics}


def calculate_selected_metrics_for_cells(cells_df: pd.DataFrame, features: Iterable[str]) -> Dict[str, float]:
    if cells_df is None or len(cells_df) < 10:
        return {}

    requested = list(features)
    coords = cells_df[["x", "y"]].to_numpy(float, copy=False)
    cell_types = cells_df["CK"].to_numpy(int, copy=False)
    metric_fns = _get_spatial_metric_functions()
    out = metric_fns["behanova"](coords, cell_types)
    return {feature: float(out.get(feature, np.nan)) for feature in requested}


def compute_core_feature_table(df: pd.DataFrame, features: Iterable[str], *, min_cells: int) -> pd.DataFrame:
    rows = []
    feature_list = list(features)
    for sample_name, group in df.groupby("sample_name", sort=False):
        if len(group) < min_cells:
            continue
        metrics = calculate_selected_metrics_for_cells(group, feature_list)
        if not metrics:
            continue
        metrics["sample_name"] = sample_name
        rows.append(metrics)
    return pd.DataFrame(rows)


def load_cached_feature_table(path: Path, features: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["sample_name"] + [feature for feature in features if feature in df.columns]
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Cached metrics file {path} is missing required features: {missing}")
    return df[cols].copy()


def load_or_compute_feature_table(
    *,
    cache_path: Path | None,
    df_cells: pd.DataFrame,
    features: Iterable[str],
    min_cells: int,
    recompute: bool,
) -> pd.DataFrame:
    if cache_path is not None and cache_path.exists() and not recompute:
        print(f"[INFO] Loading cached baseline metrics: {cache_path}")
        return load_cached_feature_table(cache_path, features)
    print("[INFO] Computing baseline metrics from cells...")
    return compute_core_feature_table(df_cells, features, min_cells=min_cells)


def feature_table_to_lookup(df: pd.DataFrame, features: Iterable[str]) -> dict[str, dict[str, float]]:
    feature_list = list(features)
    lookup: dict[str, dict[str, float]] = {}
    for row in df[["sample_name"] + feature_list].itertuples(index=False):
        sample_name = str(row[0])
        lookup[sample_name] = {
            feature: float(value) if value is not None else float("nan")
            for feature, value in zip(feature_list, row[1:])
        }
    return lookup


def _feature_quantile_target(value: float, src_values: np.ndarray, ref_values: np.ndarray) -> float:
    src = np.sort(src_values[np.isfinite(src_values)])
    ref = ref_values[np.isfinite(ref_values)]
    if src.size == 0 or ref.size == 0 or not np.isfinite(value):
        return float("nan")
    rank = np.searchsorted(src, value, side="left")
    quantile = (rank + 0.5) / max(len(src), 1)
    return float(np.quantile(ref, quantile))


def build_feature_targets(
    df_internal_metrics: pd.DataFrame,
    df_external_metrics: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float]]:
    targets: dict[str, dict[str, float]] = {}
    scales: dict[str, float] = {}
    weights: dict[str, float] = {}
    total_weight = float(panel_df["abs_cohens_d"].fillna(0).sum())

    for row in panel_df.itertuples(index=False):
        feature = str(row.feature)
        src_values = pd.to_numeric(df_internal_metrics[feature], errors="coerce").to_numpy(float)
        ref_values = pd.to_numeric(df_external_metrics[feature], errors="coerce").to_numpy(float)
        scales[feature] = float(row.std_external) if np.isfinite(row.std_external) and row.std_external > 0 else 1.0
        weights[feature] = (
            float(row.abs_cohens_d) / total_weight if total_weight > 0 and np.isfinite(row.abs_cohens_d) else 1.0 / max(len(panel_df), 1)
        )
        for sample_name, value in zip(df_internal_metrics["sample_name"].astype(str), src_values):
            targets.setdefault(sample_name, {})[feature] = _feature_quantile_target(value, src_values, ref_values)
    return targets, scales, weights


def score_feature_candidate(
    candidate_metrics: dict[str, float],
    target_metrics: dict[str, float],
    feature_scales: dict[str, float],
    feature_weights: dict[str, float],
) -> float:
    total = 0.0
    used = 0.0
    for feature, target in target_metrics.items():
        value = candidate_metrics.get(feature, np.nan)
        scale = feature_scales.get(feature, 1.0)
        weight = feature_weights.get(feature, 1.0)
        if not (np.isfinite(value) and np.isfinite(target)):
            continue
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        total += weight * abs(value - target) / scale
        used += weight
    return total / used if used > 0 else float("inf")


def _stochastic_keep_mask(n_items: int, *, p_keep: float, min_cells: int, rng: np.random.Generator) -> np.ndarray | None:
    if n_items < min_cells:
        return None
    p_keep = min(1.0, max(float(p_keep), min_cells / n_items))
    keep = rng.random(n_items) < p_keep
    if keep.sum() >= min_cells:
        return keep
    forced = np.zeros(n_items, dtype=bool)
    forced[rng.choice(n_items, size=min_cells, replace=False)] = True
    return forced


def predict_keep_adjustments(
    baseline_metrics: dict[str, float],
    target_metrics: dict[str, float],
    feature_scales: dict[str, float],
    feature_weights: dict[str, float],
) -> tuple[float, float]:
    """
    Convert baseline-vs-target feature deltas into signed thinning pressure.

    Positive pressure means "thin more" (lower keep probability).
    Returns multiplicative adjustments for cancer and stroma keep rates.
    """
    stroma_pressure = 0.0
    cancer_pressure = 0.0
    used_stroma = 0.0
    used_cancer = 0.0

    for feature, target in target_metrics.items():
        value = baseline_metrics.get(feature, np.nan)
        scale = feature_scales.get(feature, 1.0)
        weight = feature_weights.get(feature, 1.0)
        if not (np.isfinite(value) and np.isfinite(target)):
            continue
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        error = (value - target) / scale

        # Warm-start heuristic only; final selection still scores full metrics.
        if feature.startswith("stroma_ripley_L_"):
            stroma_pressure += weight * error
            used_stroma += weight
        elif feature.startswith("cancer_ripley_L_"):
            cancer_pressure += weight * error
            used_cancer += weight
        elif (
            feature.startswith("stroma_given_cancer_")
            or feature.startswith("cancer_given_stroma_")
            or feature.startswith("cancer_stroma_")
        ):
            stroma_pressure += 0.5 * weight * error
            cancer_pressure += 0.5 * weight * error
            used_stroma += 0.5 * weight
            used_cancer += 0.5 * weight
        elif feature.startswith("stroma_") and (
            feature.endswith("_degree_centrality")
            or feature.endswith("_closeness_centrality")
            or feature.endswith("_average_clustering")
            or feature.endswith("_neighborhood_enrichment_zscore")
        ):
            stroma_pressure += weight * error
            used_stroma += weight
        elif feature.startswith("cancer_") and (
            feature.endswith("_degree_centrality")
            or feature.endswith("_closeness_centrality")
            or feature.endswith("_average_clustering")
            or feature.endswith("_neighborhood_enrichment_zscore")
        ):
            cancer_pressure += weight * error
            used_cancer += weight

    if used_stroma > 0:
        stroma_pressure /= used_stroma
    if used_cancer > 0:
        cancer_pressure /= used_cancer

    stroma_pressure = float(np.clip(stroma_pressure, -FEATURE_KEEP_ADJUSTMENT_MAX, FEATURE_KEEP_ADJUSTMENT_MAX))
    cancer_pressure = float(np.clip(cancer_pressure, -FEATURE_KEEP_ADJUSTMENT_MAX, FEATURE_KEEP_ADJUSTMENT_MAX))
    return float(np.exp(-cancer_pressure)), float(np.exp(-stroma_pressure))


def _binary_search_keep_mask(
    coords_filtered: np.ndarray,
    *,
    target_nnd: float,
    min_cells: int,
    rng: np.random.Generator,
    p_init: float,
    n_binsrch: int,
    tol_rel: float,
) -> np.ndarray | None:
    if len(coords_filtered) < min_cells:
        return None

    p_min = min_cells / len(coords_filtered)
    lo = max(0.0, p_min)
    hi = min(1.0, max(p_init, p_min))
    uniform = rng.random(len(coords_filtered))

    best_keep = None
    best_err = float("inf")
    for _ in range(n_binsrch):
        p_trial = 0.5 * (lo + hi)
        keep_mask = uniform < p_trial
        if keep_mask.sum() < min_cells:
            lo = p_trial
            continue

        nnd_now = _median_nnd(coords_filtered[keep_mask])
        err = abs(nnd_now - target_nnd) / target_nnd
        if err < best_err:
            best_err = err
            best_keep = keep_mask.copy()

        if nnd_now < target_nnd:
            hi = p_trial
        else:
            lo = p_trial

        if best_err <= tol_rel:
            break

    return best_keep


def thin_internal_to_match_external_nnd(
    df_internal: pd.DataFrame,
    df_external: pd.DataFrame,
    *,
    compartment: str,
    separate_target_mode: str,
    n_subsamples: int,
    min_cells: int,
    seed: int,
    n_binsrch: int = 8,
    tol_rel: float = 0.03,
) -> pd.DataFrame:
    if compartment == "separate":
        src_stats_c = compute_sample_median_nnd(df_internal[df_internal["CK"] == 1].assign(CK=1), min_cells=min_cells, compartment="all")
        ref_stats_c = compute_sample_median_nnd(df_external[df_external["CK"] == 1].assign(CK=1), min_cells=min_cells, compartment="all")
        src_stats_s = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="stroma")
        ref_stats_s = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="stroma")

        if src_stats_c.empty or ref_stats_c.empty or src_stats_s.empty or ref_stats_s.empty:
            raise ValueError("No valid per-sample cancer/stroma NND stats for separate matching.")

        src_c_vals = src_stats_c["median_nnd_all"].dropna().to_numpy(float)
        ref_c_vals = ref_stats_c["median_nnd_all"].dropna().to_numpy(float)
        src_s_vals = src_stats_s["median_nnd_stroma"].dropna().to_numpy(float)
        ref_s_vals = ref_stats_s["median_nnd_stroma"].dropna().to_numpy(float)
        if min(len(src_c_vals), len(ref_c_vals), len(src_s_vals), len(ref_s_vals)) == 0:
            raise ValueError("No finite cancer/stroma NND values for separate matching.")

        src_c_sorted = np.sort(src_c_vals)
        src_s_sorted = np.sort(src_s_vals)
        src_stats_c = src_stats_c.set_index("sample_name")
        src_stats_s = src_stats_s.set_index("sample_name")
        paired_targets = None
        if separate_target_mode in {"paired-2d", "paired-3d-fraction"}:
            paired_targets = build_separate_paired_targets(
                src_stats_c.reset_index(),
                src_stats_s.reset_index(),
                ref_stats_c,
                ref_stats_s,
                include_fraction=(separate_target_mode == "paired-3d-fraction"),
            ).set_index("sample_name")
    else:
        src_stats = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment=compartment)
        ref_stats = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment=compartment)

        metric_col = "median_nnd_all" if compartment == "all" else "median_nnd_stroma"

        if src_stats.empty or ref_stats.empty:
            raise ValueError(
                f"No valid per-sample {compartment} NND stats for source or reference cohort. "
                "Check the chosen input and min-cells threshold."
            )

        src_vals = src_stats[metric_col].dropna().to_numpy(float)
        ref_vals = ref_stats[metric_col].dropna().to_numpy(float)
        if src_vals.size == 0 or ref_vals.size == 0:
            raise ValueError(f"No finite {compartment} NND values for source or reference.")

        src_sorted = np.sort(src_vals)
    groups = {sample_name: group for sample_name, group in df_internal.groupby("sample_name", sort=False)}

    out_parts = []
    for rep in range(1, n_subsamples + 1):
        rep_rng = np.random.default_rng(seed + rep)

        iterable = groups.keys() if compartment == "separate" else [row.sample_name for row in src_stats.itertuples(index=False)]
        for sample_name in tqdm(iterable, total=len(iterable), desc=f"Matching rep {rep}/{n_subsamples}"):
            group = groups.get(sample_name)
            if group is None:
                continue

            if compartment == "all":
                row = src_stats[src_stats["sample_name"] == sample_name].iloc[0]
                nnd_src = float(getattr(row, "median_nnd_all"))
                if not (np.isfinite(nnd_src) and nnd_src > 0):
                    continue
                rank = np.searchsorted(src_sorted, nnd_src, side="left")
                quantile = (rank + 0.5) / len(src_sorted)
                nnd_tgt = float(np.quantile(ref_vals, quantile))
                if not (np.isfinite(nnd_tgt) and nnd_tgt > 0):
                    continue
                p_keep = float(min(1.0, max(0.0, (nnd_src / nnd_tgt) ** 2)))

                base = group
                coords = base[["x", "y"]].to_numpy(float, copy=False)
                keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
                base_filtered = base.loc[keep]
                if len(base_filtered) < min_cells:
                    continue
                coords_filtered = base_filtered[["x", "y"]].to_numpy(float, copy=False)
                keep_mask = _binary_search_keep_mask(
                    coords_filtered,
                    target_nnd=nnd_tgt,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                if keep_mask is None:
                    continue
                matched = base_filtered.iloc[keep_mask].copy()
            elif compartment == "stroma":
                row = src_stats[src_stats["sample_name"] == sample_name].iloc[0]
                nnd_src = float(getattr(row, "median_nnd_stroma"))
                if not (np.isfinite(nnd_src) and nnd_src > 0):
                    continue
                rank = np.searchsorted(src_sorted, nnd_src, side="left")
                quantile = (rank + 0.5) / len(src_sorted)
                nnd_tgt = float(np.quantile(ref_vals, quantile))
                if not (np.isfinite(nnd_tgt) and nnd_tgt > 0):
                    continue
                p_keep = float(min(1.0, max(0.0, (nnd_src / nnd_tgt) ** 2)))

                cancer = group[group["CK"] == 1]
                stroma = group[group["CK"] == 0]
                if len(stroma) < min_cells:
                    continue

                coords = stroma[["x", "y"]].to_numpy(float, copy=False)
                keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
                stroma_filtered = stroma.loc[keep]
                if len(stroma_filtered) < min_cells:
                    continue
                coords_filtered = stroma_filtered[["x", "y"]].to_numpy(float, copy=False)
                keep_mask = _binary_search_keep_mask(
                    coords_filtered,
                    target_nnd=nnd_tgt,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                if keep_mask is None:
                    continue
                stroma_sub = stroma_filtered.iloc[keep_mask].copy()
                matched = pd.concat([cancer, stroma_sub], axis=0)
            else:
                cancer = group[group["CK"] == 1]
                stroma = group[group["CK"] == 0]
                if len(cancer) < min_cells or len(stroma) < min_cells:
                    continue

                if sample_name not in src_stats_c.index or sample_name not in src_stats_s.index:
                    continue

                nnd_src_c = float(src_stats_c.loc[sample_name, "median_nnd_all"])
                nnd_src_s = float(src_stats_s.loc[sample_name, "median_nnd_stroma"])
                if not (np.isfinite(nnd_src_c) and nnd_src_c > 0 and np.isfinite(nnd_src_s) and nnd_src_s > 0):
                    continue

                if paired_targets is not None:
                    if sample_name not in paired_targets.index:
                        continue
                    nnd_tgt_c = float(paired_targets.loc[sample_name, "target_nnd_cancer"])
                    nnd_tgt_s = float(paired_targets.loc[sample_name, "target_nnd_stroma"])
                else:
                    rank_c = np.searchsorted(src_c_sorted, nnd_src_c, side="left")
                    rank_s = np.searchsorted(src_s_sorted, nnd_src_s, side="left")
                    quantile_c = (rank_c + 0.5) / len(src_c_sorted)
                    quantile_s = (rank_s + 0.5) / len(src_s_sorted)
                    nnd_tgt_c = float(np.quantile(ref_c_vals, quantile_c))
                    nnd_tgt_s = float(np.quantile(ref_s_vals, quantile_s))
                if not (np.isfinite(nnd_tgt_c) and nnd_tgt_c > 0 and np.isfinite(nnd_tgt_s) and nnd_tgt_s > 0):
                    continue

                p_keep_c = float(min(1.0, max(0.0, (nnd_src_c / nnd_tgt_c) ** 2)))
                p_keep_s = float(min(1.0, max(0.0, (nnd_src_s / nnd_tgt_s) ** 2)))

                coords_c = cancer[["x", "y"]].to_numpy(float, copy=False)
                coords_s = stroma[["x", "y"]].to_numpy(float, copy=False)
                keep_c0 = remove_isolated_points(coords_c, k=5, max_percentile=99.0)
                keep_s0 = remove_isolated_points(coords_s, k=5, max_percentile=99.0)
                cancer_filtered = cancer.loc[keep_c0]
                stroma_filtered = stroma.loc[keep_s0]
                if len(cancer_filtered) < min_cells or len(stroma_filtered) < min_cells:
                    continue

                keep_c = _binary_search_keep_mask(
                    cancer_filtered[["x", "y"]].to_numpy(float, copy=False),
                    target_nnd=nnd_tgt_c,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep_c,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                keep_s = _binary_search_keep_mask(
                    stroma_filtered[["x", "y"]].to_numpy(float, copy=False),
                    target_nnd=nnd_tgt_s,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep_s,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                if keep_c is None or keep_s is None:
                    continue

                cancer_sub = cancer_filtered.iloc[keep_c].copy()
                stroma_sub = stroma_filtered.iloc[keep_s].copy()
                matched = pd.concat([cancer_sub, stroma_sub], axis=0)

            suffix = f"__thin{rep}"
            matched["orig_sample_name"] = sample_name
            matched["rep"] = rep
            matched["sample_name_new"] = insert_suffix_in_canonical_sample_name(sample_name, suffix)
            out_parts.append(matched)

    if not out_parts:
        raise RuntimeError("Matching produced no cores. Check the selected compartment and thresholds.")

    return pd.concat(out_parts, axis=0)


def _keep_prob_grid(base_keep: float, include_full: bool = True) -> list[float]:
    values = {min(1.0, max(0.0, base_keep * mult)) for mult in FEATURE_SEARCH_MULTIPLIERS}
    values.add(min(1.0, max(0.0, base_keep)))
    if include_full:
        values.add(1.0)
    return sorted(values)


def _refine_keep_probs(best_keep: float, *, min_keep: float, max_keep: float) -> list[float]:
    candidates = {
        min(max_keep, max(min_keep, best_keep)),
        min(max_keep, max(min_keep, best_keep * 0.93)),
        min(max_keep, max(min_keep, best_keep * 1.07)),
    }
    return sorted(candidates)


def _warm_start_grid(pred_keep: float, *, min_keep: float, max_keep: float) -> list[float]:
    candidates = {
        min(max_keep, max(min_keep, pred_keep)),
        min(max_keep, max(min_keep, pred_keep * 0.94)),
        min(max_keep, max(min_keep, pred_keep * 1.06)),
    }
    return sorted(candidates)


def _evaluate_feature_panel_candidate(
    candidate_df: pd.DataFrame | None,
    *,
    panel_features: list[str],
    target_metrics: dict[str, float],
    feature_scales: dict[str, float],
    feature_weights: dict[str, float],
    best_score: float,
    best_matched: pd.DataFrame | None,
    min_cells: int,
) -> tuple[float, pd.DataFrame | None]:
    if candidate_df is None or len(candidate_df) < min_cells:
        return best_score, best_matched
    metrics = calculate_selected_metrics_for_cells(candidate_df, panel_features)
    score = score_feature_candidate(metrics, target_metrics, feature_scales, feature_weights)
    if np.isfinite(score) and score < best_score:
        return score, candidate_df.copy()
    return best_score, best_matched


def _match_one_sample_feature_panel(task: dict) -> pd.DataFrame | None:
    sample_name = task["sample_name"]
    group = task["group"]
    compartment = task["compartment"]
    min_cells = task["min_cells"]
    min_cells_cancer_qc = task["min_cells_cancer_qc"]
    candidate_draws = task["candidate_draws"]
    target_metrics = task["target_metrics"]
    feature_scales = task["feature_scales"]
    feature_weights = task["feature_weights"]
    panel_features = task["panel_features"]
    rep = task["rep"]
    seed = task["seed"]
    pred_keep_c = task.get("pred_keep_c", 1.0)
    pred_keep_s = task.get("pred_keep_s", 1.0)
    rep_rng = np.random.default_rng(seed)

    best_matched = None
    best_score = float("inf")
    best_probs = (1.0, 1.0)

    best_score, best_matched = _evaluate_feature_panel_candidate(
        group,
        panel_features=panel_features,
        target_metrics=target_metrics,
        feature_scales=feature_scales,
        feature_weights=feature_weights,
        best_score=best_score,
        best_matched=best_matched,
        min_cells=min_cells,
    )

    if compartment == "all":
        base_filtered = task["base_filtered"]
        p_keep = pred_keep_s
        if len(base_filtered) < min_cells:
            return None
        min_keep = min_cells / len(base_filtered)
        grid = _warm_start_grid(p_keep, min_keep=min_keep, max_keep=1.0)
        for p_all in grid:
            for _ in range(candidate_draws):
                keep = _stochastic_keep_mask(len(base_filtered), p_keep=p_all, min_cells=min_cells, rng=rep_rng)
                if keep is None:
                    continue
                score_before = best_score
                best_score, best_matched = _evaluate_feature_panel_candidate(
                    base_filtered.iloc[keep],
                    panel_features=panel_features,
                    target_metrics=target_metrics,
                    feature_scales=feature_scales,
                    feature_weights=feature_weights,
                    best_score=best_score,
                    best_matched=best_matched,
                    min_cells=min_cells,
                )
                if best_score < score_before:
                    best_probs = (p_all, 1.0)
        for p_all in _refine_keep_probs(best_probs[0], min_keep=min_keep, max_keep=1.0):
            keep = _stochastic_keep_mask(len(base_filtered), p_keep=p_all, min_cells=min_cells, rng=rep_rng)
            if keep is None:
                continue
            best_score, best_matched = _evaluate_feature_panel_candidate(
                base_filtered.iloc[keep],
                panel_features=panel_features,
                target_metrics=target_metrics,
                feature_scales=feature_scales,
                feature_weights=feature_weights,
                best_score=best_score,
                best_matched=best_matched,
                min_cells=min_cells,
            )
    elif compartment == "stroma":
        cancer = task["cancer"]
        stroma_filtered = task["stroma_filtered"]
        p_keep_s = pred_keep_s
        if len(stroma_filtered) < min_cells:
            return None
        min_keep_s = min_cells / len(stroma_filtered)
        grid = _warm_start_grid(p_keep_s, min_keep=min_keep_s, max_keep=1.0)
        for p_stroma in grid:
            for _ in range(candidate_draws):
                keep_s = _stochastic_keep_mask(len(stroma_filtered), p_keep=p_stroma, min_cells=min_cells, rng=rep_rng)
                if keep_s is None:
                    continue
                score_before = best_score
                best_score, best_matched = _evaluate_feature_panel_candidate(
                    pd.concat([cancer, stroma_filtered.iloc[keep_s]], axis=0),
                    panel_features=panel_features,
                    target_metrics=target_metrics,
                    feature_scales=feature_scales,
                    feature_weights=feature_weights,
                    best_score=best_score,
                    best_matched=best_matched,
                    min_cells=min_cells,
                )
                if best_score < score_before:
                    best_probs = (1.0, p_stroma)
        for p_stroma in _refine_keep_probs(best_probs[1], min_keep=min_keep_s, max_keep=1.0):
            keep_s = _stochastic_keep_mask(len(stroma_filtered), p_keep=p_stroma, min_cells=min_cells, rng=rep_rng)
            if keep_s is None:
                continue
            best_score, best_matched = _evaluate_feature_panel_candidate(
                pd.concat([cancer, stroma_filtered.iloc[keep_s]], axis=0),
                panel_features=panel_features,
                target_metrics=target_metrics,
                feature_scales=feature_scales,
                feature_weights=feature_weights,
                best_score=best_score,
                best_matched=best_matched,
                min_cells=min_cells,
            )
    else:
        cancer_filtered = task["cancer_filtered"]
        stroma_filtered = task["stroma_filtered"]
        p_keep_c = pred_keep_c
        p_keep_s = pred_keep_s
        if len(cancer_filtered) < min_cells_cancer_qc or len(stroma_filtered) < min_cells:
            return None
        min_keep_c = min_cells_cancer_qc / len(cancer_filtered)
        min_keep_s = min_cells / len(stroma_filtered)
        grid_c = _warm_start_grid(p_keep_c, min_keep=min_keep_c, max_keep=1.0)
        grid_s = _warm_start_grid(p_keep_s, min_keep=min_keep_s, max_keep=1.0)
        paired = [(p_cancer, p_stroma) for p_cancer, p_stroma in zip(grid_c, grid_s)]
        paired.extend([(p_keep_c, p_keep_s)])
        for p_cancer, p_stroma in paired:
            for _ in range(candidate_draws):
                keep_c = _stochastic_keep_mask(len(cancer_filtered), p_keep=p_cancer, min_cells=min_cells_cancer_qc, rng=rep_rng)
                keep_s = _stochastic_keep_mask(len(stroma_filtered), p_keep=p_stroma, min_cells=min_cells, rng=rep_rng)
                if keep_c is None or keep_s is None:
                    continue
                score_before = best_score
                best_score, best_matched = _evaluate_feature_panel_candidate(
                    pd.concat([cancer_filtered.iloc[keep_c], stroma_filtered.iloc[keep_s]], axis=0),
                    panel_features=panel_features,
                    target_metrics=target_metrics,
                    feature_scales=feature_scales,
                    feature_weights=feature_weights,
                    best_score=best_score,
                    best_matched=best_matched,
                    min_cells=min_cells,
                )
                if best_score < score_before:
                    best_probs = (p_cancer, p_stroma)
        for p_cancer, p_stroma in zip(
            _refine_keep_probs(best_probs[0], min_keep=min_keep_c, max_keep=1.0),
            _refine_keep_probs(best_probs[1], min_keep=min_keep_s, max_keep=1.0),
        ):
            keep_c = _stochastic_keep_mask(len(cancer_filtered), p_keep=p_cancer, min_cells=min_cells_cancer_qc, rng=rep_rng)
            keep_s = _stochastic_keep_mask(len(stroma_filtered), p_keep=p_stroma, min_cells=min_cells, rng=rep_rng)
            if keep_c is None or keep_s is None:
                continue
            best_score, best_matched = _evaluate_feature_panel_candidate(
                pd.concat([cancer_filtered.iloc[keep_c], stroma_filtered.iloc[keep_s]], axis=0),
                panel_features=panel_features,
                target_metrics=target_metrics,
                feature_scales=feature_scales,
                feature_weights=feature_weights,
                best_score=best_score,
                best_matched=best_matched,
                min_cells=min_cells,
            )

    if best_matched is None:
        return None

    suffix = f"__thin{rep}"
    best_matched["orig_sample_name"] = sample_name
    best_matched["rep"] = rep
    best_matched["sample_name_new"] = insert_suffix_in_canonical_sample_name(sample_name, suffix)
    return best_matched


def thin_internal_to_match_external_feature_panel(
    df_internal: pd.DataFrame,
    df_external: pd.DataFrame,
    *,
    compartment: str,
    separate_target_mode: str,
    n_subsamples: int,
    min_cells: int,
    min_cells_cancer_qc: int,
    seed: int,
    panel_df: pd.DataFrame,
    baseline_lookup: dict[str, dict[str, float]],
    target_by_sample: dict[str, dict[str, float]],
    feature_scales: dict[str, float],
    feature_weights: dict[str, float],
    candidate_draws: int = 2,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    if compartment == "separate":
        src_stats_c = compute_sample_median_nnd(df_internal[df_internal["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
        ref_stats_c = compute_sample_median_nnd(df_external[df_external["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
        src_stats_s = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="stroma")
        ref_stats_s = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="stroma")
        if src_stats_c.empty or ref_stats_c.empty or src_stats_s.empty or ref_stats_s.empty:
            raise ValueError("No valid cancer/stroma NND stats for feature-panel matching.")
        src_c_sorted = np.sort(src_stats_c["median_nnd_all"].dropna().to_numpy(float))
        src_s_sorted = np.sort(src_stats_s["median_nnd_stroma"].dropna().to_numpy(float))
        ref_c_vals = ref_stats_c["median_nnd_all"].dropna().to_numpy(float)
        ref_s_vals = ref_stats_s["median_nnd_stroma"].dropna().to_numpy(float)
        src_stats_c = src_stats_c.set_index("sample_name")
        src_stats_s = src_stats_s.set_index("sample_name")
        paired_targets = None
        if separate_target_mode in {"paired-2d", "paired-3d-fraction"}:
            paired_targets = build_separate_paired_targets(
                src_stats_c.reset_index(),
                src_stats_s.reset_index(),
                ref_stats_c,
                ref_stats_s,
                include_fraction=(separate_target_mode == "paired-3d-fraction"),
            ).set_index("sample_name")
    else:
        src_stats = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment=compartment)
        ref_stats = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment=compartment)
        metric_col = "median_nnd_all" if compartment == "all" else "median_nnd_stroma"
        if src_stats.empty or ref_stats.empty:
            raise ValueError("No valid NND stats for feature-panel matching.")
        src_sorted = np.sort(src_stats[metric_col].dropna().to_numpy(float))
        ref_vals = ref_stats[metric_col].dropna().to_numpy(float)

    panel_features = panel_df["feature"].tolist()
    groups = {sample_name: group for sample_name, group in df_internal.groupby("sample_name", sort=False)}
    out_parts = []
    if n_jobs is None or n_jobs < 1:
        n_jobs = min(8, os.cpu_count() or 1)

    for rep in range(1, n_subsamples + 1):
        iterable = groups.keys() if compartment == "separate" else [row.sample_name for row in src_stats.itertuples(index=False)]
        tasks = []
        for sample_idx, sample_name in enumerate(iterable):
            group = groups.get(sample_name)
            if group is None or sample_name not in target_by_sample:
                continue
            baseline_metrics = baseline_lookup.get(sample_name, {})
            stroma_adjust_mult = 1.0
            cancer_adjust_mult = 1.0
            if baseline_metrics:
                cancer_adjust_mult, stroma_adjust_mult = predict_keep_adjustments(
                    baseline_metrics,
                    target_by_sample[sample_name],
                    feature_scales,
                    feature_weights,
                )

            if compartment == "all":
                row = src_stats[src_stats["sample_name"] == sample_name].iloc[0]
                nnd_src = float(getattr(row, "median_nnd_all"))
                if not (np.isfinite(nnd_src) and nnd_src > 0):
                    continue
                rank = np.searchsorted(src_sorted, nnd_src, side="left")
                quantile = (rank + 0.5) / len(src_sorted)
                nnd_tgt = float(np.quantile(ref_vals, quantile))
                p_keep = float(min(1.0, max(0.0, (nnd_src / nnd_tgt) ** 2))) if np.isfinite(nnd_tgt) and nnd_tgt > 0 else 1.0

                coords = group[["x", "y"]].to_numpy(float, copy=False)
                keep0 = remove_isolated_points(coords, k=5, max_percentile=99.0)
                base_filtered = group.loc[keep0]
                if len(base_filtered) < min_cells:
                    continue
            elif compartment == "stroma":
                row = src_stats[src_stats["sample_name"] == sample_name].iloc[0]
                nnd_src = float(getattr(row, "median_nnd_stroma"))
                if not (np.isfinite(nnd_src) and nnd_src > 0):
                    continue
                rank = np.searchsorted(src_sorted, nnd_src, side="left")
                quantile = (rank + 0.5) / len(src_sorted)
                nnd_tgt = float(np.quantile(ref_vals, quantile))
                p_keep_s = float(min(1.0, max(0.0, (nnd_src / nnd_tgt) ** 2))) if np.isfinite(nnd_tgt) and nnd_tgt > 0 else 1.0

                cancer = group[group["CK"] == 1]
                stroma = group[group["CK"] == 0]
                if len(stroma) < min_cells:
                    continue
                coords_s = stroma[["x", "y"]].to_numpy(float, copy=False)
                keep_s0 = remove_isolated_points(coords_s, k=5, max_percentile=99.0)
                stroma_filtered = stroma.loc[keep_s0]
                if len(stroma_filtered) < min_cells:
                    continue
            else:
                if sample_name not in src_stats_c.index or sample_name not in src_stats_s.index:
                    continue
                cancer = group[group["CK"] == 1]
                stroma = group[group["CK"] == 0]
                if len(cancer) < min_cells_cancer_qc or len(stroma) < min_cells:
                    continue

                nnd_src_c = float(src_stats_c.loc[sample_name, "median_nnd_all"])
                nnd_src_s = float(src_stats_s.loc[sample_name, "median_nnd_stroma"])
                if not (np.isfinite(nnd_src_c) and nnd_src_c > 0 and np.isfinite(nnd_src_s) and nnd_src_s > 0):
                    continue

                if paired_targets is not None:
                    if sample_name not in paired_targets.index:
                        continue
                    nnd_tgt_c = float(paired_targets.loc[sample_name, "target_nnd_cancer"])
                    nnd_tgt_s = float(paired_targets.loc[sample_name, "target_nnd_stroma"])
                else:
                    rank_c = np.searchsorted(src_c_sorted, nnd_src_c, side="left")
                    rank_s = np.searchsorted(src_s_sorted, nnd_src_s, side="left")
                    quantile_c = (rank_c + 0.5) / len(src_c_sorted)
                    quantile_s = (rank_s + 0.5) / len(src_s_sorted)
                    nnd_tgt_c = float(np.quantile(ref_c_vals, quantile_c))
                    nnd_tgt_s = float(np.quantile(ref_s_vals, quantile_s))
                p_keep_c = float(min(1.0, max(0.0, (nnd_src_c / nnd_tgt_c) ** 2))) if np.isfinite(nnd_tgt_c) and nnd_tgt_c > 0 else 1.0
                p_keep_s = float(min(1.0, max(0.0, (nnd_src_s / nnd_tgt_s) ** 2))) if np.isfinite(nnd_tgt_s) and nnd_tgt_s > 0 else 1.0

                coords_c = cancer[["x", "y"]].to_numpy(float, copy=False)
                coords_s = stroma[["x", "y"]].to_numpy(float, copy=False)
                keep_c0 = remove_isolated_points(coords_c, k=5, max_percentile=99.0)
                keep_s0 = remove_isolated_points(coords_s, k=5, max_percentile=99.0)
                cancer_filtered = cancer.loc[keep_c0]
                stroma_filtered = stroma.loc[keep_s0]
                if len(cancer_filtered) < min_cells_cancer_qc or len(stroma_filtered) < min_cells:
                    continue
            task = {
                "sample_name": sample_name,
                "group": group,
                "compartment": compartment,
                "min_cells": min_cells,
                "min_cells_cancer_qc": min_cells_cancer_qc,
                "candidate_draws": candidate_draws,
                "target_metrics": target_by_sample[sample_name],
                "feature_scales": feature_scales,
                "feature_weights": feature_weights,
                "panel_features": panel_features,
                "rep": rep,
                "seed": seed + rep * 10000 + sample_idx,
            }
            if compartment == "all":
                task["base_filtered"] = base_filtered
                min_keep = min_cells / len(base_filtered)
                task["pred_keep_s"] = min(1.0, max(min_keep, p_keep * stroma_adjust_mult))
            elif compartment == "stroma":
                task["cancer"] = cancer
                task["stroma_filtered"] = stroma_filtered
                min_keep = min_cells / len(stroma_filtered)
                task["pred_keep_s"] = min(1.0, max(min_keep, p_keep_s * stroma_adjust_mult))
            else:
                task["cancer_filtered"] = cancer_filtered
                task["stroma_filtered"] = stroma_filtered
                min_keep_c = min_cells_cancer_qc / len(cancer_filtered)
                min_keep_s = min_cells / len(stroma_filtered)
                task["pred_keep_c"] = min(1.0, max(min_keep_c, p_keep_c * cancer_adjust_mult))
                task["pred_keep_s"] = min(1.0, max(min_keep_s, p_keep_s * stroma_adjust_mult))
            tasks.append(task)

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [ex.submit(_match_one_sample_feature_panel, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Feature-panel rep {rep}/{n_subsamples}"):
                matched = future.result()
                if matched is not None:
                    out_parts.append(matched)

    if not out_parts:
        raise RuntimeError("Feature-panel matching produced no cores. Check thresholds and panel settings.")

    return pd.concat(out_parts, axis=0)


def load_internal_cells(path: Path, method: str, keep_all_columns: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "cellpose":
        if keep_all_columns:
            raw = pd.read_csv(path, low_memory=False)
        else:
            needed = ["filename", "x", "y", "ck", "ck_cyto_mean_raw"]
            head = pd.read_csv(path, nrows=1)
            missing = [col for col in needed if col not in head.columns]
            if missing:
                raise ValueError(f"Cellpose cells file missing required columns {missing}. Found: {list(head.columns)}")
            raw = pd.read_csv(path, usecols=needed, low_memory=False)

        minimal = raw.copy()
        minimal["x"] = pd.to_numeric(minimal["x"], errors="coerce")
        minimal["y"] = pd.to_numeric(minimal["y"], errors="coerce")
        minimal["CK"] = pd.to_numeric(minimal["ck"], errors="coerce").fillna(0).astype(int)
        minimal["sample_name"] = minimal["filename"].map(extract_sample_name_from_filename)
        minimal = minimal.dropna(subset=["x", "y", "sample_name"])
        return raw, minimal[["sample_name", "x", "y", "CK"]]

    if method == "inform":
        raw = pd.read_csv(path, low_memory=False)
        required = ["Sample Name", "Cell X Position", "Cell Y Position", "Cancer"]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"inForm cells file missing required columns {missing}. Found: {list(raw.columns)}")

        minimal = raw.copy()
        minimal["sample_name"] = minimal["Sample Name"].astype(str)
        minimal["x"] = pd.to_numeric(minimal["Cell X Position"], errors="coerce")
        minimal["y"] = pd.to_numeric(minimal["Cell Y Position"], errors="coerce")
        minimal["CK"] = pd.to_numeric(minimal["Cancer"], errors="coerce").fillna(0).astype(int)
        minimal = minimal.dropna(subset=["x", "y", "sample_name"])
        return raw, minimal[["sample_name", "x", "y", "CK"]]

    raise ValueError(f"Unsupported internal method: {method}")


def load_allowed_internal_sample_names(samples_path: Path) -> set[str]:
    samples_df = pd.read_csv(samples_path, low_memory=False)
    if "sample_name" not in samples_df.columns:
        raise ValueError(f"Expected 'sample_name' column in {samples_path}, got: {list(samples_df.columns)}")
    return set(samples_df["sample_name"].astype(str).map(canonicalize_samples_csv_name))


def filter_internal_cells_to_samples(
    df_internal_raw: pd.DataFrame,
    df_internal: pd.DataFrame,
    *,
    allowed_sample_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = df_internal["sample_name"].astype(str).isin(allowed_sample_names)
    df_internal_f = df_internal.loc[keep].copy()
    df_internal_raw_f = df_internal_raw.loc[df_internal_f.index].copy()
    return df_internal_raw_f, df_internal_f


def apply_suffix_to_internal_rows(df_out_cells: pd.DataFrame, matched_df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df_out_cells.copy()
    suffixes = np.array([f"__thin{rep}" for rep in matched_df["rep"].to_numpy(int)], dtype=object)

    if method == "cellpose":
        new_values = []
        old_values = out["filename"].astype(str).to_numpy()
        for value, suffix in tqdm(zip(old_values, suffixes), total=len(old_values), desc="Renaming filenames"):
            new_values.append(insert_suffix_before_core(value, suffix))
        out["filename"] = new_values
    elif method == "inform":
        old_values = matched_df["sample_name_new"].astype(str).to_numpy()
        out["Sample Name"] = old_values
    else:
        raise ValueError(f"Unsupported internal method: {method}")

    out["orig_sample_name"] = matched_df["orig_sample_name"].values
    out["sample_name_new"] = matched_df["sample_name_new"].values
    out["thin_rep"] = matched_df["rep"].values
    return out


def build_output_samples(
    samples_path: Path,
    matched_df: pd.DataFrame,
    *,
    n_subsamples: int,
    out_path: Path,
) -> None:
    samples_df = pd.read_csv(samples_path, low_memory=False)
    if "sample_name" not in samples_df.columns:
        raise ValueError(f"Expected 'sample_name' column in {samples_path}, got: {list(samples_df.columns)}")

    parts = []
    for rep in range(1, n_subsamples + 1):
        suffix = f"__thin{rep}"
        dup = samples_df.copy()
        dup["sample_name"] = dup["sample_name"].astype(str).map(
            lambda value: insert_suffix_before_core(value, suffix) if "_Core[1," in str(value) else value
        )
        dup["thin_rep"] = rep
        dup["orig_sample_name"] = samples_df["sample_name"].astype(str).values
        parts.append(dup)

    samples_out = pd.concat(parts, ignore_index=True)
    present = set(matched_df["sample_name_new"].astype(str).unique())
    canon = samples_out["sample_name"].astype(str).map(canonicalize_samples_csv_name)
    samples_out = samples_out.loc[canon.isin(present)].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples_out.to_csv(out_path, index=True)


def _plot_log_histogram(values: np.ndarray, title: str, path: Path) -> None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return
    plt.figure()
    plt.hist(np.log10(x), bins=60)
    plt.xlabel("log10(median NND)")
    plt.ylabel("cores")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_overlay(series_list: list[pd.Series], labels: list[str], title: str, path: Path) -> None:
    arrays = [np.log10(s.dropna().to_numpy(float)) for s in series_list]
    combined = np.concatenate(arrays)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return

    bins = np.linspace(combined.min(), combined.max(), 80)
    plt.figure()
    for arr, label in zip(arrays, labels):
        plt.hist(arr[np.isfinite(arr)], bins=bins, alpha=0.5, label=label)
    plt.xlabel("log10(median NND)")
    plt.ylabel("cores")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_qc_histograms(
    df_external: pd.DataFrame,
    df_internal: pd.DataFrame,
    df_matched: pd.DataFrame,
    *,
    compartment: str,
    outdir: Path,
    internal_label: str,
    min_cells: int,
    min_cells_cancer_qc: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    tmp = df_matched[["sample_name_new", "x", "y", "CK"]].rename(columns={"sample_name_new": "sample_name"}).copy()

    if compartment == "all":
        ext_stats = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="all")
        int_stats = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="all")
        match_stats = compute_sample_median_nnd(tmp, min_cells=min_cells, compartment="all")

        _plot_log_histogram(ext_stats["median_nnd_all"].values, "External all cells: per-core median NND", outdir / "external_all_median_nnd_log10.png")
        _plot_log_histogram(int_stats["median_nnd_all"].values, f"{internal_label} original all cells: per-core median NND", outdir / "internal_orig_all_median_nnd_log10.png")
        _plot_log_histogram(match_stats["median_nnd_all"].values, f"{internal_label} matched all cells: per-core median NND", outdir / "internal_matched_all_median_nnd_log10.png")
        _plot_overlay(
            [ext_stats["median_nnd_all"], int_stats["median_nnd_all"], match_stats["median_nnd_all"]],
            ["external", f"{internal_label} orig", f"{internal_label} matched"],
            "All-cell per-core median NND distributions",
            outdir / "all_median_nnd_log10_overlay.png",
        )
        return

    ext_stroma = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="stroma")
    int_stroma = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="stroma")
    match_stroma = compute_sample_median_nnd(tmp, min_cells=min_cells, compartment="stroma")

    _plot_log_histogram(ext_stroma["median_nnd_stroma"].values, "External stroma: per-core median NND", outdir / "external_stroma_median_nnd_log10.png")
    _plot_log_histogram(int_stroma["median_nnd_stroma"].values, f"{internal_label} original stroma: per-core median NND", outdir / "internal_orig_stroma_median_nnd_log10.png")
    _plot_log_histogram(match_stroma["median_nnd_stroma"].values, f"{internal_label} matched stroma: per-core median NND", outdir / "internal_matched_stroma_median_nnd_log10.png")
    _plot_overlay(
        [ext_stroma["median_nnd_stroma"], int_stroma["median_nnd_stroma"], match_stroma["median_nnd_stroma"]],
        ["external stroma", f"{internal_label} stroma orig", f"{internal_label} stroma matched"],
        "Stroma per-core median NND distributions",
        outdir / "stroma_median_nnd_log10_overlay.png",
    )

    ext_cancer = compute_sample_median_nnd(df_external[df_external["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
    int_cancer = compute_sample_median_nnd(df_internal[df_internal["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
    match_cancer = compute_sample_median_nnd(tmp[tmp["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
    _plot_overlay(
        [ext_cancer["median_nnd_all"], int_cancer["median_nnd_all"], match_cancer["median_nnd_all"]],
        ["external cancer", f"{internal_label} cancer orig", f"{internal_label} cancer matched"],
        "Cancer per-core median NND distributions",
        outdir / "cancer_median_nnd_log10_overlay.png",
    )


def write_feature_panel_report(
    shift_summary: pd.DataFrame,
    panel_df: pd.DataFrame,
    *,
    outdir: Path,
    internal_label: str,
    external_label: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    shift_summary.to_csv(outdir / "feature_shift_summary.csv", index=False)
    panel_df.to_csv(outdir / "selected_feature_panel.csv", index=False)

    lines = [
        "# Feature-panel selection",
        "",
        f"- Internal cohort: {internal_label}",
        f"- External cohort: {external_label}",
        "",
        "## Selection rule",
        "- Candidate pool: the non-count spatial features used by subtype prediction.",
        "- Ranking: absolute Cohen's d between internal and external per-core feature distributions.",
        "- Family cap: avoid filling the panel with multiple near-duplicate features from the same family.",
        "",
        "## Selected features",
    ]
    for row in panel_df.itertuples(index=False):
        lines.append(
            f"- `{row.feature}` ({row.family}): d={row.cohens_d:.3f}, "
            f"mean_internal={row.mean_internal:.3f}, mean_external={row.mean_external:.3f}"
        )
    (outdir / "selected_feature_panel.md").write_text("\n".join(lines) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-method", choices=["cellpose", "inform"], default="cellpose")
    parser.add_argument("--match-compartment", choices=["all", "stroma", "separate"], default="all")
    parser.add_argument(
        "--separate-target-mode",
        choices=["independent", "paired-2d", "paired-3d-fraction"],
        default="independent",
        help="For --match-compartment separate: use independent marginal quantiles, paired cancer/stroma NND matching, or paired cancer/stroma NND plus tumor-fraction matching.",
    )
    parser.add_argument("--objective", choices=["nnd", "feature-panel"], default="nnd")
    parser.add_argument("--internal-cells", type=Path)
    parser.add_argument("--internal-samples", type=Path)
    parser.add_argument("--cellpose-cells", dest="internal_cells_legacy", type=Path)
    parser.add_argument("--cellpose-samples", dest="internal_samples_legacy", type=Path)
    parser.add_argument("--external-cells", type=Path, required=True)
    parser.add_argument("--external-xy-scale", type=float, default=2.0)
    parser.add_argument("--internal-metrics-cache", type=Path)
    parser.add_argument("--external-metrics-cache", type=Path, default=Path("outputs/metrics/spatial_metrics_EXTERNAL_BOMI1.csv"))
    parser.add_argument("--recompute-baseline-metrics", action="store_true")
    parser.add_argument("--n-subsamples", type=int, default=3)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--min-cells-cancer-qc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-binsrch", type=int, default=8)
    parser.add_argument("--tol-rel", type=float, default=0.03)
    parser.add_argument("--feature-panel-size", type=int, default=6)
    parser.add_argument("--feature-panel-max-per-family", type=int, default=1)
    parser.add_argument("--feature-panel-features", type=str)
    parser.add_argument("--feature-candidate-draws", type=int, default=2)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--out-cells", type=Path, required=True)
    parser.add_argument("--out-samples", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, default=Path("outputs/qc/qc_matched_internal"))
    parser.add_argument("--keep-all-columns", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    internal_cells_path = args.internal_cells or args.internal_cells_legacy
    internal_samples_path = args.internal_samples or args.internal_samples_legacy

    if internal_cells_path is None or internal_samples_path is None:
        raise ValueError("Provide --internal-cells and --internal-samples.")
    if args.n_subsamples < 1:
        raise ValueError("--n-subsamples must be >= 1")
    if args.internal_metrics_cache is None and args.internal_method == "cellpose":
        args.internal_metrics_cache = Path("outputs/metrics/spatial_metrics_cellpose.csv")
    elif args.internal_metrics_cache is None and args.internal_method == "inform":
        args.internal_metrics_cache = Path("outputs/metrics/spatial_metrics_inform.csv")

    print("[INFO] Loading external cells...")
    ext_raw = pd.read_csv(args.external_cells, low_memory=False)
    df_external = preprocess_external_cells(ext_raw).rename(columns={"Cancer": "CK"})
    df_external["x"] = pd.to_numeric(df_external["x"], errors="coerce") * float(args.external_xy_scale)
    df_external["y"] = pd.to_numeric(df_external["y"], errors="coerce") * float(args.external_xy_scale)
    df_external["CK"] = pd.to_numeric(df_external["CK"], errors="coerce").fillna(0).astype(int)
    df_external = df_external.dropna(subset=["x", "y", "sample_name"])
    print(f"[INFO] External cells: {len(df_external):,} rows, {df_external['sample_name'].nunique():,} samples")

    print(f"[INFO] Loading internal cells ({args.internal_method})...")
    df_internal_raw, df_internal = load_internal_cells(internal_cells_path, args.internal_method, args.keep_all_columns)
    allowed_sample_names = load_allowed_internal_sample_names(internal_samples_path)
    df_internal_raw, df_internal = filter_internal_cells_to_samples(
        df_internal_raw,
        df_internal,
        allowed_sample_names=allowed_sample_names,
    )
    print(f"[INFO] Internal cells: {len(df_internal):,} rows, {df_internal['sample_name'].nunique():,} samples")

    if args.objective == "nnd":
        print(f"[INFO] Matching internal {args.match_compartment} NND distribution to external...")
        df_matched = thin_internal_to_match_external_nnd(
            df_internal,
            df_external,
            compartment=args.match_compartment,
            separate_target_mode=args.separate_target_mode,
            n_subsamples=args.n_subsamples,
            min_cells=args.min_cells,
            seed=args.seed,
            n_binsrch=args.n_binsrch,
            tol_rel=args.tol_rel,
        )
    else:
        panel_features = parse_csv_list(args.feature_panel_features)
        baseline_features = panel_features or MODEL_FEATURE_COLUMNS
        print("[INFO] Computing baseline feature shifts for data-driven panel selection...")
        df_internal_metrics = load_or_compute_feature_table(
            cache_path=args.internal_metrics_cache,
            df_cells=df_internal,
            features=baseline_features,
            min_cells=args.min_cells,
            recompute=args.recompute_baseline_metrics,
        )
        df_external_metrics = load_or_compute_feature_table(
            cache_path=args.external_metrics_cache,
            df_cells=df_external,
            features=baseline_features,
            min_cells=args.min_cells,
            recompute=args.recompute_baseline_metrics,
        )
        shift_summary = summarize_feature_shift_table(df_internal_metrics, df_external_metrics, baseline_features)
        if not panel_features:
            panel_df = select_feature_panel(
                shift_summary,
                panel_size=args.feature_panel_size,
                max_per_family=args.feature_panel_max_per_family,
            )
        else:
            panel_df = select_feature_panel(
                shift_summary,
                panel_size=len(panel_features),
                max_per_family=args.feature_panel_max_per_family,
                manual_features=panel_features,
            )
        if panel_df.empty:
            raise RuntimeError("Feature-panel objective selected 0 features. Check the candidate feature pool.")
        print("[INFO] Selected feature panel:")
        for row in panel_df.itertuples(index=False):
            print(f"  - {row.feature} ({row.family}, d={row.cohens_d:.3f})")
        write_feature_panel_report(
            shift_summary,
            panel_df,
            outdir=args.qc_dir,
            internal_label=f"{args.internal_method}_BOMI2",
            external_label="BOMI1",
        )
        target_by_sample, feature_scales, feature_weights = build_feature_targets(
            df_internal_metrics,
            df_external_metrics,
            panel_df,
        )
        baseline_lookup = feature_table_to_lookup(df_internal_metrics, panel_df["feature"].tolist())
        print(f"[INFO] Matching internal {args.match_compartment} geometry to external using feature panel...")
        df_matched = thin_internal_to_match_external_feature_panel(
            df_internal,
            df_external,
            compartment=args.match_compartment,
            separate_target_mode=args.separate_target_mode,
            n_subsamples=args.n_subsamples,
            min_cells=args.min_cells,
            min_cells_cancer_qc=args.min_cells_cancer_qc,
            seed=args.seed,
            panel_df=panel_df,
            baseline_lookup=baseline_lookup,
            target_by_sample=target_by_sample,
            feature_scales=feature_scales,
            feature_weights=feature_weights,
            candidate_draws=args.feature_candidate_draws,
            n_jobs=args.n_jobs,
        )
    print(f"[INFO] Matched selection: {len(df_matched):,} cells across {df_matched['sample_name_new'].nunique():,} subsampled cores")

    print("[INFO] Building output cells CSV...")
    df_out_cells = df_internal_raw.loc[df_matched.index].copy()
    df_out_cells = apply_suffix_to_internal_rows(df_out_cells, df_matched, args.internal_method)
    args.out_cells.parent.mkdir(parents=True, exist_ok=True)
    df_out_cells.to_csv(args.out_cells, index=False)
    print(f"[INFO] Wrote: {args.out_cells} ({len(df_out_cells):,} rows)")

    print("[INFO] Building output samples CSV...")
    build_output_samples(
        internal_samples_path,
        df_matched,
        n_subsamples=args.n_subsamples,
        out_path=args.out_samples,
    )
    print(f"[INFO] Wrote: {args.out_samples}")

    print("[INFO] Writing QC plots...")
    plot_qc_histograms(
        df_external,
        df_internal,
        df_matched,
        compartment=args.match_compartment,
        outdir=args.qc_dir,
        internal_label=f"{args.internal_method}_BOMI2",
        min_cells=args.min_cells,
        min_cells_cancer_qc=args.min_cells_cancer_qc,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
