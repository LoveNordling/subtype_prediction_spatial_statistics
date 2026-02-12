#!/usr/bin/env python3
"""
compare_pointclouds_pixels.py

Compare marked cell point clouds from three pipelines (InForm, CellProfiler, Cellpose)
using the original pixel coordinates (no per-dataset normalization or rotation).

Sample ID harmonization:
  - InForm:       "Sample Name"
  - CellProfiler: "FileName_CK"
  - Cellpose:     "filename"
All are canonicalized to e.g. "BOMI2_TIL_1_[10,A]".

Metrics (absolute pixel units):
  - CK+ proportion
  - Bidirectional mean min-distance (stroma→tumor, tumor→stroma)
  - Newman assortativity on radius graph (radius in pixels)
  - Ripley’s L curve distance (L2 between curves, CK+ and CK−)
  - CK+ Hausdorff-95 between point sets
  - Type-aware macro-F1 from greedy matching within a pixel radius

Outputs:
  <out>_per_sample_pairwise.csv
  <out>_summary_mean_sd.csv

Usage:
  python compare_pointclouds_pixels.py \
    --inform BOMI2_all_cells_TIL.csv \
    --cellprofiler cellprofiler_extracted_cells_filtered_necrosis.csv \
    --cellpose cellpose_extracted_cells_fitlered_necrosis.csv \
    --match-radius 12 \
    --assort-radius 50 \
    --out-prefix comparison_results
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import cKDTree

# Your metric utilities (operate in pixel units here)
from spatial_metrics import (
    calculate_ripley_l,
    calculate_bidirectional_min_distance,
    calculate_newmans_assortativity,
)

from spatial_statistics import preprocess_external_cells
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------
# Canonical sample-name parsing
# ---------------------------

# examples:
# "BOMI2_TIL_1_Core[1,10,A]_[5091,35249]_component_data_CK.tiff" -> "BOMI2_TIL_1_[10,A]"
_core_pat = re.compile(r'^(?P<prefix>.+?)_Core\[1,(?P<pos>\d+,[A-Z])\]')
_brkt_pat = re.compile(r'^(?P<prefix>.+?)_\[(?P<pos>\d+,[A-Z])\]')

def canonicalize_from_filename(s: str) -> str:
    s = str(s)
    m = _core_pat.search(s)
    if m:
        return f"{m.group('prefix')}_[{m.group('pos')}]"
    m2 = _brkt_pat.search(s)
    if m2:
        return f"{m2.group('prefix')}_[{m2.group('pos')}]"
    raise ValueError(f"Cannot parse sample from filename: {s}")

def canonicalize_inform_name(s: str) -> str:
    s = str(s)
    m = _brkt_pat.search(s)
    if m:
        return f"{m.group('prefix')}_[{m.group('pos')}]"
    m2 = _core_pat.search(s)
    if m2:
        return f"{m2.group('prefix')}_[{m2.group('pos')}]"
    if "_[" in s and s.endswith("]"):
        return s[: s.rfind("]") + 1]
    raise ValueError(f"Cannot parse InForm sample name: {s}")

# ---------------------------
# Loading minimal columns (pixel units)
# ---------------------------

def load_inform(path: Path) -> pd.DataFrame:
    usecols = ["Sample Name", "Cell X Position", "Cell Y Position", "CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={"Sample Name":"sample_name", "Cell X Position":"x", "Cell Y Position":"y"})
    df["sample_name"] = df["sample_name"].map(canonicalize_inform_name)
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0)
    df["CK"] = (df["CK"] > 0).astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x","y"])
    return df[["sample_name","x","y","CK"]]

def load_cellprofiler(path: Path) -> pd.DataFrame:
    usecols = ["FileName_CK", "Location_Center_X", "Location_Center_Y", "CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={"Location_Center_X":"x", "Location_Center_Y":"y"})
    df["sample_name"] = df["FileName_CK"].map(canonicalize_from_filename)
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x","y"])
    return df[["sample_name","x","y","CK"]]

def load_cellpose(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=1)
    ck_col = "CK" if "CK" in head.columns else ("ck" if "ck" in head.columns else None)
    if ck_col is None:
        raise ValueError("Cellpose file missing a CK/ck column.")
    usecols = ["filename", "x", "y", ck_col]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["sample_name"] = df["filename"].map(canonicalize_from_filename)
    df = df.rename(columns={ck_col:"CK"})
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x","y"])
    return df[["sample_name","x","y","CK"]]





# ---------------------------
# Ripley vector unpack
# ---------------------------

def extract_ripley_vectors(ripley_dict: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c_pairs = [(float(k.split("_")[-1]), v) for k, v in ripley_dict.items() if k.startswith("cancer_ripley_L_")]
    s_pairs = [(float(k.split("_")[-1]), v) for k, v in ripley_dict.items() if k.startswith("stroma_ripley_L_")]
    if not c_pairs or not s_pairs:
        return np.array([]), np.array([]), np.array([])
    c_pairs.sort(key=lambda x: x[0]); s_pairs.sort(key=lambda x: x[0])
    b_c, v_c = zip(*c_pairs); b_s, v_s = zip(*s_pairs)
    b_c, b_s = np.array(b_c), np.array(b_s)
    if not np.allclose(b_c, b_s):
        common = np.intersect1d(b_c, b_s)
        if common.size == 0:
            return np.array([]), np.array([]), np.array([])
        mask_c = np.isin(b_c, common); mask_s = np.isin(b_s, common)
        return common, np.array(v_c)[mask_c], np.array(v_s)[mask_s]
    return b_c, np.array(v_c), np.array(v_s)

# ---------------------------
# Geometry helpers (pixel units)
# ---------------------------

def hd95(A: np.ndarray, B: np.ndarray) -> float:
    if A.size == 0 or B.size == 0:
        return float("nan")
    treeA, treeB = cKDTree(A), cKDTree(B)
    dAB, _ = treeB.query(A, k=1)
    dBA, _ = treeA.query(B, k=1)
    d = np.concatenate([dAB, dBA])
    return float(np.percentile(d, 95))

def greedy_f1_match(A: np.ndarray, B: np.ndarray, radius: float) -> float:
    if A.size == 0 and B.size == 0:
        return 1.0
    if A.size == 0 or B.size == 0:
        return 0.0
    treeB = cKDTree(B)
    matched_B = np.zeros(B.shape[0], dtype=bool)
    matches = 0
    for a in A:
        dist, idx = treeB.query(a, k=1, distance_upper_bound=radius)
        if np.isfinite(dist) and idx < B.shape[0] and not matched_B[idx]:
            matched_B[idx] = True
            matches += 1
    precision = matches / max(1, B.shape[0])
    recall    = matches / max(1, A.shape[0])
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

def type_aware_macro_f1(dfA: pd.DataFrame, dfB: pd.DataFrame, radius: float) -> float:
    f1s = []
    for t in (1, 0):
        A = dfA.loc[dfA["CK"] == t, ["x","y"]].to_numpy()
        B = dfB.loc[dfB["CK"] == t, ["x","y"]].to_numpy()
        f1s.append(greedy_f1_match(A, B, radius))
    return float(np.mean(f1s))


def remove_isolated_points(coords: np.ndarray,
                           k: int = 5,
                           max_percentile: float = 99.0) -> np.ndarray:
    """
    Remove spatial outliers based on k-NN distance.

    Parameters
    ----------
    coords : (N, 2) array
        Point coordinates.
    k : int
        Which nearest neighbor to consider.
    max_percentile : float
        Percentile cutoff for k-NN distance.

    Returns
    -------
    mask : (N,) boolean array
        True for points to keep.
    """
    if coords.shape[0] <= k:
        return np.ones(coords.shape[0], dtype=bool)

    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k + 1)  # +1 because first is self
    kth_dist = dists[:, k]

    cutoff = np.percentile(kth_dist, max_percentile)
    return kth_dist <= cutoff


def thin_cellpose_to_match_external_nnd(
    df_cellpose: pd.DataFrame,
    df_external: pd.DataFrame,
    n_subsamples: int = 3,
    min_cells: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Create thinned/augmented versions of Cellpose cores so that the distribution of
    per-core median NND (median_nnd_all) matches External.

    Output:
      A new DataFrame containing only thinned Cellpose cores with new sample_name values:
        <orig_sample_name>__thin1, __thin2, ...
      Also adds column: orig_sample_name

    Notes:
      - Only makes samples sparser (increases NND). If a target NND is smaller than the source,
        p will clamp to 1.0 and that core is not densified.
      - Uses your existing compute_sample_nnd_metrics(), so the same isolated-point removal applies.
    """
    # Per-core NND summaries (uses your outlier trimming inside compute_sample_nnd_metrics)
    src_stats = compute_sample_nnd_metrics(df_cellpose, min_cells=min_cells)
    ref_stats = compute_sample_nnd_metrics(df_external, min_cells=min_cells)

    src_stats = src_stats[["sample_name", "median_nnd_all"]].dropna()
    ref_nnd = ref_stats["median_nnd_all"].dropna().to_numpy()

    if len(src_stats) == 0 or len(ref_nnd) == 0:
        raise ValueError("No valid per-sample NND stats for source or reference cohort.")

    # For percentile mapping
    src_sorted = np.sort(src_stats["median_nnd_all"].to_numpy())

    rng = np.random.default_rng(seed)
    out_parts = []

    # Pre-split cellpose by sample_name for speed
    cellpose_groups = {s: g for s, g in df_cellpose.groupby("sample_name", sort=False)}

    for rep in range(1, n_subsamples + 1):
        # Different stream per replicate (still deterministic with seed)
        rep_rng = np.random.default_rng(seed + rep)

        for row in src_stats.itertuples(index=False):
            s = row.sample_name
            nnd_src = float(row.median_nnd_all)
            if not (np.isfinite(nnd_src) and nnd_src > 0):
                continue

            # Percentile of this core within Cellpose NND distribution
            rank = np.searchsorted(src_sorted, nnd_src, side="left")
            q = (rank + 0.5) / len(src_sorted)

            # Target NND at same percentile in External
            nnd_tgt = float(np.quantile(ref_nnd, q))
            if not (np.isfinite(nnd_tgt) and nnd_tgt > 0):
                continue

            # Thinning rule (Poisson-like scaling): nnd_new ≈ nnd_src / sqrt(p)
            # Want nnd_new = nnd_tgt  ->  p ≈ (nnd_src / nnd_tgt)^2
            p = (nnd_src / nnd_tgt) ** 2
            p = float(min(1.0, max(0.0, p)))

            g = cellpose_groups.get(s)
            if g is None or len(g) < min_cells:
                continue

            # Apply the same isolated-point filtering before subsampling, for consistency
            coords = g[["x", "y"]].to_numpy()
            mask = remove_isolated_points(coords, k=5, max_percentile=99.0)
            g_f = g.iloc[mask]
            if len(g_f) < min_cells:
                continue

            n = len(g_f)
            n_keep = int(np.round(p * n))

            if n_keep < min_cells:
                # Too aggressive thinning; skip this augmented core
                continue

            idx = rep_rng.choice(n, size=n_keep, replace=False)
            g_sub = g_f.iloc[idx].copy()

            g_sub["orig_sample_name"] = s
            g_sub["sample_name"] = f"{s}__thin{rep}"
            out_parts.append(g_sub)

    if not out_parts:
        raise RuntimeError("Thinning produced no cores. Check min_cells or cohort NND distributions.")

    return pd.concat(out_parts, ignore_index=True)


def _median_nnd(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return float("nan")
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return float(np.median(dists[:, 1]))


def thin_cellpose_to_match_external_nnd_calibrated(
    df_cellpose: pd.DataFrame,
    df_external: pd.DataFrame,
    n_subsamples: int = 3,
    min_cells: int = 10,
    n_binsrch: int = 8,          # iterations of binary search
    tol_rel: float = 0.03,       # relative tolerance on median NND
    seed: int = 0,
) -> pd.DataFrame:
    """
    Create thinned Cellpose cores whose median_nnd_all matches External's distribution,
    using quantile targets and per-core calibrated thinning (binary search on p).
    """

    # Targets: quantile match on median NND
    src_stats = compute_sample_nnd_metrics(df_cellpose, min_cells=min_cells)[["sample_name", "median_nnd_all"]].dropna()
    ref_nnd = compute_sample_nnd_metrics(df_external, min_cells=min_cells)["median_nnd_all"].dropna().to_numpy()

    if len(src_stats) == 0 or len(ref_nnd) == 0:
        raise ValueError("No valid NND stats for Cellpose or External.")

    src_sorted = np.sort(src_stats["median_nnd_all"].to_numpy())

    # Speed: pre-split groups
    groups = {s: g for s, g in df_cellpose.groupby("sample_name", sort=False)}

    out_parts = []

    for rep in range(1, n_subsamples + 1):
        rng = np.random.default_rng(seed + rep)

        for row in src_stats.itertuples(index=False):
            s = row.sample_name
            g0 = groups.get(s)
            if g0 is None or len(g0) < min_cells:
                continue

            # Apply same isolated-point removal first
            coords0 = g0[["x", "y"]].to_numpy()
            mask0 = remove_isolated_points(coords0, k=5, max_percentile=99.0)
            g = g0.iloc[mask0].copy()
            if len(g) < min_cells:
                continue

            coords = g[["x", "y"]].to_numpy()

            # Determine target NND by percentile mapping
            nnd_src = float(row.median_nnd_all)
            rank = np.searchsorted(src_sorted, nnd_src, side="left")
            q = (rank + 0.5) / len(src_sorted)
            nnd_tgt = float(np.quantile(ref_nnd, q))

            if not (np.isfinite(nnd_tgt) and nnd_tgt > 0):
                continue

            # If already sparser than target (nnd_src > nnd_tgt), we cannot densify by thinning
            # Keep p=1 and accept mismatch in this direction.
            # (Usually not your case if Cellpose is left-shifted.)
            if nnd_src >= nnd_tgt:
                g_out = g.copy()
                g_out["orig_sample_name"] = s
                g_out["sample_name"] = f"{s}__thin{rep}"
                out_parts.append(g_out)
                continue

            # Precompute one uniform random number per point for nested subsampling
            u = rng.random(len(g))

            # Binary search for p in [p_min, 1]
            p_min = min_cells / len(g)
            lo, hi = p_min, 1.0
            best = None
            best_err = float("inf")

            for _ in range(n_binsrch):
                p = 0.5 * (lo + hi)
                keep = u < p
                if keep.sum() < min_cells:
                    hi = p
                    continue

                nnd_now = _median_nnd(coords[keep])

                # Track best
                err = abs(nnd_now - nnd_tgt) / nnd_tgt
                if err < best_err:
                    best_err = err
                    best = keep.copy()

                # Decide direction: thinning increases NND
                if nnd_now < nnd_tgt:
                    # still too dense, thin more
                    hi = p
                else:
                    # too sparse, keep more
                    lo = p

                if best_err <= tol_rel:
                    break

            if best is None:
                continue

            g_sub = g.iloc[best].copy()
            g_sub["orig_sample_name"] = s
            g_sub["sample_name"] = f"{s}__thin{rep}"
            out_parts.append(g_sub)

    if not out_parts:
        raise RuntimeError("No thinned cores produced. Check min_cells, NND targets, and filtering.")
    return pd.concat(out_parts, ignore_index=True)


# ---------------------------
# Per-sample metrics (pixel units)
# ---------------------------

def per_sample_metrics(coords: np.ndarray, types: np.ndarray, assort_radius: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["n_cells"] = float(len(types))
    out["ck_pos_prop"] = float(types.mean() if len(types) else 0.0)

    bdm = calculate_bidirectional_min_distance(coords, types)  # pixel units
    out["mean_min_stroma_to_tumor"] = float(bdm["stroma_to_tumor_mean_dist"])
    out["mean_min_tumor_to_stroma"] = float(bdm["tumor_to_stroma_mean_dist"])

    out["assortativity_r"] = float(calculate_newmans_assortativity(coords, types, radius=assort_radius))

    rdict = calculate_ripley_l(coords, types)  # uses pixel radii internally
    bins, Lc, Ls = extract_ripley_vectors(rdict)
    out["_ripley_bins"] = bins
    out["_ripley_cancer"] = Lc
    out["_ripley_stroma"] = Ls
    return out

def pairwise_differences(mA: dict, mB: dict, dfA: pd.DataFrame, dfB: pd.DataFrame, match_radius: float) -> dict:
    diffs = {}
    diffs["abs_delta_ck_pos_prop"] = abs(mA["ck_pos_prop"] - mB["ck_pos_prop"])
    diffs["abs_delta_mean_min_stroma_to_tumor"] = abs(mA["mean_min_stroma_to_tumor"] - mB["mean_min_stroma_to_tumor"])
    diffs["abs_delta_mean_min_tumor_to_stroma"] = abs(mA["mean_min_tumor_to_stroma"] - mB["mean_min_tumor_to_stroma"])
    diffs["abs_delta_assortativity_r"] = abs(mA["assortativity_r"] - mB["assortativity_r"])

    # Ripley: L2 distance between curves (align on common bins)
    bA, LcA, LsA = mA["_ripley_bins"], mA["_ripley_cancer"], mA["_ripley_stroma"]
    bB, LcB, LsB = mB["_ripley_bins"], mB["_ripley_cancer"], mB["_ripley_stroma"]
    common = np.intersect1d(bA, bB)
    if common.size:
        mA_mask = np.isin(bA, common); mB_mask = np.isin(bB, common)
        diffs["ripley_L2_cancer"] = float(np.linalg.norm(LcA[mA_mask] - LcB[mB_mask]))
        diffs["ripley_L2_stroma"] = float(np.linalg.norm(LsA[mA_mask] - LsB[mB_mask]))
    else:
        diffs["ripley_L2_cancer"] = np.nan
        diffs["ripley_L2_stroma"] = np.nan

    # CK+ HD95
    A_pos = dfA.loc[dfA["CK"] == 1, ["x","y"]].to_numpy()
    B_pos = dfB.loc[dfB["CK"] == 1, ["x","y"]].to_numpy()
    diffs["hd95_ckpos"] = hd95(A_pos, B_pos)

    # Macro-F1 within pixel radius
    diffs[f"macro_f1_match_r{match_radius:g}"] = type_aware_macro_f1(dfA, dfB, match_radius)
    return diffs



def compute_sample_nnd_metrics(df: pd.DataFrame,
                               min_cells: int = 10) -> pd.DataFrame:
    """
    Compute nearest-neighbor density metrics per sample.

    Expected columns:
      - sample_name
      - x, y
      - CK (1 = cancer, 0 = stroma)

    Returns:
      One row per sample with median NND metrics.
    """
    rows = []

    for s, g in df.groupby("sample_name"):
        if len(g) < min_cells:
            continue

        coords = g[["x", "y"]].to_numpy()

        mask = remove_isolated_points(coords, k=5, max_percentile=99.0)
        coords = coords[mask]
        g = g.iloc[mask]
        
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        bbox_width  = x_max - x_min
        bbox_height = y_max - y_min
        bbox_area   = bbox_width * bbox_height

        tree = cKDTree(coords)

        # k=2: first neighbor is self (distance 0)
        dists, _ = tree.query(coords, k=2)
        nnd_all = dists[:, 1]

    
        n_all = len(g)
        n_cancer = int((g["CK"] == 1).sum())
        n_stroma = int((g["CK"] == 0).sum())

        out = {
            "sample_name": s,
            "n_cells": n_all,
            "n_cancer": n_cancer,
            "n_stroma": n_stroma,
            "median_nnd_all": float(np.median(nnd_all)),
        }
        out.update({
            "bbox_width": float(bbox_width),
            "bbox_height": float(bbox_height),
            "bbox_area": float(bbox_area),
        })



        for label, name in [(1, "cancer"), (0, "stroma")]:
            sub = g[g["CK"] == label]
            if len(sub) >= min_cells:
                sub_coords = sub[["x", "y"]].to_numpy()
                sub_tree = cKDTree(sub_coords)
                sub_dists, _ = sub_tree.query(sub_coords, k=2)
                out[f"median_nnd_{name}"] = float(np.median(sub_dists[:, 1]))
            else:
                out[f"median_nnd_{name}"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------
# Main
# ---------------------------

def run_external_vs_cellpose_density(
    cellpose_path: Path,
    external_cells_path: Path,
    out_prefix: str,
):
    # Load cellpose (BOMI2)
    df_cellpose = load_cellpose(cellpose_path)
    print(df_cellpose)
    # Load external (BOMI1)
    ext_raw = pd.read_csv(external_cells_path)
    df_external = preprocess_external_cells(ext_raw)
    df_external = df_external.rename(columns={"Cancer": "CK"})
    df_external[["x", "y"]] = df_external[["x", "y"]] * 2

    
    df_cellpose = thin_cellpose_to_match_external_nnd_calibrated(
        df_cellpose=df_cellpose,
        df_external=df_external,
        n_subsamples=3,
        min_cells=10,
        seed=0,
    )
    
    
    # Compute per-sample density metrics
    dens_cellpose = compute_sample_nnd_metrics(df_cellpose)
    dens_external = compute_sample_nnd_metrics(df_external)

    dens_cellpose["cohort"] = "Cellpose_BOMI2"
    dens_external["cohort"] = "External_BOMI1"

    all_dens = pd.concat([dens_cellpose, dens_external], ignore_index=True)
    all_dens.to_csv(f"{out_prefix}_nnd_metrics.csv", index=False)

    return all_dens


def plot_density_histograms(df, metric, out_png, log_bins=False, n_bins=60, clip_q=None):
    """
    log_bins=True  -> use logarithmically spaced bin edges (more resolution for small values)
    clip_q (e.g. 0.995) -> clip the right tail when choosing bins (visualization only)
    """
    x = pd.to_numeric(df[metric], errors="coerce").dropna()

    if log_bins:
        # Log bins require strictly positive values
        x = x[x > 0]

    if x.empty:
        raise ValueError(f"No valid values to plot for metric={metric} (after filtering).")

    if clip_q is not None:
        xmax = float(x.quantile(clip_q))
        x = x[x <= xmax]

    xmin = float(x.min())
    xmax = float(x.max())

    if log_bins:
        bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
    else:
        bins = n_bins

    plt.figure(figsize=(6, 4))
    sns.histplot(
        data=df,
        x=metric,
        hue="cohort",
        stat="density",
        common_norm=False,
        bins=bins,
        element="step",
    )

    if log_bins:
        plt.xscale("log")

    plt.xlabel(metric.replace("_", " "))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


    
def main():
    ap = argparse.ArgumentParser(description="Compare marked cell point clouds in pixel units.")
    ap.add_argument("--inform", type=str, required=True)
    ap.add_argument("--cellprofiler", type=str, required=True)
    ap.add_argument("--cellpose", type=str, required=True)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--match-radius", type=float, default=12.0, help="Greedy match radius in pixels.")
    ap.add_argument("--assort-radius", type=float, default=50.0, help="Radius for assortativity graph in pixels.")
    ap.add_argument("--out-prefix", type=str, default="comparison_results")
    ap.add_argument(
        "--external-vs-cellpose",
        action="store_true",
        help="Compare external cohort (BOMI1) vs Cellpose using density histograms only."
    )
    ap.add_argument(
        "--external-cells",
        type=str,
        default="BOMI1_cells_all.csv",
        help="External cohort cell CSV (e.g. BOMI1_cells_all.csv)."
    )
    args = ap.parse_args()


    if args.external_vs_cellpose:
        print("[INFO] Running external vs Cellpose density comparison only.")

        all_dens = run_external_vs_cellpose_density(
            cellpose_path=Path(args.cellpose),
            external_cells_path=Path(args.external_cells),
            out_prefix=args.out_prefix,
        )

        # Example plots
        plot_density_histograms(
            all_dens,
            metric="median_nnd_all",
            out_png=f"{args.out_prefix}_median_nnd_all.png",
            log_bins=True
        )
        plot_density_histograms(
            all_dens,
            metric="median_nnd_cancer",
            out_png=f"{args.out_prefix}_median_nnd_cancer.png",
        )
        plot_density_histograms(
            all_dens,
            metric="median_nnd_stroma",
            out_png=f"{args.out_prefix}_median_nnd_stroma.png",
        )

        plot_density_histograms(
            all_dens,
            metric="n_cells",
            out_png=f"{args.out_prefix}_n_cells_all.png",
        )

        plot_density_histograms(
            all_dens,
            metric="n_cancer",
            out_png=f"{args.out_prefix}_n_cells_cancer.png",
        )

        plot_density_histograms(
            all_dens,
            metric="n_stroma",
            out_png=f"{args.out_prefix}_n_cells_stroma.png",
        )

        plot_density_histograms(
            all_dens,
            metric="bbox_area",
            out_png=f"{args.out_prefix}_bbox_area.png",
        )


        print("[INFO] External vs Cellpose density analysis completed.")
        return

    
    print("[1/5] Loading CSVs...")
    df_inform = load_inform(Path(args.inform))
    df_cp     = load_cellprofiler(Path(args.cellprofiler))
    df_cpose  = load_cellpose(Path(args.cellpose))

    print(f"  InForm cells: {len(df_inform):,} | samples: {df_inform['sample_name'].nunique():,}")
    print(f"  CellProfiler cells: {len(df_cp):,} | samples: {df_cp['sample_name'].nunique():,}")
    print(f"  Cellpose cells: {len(df_cpose):,} | samples: {df_cpose['sample_name'].nunique():,}")

    print("[2/5] Intersecting samples...")
    common = sorted(set(df_inform["sample_name"]) & set(df_cp["sample_name"]) & set(df_cpose["sample_name"]))
    if not common:
        print("Examples (InForm):", list(df_inform['sample_name'].unique())[:5])
        print("Examples (CellProfiler):", list(df_cp['sample_name'].unique())[:5])
        print("Examples (Cellpose):", list(df_cpose['sample_name'].unique())[:5])
        raise RuntimeError("No overlapping sample_name across datasets after canonicalization.")
    print(f"  Common samples: {len(common)}")

    # Keep only common samples; no normalization
    df_inform = df_inform[df_inform["sample_name"].isin(common)]
    df_cp     = df_cp[df_cp["sample_name"].isin(common)]
    df_cpose  = df_cpose[df_cpose["sample_name"].isin(common)]

    print("[3/5] Computing per-sample metrics (pixel units)...")
    def compute_all(df: pd.DataFrame, label: str) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for s in tqdm(common, desc=f"Metrics: {label}"):
            sub = df[df["sample_name"] == s]
            if len(sub) < args.min_cells:
                continue
            coords = sub[["x","y"]].to_numpy()
            types  = sub["CK"].to_numpy().astype(int)
            out[s] = per_sample_metrics(coords, types, assort_radius=args.assort_radius)
        return out

    m_inform = compute_all(df_inform, "InForm")
    m_cp     = compute_all(df_cp, "CellProfiler")
    m_cpose  = compute_all(df_cpose, "Cellpose")

    print("[4/5] Pairwise differences...")
    pairs = [
        ("InForm", "CellProfiler", df_inform, df_cp, m_inform, m_cp),
        ("InForm", "Cellpose",     df_inform, df_cpose, m_inform, m_cpose),
        ("CellProfiler", "Cellpose", df_cp, df_cpose, m_cp, m_cpose),
    ]
    rows = []
    for Aname, Bname, dA, dB, mA_all, mB_all in pairs:
        for s in tqdm(common, desc=f"Comparing {Aname} vs {Bname}"):
            if s not in mA_all or s not in mB_all:
                continue
            subA = dA[dA["sample_name"] == s]
            subB = dB[dB["sample_name"] == s]
            if len(subA) < args.min_cells or len(subB) < args.min_cells:
                continue
            diffs = pairwise_differences(mA_all[s], mB_all[s], subA, subB, args.match_radius)
            diffs["sample_name"] = s
            diffs["pair"] = f"{Aname}_vs_{Bname}"
            rows.append(diffs)

    if not rows:
        raise RuntimeError("No pairwise rows computed; try lowering --min-cells or check inputs.")
    per_sample_df = pd.DataFrame(rows)
    per_sample_path = f"{args.out_prefix}_per_sample_pairwise.csv"
    per_sample_df.to_csv(per_sample_path, index=False)
    print(f"Saved per-sample differences: {per_sample_path}")

    print("[5/5] Summary mean ± SD...")
    metrics_cols = [c for c in per_sample_df.columns if c not in ("sample_name","pair")]
    summary = []
    for pair, grp in per_sample_df.groupby("pair"):
        n = len(grp)
        for m in metrics_cols:
            vals = pd.to_numeric(grp[m], errors="coerce").dropna()
            mean = float(vals.mean()) if len(vals) else float("nan")
            sd   = float(vals.std(ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))
            summary.append({"pair": pair, "metric": m, "n": n, "mean": mean, "sd": sd})
    summary_df = pd.DataFrame(summary)

    def fmt(m, s): return "NA" if (m != m) else f"{m:.3f} ± {s:.3f}"
    summary_df["mean_sd"] = [fmt(m, s) for m, s in zip(summary_df["mean"], summary_df["sd"])]
    summary_wide = summary_df.pivot(index="metric", columns="pair", values="mean_sd").reset_index()

    summary_path = f"{args.out_prefix}_summary_mean_sd.csv"
    summary_wide.to_csv(summary_path, index=False)
    print(f"Saved summary (mean ± SD): {summary_path}")

    print("\n=== Average differences (mean ± SD; pixel units) ===")
    print(summary_wide.to_string(index=False))

if __name__ == "__main__":
    main()
