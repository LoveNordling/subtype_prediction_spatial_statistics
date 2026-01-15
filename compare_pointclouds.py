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

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare marked cell point clouds in pixel units.")
    ap.add_argument("--inform", type=str, required=True)
    ap.add_argument("--cellprofiler", type=str, required=True)
    ap.add_argument("--cellpose", type=str, required=True)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--match-radius", type=float, default=12.0, help="Greedy match radius in pixels.")
    ap.add_argument("--assort-radius", type=float, default=50.0, help="Radius for assortativity graph in pixels.")
    ap.add_argument("--out-prefix", type=str, default="comparison_results")
    args = ap.parse_args()

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
