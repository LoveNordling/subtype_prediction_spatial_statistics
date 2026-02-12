#!/usr/bin/env python3
"""
Make a *new* Cellpose dataset (cells.csv + samples.csv) whose per-core median nearest-neighbor
distance (NND) distribution is matched to an external cohort, via calibrated thinning/subsampling.

Design goals
- Compatible with spatial_statistics.py "cellpose" mode without code changes, by:
  - writing a new Cellpose cells CSV that keeps the expected raw columns (filename,x,y,ck,ck_cyto_mean_raw)
  - inserting "__thin{k}" into the *filename prefix* before "_Core[1," so that preprocess_cellpose_data()
    will produce unique sample_name values for the subsamples
  - duplicating samples.csv rows accordingly (same patient ID, new sample_name strings)
- Avoid leakage:
  - patient ID is preserved (replicates share the same ID)
  - NOTE: you must still do patient-level splitting / GroupKFold by ID in any learning step, otherwise
    replicated cores will leak across folds.

How thinning works (same spirit as compare_pointclouds.py)
- Compute per-core median NND on Cellpose and external cohorts.
- For each Cellpose core, find its percentile within the Cellpose NND distribution (q).
- Set the target NND as the q-quantile of the external NND distribution.
- Thin within the core using p ≈ (nnd_src / nnd_tgt)^2 (Poisson-like scaling: NND ~ 1/sqrt(density)).
- Repeat for n_subsamples, producing "__thin1", "__thin2", ...

Usage example
python make_cellpose_matched_to_external.py \
  --cellpose-cells cellpose_extracted_cells_fitlered_necrosis.csv \
  --cellpose-samples ../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv \
  --external-cells BOMI1_cells_all.csv \
  --external-xy-scale 2.0 \
  --n-subsamples 3 \
  --out-cells cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1.csv \
  --out-samples ../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1.csv \
  --qc-dir qc_matchedBOMI1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Import the exact external preprocessing used by your pipeline.
# (Defined in spatial_statistics.py)
from spatial_statistics import preprocess_external_cells


_CORE_PAT = re.compile(r"(.+?)_Core\[1,(\d+,[A-Z])\]")


def extract_sample_name_from_filename(filename: str) -> str:
    """
    Mirrors spatial_statistics.preprocess_cellpose_data() naming:
      'BOMI2_TIL_2_Core[1,10,A]_[...]_component_data_CK.tiff' -> 'BOMI2_TIL_2_[10,A]'
    """
    m = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", str(filename))
    if not m:
        raise ValueError(f"Could not parse sample name from filename: {filename}")
    return f"{m.group(1)}_[{m.group(2)}]"


def insert_suffix_before_core(filename: str, suffix: str) -> str:
    """
    Insert suffix into the filename prefix so sample_name parsing remains consistent.

    Example:
      'BOMI2_TIL_1_Core[1,1,A]_[...]_component_data_CK.tiff'
      -> 'BOMI2_TIL_1__thin1_Core[1,1,A]_[...]_component_data_CK.tiff'
    """
    s = str(filename)
    if "_Core[1," not in s:
        raise ValueError(f"Expected '_Core[1,' in filename, got: {s}")
    return s.replace("_Core[1,", f"{suffix}_Core[1,", 1)


def insert_suffix_in_canonical_sample_name(sample_name: str, suffix: str) -> str:
    """
    Canonical sample_name is like 'BOMI2_TIL_1_[1,A]'. We want:
      'BOMI2_TIL_1__thin1_[1,A]'

    This matches what spatial_statistics.preprocess_cellpose_data() will produce after we
    insert suffix into filename prefix.
    """
    s = str(sample_name)
    token = "_["
    if token not in s:
        raise ValueError(f"Expected canonical sample like '<prefix>_[<pos>]', got: {s}")
    prefix, rest = s.split(token, 1)
    return f"{prefix}{suffix}{token}{rest}"


def remove_isolated_points(coords: np.ndarray, k: int = 5, max_percentile: float = 99.0) -> np.ndarray:
    """
    Remove spatial outliers based on k-NN distance (keep <= max_percentile of kth-NN distance).
    """
    if coords.shape[0] <= k:
        return np.ones(coords.shape[0], dtype=bool)
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k + 1)  # first neighbor is self
    kth = dists[:, k]
    cutoff = np.percentile(kth, max_percentile)
    return kth <= cutoff


def _median_nnd(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return float("nan")
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return float(np.median(dists[:, 1]))


def compute_sample_nnd_metrics(df: pd.DataFrame, *, min_cells: int = 10) -> pd.DataFrame:
    """
    Per-sample median NND over *all* cells (Cancer + stroma). Uses the same isolated-point trimming
    as in compare_pointclouds.py.

    Expects columns: sample_name, x, y
    """
    rows = []
    for s, g in df.groupby("sample_name", sort=False):
        if len(g) < min_cells:
            continue
        coords = g[["x", "y"]].to_numpy(float, copy=False)
        keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
        coords_f = coords[keep]
        if coords_f.shape[0] < min_cells:
            continue
        rows.append({"sample_name": s, "median_nnd_all": _median_nnd(coords_f), "n_cells": int(coords_f.shape[0])})
    return pd.DataFrame(rows)


def thin_cellpose_to_match_external_nnd(
    df_cellpose_min: pd.DataFrame,
    df_external_min: pd.DataFrame,
    *,
    n_subsamples: int = 3,
    min_cells: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Produce thinned versions of Cellpose cores so that the *distribution* of per-core median NND
    matches the external cohort.

    Input:
      df_cellpose_min: columns [sample_name, x, y, CK] with original (canonical) sample_name
      df_external_min: columns [sample_name, x, y, CK]

    Output:
      A concatenated DataFrame with:
        - index = raw row index from df_cellpose_min (preserved)
        - columns include the same +:
            orig_sample_name, rep (int), sample_name_new
      NOTE: sample_name_new is in the "prefix__thinK_[pos]" style (prefix insertion).
    """
    src_stats = compute_sample_nnd_metrics(df_cellpose_min, min_cells=min_cells)
    ref_stats = compute_sample_nnd_metrics(df_external_min, min_cells=min_cells)

    if src_stats.empty or ref_stats.empty:
        raise ValueError("No valid per-sample NND stats for source or reference cohort (min_cells too high?).")

    src_vals = src_stats["median_nnd_all"].dropna().to_numpy(float)
    ref_vals = ref_stats["median_nnd_all"].dropna().to_numpy(float)

    if src_vals.size == 0 or ref_vals.size == 0:
        raise ValueError("No finite median NND values for source or reference.")

    src_sorted = np.sort(src_vals)

    # Pre-split groups for speed
    groups = {s: g for s, g in df_cellpose_min.groupby("sample_name", sort=False)}

    out_parts = []
    for rep in range(1, n_subsamples + 1):
        rep_rng = np.random.default_rng(seed + rep)

        for row in src_stats.itertuples(index=False):
            s = row.sample_name
            nnd_src = float(row.median_nnd_all)
            if not (np.isfinite(nnd_src) and nnd_src > 0):
                continue

            # Percentile within Cellpose NND distribution
            rank = np.searchsorted(src_sorted, nnd_src, side="left")
            q = (rank + 0.5) / len(src_sorted)

            # Target NND at same percentile in external
            nnd_tgt = float(np.quantile(ref_vals, q))
            if not (np.isfinite(nnd_tgt) and nnd_tgt > 0):
                continue

            # Thinning: nnd_new ≈ nnd_src / sqrt(p)  =>  p ≈ (nnd_src / nnd_tgt)^2
            p = float((nnd_src / nnd_tgt) ** 2)
            p = float(min(1.0, max(0.0, p)))

            g = groups.get(s)
            if g is None or len(g) < min_cells:
                continue

            # Apply isolated-point filtering before sampling
            coords = g[["x", "y"]].to_numpy(float, copy=False)
            keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
            g_f = g.loc[keep]
            if len(g_f) < min_cells:
                continue

            n = len(g_f)
            n_keep = int(np.round(p * n))
            if n_keep < min_cells:
                continue

            # Sample without replacement
            take_pos = rep_rng.choice(n, size=n_keep, replace=False)
            g_sub = g_f.iloc[take_pos].copy()

            suffix = f"__thin{rep}"
            g_sub["orig_sample_name"] = s
            g_sub["rep"] = rep
            g_sub["sample_name_new"] = insert_suffix_in_canonical_sample_name(s, suffix)
            out_parts.append(g_sub)

    if not out_parts:
        raise RuntimeError("Thinning produced no cores. Check min_cells or cohort NND distributions.")

    out = pd.concat(out_parts, axis=0)
    # Ensure index is raw index from the input df_cellpose_min (so we can subselect from the raw CSV)
    assert out.index.is_unique is False or out.index.size > 0  # duplicates possible across reps (same raw index reused)
    return out


def plot_nnd_histograms(df_ext: pd.DataFrame, df_cp: pd.DataFrame, df_cp_thin: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    ext_stats = compute_sample_nnd_metrics(df_ext, min_cells=10)
    cp_stats = compute_sample_nnd_metrics(df_cp, min_cells=10)

    # For thinned, use sample_name_new as sample_name for per-core stats
    # Important: avoid duplicate column names (df_cp_thin already has a "sample_name" column)
    tmp = df_cp_thin[["sample_name_new", "x", "y"]].rename(columns={"sample_name_new": "sample_name"}).copy()
    thin_stats = compute_sample_nnd_metrics(tmp, min_cells=10)

    def _plot_one(values, title, path):
        x = np.asarray(values, float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size == 0:
            return
        lx = np.log10(x)
        plt.figure()
        plt.hist(lx, bins=60)
        plt.xlabel("log10(median NND)")
        plt.ylabel("cores")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    _plot_one(ext_stats["median_nnd_all"].values, "External: per-core median NND", outdir / "external_median_nnd_log10.png")
    _plot_one(cp_stats["median_nnd_all"].values, "Cellpose ORIGINAL: per-core median NND", outdir / "cellpose_orig_median_nnd_log10.png")
    _plot_one(thin_stats["median_nnd_all"].values, "Cellpose THINNED: per-core median NND", outdir / "cellpose_thinned_median_nnd_log10.png")

    # Combined overlay (same bin edges)
    x_ext = np.log10(ext_stats["median_nnd_all"].dropna().to_numpy(float))
    x_cp  = np.log10(cp_stats["median_nnd_all"].dropna().to_numpy(float))
    x_th  = np.log10(thin_stats["median_nnd_all"].dropna().to_numpy(float))

    x_all = np.concatenate([x_ext, x_cp, x_th])
    x_all = x_all[np.isfinite(x_all)]
    if x_all.size:
        bins = np.linspace(x_all.min(), x_all.max(), 80)
        plt.figure()
        plt.hist(x_ext, bins=bins, alpha=0.5, label="external")
        plt.hist(x_cp,  bins=bins, alpha=0.5, label="cellpose orig")
        plt.hist(x_th,  bins=bins, alpha=0.5, label="cellpose thinned")
        plt.xlabel("log10(median NND)")
        plt.ylabel("cores")
        plt.title("Per-core median NND distributions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "median_nnd_log10_overlay.png", dpi=200)
        plt.close()


def canonicalize_samples_csv_name(s: str) -> str:
    """
    Mirrors spatial_statistics.preprocess_samples():
      - cut off the last underscore part
      - replace 'Core[1,' with '['
    """
    x = str(s)
    if "_" not in x:
        return x
    x = x[: x.rfind("_")]
    x = x.replace("Core[1,", "[")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellpose-cells", type=Path, required=True)
    ap.add_argument("--cellpose-samples", type=Path, required=True)
    ap.add_argument("--external-cells", type=Path, required=True)

    ap.add_argument("--external-xy-scale", type=float, default=2.0,
                    help="Multiply external x/y by this factor (e.g. 2.0 if Cellpose is in pixels and external is µm at 2 px/µm).")

    ap.add_argument("--n-subsamples", type=int, default=3)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out-cells", type=Path, required=True)
    ap.add_argument("--out-samples", type=Path, required=True)
    ap.add_argument("--qc-dir", type=Path, default=Path("qc_matched_cellpose"))

    ap.add_argument("--keep-all-columns", action="store_true",
                    help="If set: read and write all Cellpose columns (bigger files). Default keeps only what spatial_statistics cellpose mode needs.")
    args = ap.parse_args()

    assert args.n_subsamples >= 1

    # ---- Load external cohort (standardize to x,y,Cancer,sample_name) ----
    print("[INFO] Loading external cells...")
    ext_raw = pd.read_csv(args.external_cells, low_memory=False)
    df_ext = preprocess_external_cells(ext_raw)  # x,y,Cancer,sample_name fileciteturn10file17
    df_ext = df_ext.rename(columns={"Cancer": "CK"})  # align naming with compare_pointclouds
    df_ext["x"] = pd.to_numeric(df_ext["x"], errors="coerce") * float(args.external_xy_scale)
    df_ext["y"] = pd.to_numeric(df_ext["y"], errors="coerce") * float(args.external_xy_scale)
    df_ext["CK"] = df_ext["CK"].astype(int)
    df_ext = df_ext.dropna(subset=["x", "y"])
    print(f"[INFO] External cells: {len(df_ext):,} rows, {df_ext['sample_name'].nunique():,} samples")

    # ---- Load Cellpose cells ----
    print("[INFO] Loading Cellpose cells...")
    if args.keep_all_columns:
        df_cp_raw = pd.read_csv(args.cellpose_cells, low_memory=False)
    else:
        # Only what's needed by spatial_statistics.preprocess_cellpose_data() fileciteturn10file13
        needed = ["filename", "x", "y", "ck", "ck_cyto_mean_raw"]
        df_head = pd.read_csv(args.cellpose_cells, nrows=1)
        missing = [c for c in needed if c not in df_head.columns]
        if missing:
            raise ValueError(f"Cellpose cells file missing required columns {missing}. Found: {list(df_head.columns)}")
        df_cp_raw = pd.read_csv(args.cellpose_cells, usecols=needed, low_memory=False)

    # Standardize minimal view for thinning
    df_cp_min = df_cp_raw.copy()
    df_cp_min["x"] = pd.to_numeric(df_cp_min["x"], errors="coerce")
    df_cp_min["y"] = pd.to_numeric(df_cp_min["y"], errors="coerce")
    df_cp_min["CK"] = pd.to_numeric(df_cp_min["ck"], errors="coerce").fillna(0).astype(int)
    df_cp_min["sample_name"] = df_cp_min["filename"].map(extract_sample_name_from_filename)
    df_cp_min = df_cp_min.dropna(subset=["x", "y", "sample_name"])
    df_cp_min = df_cp_min[["sample_name", "x", "y", "CK"]]
    print(f"[INFO] Cellpose cells: {len(df_cp_min):,} rows, {df_cp_min['sample_name'].nunique():,} samples")

    # ---- QC pre ----
    print("[INFO] Writing QC plots (before thinning)...")
    args.qc_dir.mkdir(parents=True, exist_ok=True)

    # ---- Thinning ----
    print("[INFO] Thinning Cellpose to match external NND distribution...")
    df_thin = thin_cellpose_to_match_external_nnd(
        df_cp_min,
        df_ext.rename(columns={"sample_name": "sample_name"}),  # already
        n_subsamples=args.n_subsamples,
        min_cells=args.min_cells,
        seed=args.seed,
    )
    # df_thin includes orig_sample_name, rep, sample_name_new and has indices pointing to df_cp_min (same as df_cp_raw)
    print(f"[INFO] Thinned selection: {len(df_thin):,} cells across {df_thin['sample_name_new'].nunique():,} subsampled cores")

    # ---- Build output cells file ----
    print("[INFO] Building output Cellpose cells CSV...")
    # Subselect raw rows by index (preserved)
    df_out_cells = df_cp_raw.loc[df_thin.index].copy()

    # Insert suffix into filename based on rep
    rep_arr = df_thin["rep"].to_numpy(int)
    suffixes = np.array([f"__thin{r}" for r in rep_arr], dtype=object)

    # Vectorized-ish loop (string replace isn't truly vectorized)
    new_filenames = []
    fn_series = df_out_cells["filename"].astype(str).to_numpy()
    for fn, suf in tqdm(list(zip(fn_series, suffixes)), desc="Renaming filenames", total=len(fn_series)):
        new_filenames.append(insert_suffix_before_core(fn, suf))
    df_out_cells["filename"] = new_filenames

    # Optionally include a sanity column (harmless for spatial_statistics, which selects only a subset)
    df_out_cells["orig_sample_name"] = df_thin["orig_sample_name"].values
    df_out_cells["sample_name_new"] = df_thin["sample_name_new"].values
    df_out_cells["thin_rep"] = df_thin["rep"].values

    args.out_cells.parent.mkdir(parents=True, exist_ok=True)
    df_out_cells.to_csv(args.out_cells, index=False)
    print(f"[INFO] Wrote: {args.out_cells} ({len(df_out_cells):,} rows)")

    # ---- Build output samples.csv ----
    print("[INFO] Building output samples CSV...")
    df_samples = pd.read_csv(args.cellpose_samples, low_memory=False)
    if "sample_name" not in df_samples.columns:
        raise ValueError(f"Expected 'sample_name' column in {args.cellpose_samples}, got: {list(df_samples.columns)}")

    # Duplicate rows for each rep and inject suffix into sample_name string
    parts = []
    for rep in range(1, args.n_subsamples + 1):
        suf = f"__thin{rep}"
        g = df_samples.copy()
        # inject into the filename-like sample_name string
        g["sample_name"] = g["sample_name"].astype(str).map(lambda s: insert_suffix_before_core(s, suf) if "_Core[1," in str(s) else s)
        g["thin_rep"] = rep
        g["orig_sample_name"] = df_samples["sample_name"].astype(str).values
        parts.append(g)

    df_samples_out = pd.concat(parts, ignore_index=True)

    # Filter samples to those that actually have cells in df_out_cells
    present = set(df_thin["sample_name_new"].astype(str).unique())
    canon = df_samples_out["sample_name"].astype(str).map(canonicalize_samples_csv_name)
    keep = canon.isin(present)
    df_samples_out = df_samples_out.loc[keep].copy()

    # Reindex for a clean leading index column (matches your existing samples.csv style)
    args.out_samples.parent.mkdir(parents=True, exist_ok=True)
    df_samples_out.to_csv(args.out_samples, index=True)
    print(f"[INFO] Wrote: {args.out_samples} ({len(df_samples_out):,} rows)")

    # ---- QC plots after thinning ----
    print("[INFO] Writing QC plots (after thinning)...")
    plot_nnd_histograms(df_ext, df_cp_min, df_thin, args.qc_dir)

    # ---- Final sanity: show what spatial_statistics will parse ----
    # The key is that:
    #   - preprocess_cellpose_data() extracts sample_name from filename prefix fileciteturn10file13
    #   - preprocess_samples() canonicalizes samples.csv names similarly fileciteturn10file7
    # so they should match.
    print("[INFO] Done.")
    print("Tip: In spatial_statistics.py, use --cellpose with paths changed to the new out-cells and out-samples.")


if __name__ == "__main__":
    main()
