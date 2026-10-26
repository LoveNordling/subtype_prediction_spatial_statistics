#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu


METHOD_ORDER = ["CellProfiler", "Cellpose", "InForm"]
METHOD_COLORS = {
    "CellProfiler": "#1b9e77",
    "Cellpose": "#d95f02",
    "InForm": "#7570b3",
}


def canonicalize_from_filename(s: str) -> str:
    s = str(s)
    m = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", s)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    m = re.search(r"(.+?)_\[(\d+,[A-Z])\]", s)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    raise ValueError(f"Cannot parse sample from filename: {s}")


def canonicalize_inform_name(s: str) -> str:
    s = str(s)
    m = re.search(r"(.+?)_\[(\d+,[A-Z])\]", s)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    m = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", s)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    raise ValueError(f"Cannot parse InForm sample name: {s}")


def load_inform(path: str) -> pd.DataFrame:
    usecols = ["Sample Name", "Cell X Position", "Cell Y Position", "CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={"Sample Name": "sample_name", "Cell X Position": "x", "Cell Y Position": "y"})
    df["sample_name"] = df["sample_name"].map(canonicalize_inform_name)
    df["x"] = pd.to_numeric(df["x"], errors="coerce") * 2.0
    df["y"] = pd.to_numeric(df["y"], errors="coerce") * 2.0
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0)
    df["CK"] = (df["CK"] > 0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_cellprofiler(path: str) -> pd.DataFrame:
    usecols = ["FileName_CK", "Location_Center_X", "Location_Center_Y", "Intensity_MeanIntensity_CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["sample_name"] = df["FileName_CK"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["Location_Center_X"], errors="coerce")
    df["y"] = pd.to_numeric(df["Location_Center_Y"], errors="coerce")
    intens = pd.to_numeric(df["Intensity_MeanIntensity_CK"], errors="coerce")
    thr = threshold_otsu(intens.dropna().to_numpy())
    df["CK"] = (intens >= thr).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_cellpose(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=1)
    ck_col = "CK" if "CK" in head.columns else ("ck" if "ck" in head.columns else None)
    if ck_col is None:
        raise ValueError("Cellpose file missing CK/ck column")
    usecols = ["filename", "x", "y", ck_col]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["sample_name"] = df["filename"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df[ck_col], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def median_nnd(arr_xy: np.ndarray) -> float:
    if arr_xy.shape[0] < 2:
        return float("nan")
    tree = cKDTree(arr_xy)
    dists, _ = tree.query(arr_xy, k=2)
    nn = dists[:, 1]
    if nn.size == 0:
        return float("nan")
    return float(np.median(nn))


def per_sample_metrics(df: pd.DataFrame, method_name: str, keep_samples: set) -> pd.DataFrame:
    rows = []
    grouped = df[df["sample_name"].isin(keep_samples)].groupby("sample_name", sort=False)
    for sample, g in grouped:
        xy = g[["x", "y"]].to_numpy(dtype=float)
        is_tumor = g["CK"].to_numpy(dtype=int) == 1
        xy_t = xy[is_tumor]
        xy_s = xy[~is_tumor]

        n_all = int(len(xy))
        n_t = int(len(xy_t))
        n_s = int(len(xy_s))

        x_min = float(np.min(xy[:, 0])) if n_all else float("nan")
        x_max = float(np.max(xy[:, 0])) if n_all else float("nan")
        y_min = float(np.min(xy[:, 1])) if n_all else float("nan")
        y_max = float(np.max(xy[:, 1])) if n_all else float("nan")
        bbox_area = float((x_max - x_min) * (y_max - y_min)) if n_all else float("nan")

        med_nnd_all = median_nnd(xy)
        med_nnd_t = median_nnd(xy_t)
        med_nnd_s = median_nnd(xy_s)

        rows.append(
            {
                "sample_name": sample,
                "method": method_name,
                "n_cells_all": n_all,
                "n_cells_tumor": n_t,
                "n_cells_stroma": n_s,
                "tumor_fraction": (n_t / n_all) if n_all > 0 else float("nan"),
                "bbox_area": bbox_area,
                "density_all": (n_all / bbox_area) if bbox_area > 0 else float("nan"),
                "median_nnd_all": med_nnd_all,
                "median_nnd_tumor": med_nnd_t,
                "median_nnd_stroma": med_nnd_s,
            }
        )
    return pd.DataFrame(rows)


def _finite_vals(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def plot_hist_overlay(df: pd.DataFrame, metric: str, out_path: Path, title: str, bins: int = 45, log10: bool = False) -> None:
    plt.figure(figsize=(8.5, 6.2))

    all_vals = []
    per_method_vals = {}
    for method in METHOD_ORDER:
        vals = _finite_vals(df.loc[df["method"] == method, metric])
        if log10:
            vals = vals[vals > 0]
            vals = np.log10(vals)
        if vals.size == 0:
            continue
        per_method_vals[method] = vals
        all_vals.append(vals)

    if not all_vals:
        return

    combo = np.concatenate(all_vals)
    lo, hi = np.quantile(combo, [0.01, 0.99])
    if hi <= lo:
        lo, hi = np.min(combo), np.max(combo)

    for method in METHOD_ORDER:
        vals = per_method_vals.get(method)
        if vals is None:
            continue
        v = np.clip(vals, lo, hi)
        plt.hist(
            v,
            bins=bins,
            density=True,
            alpha=0.38,
            label=f"{method} (n={len(vals)})",
            color=METHOD_COLORS[method],
            edgecolor="none",
        )

    xlab = f"log10({metric})" if log10 else metric
    plt.xlabel(xlab)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate per-sample comparison histograms for supervisor meeting.")
    ap.add_argument("--inform", default="data/raw/BOMI2_all_cells_TIL.csv")
    ap.add_argument("--cellprofiler", default="data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv")
    ap.add_argument("--cellpose", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    ap.add_argument("--out-dir", default="outputs/figures/supervisor_meeting_2026_02_23")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    df_i = load_inform(args.inform)
    df_cp = load_cellprofiler(args.cellprofiler)
    df_cpose = load_cellpose(args.cellpose)

    common = set(df_i["sample_name"].unique()) & set(df_cp["sample_name"].unique()) & set(df_cpose["sample_name"].unique())
    print(f"[INFO] Common samples across all methods: {len(common)}")

    m_cp = per_sample_metrics(df_cp, "CellProfiler", common)
    m_cpose = per_sample_metrics(df_cpose, "Cellpose", common)
    m_i = per_sample_metrics(df_i, "InForm", common)
    all_metrics = pd.concat([m_cp, m_cpose, m_i], ignore_index=True)

    all_metrics.to_csv(out_dir / "histogram_metrics_per_sample.csv", index=False)

    plot_specs = [
        ("n_cells_all", False, "Per-sample total cell count distribution"),
        ("n_cells_tumor", False, "Per-sample tumor cell count distribution"),
        ("n_cells_stroma", False, "Per-sample stroma cell count distribution"),
        ("tumor_fraction", False, "Per-sample tumor fraction distribution"),
        ("bbox_area", True, "Per-sample bounding-box area distribution"),
        ("density_all", True, "Per-sample cell density distribution"),
        ("median_nnd_all", True, "Per-sample median NND (all cells)"),
        ("median_nnd_tumor", True, "Per-sample median NND (tumor cells)"),
        ("median_nnd_stroma", True, "Per-sample median NND (stroma cells)"),
    ]

    for metric, log10, title in plot_specs:
        suffix = "log10" if log10 else "raw"
        out_png = out_dir / f"hist_{metric}_{suffix}.png"
        plot_hist_overlay(all_metrics, metric, out_png, title, bins=50, log10=log10)
        print(f"[INFO] Saved {out_png.name}")

    print(f"[DONE] Wrote histograms and metric CSV to {out_dir}")


if __name__ == "__main__":
    main()
