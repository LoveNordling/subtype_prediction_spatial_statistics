#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from make_cellpose_matched_to_external_v2 import compute_sample_median_nnd
from patient_scatter_plot_utils import load_bomi1_cells
from spatial_artifact_utils import read_split_ids, subset_by_ids
from spatial_data_pipeline import ExternalPaths, InternalPaths
from spatial_utils import (
    CountQC,
    aggregate_cores_to_patient,
    filter_metrics_by_counts,
    get_feature_columns,
    preprocess_external_cells,
)


COLOR_STROMA = "#0072B2"
COLOR_TUMOR = "#E64B35"
FOLLOW_COLOR = "#3FE0D0"
COLOR_HE = "#D55E00"
COLOR_MIF = "#1F9E9C"
COLOR_MATCHED = "#4DAF4A"
DEFAULT_BOMI2_SLIDE = "BOMI2_TIL_1"
DEFAULT_BOMI1_CELLS = "data/raw/BOMI1_cells_all_cellvitpp_2026_03_26.csv"
DEFAULT_BOMI1_META = "data/reference/BOMI1_clinical_data_LUADvsSqCC.csv"
DEFAULT_BOMI1_ALL_CORES_DIR = "../BOMI1_TMA/all_cores"
DEFAULT_BOMI1_HARMONIZATION_CELLS = DEFAULT_BOMI1_CELLS


def canonicalize_from_filename(text: str) -> str:
    value = str(text)
    match = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", value)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    match = re.search(r"(.+?)_\[(\d+,[A-Z])\]", value)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    raise ValueError(f"Cannot parse sample name from: {text}")


def parse_sample_position(sample_name: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"(BOMI2_TIL_\d+)_\[(\d+),([A-Z])\]", str(sample_name))
    if match is None:
        raise ValueError(f"Cannot parse slide/position from sample name: {sample_name}")
    return match.group(1), int(match.group(2)), match.group(3)


def parse_origin_from_image_path(image_path: str) -> tuple[int, int]:
    match = re.search(r"_\[(\d+),(\d+)\]_component_data\.tif$", str(image_path))
    if match is None:
        raise ValueError(f"Cannot parse origin from image path: {image_path}")
    return int(match.group(1)), int(match.group(2))


def scale_channel_to_uint8(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.5))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return np.round(255.0 * scaled).astype(np.uint8)


def downsample_uint8_image(image: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None or max_dim <= 0:
        return image
    h, w = image.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / max_dim)))
    if step == 1:
        return image
    return image[::step, ::step, :]


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, 0.0, 1.0)
        return np.round(arr * 255.0).astype(np.uint8)
    return arr.astype(np.uint8)


def load_bomi2_ck_dapi_image(path: str | Path, max_dim: int | None = None) -> tuple[np.ndarray, int, int]:
    arr = tifffile.imread(Path(path), key=[0, 6])
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise ValueError(f"Unexpected BOMI2 image shape for {path}: {arr.shape}")
    dapi = scale_channel_to_uint8(arr[0])
    ck = scale_channel_to_uint8(arr[1])
    rgb = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.uint8)
    rgb[..., 1] = ck
    rgb[..., 2] = dapi
    original_width = int(rgb.shape[1])
    original_height = int(rgb.shape[0])
    return downsample_uint8_image(rgb, max_dim=max_dim), original_width, original_height


def load_rgb_image(path: str | Path, max_dim: int | None = None) -> tuple[np.ndarray, int, int]:
    arr = to_uint8_rgb(mpimg.imread(Path(path)))
    original_width = int(arr.shape[1])
    original_height = int(arr.shape[0])
    return downsample_uint8_image(arr, max_dim=max_dim), original_width, original_height


def load_comparison_index(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"sample_name", "image_path", "out_path"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Comparison index missing columns {sorted(missing)}: {path}")
    df["sample_name"] = df["sample_name"].astype(str)
    df["image_path"] = df["image_path"].astype(str)
    df["out_path"] = df["out_path"].astype(str)
    slide_info = df["sample_name"].map(parse_sample_position)
    df["slide_id"] = [item[0] for item in slide_info]
    df["core_col"] = [item[1] for item in slide_info]
    df["core_row"] = [item[2] for item in slide_info]
    origins = df["image_path"].map(parse_origin_from_image_path)
    df["origin_x"] = [item[0] for item in origins]
    df["origin_y"] = [item[1] for item in origins]
    return df


def choose_follow_sample(slide_df: pd.DataFrame, requested_sample: str | None) -> str:
    if requested_sample is not None:
        if requested_sample not in set(slide_df["sample_name"]):
            raise ValueError(f"Requested follow sample not present on selected slide: {requested_sample}")
        return requested_sample
    first_row = slide_df.loc[slide_df["core_row"] == "A"].sort_values(["core_col", "sample_name"]).reset_index(drop=True)
    if len(first_row) >= 4:
        return str(first_row.iloc[3]["sample_name"])
    if not first_row.empty:
        return str(first_row.iloc[0]["sample_name"])
    return str(slide_df.sort_values(["core_row", "core_col", "sample_name"]).iloc[0]["sample_name"])


def load_bomi2_method_cells(path: str, source: str) -> pd.DataFrame:
    if source == "inform":
        df = pd.read_csv(path, usecols=["Sample Name", "Cell X Position", "Cell Y Position", "CK"], low_memory=False)
        df = df.rename(columns={"Sample Name": "sample_name", "Cell X Position": "x", "Cell Y Position": "y"})
        df["sample_name"] = df["sample_name"].map(canonicalize_from_filename)
    elif source == "cellpose":
        head = pd.read_csv(path, nrows=1)
        ck_col = "CK" if "CK" in head.columns else ("ck" if "ck" in head.columns else None)
        if ck_col is None:
            raise ValueError(f"Cellpose file missing CK/ck column: {path}")
        df = pd.read_csv(path, usecols=["filename", "x", "y", ck_col], low_memory=False)
        df["sample_name"] = df["filename"].map(canonicalize_from_filename)
        df["CK"] = df[ck_col]
    elif source == "cellprofiler":
        df = pd.read_csv(path, usecols=["FileName_CK", "Location_Center_X", "Location_Center_Y", "CK"], low_memory=False)
        df["sample_name"] = df["FileName_CK"].map(canonicalize_from_filename)
        df["x"] = df["Location_Center_X"]
        df["y"] = df["Location_Center_Y"]
    else:
        raise ValueError(f"Unsupported BOMI2 source: {source}")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["sample_name", "x", "y"])


def pick_bomi1_sample(bomi1_cells: pd.DataFrame, requested_sample: str | None) -> str:
    available = sorted(bomi1_cells["sample_name"].astype(str).unique())
    if not available:
        raise ValueError("No BOMI1 samples available")
    if requested_sample is not None:
        if requested_sample not in set(available):
            raise ValueError(f"Requested BOMI1 sample not found: {requested_sample}")
        return requested_sample
    a7a = [sample for sample in available if sample.startswith("ULunA7A_")]
    if a7a:
        return a7a[0]
    return available[0]


def find_bomi1_core_image(sample_name: str, all_cores_dir: str) -> Path:
    base = Path(all_cores_dir)
    for suffix in (".jpg", ".png", ".tif", ".tiff"):
        candidate = base / f"{sample_name}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(base.glob(f"{sample_name}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find BOMI1 core image for {sample_name} in {all_cores_dir}")


def save_clean_scatter(
    g: pd.DataFrame,
    out_path: Path,
    *,
    dot_size: float,
    transparent: bool,
) -> None:
    g = g.copy()
    x_min = float(g["x"].min())
    x_max = float(g["x"].max())
    y_min = float(g["y"].min())
    y_max = float(g["y"].max())

    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)
    fig_w = 7.0
    fig_h = fig_w * (height / width)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    ax.set_facecolor((0, 0, 0, 0) if transparent else "white")

    stroma = g.loc[g["CK"] == 0]
    tumor = g.loc[g["CK"] == 1]
    if not stroma.empty:
        ax.scatter(stroma["x"], stroma["y"], s=dot_size, c=COLOR_STROMA, linewidths=0, alpha=0.92, rasterized=True)
    if not tumor.empty:
        ax.scatter(tumor["x"], tumor["y"], s=dot_size, c=COLOR_TUMOR, linewidths=0, alpha=0.92, rasterized=True)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=transparent)
    plt.close(fig)


def save_follow_core_image(
    image_rgb: np.ndarray,
    out_path: Path,
    *,
    transparent: bool,
) -> None:
    h, w = image_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 300.0, h / 300.0), dpi=300)
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    ax.imshow(image_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=transparent)
    plt.close(fig)


def strip_thin_suffix(sample_name: str) -> str:
    return re.sub(r"__thin\d+(?=_\[)", "", str(sample_name))


def load_external_bomi1_cells_for_harmonization(path: str, xy_scale: float) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = preprocess_external_cells(raw).rename(columns={"Cancer": "CK"})
    df["x"] = pd.to_numeric(df["x"], errors="coerce") * float(xy_scale)
    df["y"] = pd.to_numeric(df["y"], errors="coerce") * float(xy_scale)
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["sample_name", "x", "y"])


def _point_cloud_bounds(g: pd.DataFrame, pad_frac: float = 0.03) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(g["x"].min())
    x_max = float(g["x"].max())
    y_min = float(g["y"].min())
    y_max = float(g["y"].max())
    dx = max(1.0, x_max - x_min)
    dy = max(1.0, y_max - y_min)
    x_pad = max(dx * pad_frac, 1.0)
    y_pad = max(dy * pad_frac, 1.0)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_point_cloud_panel(
    ax: plt.Axes,
    g: pd.DataFrame,
    *,
    title: str,
    subtitle: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    dot_size: float = 1.0,
) -> None:
    stroma = g.loc[g["CK"] == 0]
    tumor = g.loc[g["CK"] == 1]
    if not stroma.empty:
        ax.scatter(stroma["x"], stroma["y"], s=dot_size, c=COLOR_STROMA, linewidths=0, alpha=0.9, rasterized=True)
    if not tumor.empty:
        ax.scatter(tumor["x"], tumor["y"], s=dot_size, c=COLOR_TUMOR, linewidths=0, alpha=0.9, rasterized=True)

    if xlim is None or ylim is None:
        xlim, ylim = _point_cloud_bounds(g)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if subtitle:
        ax.text(0.5, -0.10, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_feature_label(name: str) -> str:
    name = str(name)
    if name.startswith("cancer_ripley_L_"):
        suffix = name.split("_")[-1]
        return "Cancer Ripley L AUC" if suffix == "auc" else f"Cancer Ripley L{int(float(suffix))}"
    if name.startswith("stroma_ripley_L_"):
        suffix = name.split("_")[-1]
        return "Stroma Ripley L AUC" if suffix == "auc" else f"Stroma Ripley L{int(float(suffix))}"
    if name == "newmans_assortativity":
        return "Newman assortativity"
    if name == "cancer_degree_centrality":
        return "Cancer degree centrality"
    if name == "stroma_degree_centrality":
        return "Stroma degree centrality"
    if name == "cancer_closeness_centrality":
        return "Cancer closeness centrality"
    if name == "stroma_closeness_centrality":
        return "Stroma closeness centrality"
    if name == "cancer_average_clustering":
        return "Cancer average clustering"
    if name == "stroma_average_clustering":
        return "Stroma average clustering"
    if name == "cancer_stroma_cooccurrence_auc":
        return "Cancer-stroma co-occurrence AUC"
    if name == "cancer_stroma_cooccurrence_deviation_auc":
        return "Cancer-stroma co-occurrence deviation AUC"
    if name == "cancer_cancer_neighborhood_enrichment_zscore":
        return "Cancer-cancer enrich. z"
    if name == "cancer_stroma_neighborhood_enrichment_zscore":
        return "Cancer-stroma enrich. z"
    if name == "stroma_stroma_neighborhood_enrichment_zscore":
        return "Stroma-stroma enrich. z"
    if name == "cancer_stroma_object_object_pearson_correlation":
        return "Cancer-stroma object-object correlation"
    return name.replace("_", " ")


def feature_family_key(name: str) -> str:
    name = str(name)
    if name.startswith("cancer_ripley_L_"):
        return "cancer_ripley"
    if name.startswith("stroma_ripley_L_"):
        return "stroma_ripley"
    if name == "newmans_assortativity":
        return "assortativity"
    if name.endswith("_degree_centrality") or name.endswith("_closeness_centrality") or name.endswith("_average_clustering"):
        return "centrality"
    if "cooccurrence" in name:
        return "cooccurrence"
    if name.endswith("_neighborhood_enrichment_zscore"):
        return "enrichment"
    if name.endswith("_object_object_pearson_correlation"):
        return "object_object"
    return name


def fit_meanstd_alignment(source_train: pd.DataFrame, target_train: pd.DataFrame, feat_cols: list[str]) -> dict[str, np.ndarray]:
    src = source_train[feat_cols].to_numpy(float)
    tgt = target_train[feat_cols].to_numpy(float)
    src_mean = np.nanmean(src, axis=0)
    tgt_mean = np.nanmean(tgt, axis=0)
    src_std = np.nanstd(src, axis=0)
    tgt_std = np.nanstd(tgt, axis=0)
    src_std = np.where(np.isfinite(src_std) & (src_std > 0), src_std, 1.0)
    tgt_std = np.where(np.isfinite(tgt_std) & (tgt_std > 0), tgt_std, 1.0)
    return {
        "source_mean": src_mean,
        "source_std": src_std,
        "target_mean": tgt_mean,
        "target_std": tgt_std,
    }


def apply_meanstd_alignment(df: pd.DataFrame, feat_cols: list[str], alignment: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    vals = out[feat_cols].to_numpy(float)
    vals = ((vals - alignment["source_mean"]) / alignment["source_std"]) * alignment["target_std"] + alignment["target_mean"]
    out.loc[:, feat_cols] = vals
    return out


def save_density_harmonization_figure(
    *,
    inform_original_path: str,
    inform_matched_path: str,
    external_bomi1_path: str,
    external_xy_scale: float,
    sample_name: str,
    target_external_sample_name: str | None,
    out_path: Path,
    min_cells: int,
    dot_size: float,
) -> dict[str, object]:
    df_orig = load_bomi2_method_cells(inform_original_path, "inform")
    df_matched = load_bomi2_method_cells(inform_matched_path, "inform")
    df_external = load_external_bomi1_cells_for_harmonization(external_bomi1_path, xy_scale=external_xy_scale)

    df_matched = df_matched.copy()
    df_matched["orig_sample_name"] = df_matched["sample_name"].map(strip_thin_suffix)
    df_matched["thin_rep"] = (
        df_matched["sample_name"].astype(str).str.extract(r"__thin(\d+)_", expand=False).fillna("1").astype(int)
    )
    df_matched_rep1 = df_matched.loc[df_matched["thin_rep"] == 1].copy()

    orig_stats = compute_sample_median_nnd(df_orig, min_cells=min_cells, compartment="all")
    matched_stats = compute_sample_median_nnd(
        df_matched_rep1[["sample_name", "x", "y", "CK"]], min_cells=min_cells, compartment="all"
    )
    external_stats = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="all")

    if orig_stats.empty or matched_stats.empty or external_stats.empty:
        raise RuntimeError("Could not compute median_nnd_all distributions for density harmonization figure.")

    plot_rows = []
    for cohort_label, stat_df in [
        ("H&E CellViT", external_stats),
        ("mIF inForm", orig_stats),
        ("mIF inForm matched", matched_stats),
    ]:
        tmp = stat_df.copy()
        tmp["cohort"] = cohort_label
        plot_rows.append(tmp)
    plot_df = pd.concat(plot_rows, ignore_index=True)

    g_orig = df_orig.loc[df_orig["sample_name"] == sample_name].copy()
    if g_orig.empty:
        raise ValueError(f"No original inForm cells found for sample {sample_name}")
    g_matched = df_matched_rep1.loc[df_matched_rep1["orig_sample_name"] == sample_name, ["sample_name", "x", "y", "CK"]].copy()
    if g_matched.empty:
        raise ValueError(f"No matched inForm cells found for sample {sample_name}")

    src_sorted = np.sort(orig_stats["median_nnd_all"].dropna().to_numpy(float))
    ref_metric = external_stats["median_nnd_all"].dropna().to_numpy(float)
    sample_nnd_row = orig_stats.loc[orig_stats["sample_name"] == sample_name]
    if sample_nnd_row.empty:
        raise ValueError(f"No median_nnd_all row found for sample {sample_name}")
    sample_nnd = float(sample_nnd_row["median_nnd_all"].iloc[0])
    rank = np.searchsorted(src_sorted, sample_nnd, side="left")
    quantile = (rank + 0.5) / len(src_sorted)
    target_nnd = float(np.quantile(ref_metric, quantile))
    external_stats = external_stats.copy()
    external_stats["abs_delta"] = (pd.to_numeric(external_stats["median_nnd_all"], errors="coerce") - target_nnd).abs()
    if target_external_sample_name:
        target_rows = external_stats.loc[external_stats["sample_name"].astype(str) == str(target_external_sample_name)]
        if target_rows.empty:
            raise ValueError(f"No median_nnd_all row found for target H&E sample {target_external_sample_name}")
        matched_external_row = target_rows.iloc[0]
    else:
        matched_external_row = external_stats.sort_values(["abs_delta", "median_nnd_all"], ascending=[True, True]).iloc[0]
    matched_external_sample_name = str(matched_external_row["sample_name"])
    g_external = df_external.loc[df_external["sample_name"] == matched_external_sample_name].copy()
    if g_external.empty:
        raise ValueError(f"No external H&E cells found for sample {matched_external_sample_name}")

    matched_sample_name = str(g_matched["sample_name"].iloc[0])
    xlim, ylim = _point_cloud_bounds(g_orig)
    orig_nnd = compute_sample_median_nnd(g_orig, min_cells=min_cells, compartment="all")
    matched_nnd = compute_sample_median_nnd(g_matched, min_cells=min_cells, compartment="all")
    external_nnd = compute_sample_median_nnd(g_external, min_cells=min_cells, compartment="all")
    orig_nnd_val = float(orig_nnd["median_nnd_all"].iloc[0]) if not orig_nnd.empty else float("nan")
    matched_nnd_val = float(matched_nnd["median_nnd_all"].iloc[0]) if not matched_nnd.empty else float("nan")
    external_nnd_val = float(external_nnd["median_nnd_all"].iloc[0]) if not external_nnd.empty else float("nan")

    fig = plt.figure(figsize=(17.4, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.0, 1.0, 1.0], wspace=0.24)
    ax_dist = fig.add_subplot(gs[0, 0])
    ax_before = fig.add_subplot(gs[0, 1])
    ax_target = fig.add_subplot(gs[0, 2])
    ax_after = fig.add_subplot(gs[0, 3])

    cohorts = ["H&E CellViT", "mIF inForm", "mIF inForm matched"]
    colors = {
        "H&E CellViT": COLOR_HE,
        "mIF inForm": COLOR_MIF,
        "mIF inForm matched": COLOR_MATCHED,
    }
    positions = np.arange(1, len(cohorts) + 1)
    data = [plot_df.loc[plot_df["cohort"] == cohort, "median_nnd_all"].to_numpy(float) for cohort in cohorts]
    box = ax_dist.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.4},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, cohort in zip(box["boxes"], cohorts):
        patch.set_facecolor(colors[cohort])
        patch.set_alpha(0.8)
        patch.set_edgecolor(colors[cohort])
    rng = np.random.default_rng(0)
    for pos, cohort in zip(positions, cohorts):
        vals = plot_df.loc[plot_df["cohort"] == cohort, "median_nnd_all"].to_numpy(float)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax_dist.scatter(np.full(len(vals), pos) + jitter, vals, s=10, color=colors[cohort], alpha=0.35, linewidths=0)
    ax_dist.set_yscale("log")
    ax_dist.set_xticks(positions)
    ax_dist.set_xticklabels(["H&E", "mIF", "Matched"], rotation=0, ha="center", fontsize=14, fontweight="bold")
    ax_dist.set_ylabel("Median nearest-neighbor distance", fontsize=14, fontweight="bold")
    ax_dist.set_title("Per-core density shift", fontsize=15, fontweight="bold", pad=10)
    ax_dist.grid(axis="y", alpha=0.2)
    ax_dist.set_yticklabels([])
    ax_dist.tick_params(axis="y", length=0)

    plot_point_cloud_panel(
        ax_before,
        g_orig,
        title="Original mIF core",
        subtitle=f"Median NND {orig_nnd_val:.1f}",
        xlim=xlim,
        ylim=ylim,
        dot_size=dot_size,
    )
    plot_point_cloud_panel(
        ax_target,
        g_external,
        title="Percentile-matched H&E core\n(target)",
        subtitle=f"Median NND {external_nnd_val:.1f}",
        dot_size=dot_size,
    )
    plot_point_cloud_panel(
        ax_after,
        g_matched,
        title="Matched mIF core",
        subtitle=f"Median NND {matched_nnd_val:.1f}",
        xlim=xlim,
        ylim=ylim,
        dot_size=dot_size,
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STROMA, markeredgecolor="none", markersize=7, label="Stroma"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TUMOR, markeredgecolor="none", markersize=7, label="Tumor"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.77, -0.02),
        ncol=2,
        frameon=False,
        prop={"size": 14, "weight": "bold"},
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "sample_name": sample_name,
        "matched_sample_name": matched_sample_name,
        "matched_external_sample_name": matched_external_sample_name,
        "matched_external_selection": "manual" if target_external_sample_name else "nearest_percentile",
        "matched_external_quantile": quantile,
        "original_cells": int(len(g_orig)),
        "matched_cells": int(len(g_matched)),
        "matched_external_cells": int(len(g_external)),
        "original_median_nnd_all": orig_nnd_val,
        "matched_median_nnd_all": matched_nnd_val,
        "matched_external_median_nnd_all": external_nnd_val,
        "n_external_cores": int(len(external_stats)),
        "n_internal_cores": int(len(orig_stats)),
        "n_matched_cores": int(len(matched_stats)),
    }


def save_feature_alignment_figure(
    *,
    shift_summary_path: str,
    diagnostics_path: str,
    out_path: Path,
    internal_metrics_path: str,
    external_metrics_path: str,
    patient_table: str = "aligned_BOMI2_to_BOMI1_meanstd",
    n_features: int = 4,
) -> dict[str, object]:
    shift_df = pd.read_csv(shift_summary_path)
    diag_df = pd.read_csv(diagnostics_path)
    excluded_features = {"stroma_stroma_neighborhood_enrichment_zscore"}

    subset = shift_df.loc[shift_df["Patient table"] == patient_table].copy()
    if subset.empty:
        raise ValueError(f"No rows found for patient table {patient_table} in {shift_summary_path}")
    subset = subset.loc[~subset["feature"].isin(excluded_features)].copy()
    subset = subset.sort_values("abs_smd_before", ascending=False).copy()
    selected_rows = []
    seen_families: set[str] = set()
    for row in subset.itertuples(index=False):
        family = feature_family_key(row.feature)
        if family in seen_families:
            continue
        selected_rows.append(row._asdict())
        seen_families.add(family)
        if len(selected_rows) >= n_features:
            break
    if len(selected_rows) < n_features:
        used = {str(row["feature"]) for row in selected_rows}
        for row in subset.itertuples(index=False):
            if str(row.feature) in used:
                continue
            selected_rows.append(row._asdict())
            if len(selected_rows) >= n_features:
                break
    subset = pd.DataFrame(selected_rows)
    subset["feature_label"] = subset["feature"].map(format_feature_label)

    diag_row = diag_df.loc[diag_df["Patient table"] == patient_table]
    if diag_row.empty:
        raise ValueError(f"No diagnostics row found for patient table {patient_table} in {diagnostics_path}")
    diag_row = diag_row.iloc[0]

    internal_defaults = InternalPaths()
    external_defaults = ExternalPaths()
    ext_train_csv = external_defaults.external_train_csv or str(Path(external_defaults.external_split_dir) / "train_val.csv")
    qc = CountQC()

    int_metrics = pd.read_csv(internal_metrics_path, low_memory=False)
    ext_metrics = pd.read_csv(external_metrics_path, low_memory=False)
    int_metrics = filter_metrics_by_counts(int_metrics, qc=qc, cohort_name="INTERNAL/BOMI2")
    ext_metrics = filter_metrics_by_counts(ext_metrics, qc=qc, cohort_name="EXTERNAL/BOMI1")
    int_patient = aggregate_cores_to_patient(int_metrics)
    ext_patient = aggregate_cores_to_patient(ext_metrics)
    train_ids_int = read_split_ids(internal_defaults.internal_train_csv)
    train_ids_ext = read_split_ids(ext_train_csv)
    df_train_int = subset_by_ids(int_patient, train_ids_int)
    df_train_ext = subset_by_ids(ext_patient, train_ids_ext)

    feat_cols = get_feature_columns(df_train_int)
    alignment = fit_meanstd_alignment(df_train_int, df_train_ext, feat_cols)
    df_train_int_aligned = apply_meanstd_alignment(df_train_int, feat_cols, alignment)

    n_panels = len(subset)
    fig = plt.figure(figsize=(16.8, 3.9))
    left_count = int(math.ceil(n_panels / 2))
    right_count = n_panels - left_count
    gs = fig.add_gridspec(
        nrows=2,
        ncols=n_panels + 1,
        width_ratios=[1.0] * left_count + [0.22] + [1.0] * right_count,
        hspace=0.34,
        wspace=0.10,
    )
    colors = {
        "H&E target": COLOR_HE,
        "mIF original": COLOR_MIF,
        "mIF aligned": COLOR_MATCHED,
    }
    left_axes = []
    right_axes = []
    spacer_col = left_count
    arrow_ax = fig.add_subplot(gs[:, spacer_col])
    arrow_ax.axis("off")
    arrow_ax.annotate(
        "",
        xy=(0.5, 0.43),
        xytext=(0.5, 0.57),
        arrowprops={"arrowstyle": "->", "lw": 3.2, "color": "#000000"},
        xycoords="axes fraction",
    )
    arrow_ax.text(0.5, 0.78, "...", ha="center", va="center", fontsize=24, fontweight="bold", color="#000000")
    arrow_ax.text(0.5, 0.22, "...", ha="center", va="center", fontsize=24, fontweight="bold", color="#000000")

    for col_idx, row in enumerate(subset.itertuples(index=False)):
        grid_col = col_idx if col_idx < left_count else col_idx + 1
        ax_left = fig.add_subplot(gs[0, grid_col])
        ax_right = fig.add_subplot(gs[1, grid_col])
        left_axes.append(ax_left)
        right_axes.append(ax_right)
        feature = str(row.feature)
        label = str(row.feature_label)
        title_label = label.replace("Cancer-stroma co-occurrence AUC", "Cancer-stroma\nco-occurrence AUC")
        title_label = title_label.replace(
            "Cancer-stroma co-occurrence deviation AUC",
            "Cancer-stroma\nco-occurrence\ndeviation AUC",
        )
        title_label = title_label.replace("Cancer-stroma object-object correlation", "Cancer-stroma\nobject-object\ncorrelation")
        vals_target = pd.to_numeric(df_train_ext[feature], errors="coerce").dropna().to_numpy(float)
        vals_orig = pd.to_numeric(df_train_int[feature], errors="coerce").dropna().to_numpy(float)
        vals_aligned = pd.to_numeric(df_train_int_aligned[feature], errors="coerce").dropna().to_numpy(float)
        combined = np.concatenate([vals_target, vals_orig, vals_aligned])
        lo = float(np.nanpercentile(combined, 1.0))
        hi = float(np.nanpercentile(combined, 99.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(combined))
            hi = float(np.nanmax(combined))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 22)
        ax_left.hist(vals_target, bins=bins, density=True, color=colors["H&E target"], alpha=0.22)
        ax_left.hist(vals_orig, bins=bins, density=True, histtype="step", linewidth=1.8, color=colors["mIF original"])
        ax_right.hist(vals_target, bins=bins, density=True, color=colors["H&E target"], alpha=0.22)
        ax_right.hist(vals_aligned, bins=bins, density=True, histtype="step", linewidth=1.8, color=colors["mIF aligned"])

        ax_left.set_title(title_label, fontsize=16, fontweight="bold", pad=6)
        for ax in [ax_left, ax_right]:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="both", length=0)
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)

    fig.text(0.03, 0.76, "Before", ha="left", va="center", fontsize=16, fontweight="bold")
    fig.text(0.03, 0.30, "After mean/std\nalignment", ha="left", va="center", fontsize=16, fontweight="bold")

    handles = [
        Line2D([0], [0], color=colors["H&E target"], linewidth=6.0, alpha=0.22, label="H&E target"),
        Line2D([0], [0], color=colors["mIF original"], linewidth=2.0, label="mIF original"),
        Line2D([0], [0], color=colors["mIF aligned"], linewidth=2.0, label="mIF aligned"),
    ]
    legend = fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=16)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.tight_layout(rect=[0.05, 0.08, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "patient_table": patient_table,
        "top_features": subset["feature"].tolist(),
        "domain_auc_train_before": float(diag_row["domain_auc_train_before"]),
        "domain_auc_train_after": float(diag_row["domain_auc_train_after"]),
    }


def save_fake_tma(
    slide_df: pd.DataFrame,
    out_path: Path,
    *,
    canvas_size: int,
    margin: int,
    per_core_max_dim: int,
) -> None:
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    tiles = []
    for row in tqdm(slide_df.itertuples(index=False), total=len(slide_df), desc="Loading TMA cores", unit="core"):
        image_rgb, original_width, original_height = load_bomi2_ck_dapi_image(row.image_path, max_dim=per_core_max_dim)
        tiles.append(
            {
                "sample_name": row.sample_name,
                "core_col": int(row.core_col),
                "core_row": str(row.core_row),
                "image_rgb": image_rgb,
                "width": int(image_rgb.shape[1]),
                "height": int(image_rgb.shape[0]),
                "original_width": original_width,
                "original_height": original_height,
            }
        )

    row_labels = sorted({item["core_row"] for item in tiles})
    col_numbers = sorted({item["core_col"] for item in tiles})
    row_index = {label: idx for idx, label in enumerate(row_labels)}
    col_index = {value: idx for idx, value in enumerate(col_numbers)}

    max_tile_w = max(item["width"] for item in tiles)
    max_tile_h = max(item["height"] for item in tiles)
    n_cols = len(col_numbers)
    n_rows = len(row_labels)
    inner_w = max(1, canvas_size - 2 * margin)
    inner_h = max(1, canvas_size - 2 * margin)
    pitch_x = inner_w / max(1, n_cols)
    pitch_y = inner_h / max(1, n_rows)
    tile_scale = min(
        1.8 * pitch_x / max(1, max_tile_w),
        1.8 * pitch_y / max(1, max_tile_h),
        1.0,
    )

    for item in tiles:
        scaled = item["image_rgb"]
        if tile_scale < 1.0:
            step = max(1, int(math.ceil(1.0 / tile_scale)))
            scaled = scaled[::step, ::step, :]
        scaled_h, scaled_w = scaled.shape[:2]
        center_x = margin + (col_index[item["core_col"]] + 0.5) * pitch_x
        center_y = margin + (row_index[item["core_row"]] + 0.5) * pitch_y
        x0 = int(round(center_x - 0.5 * scaled_w))
        y0 = int(round(center_y - 0.5 * scaled_h))
        x1 = min(canvas_size, x0 + scaled_w)
        y1 = min(canvas_size, y0 + scaled_h)
        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        canvas[dst_y0:y1, dst_x0:x1] = scaled[src_y0 : src_y0 + (y1 - dst_y0), src_x0 : src_x0 + (x1 - dst_x0)]

    fig, ax = plt.subplots(figsize=(canvas_size / 200.0, canvas_size / 200.0), dpi=200, facecolor="black")
    ax.imshow(canvas)
    ax.set_xlim(0, canvas_size)
    ax.set_ylim(canvas_size, 0)
    ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)


def ensure_requested_work(args: argparse.Namespace) -> None:
    if args.make_all:
        args.make_fake_tma = True
        args.make_follow_core_image = True
        args.make_bomi2_scatters = True
        args.make_bomi1_scatter = True
        args.make_bomi1_core_image = True
        args.make_density_harmonization = True
        args.make_feature_alignment = True
    if not any([
        args.make_fake_tma,
        args.make_follow_core_image,
        args.make_bomi2_scatters,
        args.make_bomi1_scatter,
        args.make_bomi1_core_image,
        args.make_density_harmonization,
        args.make_feature_alignment,
    ]):
        raise SystemExit("No figure tasks selected. Use --make-all or one of the --make-* flags.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate optional paper-ready spatial figures in a dedicated output folder.")
    ap.add_argument("--comparison-index", default="outputs/figures/bomi2_segmentation_method_comparison/comparison_plot_index.csv")
    ap.add_argument("--bomi2-inform", default="data/raw/BOMI2_all_cells_TIL.csv")
    ap.add_argument("--bomi2-inform-matched", default="data/interim/BOMI2_all_cells_TIL__matchedBOMI1.csv")
    ap.add_argument("--bomi2-cellpose", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    ap.add_argument("--bomi2-cellprofiler", default="data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv")
    ap.add_argument("--bomi1-cells", default=DEFAULT_BOMI1_CELLS)
    ap.add_argument("--bomi1-harmonization-cells", default=DEFAULT_BOMI1_HARMONIZATION_CELLS)
    ap.add_argument("--bomi1-meta", default=DEFAULT_BOMI1_META)
    ap.add_argument("--bomi1-all-cores-dir", default=DEFAULT_BOMI1_ALL_CORES_DIR)
    ap.add_argument("--external-xy-scale", type=float, default=2.0)
    ap.add_argument("--bomi2-slide", default=DEFAULT_BOMI2_SLIDE)
    ap.add_argument("--follow-sample", help="Optional BOMI2 core to follow, e.g. BOMI2_TIL_1_[1,A]")
    ap.add_argument("--bomi1-sample", help="Optional BOMI1 core to plot, e.g. ULunA4A_A-1")
    ap.add_argument("--density-harmonization-target-sample", help="Optional H&E core to use as the density harmonization target example.")
    ap.add_argument("--out-dir", default="outputs/figures/paper_figures")
    ap.add_argument("--tma-canvas-size", type=int, default=1000)
    ap.add_argument("--tma-margin", type=int, default=40)
    ap.add_argument("--tma-core-max-dim", type=int, default=140)
    ap.add_argument("--scatter-dot-size", type=float, default=6.0)
    ap.add_argument("--full-core-max-dim", type=int, default=2200)
    ap.add_argument("--bomi1-core-max-dim", type=int, default=1200)
    ap.add_argument("--transparent-scatters", action="store_true")
    ap.add_argument("--harmonization-min-cells", type=int, default=30)
    ap.add_argument("--feature-alignment-shift-summary", default="outputs/metrics/feature_alignment_shift_summary.csv")
    ap.add_argument("--feature-alignment-diagnostics", default="outputs/metrics/feature_alignment_diagnostics.csv")
    ap.add_argument("--feature-alignment-internal-metrics", default="outputs/metrics/spatial_metrics_inform.csv")
    ap.add_argument("--feature-alignment-external-metrics", default="outputs/metrics/spatial_metrics_EXTERNAL_BOMI1.csv")
    ap.add_argument("--feature-alignment-patient-table", default="aligned_BOMI2_to_BOMI1_meanstd")
    ap.add_argument("--feature-alignment-top-n", type=int, default=4)
    ap.add_argument("--make-fake-tma", action="store_true")
    ap.add_argument("--make-follow-core-image", action="store_true")
    ap.add_argument("--make-bomi2-scatters", action="store_true")
    ap.add_argument("--make-bomi1-scatter", action="store_true")
    ap.add_argument("--make-bomi1-core-image", action="store_true")
    ap.add_argument("--make-density-harmonization", action="store_true")
    ap.add_argument("--make-feature-alignment", action="store_true")
    ap.add_argument("--make-all", action="store_true")
    args = ap.parse_args()

    ensure_requested_work(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = load_comparison_index(args.comparison_index)
    slide_df = comparison_df.loc[comparison_df["slide_id"] == args.bomi2_slide].copy()
    if slide_df.empty:
        raise ValueError(f"No entries found for BOMI2 slide {args.bomi2_slide}")
    slide_df = slide_df.sort_values(["core_row", "core_col", "sample_name"]).reset_index(drop=True)
    follow_sample = choose_follow_sample(slide_df, args.follow_sample)
    follow_row = slide_df.loc[slide_df["sample_name"] == follow_sample].iloc[0]

    manifest: dict[str, object] = {
        "bomi2_slide": args.bomi2_slide,
        "follow_sample": follow_sample,
        "output_dir": str(out_dir),
    }

    if args.make_fake_tma:
        fake_tma_dir = out_dir / "01_fake_tma"
        fake_tma_dir.mkdir(parents=True, exist_ok=True)
        fake_tma_path = fake_tma_dir / f"fake_tma_{args.bomi2_slide}.png"
        save_fake_tma(
            slide_df,
            fake_tma_path,
            canvas_size=args.tma_canvas_size,
            margin=args.tma_margin,
            per_core_max_dim=args.tma_core_max_dim,
        )
        manifest["fake_tma"] = str(fake_tma_path)

    if args.make_follow_core_image:
        follow_dir = out_dir / "02_follow_core_image"
        follow_dir.mkdir(parents=True, exist_ok=True)
        image_rgb, _, _ = load_bomi2_ck_dapi_image(follow_row.image_path, max_dim=args.full_core_max_dim)
        follow_path = follow_dir / f"{follow_sample.replace('[', '_').replace(']', '').replace(',', '_')}_ck_dapi.png"
        save_follow_core_image(
            image_rgb,
            follow_path,
            transparent=True,
        )
        manifest["follow_core_image"] = str(follow_path)

    if args.make_bomi2_scatters:
        scatter_dir = out_dir / "03_bomi2_scatters"
        scatter_dir.mkdir(parents=True, exist_ok=True)
        method_specs = [
            ("inform", args.bomi2_inform),
            ("cellpose", args.bomi2_cellpose),
            ("cellprofiler", args.bomi2_cellprofiler),
        ]
        scatter_outputs = {}
        for method_name, path in method_specs:
            print(f"[INFO] Loading BOMI2 {method_name} cells")
            cells = load_bomi2_method_cells(path, method_name)
            sample_cells = cells.loc[cells["sample_name"] == follow_sample].copy()
            if sample_cells.empty:
                raise ValueError(f"No {method_name} cells found for follow sample {follow_sample}")
            out_path = scatter_dir / f"{follow_sample.replace('[', '_').replace(']', '').replace(',', '_')}_{method_name}_scatter.png"
            save_clean_scatter(
                sample_cells,
                out_path,
                dot_size=args.scatter_dot_size,
                transparent=args.transparent_scatters,
            )
            scatter_outputs[method_name] = str(out_path)
        manifest["bomi2_scatters"] = scatter_outputs

    if args.make_bomi1_scatter:
        bomi1_dir = out_dir / "04_bomi1_scatter"
        bomi1_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Loading BOMI1 cells")
        bomi1_cells = load_bomi1_cells(args.bomi1_cells, args.bomi1_meta)
        bomi1_sample = pick_bomi1_sample(bomi1_cells, args.bomi1_sample)
        bomi1_sample_cells = bomi1_cells.loc[bomi1_cells["sample_name"] == bomi1_sample].copy()
        out_path = bomi1_dir / f"{bomi1_sample}_scatter.png"
        save_clean_scatter(
            bomi1_sample_cells,
            out_path,
            dot_size=args.scatter_dot_size,
            transparent=args.transparent_scatters,
        )
        manifest["bomi1_sample"] = bomi1_sample
        manifest["bomi1_scatter"] = str(out_path)

    if args.make_bomi1_core_image:
        core_dir = out_dir / "05_bomi1_core_image"
        core_dir.mkdir(parents=True, exist_ok=True)
        if "bomi1_sample" in manifest:
            bomi1_sample = str(manifest["bomi1_sample"])
        else:
            print("[INFO] Loading BOMI1 cells")
            bomi1_cells = load_bomi1_cells(args.bomi1_cells, args.bomi1_meta)
            bomi1_sample = pick_bomi1_sample(bomi1_cells, args.bomi1_sample)
            manifest["bomi1_sample"] = bomi1_sample
        core_image_path = find_bomi1_core_image(bomi1_sample, args.bomi1_all_cores_dir)
        core_rgb, _, _ = load_rgb_image(core_image_path, max_dim=args.bomi1_core_max_dim)
        out_path = core_dir / f"{bomi1_sample}_core.png"
        save_follow_core_image(
            core_rgb,
            out_path,
            transparent=False,
        )
        manifest["bomi1_core_image"] = str(out_path)

    if args.make_density_harmonization:
        density_dir = out_dir / "07_density_harmonization"
        density_dir.mkdir(parents=True, exist_ok=True)
        density_path = density_dir / "inform_density_harmonization.png"
        density_summary = save_density_harmonization_figure(
            inform_original_path=args.bomi2_inform,
            inform_matched_path=args.bomi2_inform_matched,
            external_bomi1_path=args.bomi1_harmonization_cells,
            external_xy_scale=args.external_xy_scale,
            sample_name=follow_sample,
            target_external_sample_name=args.density_harmonization_target_sample,
            out_path=density_path,
            min_cells=args.harmonization_min_cells,
            dot_size=max(0.6, args.scatter_dot_size * 0.25),
        )
        manifest["density_harmonization"] = {
            "path": str(density_path),
            **density_summary,
        }

    if args.make_feature_alignment:
        alignment_dir = out_dir / "08_feature_alignment"
        alignment_dir.mkdir(parents=True, exist_ok=True)
        alignment_path = alignment_dir / "meanstd_feature_alignment.png"
        alignment_summary = save_feature_alignment_figure(
            shift_summary_path=args.feature_alignment_shift_summary,
            diagnostics_path=args.feature_alignment_diagnostics,
            internal_metrics_path=args.feature_alignment_internal_metrics,
            external_metrics_path=args.feature_alignment_external_metrics,
            out_path=alignment_path,
            patient_table=args.feature_alignment_patient_table,
            n_features=args.feature_alignment_top_n,
        )
        manifest["feature_alignment"] = {
            "path": str(alignment_path),
            **alignment_summary,
        }

    manifest_path = out_dir / "paper_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote paper figures to {out_dir}")


if __name__ == "__main__":
    main()
