#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from spatial_metrics import BEHANOVA_FEATURE_COLUMNS
from spatial_utils import aggregate_cores_to_patient


META_COLS = {"ID", "sample_name", "label"}
COUNT_COLS = {"cell_count", "cancer_count", "stroma_count"}
DEFAULT_METHOD_TO_PATH = {
    "inform": "outputs/metrics/spatial_metrics_inform.csv",
    "cellpose": "outputs/metrics/spatial_metrics_cellpose.csv",
    "cellprofiler": "outputs/metrics/spatial_metrics_cellprofiler.csv",
}
DEFAULT_METHOD_PAIRS = [
    ("inform", "cellpose"),
    ("inform", "cellprofiler"),
    ("cellpose", "cellprofiler"),
]
METHOD_PAIR_LABELS = {
    ("inform", "cellpose"): "inForm vs Cellpose",
    ("inform", "cellprofiler"): "inForm vs CellProfiler",
    ("cellpose", "cellprofiler"): "Cellpose vs CellProfiler",
}
METHOD_PAIR_COLORS = {
    ("inform", "cellpose"): "#1b9e77",
    ("inform", "cellprofiler"): "#d95f02",
    ("cellpose", "cellprofiler"): "#7570b3",
}
REPRESENTATIVE_METRICS = [
    "cancer_ripley_L_auc",
    "newmans_assortativity",
    "cancer_degree_centrality",
    "cancer_stroma_cooccurrence_auc",
    "cancer_stroma_neighborhood_enrichment_zscore",
    "cancer_stroma_object_object_pearson_correlation",
]
FAMILY_PAIR_FOR_POOLED_SCATTER = ("inform", "cellpose")
FIGURE_TITLE_FONTSIZE = 25
SUBPLOT_TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 15
ANNOTATION_FONTSIZE = 15
LEGEND_FONTSIZE = 15
FAMILY_ORDER = [
    "Ripley",
    "Newman's Assortativity",
    "Graph Centrality / Clustering",
    "Cell-Type Co-Occurrence Ratio",
    "Neighborhood Enrichment",
    "Object-Object Correlation",
]
SUPPLEMENTARY_SCATTER_GROUPS = [
    ("single_object_global", "Single-object global statistics", ["Ripley", "Newman's Assortativity"]),
    ("graph_centrality_clustering", "Graph centrality / clustering", ["Graph Centrality / Clustering"]),
    (
        "two_object_interaction",
        "Two-object interaction statistics",
        ["Cell-Type Co-Occurrence Ratio", "Neighborhood Enrichment", "Object-Object Correlation"],
    ),
]


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def canonicalize_bomi2_sample_name(text: str) -> str:
    value = str(text)
    match = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", value)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    match = re.search(r"(.+?)_\[(\d+,[A-Z])\]", value)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    raise ValueError(f"Cannot parse BOMI2 sample name from: {text}")


def metric_family(metric: str) -> str:
    if "ripley" in metric:
        return "Ripley"
    if metric == "newmans_assortativity":
        return "Newman's Assortativity"
    if "centrality" in metric or metric.endswith("_average_clustering"):
        return "Graph Centrality / Clustering"
    if "cooccurrence" in metric:
        return "Cell-Type Co-Occurrence Ratio"
    if metric.endswith("_neighborhood_enrichment_zscore"):
        return "Neighborhood Enrichment"
    if metric.endswith("_object_object_pearson_correlation"):
        return "Object-Object Correlation"
    return "Other"


def pretty_metric_name(metric: str) -> str:
    replacements = {
        "newmans_assortativity": "Newman's assortativity",
        "cancer_degree_centrality": "Cancer group degree centrality",
        "stroma_degree_centrality": "Stroma group degree centrality",
        "cancer_closeness_centrality": "Cancer group closeness centrality",
        "stroma_closeness_centrality": "Stroma group closeness centrality",
        "cancer_average_clustering": "Cancer average clustering",
        "stroma_average_clustering": "Stroma average clustering",
        "cancer_cancer_neighborhood_enrichment_zscore": "Cancer-cancer neighborhood enrichment z-score",
        "cancer_stroma_neighborhood_enrichment_zscore": "Cancer-stroma neighborhood enrichment z-score",
        "stroma_stroma_neighborhood_enrichment_zscore": "Stroma-stroma neighborhood enrichment z-score",
        "cancer_stroma_object_object_pearson_correlation": "Cancer-stroma object-object Pearson correlation",
    }
    if metric in replacements:
        return replacements[metric]
    if metric.startswith("cancer_ripley_L_"):
        suffix = metric.split("_")[-1]
        return "Cancer Ripley's L AUC" if suffix == "auc" else f"Cancer Ripley's L ({suffix})"
    if metric.startswith("stroma_ripley_L_"):
        suffix = metric.split("_")[-1]
        return "Stroma Ripley's L AUC" if suffix == "auc" else f"Stroma Ripley's L ({suffix})"
    if metric == "cancer_stroma_cooccurrence_auc":
        return "Cancer-stroma co-occurrence ratio AUC"
    if metric == "cancer_stroma_cooccurrence_deviation_auc":
        return "Cancer-stroma co-occurrence deviation AUC"
    return metric.replace("_", " ")


def wrap_label(value: str, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))


def load_metrics_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"Loaded empty metrics table: {path}")

    required = {"ID", "label", "cell_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics table {path} is missing required columns: {sorted(missing)}")
    return df


def load_allowed_bomi2_samples(path: str) -> set[str]:
    df = pd.read_csv(path, usecols=["sample_name"], low_memory=False)
    return set(df["sample_name"].astype(str).map(canonicalize_bomi2_sample_name))


def filter_to_allowed_samples(df: pd.DataFrame, allowed_samples: set[str], method_name: str) -> pd.DataFrame:
    before = len(df)
    out = df[df["sample_name"].astype(str).isin(allowed_samples)].copy()
    after = len(out)
    print(f"[INFO] {method_name}: kept {after}/{before} rows after study-core filter")
    return out


def aggregate_metrics(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if level == "core":
        return df.copy()
    if level == "patient":
        return aggregate_cores_to_patient(df)
    raise ValueError(f"Unsupported aggregation level: {level}")


def get_metric_columns(frames: Sequence[pd.DataFrame], include_counts: bool = False) -> list[str]:
    shared: set[str] | None = None
    excluded = set(META_COLS)
    if not include_counts:
        excluded |= COUNT_COLS

    for df in frames:
        numeric_cols = {
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col not in excluded
        }
        shared = numeric_cols if shared is None else shared & numeric_cols

    return sorted(shared or [])


def get_behanova_metric_columns(frames: Sequence[pd.DataFrame], include_counts: bool = False) -> list[str]:
    """Return the active manuscript metric panel shared by all input frames."""

    metric_cols = [
        metric
        for metric in BEHANOVA_FEATURE_COLUMNS
        if all(metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]) for df in frames)
    ]
    if include_counts:
        count_cols = [
            metric
            for metric in ("cell_count", "cancer_count", "stroma_count")
            if all(metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]) for df in frames)
        ]
        return count_cols + metric_cols
    return metric_cols


def build_pair_table(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    metric: str,
    left_name: str,
    right_name: str,
    id_col: str,
) -> pd.DataFrame:
    left = left_df[[id_col, metric]].rename(columns={metric: f"{metric}__{left_name}"})
    right = right_df[[id_col, metric]].rename(columns={metric: f"{metric}__{right_name}"})
    merged = left.merge(right, on=id_col, how="inner")
    x_col = f"{metric}__{left_name}"
    y_col = f"{metric}__{right_name}"
    merged[x_col] = pd.to_numeric(merged[x_col], errors="coerce")
    merged[y_col] = pd.to_numeric(merged[y_col], errors="coerce")
    merged = merged.dropna(subset=[x_col, y_col]).reset_index(drop=True)
    return merged


def compute_pairwise_correlations(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_arr = pd.to_numeric(x, errors="coerce").to_numpy(float)
    y_arr = pd.to_numeric(y, errors="coerce").to_numpy(float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size < 2:
        return float("nan"), float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x_arr, y_arr)[0, 1])
    spearman = float(pd.Series(x_arr).corr(pd.Series(y_arr), method="spearman"))
    return pearson, spearman


def identify_pair_outliers(
    pair_df: pd.DataFrame,
    *,
    metric: str,
    left_name: str,
    right_name: str,
    id_col: str,
    top_n: int,
) -> pd.DataFrame:
    if pair_df.empty or top_n <= 0:
        return pd.DataFrame(
            columns=[
                "metric",
                "left_method",
                "right_method",
                id_col,
                "left_value",
                "right_value",
                "abs_diff",
                "outlier_score",
                "outlier_rank",
            ]
        )

    x_col = f"{metric}__{left_name}"
    y_col = f"{metric}__{right_name}"
    out = pair_df[[id_col, x_col, y_col]].copy()
    out["left_value"] = pd.to_numeric(out[x_col], errors="coerce")
    out["right_value"] = pd.to_numeric(out[y_col], errors="coerce")
    out = out.dropna(subset=["left_value", "right_value"]).copy()
    if out.empty:
        return pd.DataFrame()

    out["abs_diff"] = (out["right_value"] - out["left_value"]).abs()
    scale = float(np.nanmedian(out["abs_diff"]))
    if not np.isfinite(scale) or math.isclose(scale, 0.0):
        scale = float(np.nanstd(out["abs_diff"]))
    if not np.isfinite(scale) or math.isclose(scale, 0.0):
        scale = 1.0
    out["outlier_score"] = out["abs_diff"] / scale
    out = out.sort_values(["outlier_score", "abs_diff"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    out["outlier_rank"] = np.arange(1, len(out) + 1)
    out["metric"] = metric
    out["left_method"] = left_name
    out["right_method"] = right_name
    out = out.rename(columns={id_col: "entity_id"})
    return out[
        [
            "metric",
            "left_method",
            "right_method",
            "entity_id",
            "left_value",
            "right_value",
            "abs_diff",
            "outlier_score",
            "outlier_rank",
        ]
    ]


def _safe_axis_limits(values: Iterable[float], pad_frac: float = 0.04) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if math.isclose(lo, hi):
        delta = 1.0 if math.isclose(lo, 0.0) else abs(lo) * 0.05
        return (lo - delta, hi + delta)
    pad = (hi - lo) * pad_frac
    return (lo - pad, hi + pad)


def plot_metric_figure(
    metric: str,
    method_pairs: Sequence[tuple[str, str]],
    aggregated: dict[str, pd.DataFrame],
    out_path: Path,
    id_col: str,
) -> list[dict[str, object]]:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), constrained_layout=True)
    rows: list[dict[str, object]] = []

    for ax, (left_name, right_name) in zip(axes, method_pairs):
        pair_df = build_pair_table(
            left_df=aggregated[left_name],
            right_df=aggregated[right_name],
            metric=metric,
            left_name=left_name,
            right_name=right_name,
            id_col=id_col,
        )
        x_col = f"{metric}__{left_name}"
        y_col = f"{metric}__{right_name}"
        pearson_r, spearman_rho = compute_pairwise_correlations(pair_df[x_col], pair_df[y_col])

        x = pair_df[x_col].to_numpy(float) if not pair_df.empty else np.array([])
        y = pair_df[y_col].to_numpy(float) if not pair_df.empty else np.array([])
        lims = _safe_axis_limits(np.concatenate([x, y]) if x.size or y.size else np.array([0.0, 1.0]))

        ax.scatter(x, y, s=34, alpha=0.8, color="#1f77b4", edgecolor="white", linewidth=0.4)
        ax.plot(lims, lims, linestyle="--", linewidth=1.0, color="0.55")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"{left_name} vs {right_name}", fontsize=SUBPLOT_TITLE_FONTSIZE)
        ax.set_xlabel(left_name, fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(right_name, fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.text(
            0.03,
            0.97,
            f"Pearson r = {pearson_r:.3f}\nSpearman rho = {spearman_rho:.3f}\nn = {len(pair_df)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONTSIZE,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )

        rows.append(
            {
                "metric": metric,
                "left_method": left_name,
                "right_method": right_name,
                "n": int(len(pair_df)),
                "pearson_r": pearson_r,
                "spearman_rho": spearman_rho,
            }
        )

    fig.suptitle(pretty_metric_name(metric), fontsize=FIGURE_TITLE_FONTSIZE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return rows


def load_summary(summary_path: str) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    df = df[df["pearson_r"].notna()].copy()
    df["family"] = df["metric"].map(metric_family)
    df["pair"] = list(zip(df["left_method"], df["right_method"]))
    return df


def select_family_representative_metrics(summary_df: pd.DataFrame) -> list[str]:
    """Pick one non-count metric per family closest to that family's median agreement."""

    metric_stats = (
        summary_df[summary_df["family"].isin(FAMILY_ORDER)]
        .groupby(["family", "metric"], as_index=False)
        .agg(mean_spearman=("spearman_rho", "mean"))
    )
    if metric_stats.empty:
        return []

    selected: list[str] = []
    for family in FAMILY_ORDER:
        fam = metric_stats[metric_stats["family"] == family].copy()
        if fam.empty:
            continue
        median_spearman = float(fam["mean_spearman"].median())
        fam["distance_to_family_median"] = (fam["mean_spearman"] - median_spearman).abs()
        fam = fam.sort_values(["distance_to_family_median", "metric"])
        selected.append(str(fam.iloc[0]["metric"]))
    return selected


def make_family_summary_figure(summary_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
    families = [family for family in FAMILY_ORDER if family in set(summary_df["family"])]
    x_positions = np.arange(len(families), dtype=float)
    pair_offsets = {
        ("inform", "cellpose"): -0.22,
        ("inform", "cellprofiler"): 0.0,
        ("cellpose", "cellprofiler"): 0.22,
    }

    family_stats = (
        summary_df.groupby(["family", "left_method", "right_method"])
        .agg(
            mean_spearman=("spearman_rho", "mean"),
            median_spearman=("spearman_rho", "median"),
            min_spearman=("spearman_rho", "min"),
            max_spearman=("spearman_rho", "max"),
            n_metrics=("metric", "nunique"),
        )
        .reset_index()
    )

    for pair in DEFAULT_METHOD_PAIRS:
        pair_df = summary_df[
            (summary_df["left_method"] == pair[0]) & (summary_df["right_method"] == pair[1])
        ].copy()
        xs: list[float] = []
        ys: list[float] = []
        for idx, family in enumerate(families):
            fam_vals = pair_df.loc[pair_df["family"] == family, "spearman_rho"].to_numpy(float)
            if fam_vals.size == 0:
                continue
            jitter = np.linspace(-0.05, 0.05, fam_vals.size) if fam_vals.size > 1 else np.array([0.0])
            xs.extend((x_positions[idx] + pair_offsets[pair] + jitter).tolist())
            ys.extend(fam_vals.tolist())
            ax.hlines(
                float(np.median(fam_vals)),
                x_positions[idx] + pair_offsets[pair] - 0.09,
                x_positions[idx] + pair_offsets[pair] + 0.09,
                color=METHOD_PAIR_COLORS[pair],
                linewidth=2.8,
            )
        ax.scatter(
            xs,
            ys,
            s=54,
            alpha=0.9,
            color=METHOD_PAIR_COLORS[pair],
            edgecolor="white",
            linewidth=0.5,
            label=METHOD_PAIR_LABELS[pair],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(families, rotation=20, ha="right", fontsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Spearman correlation", fontsize=AXIS_LABEL_FONTSIZE)
    y_values = summary_df["spearman_rho"].dropna().to_numpy(float)
    if y_values.size:
        y_min = max(-1.0, float(np.nanmin(y_values)) - 0.08)
        y_max = min(1.02, float(np.nanmax(y_values)) + 0.08)
        if y_max <= y_min:
            y_min, y_max = 0.0, 1.02
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_title("Segmentation-method agreement by metric family", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=LEGEND_FONTSIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return family_stats.sort_values(["family", "left_method", "right_method"]).reset_index(drop=True)


def make_raw_metric_scatter_figure(
    aggregated: dict[str, pd.DataFrame],
    metrics: list[str],
    out_path: Path,
    aggregate_level: str,
    title: str,
) -> pd.DataFrame:
    id_col = "ID" if aggregate_level == "patient" else "sample_name"
    n_rows = len(metrics)
    n_cols = len(DEFAULT_METHOD_PAIRS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(19.0, 3.8 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    rows: list[dict[str, object]] = []

    for row_idx, metric in enumerate(metrics):
        for col_idx, pair in enumerate(DEFAULT_METHOD_PAIRS):
            ax = axes[row_idx, col_idx]
            left_name, right_name = pair
            pair_df = build_pair_table(
                left_df=aggregated[left_name],
                right_df=aggregated[right_name],
                metric=metric,
                left_name=left_name,
                right_name=right_name,
                id_col=id_col,
            )
            x_col = f"{metric}__{left_name}"
            y_col = f"{metric}__{right_name}"
            pearson_r, spearman_rho = compute_pairwise_correlations(pair_df[x_col], pair_df[y_col])
            x = pair_df[x_col].to_numpy(float) if not pair_df.empty else np.array([])
            y = pair_df[y_col].to_numpy(float) if not pair_df.empty else np.array([])
            lims = _safe_axis_limits(np.concatenate([x, y]) if x.size or y.size else np.array([0.0, 1.0]))

            ax.scatter(
                x,
                y,
                s=28,
                alpha=0.78,
                color=METHOD_PAIR_COLORS[pair],
                edgecolor="white",
                linewidth=0.4,
            )
            ax.plot(lims, lims, linestyle="--", linewidth=1.0, color="0.55")
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            ax.grid(True, alpha=0.22, linewidth=0.6)
            ax.text(
                0.03,
                0.97,
                f"r={pearson_r:.3f}\nrho={spearman_rho:.3f}\nn={len(pair_df)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION_FONTSIZE,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
            )

            if row_idx == 0:
                ax.set_title(METHOD_PAIR_LABELS[pair], fontsize=SUBPLOT_TITLE_FONTSIZE)
            if row_idx == n_rows - 1:
                ax.set_xlabel(left_name, fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{wrap_label(pretty_metric_name(metric), width=22)}\n{right_name}",
                    fontsize=AXIS_LABEL_FONTSIZE,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=58,
                )
            else:
                ax.set_ylabel(right_name, fontsize=AXIS_LABEL_FONTSIZE)

            rows.append(
                {
                    "metric": metric,
                    "left_method": left_name,
                    "right_method": right_name,
                    "pearson_r": pearson_r,
                    "spearman_rho": spearman_rho,
                    "n": int(len(pair_df)),
                }
            )

    fig.suptitle(title, fontsize=FIGURE_TITLE_FONTSIZE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return pd.DataFrame(rows)


def make_supplementary_raw_scatter_figures(
    summary_df: pd.DataFrame,
    aggregated: dict[str, pd.DataFrame],
    paper_dir: Path,
    aggregate_level: str,
) -> tuple[pd.DataFrame, list[str]]:
    all_rows: list[pd.DataFrame] = []
    selected_metrics: list[str] = []

    for slug, title, families in SUPPLEMENTARY_SCATTER_GROUPS:
        metrics = [
            metric
            for family in families
            for metric in sorted(summary_df.loc[summary_df["family"] == family, "metric"].unique().tolist())
        ]
        if not metrics:
            continue
        selected_metrics.extend(metrics)
        group_df = make_raw_metric_scatter_figure(
            aggregated=aggregated,
            metrics=metrics,
            out_path=paper_dir / f"all_metric_scatter_panels_{slug}.png",
            aggregate_level=aggregate_level,
            title=title,
        )
        group_df.insert(0, "supplementary_group", slug)
        all_rows.append(group_df)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return out, selected_metrics


def _percentile_rank(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    keep = values.notna()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    n = int(keep.sum())
    if n == 0:
        return out
    if n == 1:
        out.loc[keep] = 0.5
        return out
    out.loc[keep] = (values.loc[keep].rank(method="average") - 1.0) / float(n - 1)
    return out


def make_family_percentile_scatter_figure(
    summary_df: pd.DataFrame,
    aggregated: dict[str, pd.DataFrame],
    out_path: Path,
    aggregate_level: str,
    method_pair: tuple[str, str] = FAMILY_PAIR_FOR_POOLED_SCATTER,
) -> pd.DataFrame:
    """Plot all metrics in each family after within-metric percentile scaling."""

    id_col = "ID" if aggregate_level == "patient" else "sample_name"
    left_name, right_name = method_pair
    families = [family for family in FAMILY_ORDER if family in set(summary_df["family"])]
    n_cols = 3
    n_rows = int(math.ceil(len(families) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14.4, 4.1 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    rows: list[dict[str, object]] = []
    pooled_rows: list[dict[str, object]] = []

    for ax_idx, family in enumerate(families):
        ax = axes[ax_idx]
        family_metrics = sorted(summary_df.loc[summary_df["family"] == family, "metric"].unique().tolist())
        pooled_x: list[float] = []
        pooled_y: list[float] = []

        for metric in family_metrics:
            pair_df = build_pair_table(
                left_df=aggregated[left_name],
                right_df=aggregated[right_name],
                metric=metric,
                left_name=left_name,
                right_name=right_name,
                id_col=id_col,
            )
            x_col = f"{metric}__{left_name}"
            y_col = f"{metric}__{right_name}"
            pair_df["left_percentile"] = _percentile_rank(pair_df[x_col])
            pair_df["right_percentile"] = _percentile_rank(pair_df[y_col])
            pair_df = pair_df.dropna(subset=["left_percentile", "right_percentile"])
            if pair_df.empty:
                continue

            x = pair_df["left_percentile"].to_numpy(float)
            y = pair_df["right_percentile"].to_numpy(float)
            pooled_x.extend(x.tolist())
            pooled_y.extend(y.tolist())
            metric_pearson, metric_spearman = compute_pairwise_correlations(pair_df[x_col], pair_df[y_col])
            rows.append(
                {
                    "family": family,
                    "metric": metric,
                    "left_method": left_name,
                    "right_method": right_name,
                    "n": int(len(pair_df)),
                    "pearson_r": metric_pearson,
                    "spearman_rho": metric_spearman,
                }
            )
            ax.scatter(
                x,
                y,
                s=15,
                alpha=0.35,
                color=METHOD_PAIR_COLORS[method_pair],
                edgecolor="none",
            )

        pooled_pearson, pooled_spearman = compute_pairwise_correlations(
            pd.Series(pooled_x),
            pd.Series(pooled_y),
        )
        pooled_rows.append(
            {
                "family": family,
                "metric": "__pooled_family_percentiles__",
                "left_method": left_name,
                "right_method": right_name,
                "n": int(len(pooled_x)),
                "pearson_r": pooled_pearson,
                "spearman_rho": pooled_spearman,
            }
        )
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.55")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.set_title(family, fontsize=SUBPLOT_TITLE_FONTSIZE)
        ax.set_xlabel(f"{left_name} percentile", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f"{right_name} percentile", fontsize=AXIS_LABEL_FONTSIZE)
        ax.text(
            0.04,
            0.96,
            f"pooled rho={pooled_spearman:.3f}\nmetrics={len(family_metrics)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONTSIZE,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )

    for ax in axes[len(families):]:
        ax.axis("off")

    fig.suptitle(
        f"Family-level patient rank agreement ({METHOD_PAIR_LABELS[method_pair]})",
        fontsize=FIGURE_TITLE_FONTSIZE,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    out = pd.DataFrame(rows)
    if pooled_rows:
        out = pd.concat([out, pd.DataFrame(pooled_rows)], ignore_index=True)
    return out


def build_text_outputs(summary_df: pd.DataFrame, out_path: Path) -> None:
    pair_overall = (
        summary_df.groupby(["left_method", "right_method"])
        .agg(mean_pearson=("pearson_r", "mean"), mean_spearman=("spearman_rho", "mean"))
        .reset_index()
    )
    worst_pair = pair_overall.sort_values("mean_spearman").iloc[0]

    metric_means = (
        summary_df.groupby("metric")
        .agg(mean_pearson=("pearson_r", "mean"), mean_spearman=("spearman_rho", "mean"))
        .reset_index()
        .sort_values("mean_spearman")
    )
    low_metrics = metric_means.head(min(3, len(metric_means)))
    high_metrics = metric_means.tail(min(3, len(metric_means))).sort_values("mean_spearman", ascending=False)

    family_overall = (
        summary_df.groupby("family")
        .agg(mean_spearman=("spearman_rho", "mean"))
        .reset_index()
        .sort_values("mean_spearman")
    )
    low_family = family_overall.iloc[0]
    high_family = family_overall.iloc[-1]

    low_metric_text = ", ".join(pretty_metric_name(metric) for metric in low_metrics["metric"])
    high_metric_text = ", ".join(pretty_metric_name(metric) for metric in high_metrics["metric"])

    text = f"""# Segmentation Method Robustness Figures

## Figure 1

**Segmentation-method robustness differs by spatial metric family.** Each point denotes one patient-aggregated spatial metric, grouped by spatial-statistic family and plotted as the Spearman correlation between two segmentation methods across patients. Horizontal bars indicate the median correlation within each family and segmentation-method pair.

## Figure 2

**Patient-rank agreement summarizes family-level robustness for the weakest segmentation-method comparison.** The inForm-Cellpose comparison is shown because it had the lowest mean Spearman agreement across the evaluated segmentation-method pairs. For each metric separately, patients were ranked within each segmentation method and converted to percentile ranks from 0 to 1, where 0 denotes the lowest-ranked patient and 1 denotes the highest-ranked patient for that metric and segmentation method. Metrics belonging to the same family were then combined in one subplot, so each point represents one patient-metric observation. The diagonal indicates identical patient ordering between inForm and Cellpose.

## Supplementary Figure: Single-Object Global Statistics

**Raw-scale agreement for single-object global statistics.** Rows show cancer Ripley's L AUC, stroma Ripley's L AUC, and Newman's assortativity. Columns compare inForm with Cellpose, inForm with CellProfiler, and Cellpose with CellProfiler. Each point represents one patient. Axes show patient-level metric values from the two segmentation methods being compared.

## Supplementary Figure: Graph Centrality / Clustering

**Raw-scale agreement for graph centrality and clustering metrics.** Rows show cancer and stroma average clustering, closeness centrality, and degree centrality. Columns compare inForm with Cellpose, inForm with CellProfiler, and Cellpose with CellProfiler. Each point represents one patient. Axes show patient-level metric values from the two segmentation methods being compared.

## Supplementary Figure: Two-Object Interaction Statistics

**Raw-scale agreement for two-object interaction statistics.** Rows show cancer-stroma co-occurrence AUC, cancer-stroma co-occurrence deviation AUC, cancer-cancer, cancer-stroma, and stroma-stroma neighborhood enrichment z-scores, and cancer-stroma object-object Pearson correlation. Columns compare inForm with Cellpose, inForm with CellProfiler, and Cellpose with CellProfiler. Each point represents one patient. Axes show patient-level metric values from the two segmentation methods being compared.

## Short Results Text

Patient-level agreement between segmentation methods was high overall, but robustness depended strongly on metric family. The metric panel was restricted to established single-object and two-object spatial statistics: Ripley's function, Newman's assortativity, graph centrality/clustering scores, cell-type co-occurrence ratio, neighborhood enrichment, and object-object correlation analysis. The percentile-rank scatter summarizes whether two segmentation methods preserve patient ordering within each family, while avoiding direct comparison of raw metric values measured on different scales.

The weakest segmentation-method pairing was {METHOD_PAIR_LABELS[(worst_pair['left_method'], worst_pair['right_method'])]} (mean Pearson {worst_pair['mean_pearson']:.3f}; mean Spearman {worst_pair['mean_spearman']:.3f}). At the metric-family level, {high_family['family']} was the most robust family (mean Spearman {high_family['mean_spearman']:.3f}), whereas {low_family['family']} was the least robust (mean Spearman {low_family['mean_spearman']:.3f}). Among individual features, the lowest-agreement metrics were {low_metric_text}, while the strongest-agreement metrics included {high_metric_text}.
"""
    out_path.write_text(text)


def write_outlier_name_lists(outlier_df: pd.DataFrame, out_path: Path) -> None:
    if outlier_df.empty:
        out_path.write_text("# Outlier names\n\nNo outliers were identified.\n", encoding="utf-8")
        return

    lines = ["# Outlier names", ""]
    grouped = outlier_df.groupby(["metric", "left_method", "right_method"], sort=True)
    for (metric, left_method, right_method), group in grouped:
        lines.append(f"## {pretty_metric_name(metric)} ({left_method} vs {right_method})")
        for row in group.sort_values("outlier_rank").itertuples(index=False):
            lines.append(
                f"- {row.entity_id} "
                f"(rank {int(row.outlier_rank)}, |delta|={row.abs_diff:.4g}, score={row.outlier_score:.3f})"
            )
        lines.append("")
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate per-metric scatter plots comparing patient- or core-level spatial metrics "
            "between inForm, Cellpose, and CellProfiler."
        )
    )
    ap.add_argument("--inform", default=DEFAULT_METHOD_TO_PATH["inform"])
    ap.add_argument("--cellpose", default=DEFAULT_METHOD_TO_PATH["cellpose"])
    ap.add_argument("--cellprofiler", default=DEFAULT_METHOD_TO_PATH["cellprofiler"])
    ap.add_argument(
        "--bomi2-samples-labels",
        default="../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/samples_labels.csv",
        help="Study-core sample list used to filter BOMI2 metrics before comparison.",
    )
    ap.add_argument(
        "--aggregate-level",
        choices=["patient", "core"],
        default="patient",
        help="Aggregate per-core metrics to patient level before plotting. Default: patient.",
    )
    ap.add_argument(
        "--include-counts",
        action="store_true",
        help="Also render count-based metrics such as cell_count, cancer_count, and stroma_count.",
    )
    ap.add_argument(
        "--metrics",
        nargs="*",
        help="Optional explicit metric list. Defaults to the selected --metric-panel.",
    )
    ap.add_argument(
        "--metric-panel",
        choices=["behanova", "shared"],
        default="behanova",
        help=(
            "Metric panel to plot. 'behanova' uses the active manuscript feature set; "
            "'shared' uses all shared numeric columns after exclusions."
        ),
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/figures/segmentation_metric_correlations",
        help="Directory where per-metric PNG figures and the summary CSV will be written.",
    )
    ap.add_argument(
        "--make-paper-summary",
        action="store_true",
        help="Also create condensed family-level and representative paper figures plus draft captions.",
    )
    ap.add_argument(
        "--outlier-top-n",
        type=int,
        default=5,
        help="How many strongest outlier names to save per metric and method pair. Default: 5.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    method_to_path = {
        "inform": args.inform,
        "cellpose": args.cellpose,
        "cellprofiler": args.cellprofiler,
    }

    print(f"[INFO] Loading allowed BOMI2 study cores from {args.bomi2_samples_labels}")
    allowed_samples = load_allowed_bomi2_samples(args.bomi2_samples_labels)
    print(f"[INFO] Allowed study cores: {len(allowed_samples)}")

    print("[INFO] Loading metrics tables")
    loaded = {name: load_metrics_table(path) for name, path in method_to_path.items()}
    loaded = {
        name: filter_to_allowed_samples(df, allowed_samples, method_name=name)
        for name, df in loaded.items()
    }

    print(f"[INFO] Aggregating metrics to {args.aggregate_level} level")
    aggregated = {name: aggregate_metrics(df, args.aggregate_level) for name, df in loaded.items()}

    if args.metric_panel == "behanova":
        metric_cols = get_behanova_metric_columns(list(aggregated.values()), include_counts=args.include_counts)
    else:
        metric_cols = get_metric_columns(list(aggregated.values()), include_counts=args.include_counts)

    if args.metrics:
        requested = list(dict.fromkeys(args.metrics))
        missing = sorted(set(requested) - set(metric_cols))
        if missing:
            raise ValueError(f"Requested metrics are not shared numeric columns: {missing}")
        metric_cols = requested

    if not metric_cols:
        raise RuntimeError("No shared numeric metrics found to plot")

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for old_png in figures_dir.glob("*.png"):
        old_png.unlink()

    print(f"[INFO] Rendering {len(metric_cols)} metric figures to {figures_dir}")
    summary_rows: list[dict[str, object]] = []
    outlier_frames: list[pd.DataFrame] = []
    id_col = "ID" if args.aggregate_level == "patient" else "sample_name"
    for metric in tqdm(metric_cols, desc="Plotting metrics", unit="metric"):
        out_path = figures_dir / f"{sanitize_filename(metric)}.png"
        summary_rows.extend(
            plot_metric_figure(
                metric=metric,
                method_pairs=DEFAULT_METHOD_PAIRS,
                aggregated=aggregated,
                out_path=out_path,
                id_col=id_col,
            )
        )
        for left_name, right_name in DEFAULT_METHOD_PAIRS:
            pair_df = build_pair_table(
                left_df=aggregated[left_name],
                right_df=aggregated[right_name],
                metric=metric,
                left_name=left_name,
                right_name=right_name,
                id_col=id_col,
            )
            outlier_frames.append(
                identify_pair_outliers(
                    pair_df,
                    metric=metric,
                    left_name=left_name,
                    right_name=right_name,
                    id_col=id_col,
                    top_n=args.outlier_top_n,
                )
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["metric", "left_method", "right_method"]).reset_index(drop=True)
    summary_path = out_dir / "pairwise_correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[INFO] Wrote summary to {summary_path}")

    outlier_df = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else pd.DataFrame()
    outlier_csv_path = out_dir / "pairwise_outlier_names.csv"
    outlier_md_path = out_dir / "pairwise_outlier_names.md"
    outlier_df.to_csv(outlier_csv_path, index=False)
    write_outlier_name_lists(outlier_df, outlier_md_path)
    print(f"[INFO] Wrote outlier names to {outlier_csv_path}")
    print(f"[INFO] Wrote outlier name list to {outlier_md_path}")

    if args.make_paper_summary:
        paper_dir = out_dir / "paper_summary"
        paper_dir.mkdir(parents=True, exist_ok=True)
        summary_with_family = load_summary(str(summary_path))

        print("[INFO] Creating paper-summary family figure")
        family_stats = make_family_summary_figure(
            summary_with_family,
            paper_dir / "metric_family_correlation_summary.png",
        )
        family_stats.to_csv(paper_dir / "metric_family_correlation_summary.csv", index=False)

        print("[INFO] Creating family-level percentile scatter panel")
        family_percentile_df = make_family_percentile_scatter_figure(
            summary_df=summary_with_family,
            aggregated=aggregated,
            out_path=paper_dir / "metric_family_percentile_scatter_panels.png",
            aggregate_level=args.aggregate_level,
        )
        family_percentile_df.to_csv(paper_dir / "metric_family_percentile_scatter_panels.csv", index=False)

        print("[INFO] Creating supplementary all-metric scatter panels")
        supplementary_df, supplementary_metrics = make_supplementary_raw_scatter_figures(
            summary_df=summary_with_family,
            aggregated=aggregated,
            paper_dir=paper_dir,
            aggregate_level=args.aggregate_level,
        )
        supplementary_df.to_csv(paper_dir / "all_metric_scatter_panels.csv", index=False)
        if not outlier_df.empty:
            representative_outliers = outlier_df[outlier_df["metric"].isin(supplementary_metrics)].copy()
            representative_outliers.to_csv(paper_dir / "all_metric_scatter_outlier_names.csv", index=False)
            write_outlier_name_lists(
                representative_outliers,
                paper_dir / "all_metric_scatter_outlier_names.md",
            )

        print("[INFO] Writing paper-summary captions and results text")
        build_text_outputs(
            summary_df=summary_with_family,
            out_path=paper_dir / "captions_and_results.md",
        )


if __name__ == "__main__":
    main()
