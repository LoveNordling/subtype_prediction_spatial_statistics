#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from make_cellpose_matched_to_external_v2 import compute_sample_median_nnd
from spatial_utils import preprocess_external_cells


COLOR_STROMA = "#0072B2"
COLOR_TUMOR = "#E64B35"


def canonicalize_from_filename(s: str) -> str:
    s = str(s)
    match = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", s)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    match = re.search(r"(.+?)_\[(\d+,[A-Z])\]", s)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    raise ValueError(f"Cannot parse sample from filename: {s}")


def load_original_cellpose(path: str) -> pd.DataFrame:
    usecols = ["filename", "x", "y", "CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["sample_name"] = df["filename"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_matched_cellpose(path: str) -> pd.DataFrame:
    usecols = ["orig_sample_name", "sample_name_new", "thin_rep", "x", "y", "ck"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["orig_sample_name"] = df["orig_sample_name"].astype(str)
    df["sample_name_new"] = df["sample_name_new"].astype(str)
    df["thin_rep"] = pd.to_numeric(df["thin_rep"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["ck"], errors="coerce").fillna(0).astype(int)
    return df[["orig_sample_name", "sample_name_new", "thin_rep", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_external_bomi1(path: str, xy_scale: float) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = preprocess_external_cells(raw).rename(columns={"Cancer": "CK"})
    df["x"] = pd.to_numeric(df["x"], errors="coerce") * float(xy_scale)
    df["y"] = pd.to_numeric(df["y"], errors="coerce") * float(xy_scale)
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y", "sample_name"])


def compute_percentile_matched_external_samples(
    df_internal: pd.DataFrame,
    df_external: pd.DataFrame,
    *,
    compartment: str,
    min_cells: int,
) -> pd.DataFrame:
    src_stats = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment=compartment)
    ref_stats = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment=compartment)
    if src_stats.empty or ref_stats.empty:
        raise RuntimeError("Could not compute source/reference NND tables for percentile matching.")

    metric_col = "median_nnd_all" if compartment == "all" else "median_nnd_stroma"
    src_sorted = np.sort(src_stats[metric_col].dropna().to_numpy(float))
    ref_metric = ref_stats[metric_col].dropna().to_numpy(float)
    ref_stats = ref_stats.copy()

    rows = []
    for row in src_stats.itertuples(index=False):
        nnd_src = float(getattr(row, metric_col))
        if not (np.isfinite(nnd_src) and nnd_src > 0):
            continue
        rank = np.searchsorted(src_sorted, nnd_src, side="left")
        quantile = (rank + 0.5) / len(src_sorted)
        target_nnd = float(np.quantile(ref_metric, quantile))
        ref_stats["abs_delta"] = (pd.to_numeric(ref_stats[metric_col], errors="coerce") - target_nnd).abs()
        best = ref_stats.sort_values(["abs_delta", metric_col], ascending=[True, True]).iloc[0]
        rows.append(
            {
                "sample_name": row.sample_name,
                "internal_nnd": nnd_src,
                "quantile": quantile,
                "target_external_nnd": target_nnd,
                "matched_external_sample_name": str(best["sample_name"]),
                "matched_external_nnd": float(best[metric_col]),
                "matched_external_abs_delta": float(best["abs_delta"]),
            }
        )
    return pd.DataFrame(rows)


def _bounds_for_group(g: pd.DataFrame, pad_frac: float = 0.03) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min = float(g["x"].min())
    x_max = float(g["x"].max())
    y_min = float(g["y"].min())
    y_max = float(g["y"].max())
    dx = x_max - x_min
    dy = y_max - y_min
    x_pad = max(dx * pad_frac, 1.0)
    y_pad = max(dy * pad_frac, 1.0)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_scatter_panel(ax, g: pd.DataFrame, title: str) -> None:
    stroma = g[g["CK"] == 0]
    tumor = g[g["CK"] == 1]

    if len(stroma) > 0:
        ax.scatter(stroma["x"], stroma["y"], color=COLOR_STROMA, s=1.2, alpha=0.9, linewidths=0)
    if len(tumor) > 0:
        ax.scatter(tumor["x"], tumor["y"], color=COLOR_TUMOR, s=1.2, alpha=0.9, linewidths=0)

    xlim, ylim = _bounds_for_group(g)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STROMA, markeredgecolor="none", markersize=6, label="Stroma"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TUMOR, markeredgecolor="none", markersize=6, label="Tumor"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.9)


def plot_missing_panel(ax, title: str) -> None:
    ax.text(0.5, 0.5, "Not present\nin current file", ha="center", va="center", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(True)


def safe_name(sample: str) -> str:
    return sample.replace("[", "").replace("]", "").replace(",", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot BOMI2 Cellpose cores, subsampled variants, and percentile-matched BOMI1 cores.")
    ap.add_argument("--original", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    ap.add_argument("--matched-global", default="data/interim/cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1.csv")
    ap.add_argument("--matched-separate", default="data/interim/cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1_separate.csv")
    ap.add_argument("--matched-featurepanel", default="data/interim/cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1_featurepanel_quick.csv")
    ap.add_argument("--external", default="data/raw/BOMI1_cells_all.csv")
    ap.add_argument("--external-xy-scale", type=float, default=2.0)
    ap.add_argument("--match-compartment", choices=["all", "stroma"], default="all")
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--out-dir", default="outputs/figures/cellpose_subsample_with_bomi1_match")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--sample", action="append", default=[])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading original, matched, and BOMI1 external cell tables")
    df_original = load_original_cellpose(args.original)
    df_global = load_matched_cellpose(args.matched_global)
    df_separate = load_matched_cellpose(args.matched_separate)
    df_featurepanel = load_matched_cellpose(args.matched_featurepanel)
    df_external = load_external_bomi1(args.external, xy_scale=args.external_xy_scale)

    external_matches = compute_percentile_matched_external_samples(
        df_original,
        df_external,
        compartment=args.match_compartment,
        min_cells=args.min_cells,
    )
    external_matches.to_csv(out_dir / "matched_external_samples.csv", index=False)

    groups_original = {k: g for k, g in df_original.groupby("sample_name", sort=False)}
    groups_global = {(k1, int(k2)): g for (k1, k2), g in df_global.groupby(["orig_sample_name", "thin_rep"], sort=False)}
    groups_separate = {(k1, int(k2)): g for (k1, k2), g in df_separate.groupby(["orig_sample_name", "thin_rep"], sort=False)}
    groups_featurepanel = {(k1, int(k2)): g for (k1, k2), g in df_featurepanel.groupby(["orig_sample_name", "thin_rep"], sort=False)}
    groups_external = {k: g for k, g in df_external.groupby("sample_name", sort=False)}

    matched_lookup = {
        row.sample_name: row for row in external_matches.itertuples(index=False)
        if row.matched_external_sample_name in groups_external
    }
    common = []
    for sample in sorted(set(groups_original) & set(matched_lookup)):
        has_global = all((sample, rep) in groups_global for rep in (1, 2, 3))
        has_separate = all((sample, rep) in groups_separate for rep in (1, 2, 3))
        if has_global and has_separate:
            common.append(sample)
    if not common:
        raise RuntimeError("No common samples across original, matched variants, and percentile-matched BOMI1 cores.")

    if args.sample:
        selected = [sample for sample in args.sample if sample in common]
    else:
        counts = []
        for sample in common:
            row = matched_lookup[sample]
            ext_name = row.matched_external_sample_name
            n_original = len(groups_original[sample])
            n_global_r1 = len(groups_global[(sample, 1)])
            n_global_r2 = len(groups_global[(sample, 2)])
            n_global_r3 = len(groups_global[(sample, 3)])
            n_separate_r1 = len(groups_separate[(sample, 1)])
            n_separate_r2 = len(groups_separate[(sample, 2)])
            n_separate_r3 = len(groups_separate[(sample, 3)])
            n_featurepanel_r1 = len(groups_featurepanel[(sample, 1)]) if (sample, 1) in groups_featurepanel else np.nan
            n_external = len(groups_external[ext_name])
            counts.append(
                {
                    "sample_name": sample,
                    "matched_external_sample_name": ext_name,
                    "n_original": n_original,
                    "n_global_r1": n_global_r1,
                    "n_global_r2": n_global_r2,
                    "n_global_r3": n_global_r3,
                    "n_separate_r1": n_separate_r1,
                    "n_separate_r2": n_separate_r2,
                    "n_separate_r3": n_separate_r3,
                    "n_featurepanel_r1": n_featurepanel_r1,
                    "n_external": n_external,
                    "retention_global_r1": n_global_r1 / n_original if n_original else float("nan"),
                    "retention_global_r2": n_global_r2 / n_original if n_original else float("nan"),
                    "retention_global_r3": n_global_r3 / n_original if n_original else float("nan"),
                    "retention_separate_r1": n_separate_r1 / n_original if n_original else float("nan"),
                    "retention_separate_r2": n_separate_r2 / n_original if n_original else float("nan"),
                    "retention_separate_r3": n_separate_r3 / n_original if n_original else float("nan"),
                    "retention_featurepanel_r1": n_featurepanel_r1 / n_original if n_original and pd.notna(n_featurepanel_r1) else float("nan"),
                    "internal_nnd": row.internal_nnd,
                    "target_external_nnd": row.target_external_nnd,
                    "matched_external_nnd": row.matched_external_nnd,
                    "matched_external_abs_delta": row.matched_external_abs_delta,
                }
            )
        counts_df = pd.DataFrame(counts).sort_values("n_original", ascending=False).reset_index(drop=True)
        counts_df.to_csv(out_dir / "selected_samples_counts.csv", index=False)
        selected = counts_df.head(args.n_samples)["sample_name"].tolist()

    print(f"[INFO] Writing {len(selected)} sample figures to {out_dir}")
    for idx, sample in enumerate(selected, start=1):
        ext_row = matched_lookup[sample]
        matched_external = ext_row.matched_external_sample_name

        g_original = groups_original[sample]
        g_external = groups_external[matched_external]
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        axes = axes.ravel()
        panels: list[tuple[str, pd.DataFrame | None]] = [
            (f"Original BOMI2\nn={len(g_original)}", g_original),
            (
                f"Global r1\nn={len(groups_global[(sample, 1)])} ({len(groups_global[(sample, 1)]) / len(g_original):.1%})",
                groups_global[(sample, 1)],
            ),
            (
                f"Global r2\nn={len(groups_global[(sample, 2)])} ({len(groups_global[(sample, 2)]) / len(g_original):.1%})",
                groups_global[(sample, 2)],
            ),
            (
                f"Global r3\nn={len(groups_global[(sample, 3)])} ({len(groups_global[(sample, 3)]) / len(g_original):.1%})",
                groups_global[(sample, 3)],
            ),
            (
                f"Separate r1\nn={len(groups_separate[(sample, 1)])} ({len(groups_separate[(sample, 1)]) / len(g_original):.1%})",
                groups_separate[(sample, 1)],
            ),
            (
                f"Separate r2\nn={len(groups_separate[(sample, 2)])} ({len(groups_separate[(sample, 2)]) / len(g_original):.1%})",
                groups_separate[(sample, 2)],
            ),
            (
                f"Separate r3\nn={len(groups_separate[(sample, 3)])} ({len(groups_separate[(sample, 3)]) / len(g_original):.1%})",
                groups_separate[(sample, 3)],
            ),
            (
                f"Feature-panel quick r1\nn={len(groups_featurepanel[(sample, 1)])} ({len(groups_featurepanel[(sample, 1)]) / len(g_original):.1%})",
                groups_featurepanel[(sample, 1)],
            ) if (sample, 1) in groups_featurepanel else ("Feature-panel quick r1", None),
            ("Feature-panel quick r2", groups_featurepanel.get((sample, 2))),
            ("Feature-panel quick r3", groups_featurepanel.get((sample, 3))),
            (
                f"BOMI1 matched core\n{matched_external}\nn={len(g_external)}",
                g_external,
            ),
        ]

        for ax, (title, group) in zip(axes, panels):
            if group is None or len(group) == 0:
                plot_missing_panel(ax, title)
            else:
                plot_scatter_panel(ax, group, title)
        for ax in axes[len(panels):]:
            ax.axis("off")

        fig.suptitle(
            (
                f"{sample} | {args.match_compartment} NND "
                f"{ext_row.internal_nnd:.2f} -> target {ext_row.target_external_nnd:.2f} -> "
                f"{matched_external} ({ext_row.matched_external_nnd:.2f})"
            ),
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"{idx:02d}_{safe_name(sample)}.png", dpi=170)
        plt.close(fig)

        if idx % 5 == 0 or idx == len(selected):
            print(f"[INFO] Saved {idx}/{len(selected)}")

    print(f"[DONE] Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
