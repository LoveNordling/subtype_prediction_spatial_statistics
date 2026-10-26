#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from make_cellpose_matched_to_external_v2 import build_separate_paired_targets, compute_sample_median_nnd
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
    df = pd.read_csv(path, usecols=["filename", "x", "y", "CK"], low_memory=False)
    df["sample_name"] = df["filename"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_matched_cellpose(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["orig_sample_name", "thin_rep", "x", "y", "ck"], low_memory=False)
    df["orig_sample_name"] = df["orig_sample_name"].astype(str)
    df["thin_rep"] = pd.to_numeric(df["thin_rep"], errors="coerce").astype("Int64")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["ck"], errors="coerce").fillna(0).astype(int)
    return df[["orig_sample_name", "thin_rep", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_external_bomi1(path: str, xy_scale: float) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    df = preprocess_external_cells(raw).rename(columns={"Cancer": "CK"})
    df["x"] = pd.to_numeric(df["x"], errors="coerce") * float(xy_scale)
    df["y"] = pd.to_numeric(df["y"], errors="coerce") * float(xy_scale)
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y", "sample_name"])


def compute_paired2d_matches(df_internal: pd.DataFrame, df_external: pd.DataFrame, *, min_cells: int, min_cells_cancer_qc: int) -> pd.DataFrame:
    src_stats_c = compute_sample_median_nnd(
        df_internal[df_internal["CK"] == 1].assign(CK=1),
        min_cells=min_cells_cancer_qc,
        compartment="all",
    )
    src_stats_s = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="stroma")
    ref_stats_c = compute_sample_median_nnd(
        df_external[df_external["CK"] == 1].assign(CK=1),
        min_cells=min_cells_cancer_qc,
        compartment="all",
    )
    ref_stats_s = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="stroma")
    return build_separate_paired_targets(src_stats_c, src_stats_s, ref_stats_c, ref_stats_s)


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
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STROMA, markeredgecolor="none", markersize=6, label="Stroma"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TUMOR, markeredgecolor="none", markersize=6, label="Tumor"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.9)


def safe_name(sample: str) -> str:
    return sample.replace("[", "").replace("]", "").replace(",", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot original BOMI2 core, 3 separate subsamples, and the paired-2d matched BOMI1 core.")
    ap.add_argument("--original", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    ap.add_argument("--matched-separate", default="data/interim/cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1_separate.csv")
    ap.add_argument("--external", default="data/raw/BOMI1_cells_all.csv")
    ap.add_argument("--external-xy-scale", type=float, default=2.0)
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--min-cells-cancer-qc", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--out-dir", default="outputs/figures/cellpose_separate_paired2d_with_bomi1")
    ap.add_argument("--sample", action="append", default=[])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading original BOMI2, separate subsamples, and BOMI1 cells")
    df_original = load_original_cellpose(args.original)
    df_separate = load_matched_cellpose(args.matched_separate)
    df_external = load_external_bomi1(args.external, xy_scale=args.external_xy_scale)

    print("[INFO] Computing paired-2d BOMI1 matches")
    paired_matches = compute_paired2d_matches(
        df_original,
        df_external,
        min_cells=args.min_cells,
        min_cells_cancer_qc=args.min_cells_cancer_qc,
    )
    paired_matches.to_csv(out_dir / "matched_external_samples.csv", index=False)

    groups_original = {k: g for k, g in df_original.groupby("sample_name", sort=False)}
    groups_separate = {(k1, int(k2)): g for (k1, k2), g in df_separate.groupby(["orig_sample_name", "thin_rep"], sort=False)}
    groups_external = {k: g for k, g in df_external.groupby("sample_name", sort=False)}
    match_lookup = {
        row.sample_name: row for row in paired_matches.itertuples(index=False)
        if row.matched_external_sample_name in groups_external
    }

    common = []
    for sample in sorted(set(groups_original) & set(match_lookup)):
        if all((sample, rep) in groups_separate for rep in (1, 2, 3)):
            common.append(sample)
    if not common:
        raise RuntimeError("No common samples with original, separate reps 1-3, and paired-2d BOMI1 matches.")

    counts = []
    for sample in common:
        row = match_lookup[sample]
        counts.append(
            {
                "sample_name": sample,
                "matched_external_sample_name": row.matched_external_sample_name,
                "n_original": len(groups_original[sample]),
                "n_separate_r1": len(groups_separate[(sample, 1)]),
                "n_separate_r2": len(groups_separate[(sample, 2)]),
                "n_separate_r3": len(groups_separate[(sample, 3)]),
                "n_external": len(groups_external[row.matched_external_sample_name]),
                "src_nnd_cancer": row.src_nnd_cancer,
                "src_nnd_stroma": row.src_nnd_stroma,
                "target_nnd_cancer": row.target_nnd_cancer,
                "target_nnd_stroma": row.target_nnd_stroma,
                "matched_external_pair_distance": row.matched_external_pair_distance,
            }
        )
    counts_df = pd.DataFrame(counts).sort_values("n_original", ascending=False).reset_index(drop=True)
    counts_df.to_csv(out_dir / "selected_samples_counts.csv", index=False)

    if args.sample:
        selected = [sample for sample in args.sample if sample in common]
    else:
        selected = counts_df.head(args.n_samples)["sample_name"].tolist()

    print(f"[INFO] Writing {len(selected)} figures to {out_dir}")
    for idx, sample in enumerate(selected, start=1):
        row = match_lookup[sample]
        matched_external = row.matched_external_sample_name
        groups = [
            ("Original BOMI2", groups_original[sample]),
            ("Separate r1", groups_separate[(sample, 1)]),
            ("Separate r2", groups_separate[(sample, 2)]),
            ("Separate r3", groups_separate[(sample, 3)]),
            (f"BOMI1 paired-2d match\n{matched_external}", groups_external[matched_external]),
        ]

        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        for ax, (label, group) in zip(axes, groups):
            title = label
            if label != "Original BOMI2":
                title = f"{label}\nn={len(group)}"
            else:
                title = f"{label}\nn={len(group)}"
            plot_scatter_panel(ax, group, title)

        fig.suptitle(
            (
                f"{sample} | cancer NND {row.src_nnd_cancer:.2f}->{row.target_nnd_cancer:.2f} | "
                f"stroma NND {row.src_nnd_stroma:.2f}->{row.target_nnd_stroma:.2f}"
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
