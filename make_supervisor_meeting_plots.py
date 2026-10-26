#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from skimage.filters import threshold_otsu

COLOR_STROMA = "#0072B2"
COLOR_TUMOR = "#E64B35"


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
    return df.dropna(subset=["x", "y"])


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


def _plot_panel(ax, g: pd.DataFrame, title: str, xlim: tuple, ylim: tuple, marker_size: float = 1.2) -> None:
    stroma = g[g["CK"] == 0]
    tumor = g[g["CK"] == 1]

    if len(stroma) > 0:
        ax.scatter(
            stroma["x"],
            stroma["y"],
            color=COLOR_STROMA,
            s=marker_size,
            alpha=0.92,
            linewidths=0,
            label="Stroma",
        )
    if len(tumor) > 0:
        ax.scatter(
            tumor["x"],
            tumor["y"],
            color=COLOR_TUMOR,
            s=marker_size,
            alpha=0.92,
            linewidths=0,
            label="Tumor",
        )

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


def build_zoom_bounds(x_min: float, x_max: float, y_min: float, y_max: float, cx: float, cy: float, zoom_factor: float) -> tuple:
    width = (x_max - x_min) / zoom_factor
    height = (y_max - y_min) / zoom_factor
    zx_min = cx - width / 2.0
    zx_max = cx + width / 2.0
    zy_min = cy - height / 2.0
    zy_max = cy + height / 2.0
    return (zx_min, zx_max, zy_min, zy_max)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create supervisor meeting comparison plots (full + 10x zoom).")
    ap.add_argument("--inform", default="data/raw/BOMI2_all_cells_TIL.csv")
    ap.add_argument("--cellprofiler", default="data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv")
    ap.add_argument("--cellpose", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    ap.add_argument("--out-dir", default="outputs/figures/supervisor_meeting_2026_02_23")
    ap.add_argument("--n-samples", type=int, default=36, help="Number of shared samples to plot (sorted by total shared cells).")
    ap.add_argument("--zoom-factor", type=float, default=10.0, help="Linear zoom factor; 10 => 10x zoom.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading cells...")
    df_i = load_inform(args.inform)
    df_cp = load_cellprofiler(args.cellprofiler)
    df_cpose = load_cellpose(args.cellpose)

    g_i = {k: g for k, g in df_i.groupby("sample_name", sort=False)}
    g_cp = {k: g for k, g in df_cp.groupby("sample_name", sort=False)}
    g_cpose = {k: g for k, g in df_cpose.groupby("sample_name", sort=False)}

    common = sorted(set(g_i.keys()) & set(g_cp.keys()) & set(g_cpose.keys()))
    if not common:
        raise RuntimeError("No common sample names across all three methods.")

    counts = []
    for s in common:
        n_i = len(g_i[s])
        n_cp = len(g_cp[s])
        n_cpose = len(g_cpose[s])
        counts.append((s, n_i, n_cp, n_cpose, n_i + n_cp + n_cpose))
    counts_df = pd.DataFrame(counts, columns=["sample_name", "n_inform", "n_cellprofiler", "n_cellpose", "n_total"])
    counts_df = counts_df.sort_values("n_total", ascending=False).reset_index(drop=True)

    selected = counts_df.head(args.n_samples)["sample_name"].tolist()
    counts_df.to_csv(out_dir / "selected_samples_counts.csv", index=False)

    print(f"[INFO] Common samples: {len(common)} | plotting top {len(selected)} by shared cell count")

    for idx, sample in enumerate(selected, start=1):
        gi = g_i[sample]
        gcp = g_cp[sample]
        gcpose = g_cpose[sample]

        x_min = float(min(gi["x"].min(), gcp["x"].min(), gcpose["x"].min()))
        x_max = float(max(gi["x"].max(), gcp["x"].max(), gcpose["x"].max()))
        y_min = float(min(gi["y"].min(), gcp["y"].min(), gcpose["y"].min()))
        y_max = float(max(gi["y"].max(), gcp["y"].max(), gcpose["y"].max()))

        cx = float(gi["x"].median())
        cy = float(gi["y"].median())
        zx_min, zx_max, zy_min, zy_max = build_zoom_bounds(x_min, x_max, y_min, y_max, cx, cy, args.zoom_factor)

        fig, axs = plt.subplots(2, 3, figsize=(18, 11))

        _plot_panel(axs[0, 0], gcp, "CellProfiler (full)", (x_min, x_max), (y_min, y_max), marker_size=1.2)
        _plot_panel(axs[0, 1], gcpose, "Cellpose (full)", (x_min, x_max), (y_min, y_max), marker_size=1.2)
        _plot_panel(axs[0, 2], gi, "InForm (full)", (x_min, x_max), (y_min, y_max), marker_size=1.2)

        _plot_panel(
            axs[1, 0], gcp, f"CellProfiler ({int(args.zoom_factor)}x zoom)", (zx_min, zx_max), (zy_min, zy_max), marker_size=54.0
        )
        _plot_panel(
            axs[1, 1], gcpose, f"Cellpose ({int(args.zoom_factor)}x zoom)", (zx_min, zx_max), (zy_min, zy_max), marker_size=54.0
        )
        _plot_panel(
            axs[1, 2], gi, f"InForm ({int(args.zoom_factor)}x zoom)", (zx_min, zx_max), (zy_min, zy_max), marker_size=54.0
        )

        box_w = zx_max - zx_min
        box_h = zy_max - zy_min
        for ax in axs[0, :]:
            ax.add_patch(
                Rectangle(
                    (zx_min, zy_min),
                    box_w,
                    box_h,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.8,
                    linestyle="--",
                )
            )

        fig.suptitle(
            f"{sample} | counts: CP={len(gcp)}, Cellpose={len(gcpose)}, InForm={len(gi)}",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        safe = sample.replace("[", "").replace("]", "").replace(",", "_")
        out_png = out_dir / f"{idx:02d}_{safe}.png"
        fig.savefig(out_png, dpi=170)
        plt.close(fig)

        if idx % 5 == 0 or idx == len(selected):
            print(f"[INFO] Saved {idx}/{len(selected)}")

    print(f"[DONE] Wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
