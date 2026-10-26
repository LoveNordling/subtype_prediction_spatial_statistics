#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from tqdm import tqdm

from plot_patient_core_panels import (
    build_zoom_bounds,
    canonicalize_bomi2_sample_name,
    load_bomi2_image,
    plot_image_panel,
    plot_scatter_panel,
)


def canonicalize_from_filename(text: str) -> str:
    value = str(text)
    m = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", value)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    m = re.search(r"(.+?)_\[(\d+,[A-Z])\]", value)
    if m:
        return f"{m.group(1)}_[{m.group(2)}]"
    raise ValueError(f"Cannot parse sample name from filename: {text}")


def load_inform_cells(path: str) -> pd.DataFrame:
    usecols = ["Sample Name", "Cell X Position", "Cell Y Position", "CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(
        columns={
            "Sample Name": "sample_name",
            "Cell X Position": "x",
            "Cell Y Position": "y",
        }
    )
    df["sample_name"] = df["sample_name"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_cellprofiler_cells(path: str) -> pd.DataFrame:
    usecols = ["FileName_CK", "Location_Center_X", "Location_Center_Y", "CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["sample_name"] = df["FileName_CK"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["Location_Center_X"], errors="coerce")
    df["y"] = pd.to_numeric(df["Location_Center_Y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_cellpose_cells(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=1)
    ck_col = "CK" if "CK" in head.columns else ("ck" if "ck" in head.columns else None)
    if ck_col is None:
        raise ValueError("Cellpose file missing CK/ck column")
    df = pd.read_csv(path, usecols=["filename", "x", "y", ck_col], low_memory=False)
    df["sample_name"] = df["filename"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df[ck_col], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_allowed_bomi2_samples(samples_labels_path: str) -> set[str]:
    df = pd.read_csv(samples_labels_path, usecols=["sample_name"], low_memory=False)
    return set(df["sample_name"].astype(str).map(canonicalize_bomi2_sample_name))


def build_bomi2_meta(samples_path: str, image_dir: str, allowed_samples: set[str]) -> pd.DataFrame:
    samples = pd.read_csv(samples_path, low_memory=False)
    samples["sample_name"] = samples["sample_name"].astype(str)
    samples["sample_name_canonical"] = samples["sample_name"].map(canonicalize_bomi2_sample_name)
    samples = samples.loc[samples["sample_name_canonical"].isin(allowed_samples)].copy()
    samples["image_path"] = samples["sample_name"].map(lambda s: str(Path(image_dir) / f"{s}_component_data.tif"))
    return samples.rename(columns={"ID": "patient_id"})


def sanitize_sample_name(sample_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_name)).strip("_")


def add_zoom_box(ax, zoom_box: tuple[float, float, float, float]) -> None:
    zx0, zx1, zy0, zy1 = zoom_box
    ax.add_patch(
        Rectangle(
            (zx0, zy0),
            zx1 - zx0,
            zy1 - zy0,
            fill=False,
            edgecolor="lime",
            linewidth=1.8,
            linestyle="--",
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create per-core BOMI2 comparison figures for inForm, Cellpose, and CellProfiler."
    )
    ap.add_argument("--inform", default="data/raw/BOMI2_all_cells_TIL.csv")
    ap.add_argument("--cellpose", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    ap.add_argument("--cellprofiler", default="data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv")
    ap.add_argument("--bomi2-samples", default="../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv")
    ap.add_argument(
        "--bomi2-samples-labels",
        default="../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/samples_labels.csv",
    )
    ap.add_argument("--bomi2-image-dir", default="/media/love/My Passport/BOMI2_inform_TIF_output")
    ap.add_argument("--image-max-dim", type=int, default=800)
    ap.add_argument("--zoom-factor", type=float, default=4.0)
    ap.add_argument("--max-scatter-points", type=int)
    ap.add_argument("--out-dpi", type=int, default=140)
    ap.add_argument("--sample", action="append", default=[], help="Optional canonical sample name(s), e.g. BOMI2_TIL_1_[2,J]")
    ap.add_argument("--limit", type=int, help="Optional maximum number of cores to render after filtering")
    ap.add_argument("--image-cache-dir", default="outputs/figures/bomi2_segmentation_method_comparison/_image_cache")
    ap.add_argument("--out-dir", default="outputs/figures/bomi2_segmentation_method_comparison")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading cell tables")
    inform_cells = load_inform_cells(args.inform)
    cellpose_cells = load_cellpose_cells(args.cellpose)
    cellprofiler_cells = load_cellprofiler_cells(args.cellprofiler)

    inform_groups = {k: g for k, g in inform_cells.groupby("sample_name", sort=False)}
    cellpose_groups = {k: g for k, g in cellpose_cells.groupby("sample_name", sort=False)}
    cellprofiler_groups = {k: g for k, g in cellprofiler_cells.groupby("sample_name", sort=False)}

    allowed_samples = load_allowed_bomi2_samples(args.bomi2_samples_labels)
    bomi2_meta = build_bomi2_meta(args.bomi2_samples, args.bomi2_image_dir, allowed_samples=allowed_samples)

    common_samples = set(inform_groups) & set(cellpose_groups) & set(cellprofiler_groups)
    bomi2_meta = bomi2_meta.loc[bomi2_meta["sample_name_canonical"].isin(common_samples)].copy()

    if args.sample:
        requested = {str(s) for s in args.sample}
        bomi2_meta = bomi2_meta.loc[bomi2_meta["sample_name_canonical"].isin(requested)].copy()

    if args.limit is not None:
        bomi2_meta = bomi2_meta.head(args.limit).copy()

    bomi2_meta = bomi2_meta.drop_duplicates(subset=["sample_name_canonical"]).reset_index(drop=True)
    if bomi2_meta.empty:
        raise RuntimeError("No BOMI2 samples remained after applying the labeled-cohort and common-sample filters")

    print(
        "[INFO] Rendering comparison panels for "
        f"{len(bomi2_meta)} cores across "
        f"{bomi2_meta['patient_id'].astype(str).nunique()} patients"
    )

    rows = []
    image_cache_dir = Path(args.image_cache_dir)
    for idx, row in enumerate(tqdm(bomi2_meta.itertuples(index=False), total=len(bomi2_meta), desc="Saving comparison figures", unit="core"), start=1):
        sample_name = str(row.sample_name_canonical)
        patient_id = str(row.patient_id)
        image_path = Path(str(row.image_path))

        inform_g = inform_groups[sample_name]
        cellpose_g = cellpose_groups[sample_name]
        cellprofiler_g = cellprofiler_groups[sample_name]

        x_max = max(
            float(inform_g["x"].max()),
            float(cellpose_g["x"].max()),
            float(cellprofiler_g["x"].max()),
        )
        y_max = max(
            float(inform_g["y"].max()),
            float(cellpose_g["y"].max()),
            float(cellprofiler_g["y"].max()),
        )

        image_spec = load_bomi2_image(image_path, args.image_max_dim, image_cache_dir)
        if image_spec is not None:
            x_max = max(x_max, float(image_spec.original_width))
            y_max = max(y_max, float(image_spec.original_height))
        xlim = (0.0, x_max)
        ylim = (0.0, y_max)

        zoom_box = build_zoom_bounds(cellpose_g, zoom_factor=args.zoom_factor)

        fig = plt.figure(figsize=(20, 11), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.22)
        top = outer[0].subgridspec(1, 3, wspace=0.18, width_ratios=[1.0, 1.0, 1.0])
        bottom = outer[1].subgridspec(1, 4, wspace=0.18, width_ratios=[1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])

        ax_whole_inform = fig.add_subplot(top[0, 0])
        ax_whole_cellpose = fig.add_subplot(top[0, 1])
        ax_whole_cellprofiler = fig.add_subplot(top[0, 2])
        ax_image = fig.add_subplot(bottom[0, 0])
        ax_zoom_inform = fig.add_subplot(bottom[0, 1])
        ax_zoom_cellpose = fig.add_subplot(bottom[0, 2])
        ax_zoom_cellprofiler = fig.add_subplot(bottom[0, 3])

        plot_scatter_panel(
            ax_whole_inform,
            inform_g,
            f"InForm whole\nn={len(inform_g)}",
            xlim,
            ylim,
            marker_size=1.2,
            zoom_box=zoom_box,
            max_points=args.max_scatter_points,
        )
        plot_scatter_panel(
            ax_whole_cellpose,
            cellpose_g,
            f"Cellpose whole\nn={len(cellpose_g)}",
            xlim,
            ylim,
            marker_size=1.2,
            zoom_box=zoom_box,
            max_points=args.max_scatter_points,
        )
        plot_scatter_panel(
            ax_whole_cellprofiler,
            cellprofiler_g,
            f"CellProfiler whole\nn={len(cellprofiler_g)}",
            xlim,
            ylim,
            marker_size=1.2,
            zoom_box=zoom_box,
            max_points=args.max_scatter_points,
        )

        plot_image_panel(
            ax_image,
            image_spec,
            "CK-DAPI image",
            xlim,
            ylim,
        )
        add_zoom_box(ax_image, zoom_box)

        plot_scatter_panel(
            ax_zoom_inform,
            inform_g,
            f"InForm zoom ({args.zoom_factor:.1f}x)",
            (zoom_box[0], zoom_box[1]),
            (zoom_box[2], zoom_box[3]),
            marker_size=10.0,
            max_points=args.max_scatter_points,
        )
        plot_scatter_panel(
            ax_zoom_cellpose,
            cellpose_g,
            f"Cellpose zoom ({args.zoom_factor:.1f}x)",
            (zoom_box[0], zoom_box[1]),
            (zoom_box[2], zoom_box[3]),
            marker_size=10.0,
            max_points=args.max_scatter_points,
        )
        plot_scatter_panel(
            ax_zoom_cellprofiler,
            cellprofiler_g,
            f"CellProfiler zoom ({args.zoom_factor:.1f}x)",
            (zoom_box[0], zoom_box[1]),
            (zoom_box[2], zoom_box[3]),
            marker_size=10.0,
            max_points=args.max_scatter_points,
        )

        fig.suptitle(f"BOMI2 core {sample_name} | patient {patient_id}", fontsize=15)

        out_path = out_dir / f"{idx:03d}_{sanitize_sample_name(sample_name)}.png"
        fig.savefig(out_path, dpi=args.out_dpi)
        plt.close(fig)

        rows.append(
            {
                "patient_id": patient_id,
                "sample_name": sample_name,
                "image_path": str(image_path),
                "inform_count": len(inform_g),
                "cellpose_count": len(cellpose_g),
                "cellprofiler_count": len(cellprofiler_g),
                "out_path": str(out_path),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "comparison_plot_index.csv", index=False)
    print(f"[DONE] Wrote method comparison figures to {out_dir}")


if __name__ == "__main__":
    main()
