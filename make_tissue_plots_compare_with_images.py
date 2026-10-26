#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tifffile
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm


COLOR_STROMA = "#0072B2"
COLOR_TUMOR = "#E64B35"
SEG_CMAP = ListedColormap(["#E64B35", "#111111", "#0072B2"])


def canonicalize_from_filename(s: str) -> str:
    s = str(s)
    match = re.search(r"(.+?)_Core\[1,(\d+,[A-Z])\]", s)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    match = re.search(r"(.+?)_\[(\d+,[A-Z])\]", s)
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    raise ValueError(f"Cannot parse sample from filename: {s}")


def load_inform(path: str) -> pd.DataFrame:
    usecols = ["Sample Name", "Raw Sample Name", "ImageID", "Cell X Position", "Cell Y Position", "CK"]
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
    return df.dropna(subset=["x", "y"])


def load_cellprofiler(path: str) -> pd.DataFrame:
    usecols = ["FileName_CK", "ImageID", "Location_Center_X", "Location_Center_Y", "CK"]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["sample_name"] = df["FileName_CK"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["Location_Center_X"], errors="coerce")
    df["y"] = pd.to_numeric(df["Location_Center_Y"], errors="coerce")
    df["CK"] = pd.to_numeric(df["CK"], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "ImageID", "x", "y", "CK"]].dropna(subset=["x", "y"])


def load_cellpose(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=1)
    ck_col = "CK" if "CK" in head.columns else ("ck" if "ck" in head.columns else None)
    if ck_col is None:
        raise ValueError("Cellpose file missing CK/ck column")
    usecols = ["filename", "ImageID", "x", "y", ck_col]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["sample_name"] = df["filename"].map(canonicalize_from_filename)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["CK"] = pd.to_numeric(df[ck_col], errors="coerce").fillna(0).astype(int)
    return df[["sample_name", "ImageID", "x", "y", "CK"]].dropna(subset=["x", "y"])


def build_sample_image_lookup(df_inform: pd.DataFrame, image_dir: Path) -> dict[str, dict[str, Path]]:
    lookup: dict[str, dict[str, Path]] = {}
    meta = (
        df_inform[["sample_name", "ImageID"]]
        .dropna()
        .drop_duplicates(subset=["sample_name"])
    )
    for row in meta.itertuples(index=False):
        image_id = str(row.ImageID)
        lookup[str(row.sample_name)] = {
            "original": image_dir / f"{image_id}.tif",
            "composite": image_dir / f"{image_id}_composite_image.tif",
            "segmap": image_dir / f"{image_id}_binary_seg_maps.tif",
        }
    return lookup


def read_tiff_for_display(path: Path):
    if not path.exists():
        return None

    img = tifffile.imread(path)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] <= 8:
        channels = img[: min(3, img.shape[0]), :, :]
        return channels.transpose(1, 2, 0)
    if img.ndim == 3 and img.shape[-1] <= 8:
        return img[:, :, : min(3, img.shape[-1])]
    return img[0] if img.ndim > 3 else img


def read_segmentation_map(path: Path):
    if not path.exists():
        return None
    img = tifffile.imread(path)
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] <= 8:
        return img[0]
    if img.ndim == 3 and img.shape[-1] <= 8:
        return img[:, :, 0]
    return None


def plot_image_panel(ax, image_path: Path, title: str) -> None:
    image = read_tiff_for_display(image_path)
    if image is None:
        ax.text(0.5, 0.5, f"Missing\n{image_path.name}", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return

    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_segmentation_panel(ax, seg_path: Path, title: str) -> None:
    seg = read_segmentation_map(seg_path)
    if seg is None:
        ax.text(0.5, 0.5, f"Missing\n{seg_path.name}", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return

    ax.imshow(seg, cmap=SEG_CMAP, interpolation="nearest", vmin=0, vmax=2)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_scatter_panel(ax, g: pd.DataFrame, title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    stroma = g[g["CK"] == 0]
    tumor = g[g["CK"] == 1]

    if len(stroma) > 0:
        ax.scatter(stroma["x"], stroma["y"], color=COLOR_STROMA, s=1.2, alpha=0.9, linewidths=0)
    if len(tumor) > 0:
        ax.scatter(tumor["x"], tumor["y"], color=COLOR_TUMOR, s=1.2, alpha=0.9, linewidths=0)

    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STROMA, markeredgecolor="none", markersize=6, label="CK-"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TUMOR, markeredgecolor="none", markersize=6, label="CK+"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-sample comparison figures with images and CK-colored scatters.")
    parser.add_argument("--inform", default="data/raw/BOMI2_all_cells_TIL.csv")
    parser.add_argument("--cellprofiler", default="data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv")
    parser.add_argument("--cellpose", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    parser.add_argument("--image-dir", default="/media/love/My Passport/BOMI2_inform_output_2026_03_12")
    parser.add_argument("--out-dir", default="outputs/figures/tissue_plots_compare_with_images")
    parser.add_argument("--n-samples", type=int, default=36)
    parser.add_argument("--sample", action="append", default=[], help="Optional canonical sample name(s), e.g. BOMI2_TIL_1_[2,J]")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading cell tables")
    df_inform = load_inform(args.inform)
    df_cellprofiler = load_cellprofiler(args.cellprofiler)
    df_cellpose = load_cellpose(args.cellpose)

    image_lookup = build_sample_image_lookup(df_inform, Path(args.image_dir))

    groups_inform = {k: g for k, g in df_inform.groupby("sample_name", sort=False)}
    groups_cellprofiler = {k: g for k, g in df_cellprofiler.groupby("sample_name", sort=False)}
    groups_cellpose = {k: g for k, g in df_cellpose.groupby("sample_name", sort=False)}

    common = sorted(set(groups_inform) & set(groups_cellprofiler) & set(groups_cellpose))
    if not common:
        raise RuntimeError("No common sample names across inform, cellprofiler, and cellpose")

    if args.sample:
        selected = [sample for sample in args.sample if sample in common]
        print(f"[INFO] Requested {len(args.sample)} samples, found {len(selected)} in common set")
    else:
        counts = []
        for sample in common:
            counts.append(
                (
                    sample,
                    len(groups_inform[sample]),
                    len(groups_cellprofiler[sample]),
                    len(groups_cellpose[sample]),
                )
            )
        counts_df = pd.DataFrame(counts, columns=["sample_name", "n_inform", "n_cellprofiler", "n_cellpose"])
        counts_df["n_total"] = counts_df["n_inform"] + counts_df["n_cellprofiler"] + counts_df["n_cellpose"]
        counts_df = counts_df.sort_values("n_total", ascending=False).reset_index(drop=True)
        counts_df.to_csv(out_dir / "selected_samples_counts.csv", index=False)
        selected = counts_df.head(args.n_samples)["sample_name"].tolist()
        print(f"[INFO] Plotting top {len(selected)} samples by total detected cells")

    for idx, sample in enumerate(tqdm(selected, desc="Saving comparison figures", unit="sample"), start=1):
        gi = groups_inform[sample]
        gcp = groups_cellprofiler[sample]
        gcpose = groups_cellpose[sample]

        x_min = float(min(gi["x"].min(), gcp["x"].min(), gcpose["x"].min()))
        x_max = float(max(gi["x"].max(), gcp["x"].max(), gcpose["x"].max()))
        y_min = float(min(gi["y"].min(), gcp["y"].min(), gcpose["y"].min()))
        y_max = float(max(gi["y"].max(), gcp["y"].max(), gcpose["y"].max()))

        fig, axes = plt.subplots(2, 4, figsize=(22, 11))

        sample_images = image_lookup.get(sample, {})
        plot_image_panel(axes[0, 0], sample_images.get("original", Path("")), "Original image")
        plot_image_panel(axes[0, 1], sample_images.get("composite", Path("")), "Composite image")
        plot_segmentation_panel(axes[0, 2], sample_images.get("segmap", Path("")), "Segmentation map")
        axes[0, 3].axis("off")
        axes[0, 3].text(
            0.03,
            0.97,
            (
                f"Sample: {sample}\n"
                f"InForm cells: {len(gi):,}\n"
                f"CellProfiler cells: {len(gcp):,}\n"
                f"Cellpose cells: {len(gcpose):,}"
            ),
            ha="left",
            va="top",
            fontsize=11,
            transform=axes[0, 3].transAxes,
        )

        plot_scatter_panel(axes[1, 0], gcp, "CellProfiler", (x_min, x_max), (y_min, y_max))
        plot_scatter_panel(axes[1, 1], gcpose, "Cellpose", (x_min, x_max), (y_min, y_max))
        plot_scatter_panel(axes[1, 2], gi, "InForm", (x_min, x_max), (y_min, y_max))
        axes[1, 3].axis("off")
        seg_handles = [
            Patch(facecolor="#E64B35", edgecolor="none", label="Seg label 0 (red)"),
            Patch(facecolor="#111111", edgecolor="none", label="Seg label 1 (black)"),
            Patch(facecolor="#0072B2", edgecolor="none", label="Seg label 2 (blue)"),
        ]
        ck_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_STROMA, markeredgecolor="none", markersize=7, label="Cells: CK-"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_TUMOR, markeredgecolor="none", markersize=7, label="Cells: CK+"),
        ]
        legend = axes[1, 3].legend(
            handles=seg_handles + ck_handles,
            loc="upper left",
            frameon=True,
            framealpha=0.95,
            fontsize=10,
            title="Legend",
            title_fontsize=11,
        )
        axes[1, 3].add_artist(legend)

        fig.tight_layout()
        safe_name = sample.replace("[", "").replace("]", "").replace(",", "_")
        out_path = out_dir / f"{safe_name}.png"
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

    print(f"[DONE] Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
