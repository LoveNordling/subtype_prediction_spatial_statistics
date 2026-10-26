#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from patient_scatter_plot_utils import (
    ImageSpec,
    SCATTER_SIZE,
    ZOOM_SCATTER_SIZE,
    build_zoom_bounds,
    canonicalize_bomi2_sample_name,
    load_bomi1_cells,
    load_bomi1_image,
    load_bomi2_allowed_sample_names,
    load_bomi2_cells,
    load_bomi2_image,
    plot_scatter_panel,
)


def build_bomi2_meta(
    samples_path: str,
    image_dir: str,
    *,
    allowed_sample_names: set[str] | None = None,
) -> pd.DataFrame:
    samples = pd.read_csv(samples_path, low_memory=False)
    samples["sample_name"] = samples["sample_name"].astype(str)
    samples["sample_name_canonical"] = samples["sample_name"].map(canonicalize_bomi2_sample_name)
    if allowed_sample_names is not None:
        samples = samples.loc[samples["sample_name_canonical"].isin(allowed_sample_names)].copy()
    samples["image_path"] = samples["sample_name"].map(lambda s: str(Path(image_dir) / f"{s}_component_data.tif"))
    return samples.rename(columns={"ID": "patient_id"})


def build_bomi1_meta(meta_path: str, image_dir: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, low_memory=False)
    meta["sample_name"] = meta["sample_name"].astype(str)
    meta["image_path"] = meta["sample_name"].map(lambda s: str(Path(image_dir) / f"{s}.tif"))
    return meta.rename(columns={"ID": "patient_id"})


def plot_image_panel(ax, image_spec: ImageSpec | None, title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=10)
    if image_spec is None:
        ax.text(0.5, 0.5, "Missing image", ha="center", va="center", fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.imshow(
        image_spec.image,
        extent=(0, image_spec.original_width, 0, image_spec.original_height),
        origin="lower",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


def patient_bounds(sample_names: list[str], cell_groups: dict[str, pd.DataFrame], image_loader, image_paths: dict[str, str]) -> tuple[float, float, float, float]:
    x_max = 1.0
    y_max = 1.0
    for sample_name in sample_names:
        g = cell_groups.get(sample_name)
        if g is not None and not g.empty:
            x_max = max(x_max, float(g["x"].max()))
            y_max = max(y_max, float(g["y"].max()))
        img = image_loader(Path(image_paths[sample_name]))
        if img is not None:
            x_max = max(x_max, float(img.original_width))
            y_max = max(y_max, float(img.original_height))
    return (0.0, x_max), (0.0, y_max)


def render_cohort(
    *,
    cohort_name: str,
    meta_df: pd.DataFrame,
    sample_name_col: str,
    cell_groups: dict[str, pd.DataFrame],
    image_loader,
    out_dir: Path,
    image_max_dim: int,
    zoom_factor: float,
    image_cache_dir: Path | None,
    max_scatter_points: int | None,
    out_dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    grouped = list(meta_df.groupby("patient_id", sort=False))
    print(f"[INFO] Rendering {cohort_name}: {len(grouped)} patients")

    for idx, (patient_id, patient_meta) in enumerate(tqdm(grouped, desc=f"{cohort_name} patients", unit="patient"), start=1):
        sample_names = []
        image_paths = {}
        for row in patient_meta.itertuples(index=False):
            sample_name = str(getattr(row, sample_name_col))
            if sample_name not in cell_groups:
                continue
            sample_names.append(sample_name)
            image_paths[sample_name] = str(row.image_path)
        if not sample_names:
            continue

        image_specs = {
            sample_name: image_loader(Path(image_paths[sample_name]), image_max_dim, image_cache_dir)
            for sample_name in sample_names
        }
        x_max = 1.0
        y_max = 1.0
        for sample_name in sample_names:
            g = cell_groups[sample_name]
            x_max = max(x_max, float(g["x"].max()))
            y_max = max(y_max, float(g["y"].max()))
            img = image_specs[sample_name]
            if img is not None:
                x_max = max(x_max, float(img.original_width))
                y_max = max(y_max, float(img.original_height))
        xlim, ylim = (0.0, x_max), (0.0, y_max)

        n_rows = len(sample_names)
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)

        for row_idx, sample_name in enumerate(sample_names):
            image = image_specs[sample_name]
            zoom_box = build_zoom_bounds(cell_groups[sample_name], zoom_factor=zoom_factor)
            plot_image_panel(
                axes[row_idx, 0],
                image,
                f"{sample_name} image",
                xlim,
                ylim,
            )
            plot_scatter_panel(
                axes[row_idx, 1],
                cell_groups[sample_name],
                f"{sample_name} scatter\nn={len(cell_groups[sample_name])}",
                xlim,
                ylim,
                marker_size=SCATTER_SIZE,
                zoom_box=zoom_box,
                max_points=max_scatter_points,
            )
            plot_scatter_panel(
                axes[row_idx, 2],
                cell_groups[sample_name],
                f"{sample_name} zoom ({zoom_factor:.1f}x)",
                (zoom_box[0], zoom_box[1]),
                (zoom_box[2], zoom_box[3]),
                marker_size=ZOOM_SCATTER_SIZE,
                max_points=max_scatter_points,
            )

        fig.suptitle(f"{cohort_name} patient {patient_id} | cores={len(sample_names)}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"{idx:03d}_{patient_id}.png", dpi=out_dpi)
        plt.close(fig)

        rows.append({"patient_id": patient_id, "n_cores": len(sample_names), "sample_names": "|".join(sample_names)})
        if idx % 25 == 0 or idx == len(grouped):
            print(f"[INFO] {cohort_name}: saved {idx}/{len(grouped)} patients")

    pd.DataFrame(rows).to_csv(out_dir / "patient_plot_index.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create patient-wise core panels with images and scatter plots for BOMI1 and BOMI2.")
    ap.add_argument("--cohort", choices=["BOMI1", "BOMI2", "both"], default="both")
    ap.add_argument("--bomi2-cells", default="data/interim/cellpose_extracted_cells_fitlered_necrosis.csv")
    ap.add_argument("--bomi2-samples", default="../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv")
    ap.add_argument(
        "--bomi2-samples-labels",
        default="../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/samples_labels.csv",
    )
    ap.add_argument("--bomi2-image-dir", default="/media/love/My Passport/BOMI2_inform_TIF_output")
    ap.add_argument("--bomi1-cells", default="data/raw/BOMI1_cells_all_cellvitpp_2026_03_26.csv")
    ap.add_argument("--bomi1-meta", default="data/reference/BOMI1_clinical_data_LUADvsSqCC.csv")
    ap.add_argument("--bomi1-image-dir", default="~/BOMI1_TMA/BOMI1_cores_tiff_fix")
    ap.add_argument("--image-max-dim", type=int, default=800)
    ap.add_argument("--zoom-factor", type=float, default=4.0)
    ap.add_argument("--max-scatter-points", type=int)
    ap.add_argument("--out-dpi", type=int, default=120)
    ap.add_argument("--image-cache-dir", default="outputs/figures/patient_core_panels/_image_cache")
    ap.add_argument("--out-dir", default="outputs/figures/patient_core_panels")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cohort in {"BOMI2", "both"}:
        print("[INFO] Loading BOMI2 cells and metadata")
        bomi2_cells = load_bomi2_cells(args.bomi2_cells)
        allowed_sample_names = load_bomi2_allowed_sample_names(args.bomi2_samples_labels)
        bomi2_meta = build_bomi2_meta(
            args.bomi2_samples,
            args.bomi2_image_dir,
            allowed_sample_names=allowed_sample_names,
        )
        print(
            "[INFO] BOMI2 subtype-filtered cohort: "
            f"{bomi2_meta['patient_id'].astype(str).nunique()} "
            f"patients across {len(bomi2_meta)} cores"
        )
        bomi2_groups = {k: g for k, g in bomi2_cells.groupby("sample_name", sort=False)}
        render_cohort(
            cohort_name="BOMI2",
            meta_df=bomi2_meta,
            sample_name_col="sample_name_canonical",
            cell_groups=bomi2_groups,
            image_loader=load_bomi2_image,
            out_dir=out_dir / "BOMI2",
            image_max_dim=args.image_max_dim,
            zoom_factor=args.zoom_factor,
            image_cache_dir=Path(args.image_cache_dir) / "BOMI2",
            max_scatter_points=args.max_scatter_points,
            out_dpi=args.out_dpi,
        )

    if args.cohort in {"BOMI1", "both"}:
        print("[INFO] Loading BOMI1 cells and metadata")
        bomi1_cells = load_bomi1_cells(args.bomi1_cells, args.bomi1_meta)
        bomi1_meta = build_bomi1_meta(args.bomi1_meta, str(Path(args.bomi1_image_dir).expanduser()))
        bomi1_groups = {k: g for k, g in bomi1_cells.groupby("sample_name", sort=False)}
        render_cohort(
            cohort_name="BOMI1",
            meta_df=bomi1_meta,
            sample_name_col="sample_name",
            cell_groups=bomi1_groups,
            image_loader=load_bomi1_image,
            out_dir=out_dir / "BOMI1",
            image_max_dim=args.image_max_dim,
            zoom_factor=args.zoom_factor,
            image_cache_dir=Path(args.image_cache_dir) / "BOMI1",
            max_scatter_points=args.max_scatter_points,
            out_dpi=args.out_dpi,
        )
    print(f"[DONE] Wrote patient panels to {out_dir}")


if __name__ == "__main__":
    main()
