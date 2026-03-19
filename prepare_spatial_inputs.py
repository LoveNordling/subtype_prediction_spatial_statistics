from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.stats import gaussian_kde
from skimage.filters import threshold_otsu
from tqdm import tqdm


INFORM_RAW_SAMPLE_COL = "Raw Sample Name"
INFORM_IMAGE_ID_COL = "ImageID"
INFORM_SOURCE_FILE_COL = "Source File"
KEEP_SEG_LABELS = {0, 2}
DROP_SEG_LABELS = {1}

INFORM_MINIMAL_COLUMNS = [
    "Sample Name",
    INFORM_RAW_SAMPLE_COL,
    "Tissue Category",
    "Cell ID",
    "Cell X Position",
    "Cell Y Position",
    "Nucleus Area (pixels)",
    "Nucleus Compactness",
    "Nucleus Axis Ratio",
    "Nucleus DAPI Mean (Normalized Counts, Total Weighting)",
    "Cytoplasm Opal 520 Mean (Normalized Counts, Total Weighting)",
    "Cytoplasm Opal 540 Mean (Normalized Counts, Total Weighting)",
    "Cytoplasm Opal 570 Mean (Normalized Counts, Total Weighting)",
    "Cytoplasm Opal 650 Mean (Normalized Counts, Total Weighting)",
    "Cytoplasm Opal 690 Mean (Normalized Counts, Total Weighting)",
    "Cytoplasm Autofluorescence Mean (Normalized Counts, Total Weighting)",
    "Lab ID",
    "Slide ID",
    "TMA Sector",
    "TMA Row",
    "TMA Column",
    "TMA Field",
    INFORM_IMAGE_ID_COL,
    INFORM_SOURCE_FILE_COL,
]

INFORM_LEGACY_PLACEHOLDERS = [
    "CD4_Single",
    "CD4_Treg",
    "CD8_Single",
    "CD8_Treg",
    "B_cells",
    "CKSingle",
    "Stroma_other",
]


def panck_min_valley(data: pd.DataFrame, column: str, grid: int = 2048) -> tuple[float, pd.Series]:
    series = pd.to_numeric(data[column], errors="coerce")
    values = series.dropna().to_numpy(float)
    if values.size < 10:
        raise ValueError(f"Need at least 10 non-null values to threshold '{column}', got {values.size}.")

    otsu = float(threshold_otsu(values))
    xs = np.linspace(values.min(), values.max(), grid)
    kde = gaussian_kde(values)
    ys = kde(xs)

    left_mask = xs < otsu
    right_mask = xs > otsu
    if left_mask.sum() < 3 or right_mask.sum() < 3:
        cut = otsu
    else:
        peak1 = xs[left_mask][np.argmax(ys[left_mask])]
        peak2 = xs[right_mask][np.argmax(ys[right_mask])]
        between = (xs > peak1) & (xs < peak2)
        cut = otsu if not between.any() else float(xs[between][np.argmin(ys[between])])

    binary = (series >= cut).fillna(0).astype("uint8")
    return cut, binary


def canonicalize_inform_sample_name(raw_sample_name: str) -> str:
    stem = Path(str(raw_sample_name)).stem
    match = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", stem)
    if not match:
        raise ValueError(f"Could not parse canonical sample name from inForm sample: {raw_sample_name}")
    return f"{match.group(1)}_[{match.group(2)}]"


def extract_image_id(value: str) -> str | None:
    match = re.search(r"(BOMI2_TIL_.*?_Core\[\d+,\d+,[A-Z]\]_\[\d+,\d+\])", str(value))
    return match.group(1) if match else None


def build_mask_lookup(mask_dir: Path) -> dict[str, Path]:
    print(f"[INFO] Scanning mask TIFFs in {mask_dir}")
    mask_paths: dict[str, Path] = {}
    mask_files = sorted(mask_dir.glob("*_binary_seg_maps.tif"))
    for mask_path in tqdm(mask_files, desc="Indexing masks", unit="mask"):
        image_id = extract_image_id(mask_path.name)
        if image_id:
            mask_paths[image_id] = mask_path
    print(f"[INFO] Indexed {len(mask_paths):,} mask files")
    return mask_paths


def select_mask_plane(mask_array: np.ndarray, mask_plane: int) -> tuple[np.ndarray, int]:
    if mask_array.ndim == 2:
        return mask_array, 0

    if mask_array.ndim != 3:
        raise ValueError(f"Unsupported mask dimensions {mask_array.shape}; expected 2D or 3D TIFF.")

    if mask_array.shape[0] <= 8:
        plane = min(mask_plane, mask_array.shape[0] - 1)
        return mask_array[plane, :, :], plane

    if mask_array.shape[-1] <= 8:
        plane = min(mask_plane, mask_array.shape[-1] - 1)
        return mask_array[:, :, plane], plane

    raise ValueError(f"Could not infer mask axis from shape {mask_array.shape}.")


def filter_cells_by_mask(
    df: pd.DataFrame,
    *,
    image_col: str,
    x_col: str,
    y_col: str,
    mask_paths: dict[str, Path],
    mask_plane: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    print(f"[INFO] Starting mask filtering on {len(df):,} rows using column '{image_col}'")
    kept_groups: list[pd.DataFrame] = []
    stats = {
        "input_rows": int(len(df)),
        "kept_rows": 0,
        "dropped_rows": 0,
        "missing_mask_groups": 0,
        "missing_image_id_rows": 0,
        "mask_plane_min": -1,
        "mask_plane_max": -1,
    }
    used_planes: list[int] = []

    grouped = df.groupby(image_col, dropna=False, sort=False)
    for image_id, group in tqdm(grouped, desc="Filtering image groups", unit="image"):
        if pd.isna(image_id):
            stats["missing_image_id_rows"] += int(len(group))
            continue

        mask_path = mask_paths.get(str(image_id))
        if mask_path is None:
            stats["missing_mask_groups"] += 1
            stats["dropped_rows"] += int(len(group))
            continue

        mask, used_plane = select_mask_plane(tifffile.imread(mask_path), mask_plane=mask_plane)
        used_planes.append(int(used_plane))
        keep_indices: list[int] = []
        for idx, row in group.iterrows():
            x = int(round(float(row[x_col])))
            y = int(round(float(row[y_col])))

            # Preserve out-of-bounds cells to match the previous scripts.
            # For the March 12, 2026 inForm masks, plane 0 is the tissue category map.
            # Confirmed from visual QA:
            #   - seg label 1 = background -> drop
            #   - seg labels 0 and 2 = keep
            if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                keep_indices.append(idx)
            elif int(mask[y, x]) in KEEP_SEG_LABELS:
                keep_indices.append(idx)

        kept = group.loc[keep_indices].copy()
        kept_groups.append(kept)
        stats["kept_rows"] += int(len(kept))
        stats["dropped_rows"] += int(len(group) - len(kept))

    if not kept_groups:
        raise ValueError("Mask filtering removed every row; check mask paths, image IDs, and coordinates.")

    if used_planes:
        stats["mask_plane_min"] = int(min(used_planes))
        stats["mask_plane_max"] = int(max(used_planes))

    print(
        "[INFO] Mask filtering complete: "
        f"kept {stats['kept_rows']:,} / {stats['input_rows']:,} rows, "
        f"dropped {stats['dropped_rows']:,}"
    )
    return pd.concat(kept_groups, ignore_index=True), stats


def merge_inform_exports(inform_dir: Path) -> pd.DataFrame:
    files = sorted(inform_dir.glob("*_cell_seg_data.txt"))
    if not files:
        raise FileNotFoundError(f"No *_cell_seg_data.txt files found in {inform_dir}")

    print(f"[INFO] Loading inForm cores from {inform_dir}")
    print(f"[INFO] Found {len(files):,} inForm cell export files")
    merged: list[pd.DataFrame] = []
    for path in tqdm(files, desc="Loading inForm cores", unit="core"):
        df = pd.read_csv(path, sep="\t", low_memory=False)
        raw_sample = df["Sample Name"].astype(str)
        df[INFORM_RAW_SAMPLE_COL] = raw_sample
        df["Sample Name"] = raw_sample.map(canonicalize_inform_sample_name)
        df[INFORM_IMAGE_ID_COL] = raw_sample.map(lambda s: extract_image_id(Path(s).stem))
        df[INFORM_SOURCE_FILE_COL] = path.name

        for col in INFORM_MINIMAL_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA

        merged.append(df[INFORM_MINIMAL_COLUMNS].copy())

    result = pd.concat(merged, ignore_index=True)
    result["Cell X Position"] = pd.to_numeric(result["Cell X Position"], errors="coerce")
    result["Cell Y Position"] = pd.to_numeric(result["Cell Y Position"], errors="coerce")
    print(f"[INFO] Merged inForm table has {len(result):,} rows")
    return result


def add_inform_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Cancer"] = out["Tissue Category"].map({"Tumor": 1, "Stroma": 0}).fillna(0).astype(int)
    for col in INFORM_LEGACY_PLACEHOLDERS:
        out[col] = 0
    return out


def prepare_inform(
    inform_dir: Path,
    inform_merged_out: Path,
    inform_masked_out: Path,
    mask_dir: Path,
    mask_plane: int,
) -> tuple[Path, float, dict[str, int]]:
    print("[INFO] Preparing inForm cells")
    merged = merge_inform_exports(inform_dir)
    merged_with_legacy = add_inform_legacy_columns(merged)
    inform_merged_out.parent.mkdir(parents=True, exist_ok=True)
    merged_with_legacy.to_csv(inform_merged_out, index=False)
    print(f"[INFO] Saved merged unmasked inForm cells to {inform_merged_out}")

    mask_paths = build_mask_lookup(mask_dir)
    filtered, stats = filter_cells_by_mask(
        merged_with_legacy,
        image_col=INFORM_IMAGE_ID_COL,
        x_col="Cell X Position",
        y_col="Cell Y Position",
        mask_paths=mask_paths,
        mask_plane=mask_plane,
    )

    cut, ck_binary = panck_min_valley(
        filtered,
        "Cytoplasm Opal 690 Mean (Normalized Counts, Total Weighting)",
    )
    print(f"[INFO] inForm CK threshold = {cut:.6f}")
    filtered["CK"] = ck_binary
    filtered["CKSingle"] = filtered["CK"].astype(int)
    filtered["Stroma_other"] = (1 - filtered["CK"]).astype(int)
    filtered.to_csv(inform_masked_out, index=False)
    print(f"[INFO] Saved masked inForm cells to {inform_masked_out}")
    return inform_masked_out, cut, stats


def prepare_generic_source(
    *,
    input_csv: Path,
    output_csv: Path,
    mask_dir: Path,
    mask_plane: int,
    image_col: str,
    x_col: str,
    y_col: str,
    intensity_col: str,
    ck_output_cols: list[str],
) -> tuple[Path, float, dict[str, int]]:
    print(f"[INFO] Preparing source file {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)

    required_cols = {x_col, y_col, intensity_col}
    missing_required = sorted(col for col in required_cols if col not in df.columns)
    if missing_required:
        raise ValueError(
            f"{input_csv} is missing required columns for processing: {missing_required}"
        )

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col]).copy()
    print(f"[INFO] Loaded {len(df):,} rows from {input_csv}")
    if image_col not in df.columns:
        df[image_col] = pd.NA

    if df[image_col].isna().any():
        # Backfill image IDs from the original filename if needed.
        source_name_col = "FileName_CK" if "FileName_CK" in df.columns else "filename"
        if source_name_col not in df.columns:
            raise ValueError(
                f"{input_csv} is missing both '{image_col}' and a filename column needed to derive it."
            )

        missing = df[image_col].isna()
        df.loc[missing, image_col] = df.loc[missing, source_name_col].map(extract_image_id)

    mask_paths = build_mask_lookup(mask_dir)
    filtered, stats = filter_cells_by_mask(
        df,
        image_col=image_col,
        x_col=x_col,
        y_col=y_col,
        mask_paths=mask_paths,
        mask_plane=mask_plane,
    )

    cut, ck_binary = panck_min_valley(filtered, intensity_col)
    print(f"[INFO] CK threshold for {input_csv.name} = {cut:.6f}")
    for ck_col in ck_output_cols:
        filtered[ck_col] = ck_binary

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)
    print(f"[INFO] Saved processed cells to {output_csv}")
    return output_csv, cut, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the March 12 2026 inForm export, then mask and CK-threshold "
            "inForm, CellProfiler, and Cellpose cells into spatial_statistics-ready CSVs."
        )
    )
    parser.add_argument(
        "--inform-dir",
        type=Path,
        default=Path("/media/love/My Passport/BOMI2_inform_output_2026_03_12"),
        help="Directory containing raw inForm *_cell_seg_data.txt files and *_binary_seg_maps.tif masks.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory containing *_binary_seg_maps.tif. Defaults to --inform-dir.",
    )
    parser.add_argument(
        "--inform-merged-out",
        type=Path,
        default=Path("data/interim/BOMI2_all_cells_TIL_unmasked_2026_03_12.csv"),
        help="Merged raw inForm cells CSV.",
    )
    parser.add_argument(
        "--inform-masked-out",
        type=Path,
        default=Path("data/raw/BOMI2_all_cells_TIL.csv"),
        help="Masked and CK-thresholded inForm CSV used by spatial_statistics4_parallelized.py.",
    )
    parser.add_argument(
        "--cellprofiler-in",
        type=Path,
        default=Path("data/raw/Cellprofiler_raw_output.csv"),
        help="Unmasked raw CellProfiler cells CSV.",
    )
    parser.add_argument(
        "--cellprofiler-out",
        type=Path,
        default=Path("data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv"),
        help="Masked and CK-thresholded CellProfiler CSV used by spatial_statistics4_parallelized.py.",
    )
    parser.add_argument(
        "--cellpose-in",
        type=Path,
        default=Path("data/raw/cellpose_extracted_cells.csv"),
        help="Unmasked Cellpose cells CSV.",
    )
    parser.add_argument(
        "--cellpose-out",
        type=Path,
        default=Path("data/interim/cellpose_extracted_cells_fitlered_necrosis.csv"),
        help="Masked and CK-thresholded Cellpose CSV used by spatial_statistics4_parallelized.py.",
    )
    parser.add_argument(
        "--mask-plane",
        type=int,
        default=0,
        help="Zero-based TIFF plane to use from the binary segmentation maps. For the March 12, 2026 masks, use 0 for tissue classes.",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        choices=["inform", "cellprofiler", "cellpose", "all"],
        default=["all"],
        help="Which sources to process. Use e.g. '--run cellprofiler' to only regenerate CellProfiler.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mask_dir = args.mask_dir or args.inform_dir
    run_targets = {"inform", "cellprofiler", "cellpose"} if "all" in args.run else set(args.run)
    print("[INFO] Starting spatial input preparation")
    print(f"[INFO] inForm directory: {args.inform_dir}")
    print(f"[INFO] mask directory: {mask_dir}")
    print(f"[INFO] mask plane: {args.mask_plane}")
    print(f"[INFO] run targets: {sorted(run_targets)}")

    if "inform" in run_targets:
        inform_out, inform_cut, inform_stats = prepare_inform(
            inform_dir=args.inform_dir,
            inform_merged_out=args.inform_merged_out,
            inform_masked_out=args.inform_masked_out,
            mask_dir=mask_dir,
            mask_plane=args.mask_plane,
        )
        print(f"inForm merged: {args.inform_merged_out}")
        print(f"inForm masked: {inform_out}")
        print(f"inForm CK cut: {inform_cut:.6f}")
        print(f"inForm stats: {inform_stats}")

    if "cellprofiler" in run_targets:
        cp_out, cp_cut, cp_stats = prepare_generic_source(
            input_csv=args.cellprofiler_in,
            output_csv=args.cellprofiler_out,
            mask_dir=mask_dir,
            mask_plane=args.mask_plane,
            image_col="ImageID",
            x_col="Location_Center_X",
            y_col="Location_Center_Y",
            intensity_col="Intensity_MeanIntensity_CK",
            ck_output_cols=["CK"],
        )
        print("[INFO] CellProfiler processing complete")
        print(f"CellProfiler masked: {cp_out}")
        print(f"CellProfiler CK cut: {cp_cut:.6f}")
        print(f"CellProfiler stats: {cp_stats}")

    if "cellpose" in run_targets:
        cpose_out, cpose_cut, cpose_stats = prepare_generic_source(
            input_csv=args.cellpose_in,
            output_csv=args.cellpose_out,
            mask_dir=mask_dir,
            mask_plane=args.mask_plane,
            image_col="ImageID",
            x_col="x",
            y_col="y",
            intensity_col="ck_cyto_mean_raw",
            ck_output_cols=["ck", "CK"],
        )
        print("[INFO] Cellpose processing complete")
        print(f"Cellpose masked: {cpose_out}")
        print(f"Cellpose CK cut: {cpose_cut:.6f}")
        print(f"Cellpose stats: {cpose_stats}")


if __name__ == "__main__":
    main()
