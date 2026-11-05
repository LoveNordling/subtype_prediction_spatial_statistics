#!/usr/bin/env python3
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from tqdm import tqdm

# -------- CONFIG --------
INPUT_IMAGE_DIR = "/data3/love/lung_cancer_BOMI2/image_data/raw_image_data/all"
NUCLEI_MASK_DIR = "nuclei_masks"
OUTPUT_CSV_PATH = "cell_data_from_masks.csv"

CYTO_RADIUS = 6
CK_CHANNEL = 6
CK_POS_THRESHOLD = 7.0

N_WORKERS = max(1, os.cpu_count() - 1)   # tune as you like
# ------------------------

def basename_from_mask(p: str) -> str:
    name = os.path.basename(p)
    assert name.endswith("_nuclei_labels.tif"), f"Unexpected mask name: {name}"
    return name[:-len("_nuclei_labels.tif")]

def process_one(mask_path: str) -> pd.DataFrame:
    base = basename_from_mask(mask_path)
    raw_image_path = os.path.join(INPUT_IMAGE_DIR, base + ".tif")
    if not os.path.exists(raw_image_path):
        raise FileNotFoundError(f"Raw image missing for mask: {raw_image_path}")

    nuclei_labels = tiff.imread(mask_path)
    if nuclei_labels.ndim != 2:
        raise ValueError(f"Nuclei labels must be 2D, got {nuclei_labels.shape}")

    # Read only CK channel if possible; fall back to memmap
    try:
        ck = tiff.imread(raw_image_path, key=CK_CHANNEL).astype(np.float32, copy=False)
    except Exception:
        mm = tiff.memmap(raw_image_path)
        if mm.ndim != 3 or CK_CHANNEL >= mm.shape[0]:
            raise ValueError(f"Unexpected raw shape {mm.shape} or CK_CHANNEL out of range for {raw_image_path}")
        ck = mm[CK_CHANNEL].astype(np.float32, copy=False)

    if ck.shape != nuclei_labels.shape:
        raise ValueError(f"Shape mismatch: nuclei {nuclei_labels.shape} vs CK {ck.shape} for {base}")

    # Fast cytoplasm via expand_labels (no overlaps)
    expanded = expand_labels(nuclei_labels, distance=int(CYTO_RADIUS))
    cytoplasm_labels = expanded.copy()
    cytoplasm_labels[nuclei_labels > 0] = 0

    max_label = int(nuclei_labels.max())

    # Per-cell CK over cytoplasm
    cyto_ck_sum = np.bincount(cytoplasm_labels.ravel(),
                              weights=ck.ravel(),
                              minlength=max_label + 1)
    cyto_px_count = np.bincount(cytoplasm_labels.ravel(),
                                minlength=max_label + 1)
    cyto_ck_mean = np.full_like(cyto_ck_sum, np.nan, dtype=np.float64)
    valid = cyto_px_count > 0
    cyto_ck_mean[valid] = cyto_ck_sum[valid] / cyto_px_count[valid]

    # Vectorized nuclei geometry
    props = regionprops_table(nuclei_labels, properties=("label", "centroid", "area"))
    labels = props["label"].astype(np.int64)
    ys = props["centroid-0"].astype(np.float64)
    xs = props["centroid-1"].astype(np.float64)
    areas = props["area"].astype(np.int64)

    # Build DataFrame (no Python loops over pixels)
    # Safe indexing because bincount arrays are 0..max_label
    df = pd.DataFrame({
        "filename": base + ".tif",
        "cell_id": labels,
        "x": xs,
        "y": ys,
        "area_px_nucleus": areas,
        "cyto_area_px": cyto_px_count[labels],
        "ck_cyto_sum_raw": cyto_ck_sum[labels].astype(np.float64),
        "ck_cyto_mean_raw": cyto_ck_mean[labels],
    })
    # Binary flag
    ck_flag = (df["ck_cyto_mean_raw"] > CK_POS_THRESHOLD) & np.isfinite(df["ck_cyto_mean_raw"])
    df["ck"] = ck_flag.astype(np.int8)
    return df

def main():
    mask_paths = sorted(glob(os.path.join(NUCLEI_MASK_DIR, "*_nuclei_labels.tif")))
    if not mask_paths:
        raise FileNotFoundError(f"No nuclei label TIFFs found in {NUCLEI_MASK_DIR}")

    dfs = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one, p): p for p in mask_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Images"):
            dfs.append(fut.result())

    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
