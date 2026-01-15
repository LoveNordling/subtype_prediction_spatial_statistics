#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # speed: non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import tifffile
from skimage.filters import threshold_otsu
import cv2

# --------------- CONFIG ---------------
DF1_PATH = "./cellprofiler_extracted_cells_filtered_necrosis.csv"
DF2_PATH = "./cellpose_extracted_cells_fitlered_necrosis.csv"
DF3_PATH = "./BOMI2_all_cells_TIL.csv"
IMAGE_FOLDER = "/media/love/Love Extern/TESTOUT/"
OUTPUT_DIR = "tissue_plots_compare"

# channel indices to keep in the RGB compose (matching your original selection)
CHANNELS = [0, 1, 2, 3, 4, 6]
# same 6x3 “color” matrix you used, scaled like before
COLORS = np.array([
    [2, 0, 0],
    [0, 0, 3],
    [0, 3, 3],
    [3, 3, 0],
    [3, 0, 3],
    [0, 1, 0],
], dtype=np.float32)
# gamma for display only
GAMMA = 0.5
# --------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------- helpers --------
def extract_sample_name(filename: str) -> str:
    m = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", filename)
    if not m:
        raise ValueError(f"Could not parse sample name from filename: {filename}")
    return f"{m.group(1)}_[{m.group(2)}]"



# correct gamma LUT: output = x ** gamma  (gamma < 1 brightens)
def gamma_lut_uint8(gamma: float) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.clip((xs ** gamma) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return lut

GAMMA = 0.5  # as before
GAMMA_LUT = gamma_lut_uint8(GAMMA)

def multiplex2rgb_fast(image: np.ndarray) -> np.ndarray:
    """
    image: (C,H,W). Uses CHANNELS and COLORS from your config.
    Speedy + much better visibility:
      1) per-channel robust normalization (1st-99th pct)
      2) linear mix to RGB
      3) normalize to 0-255
      4) apply gamma LUT (correct direction)
    """
    sel = image[CHANNELS, :, :].astype(np.float32, copy=False)  # (K,H,W)

    # robust per-channel scaling
    K, H, W = sel.shape
    sel_reshaped = sel.reshape(K, -1)
    p1 = np.percentile(sel_reshaped, 1, axis=1)
    p99 = np.percentile(sel_reshaped, 99, axis=1)
    scale = p99 - p1
    # avoid division by zero
    scale[scale <= 0] = 1.0

    # broadcast: (K,H,W) -> scaled to ~[0,1] per channel
    sel = (sel - p1[:, None, None]) / scale[:, None, None]
    sel = np.clip(sel, 0.0, 1.0)

    # mix channels to RGB with your COLORS (Kx3) in one shot
    rgb = np.tensordot(sel, COLORS.astype(np.float32, copy=False), axes=([0], [0]))  # (H,W,3)

    # normalize mix to [0,255]
    mn = rgb.min()
    mx = rgb.max()
    if mx > mn:
        rgb = (rgb - mn) / (mx - mn)
    else:
        rgb = np.zeros_like(rgb, dtype=np.float32)
    rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)

    # apply gamma (correct direction)
    rgb_u8 = GAMMA_LUT[rgb_u8]  # LUT index per channel is fine

    return rgb_u8


# -------- load data once --------
df1 = pd.read_csv(DF1_PATH)
df2 = pd.read_csv(DF2_PATH)
df3 = pd.read_csv(DF3_PATH, low_memory=False)

# Precompute Otsu on df1 once (as you already do)
otsu_thresh1 = threshold_otsu(df1["Intensity_MeanIntensity_CK"].dropna().to_numpy())

# Add SampleName columns (vectorized apply is fine here; could also precompile regex)
df1["SampleName"] = df1["ImageID"].map(extract_sample_name)
df2["SampleName"] = df2["ImageID"].map(extract_sample_name)

# Group once (fast slicing later)
g1 = df1.groupby("SampleName", sort=False)
g2 = df2.groupby("SampleName", sort=False)
g3 = df3.groupby("Sample Name", sort=False)

# Compute intersection of available sample keys
samples = set(g1.groups.keys()) & set(g2.groups.keys()) & set(g3.groups.keys())

# Map samples to image paths in one pass
image_map = {}
for fname in os.listdir(IMAGE_FOLDER):
    if fname.endswith(".tif"):
        try:
            sname = extract_sample_name(fname)
        except ValueError:
            continue
        image_map[sname] = os.path.join(IMAGE_FOLDER, fname)

# -------- main loop --------
for sample in tqdm(sorted(samples), desc="Samples"):
    # fast, pre-sliced subsets (no re-filtering)
    s1 = g1.get_group(sample)
    s2 = g2.get_group(sample)
    s3 = g3.get_group(sample)

    # Precompute numeric colors (0/1) instead of string mapping (much faster in Matplotlib)
    c1 = (s1["Intensity_MeanIntensity_CK"].to_numpy() >= otsu_thresh1).astype(np.uint8)
    # df2 already has CK (capitalization depends on your file; adjust if needed)
    c2 = s2["CK"].to_numpy().astype(np.uint8)
    # df3 has CK in {0.0, 1.0}; dropna once when grouping might be overkill, do per-sample
    s3_valid = s3.dropna(subset=["CK"])
    c3 = s3_valid["CK"].to_numpy().astype(np.uint8)

    # Coordinates (vectorized)
    x1 = s1["Location_Center_X"].to_numpy()
    y1 = s1["Location_Center_Y"].to_numpy()

    x2 = s2["x"].to_numpy()
    y2 = s2["y"].to_numpy()

    x3 = (s3_valid["Cell X Position"].to_numpy(dtype=np.float32) * 2.0)
    y3 = (s3_valid["Cell Y Position"].to_numpy(dtype=np.float32) * 2.0)

    # Global bounds once
    x_min = np.min([x1.min(initial=np.inf), x2.min(initial=np.inf), x3.min(initial=np.inf)])
    x_max = np.max([x1.max(initial=-np.inf), x2.max(initial=-np.inf), x3.max(initial=-np.inf)])
    y_min = np.min([y1.min(initial=np.inf), y2.min(initial=np.inf), y3.min(initial=np.inf)])
    y_max = np.max([y1.max(initial=-np.inf), y2.max(initial=-np.inf), y3.max(initial=-np.inf)])

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    # Use same colormap with numeric c values
    axs[0].scatter(x1, y1, c=c1, cmap="coolwarm", s=1, alpha=0.6, vmin=0, vmax=1)
    axs[0].set_title(f"{sample} - cellprofiler")

    axs[1].scatter(x2, y2, c=c2, cmap="coolwarm", s=1, alpha=0.6, vmin=0, vmax=1)
    axs[1].set_title(f"{sample} - cellpose")

    axs[2].scatter(x3, y3, c=c3, cmap="coolwarm", s=1, alpha=0.6, vmin=0, vmax=1)
    axs[2].set_title(f"{sample} - inform")

    # Image panel
    if sample in image_map:
        img = tifffile.imread(image_map[sample])
        if img.ndim == 3 and img.shape[0] < 10:  # (C,H,W)
            pass
        elif img.ndim == 3 and img.shape[-1] < 10:  # (H,W,C)
            img = np.moveaxis(img, -1, 0)
        else:
            raise ValueError(f"Unexpected image shape for {image_map[sample]}: {img.shape}")

        # --- NEW: downsample for speed ---
        DOWNSAMPLE = 4
        if DOWNSAMPLE > 1:
            C, H, W = img.shape
            img_small = np.empty((C, H // DOWNSAMPLE, W // DOWNSAMPLE), dtype=img.dtype)
            for c in range(C):
                img_small[c] = cv2.resize(img[c], (W // DOWNSAMPLE, H // DOWNSAMPLE),
                                      interpolation=cv2.INTER_AREA)
            img = img_small
        # --------------------------------
            
        image_rgb = multiplex2rgb_fast(img)
        axs[3].imshow(image_rgb, interpolation="nearest")
        axs[3].set_title(f"{sample} - image")
        axs[3].axis("off")
    else:
        axs[3].set_title("Image not found")
        axs[3].axis("off")

    # Format axes (do it in a tiny loop)
    for ax in axs[:3]:
        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.invert_yaxis()

    plt.tight_layout()
    safe = sample.replace("[", "").replace("]", "").replace(",", "_")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{safe}.png"), dpi=150)
    plt.close()
