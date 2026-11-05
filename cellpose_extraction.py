import numpy as np
import tifffile
from cellpose import models
import cv2
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops, label
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import os
from glob import glob
from tqdm import tqdm

print("Is CUDA available?", torch.cuda.is_available())

# Input and output directories
input_dir = "/data3/love/lung_cancer_BOMI2/image_data/raw_image_data/all/"
output_plot_dir = "cell_plots"
output_mask_dir = "nuclei_masks"          # new: where we store labeled nuclei masks
output_csv_path = "cell_data.csv"

os.makedirs(output_plot_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Load Cellpose model
model = models.CellposeModel(pretrained_model="nuclei", gpu=True, model_type="nuclei")

# Collect all tif files excluding those with 'screening' in the filename
tif_files = [
    f for f in glob(os.path.join(input_dir, "*.tif"))
    if "screening" not in os.path.basename(f).lower()
]

all_results = []

for file_path in tqdm(tif_files):
    fname = os.path.basename(file_path)
    print(f"Processing {file_path}...")
    img = tifffile.imread(file_path)

    # Extract DAPI and CK channels (adjust indices if needed)
    dapi = img[0, :, :]
    ck = img[6, :, :]

    # Prepare input for Cellpose (RGB stack of DAPI is fine)
    dapi_rgb = np.stack([dapi, dapi, dapi], axis=-1)

    t_start = time.time()
    masks, _, _ = model.eval([dapi_rgb])   # list in, list out
    masks = masks[0]

    # Label nuclei
    labeled_nuclei = label(masks)  # 0 is background, 1..N are nuclei IDs

    # Save the labeled nuclei segmentation for later cytoplasm extraction
    mask_out_path = os.path.join(output_mask_dir, fname.replace(".tif", "_nuclei_labels.tif"))
    # Use uint32 to be safe if there are many objects
    tifffile.imwrite(mask_out_path, labeled_nuclei.astype(np.uint32))

    # Optional: if you want a default dilated mask for quick experiments, keep this
    # but cytoplasm strategy can be recomputed later from saved nuclei mask.
    binary_mask = labeled_nuclei > 0
    dilated_mask = binary_dilation(binary_mask, iterations=10)
    dilated_labels = label(dilated_mask)

    # CK intensity per nucleus (raw)
    max_label = labeled_nuclei.max()
    # Sum of CK within each labeled nucleus
    ck_sum = np.bincount(labeled_nuclei.ravel(), weights=ck.ravel(), minlength=max_label + 1)
    # Pixel count per nucleus
    pixel_count = np.bincount(labeled_nuclei.ravel(), minlength=max_label + 1)
    # Mean CK per nucleus; leave zeros where pixel_count is zero
    ck_mean_nucleus = np.divide(ck_sum, pixel_count, out=np.zeros_like(ck_sum), where=pixel_count != 0)

    # Collect per-cell results
    file_results = []
    for region in regionprops(labeled_nuclei):
        cell_id = region.label
        if cell_id == 0:
            continue

        y, x = region.centroid
        area_px = region.area
        ck_mean = float(ck_mean_nucleus[cell_id])
        ck_sum_val = float(ck_sum[cell_id])

        # Keep your binary CK flag (threshold can be adjusted later)
        ck_positive = int(ck_mean > 7)

        file_results.append({
            'filename': fname,
            'cell_id': int(cell_id),
            'x': float(x),
            'y': float(y),
            'area_px': int(area_px),
            'ck_mean_raw': ck_mean,   # new: raw mean CK per nucleus
            'ck_sum_raw': ck_sum_val, # new: raw summed CK per nucleus
            'ck': ck_positive         # existing binary flag
        })

    # Append to overall results
    all_results.extend(file_results)

    # Plot nuclei centroids colored by CK flag (unchanged)
    df = pd.DataFrame(file_results)
    # Choose one y-flip method; using invert_yaxis is enough
    plt.figure(figsize=(8, 8))
    plt.scatter(df['x'], df['y'], c=df['ck'], cmap='coolwarm', marker='o', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Nuclei Centroids (CK Positivity) - {fname}')
    plt.colorbar(label='CK Positive (1) / Negative (0)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_plot_path = os.path.join(output_plot_dir, fname.replace('.tif', '_plot.png'))
    plt.savefig(output_plot_path, dpi=200)
    plt.close()

    print(f"Finished {file_path} in {time.time() - t_start:.2f}s. "
          f"Saved mask to {mask_out_path} and plot to {output_plot_path}.")

# Save all results to CSV
all_df = pd.DataFrame(all_results)
all_df.to_csv(output_csv_path, index=False)
print(f"Saved all cell data to {output_csv_path}")
