import pandas as pd
import tifffile
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
# ----------------- Paths -----------------
csv_path = 'cellpose_extracted_cells.csv'
masks_dir = '/media/love/Love Extern/OUTPUT'
output_csv = 'cellpose_extracted_cells_fitlered_necrosis.csv'
vis_output_dir = "tissue_plots_cellpose"  # Save visualizations here

# ----------------- Helper Functions -----------------

def extract_identifier(filename):
    """Extracts image identifier from filename using regex"""
    match = re.search(r'BOMI2_TIL_.*?\[\d+,\d+,[A-Z]\]_\[\d+,\d+\]', filename)
    return match.group(0) if match else None

def load_mask_dict(mask_dir):
    """Build a dictionary mapping image ID to segmentation mask path"""
    mask_paths = {}
    for fname in os.listdir(mask_dir):
        if fname.endswith('_binary_seg_maps.tif'):
            identifier = extract_identifier(fname)
            if identifier:
                mask_paths[identifier] = os.path.join(mask_dir, fname)
    return mask_paths





def visualize_mask_and_cells(mask, all_cells, kept_cells, image_id, out_path):
    """Plot grayscale mask and scatter cells with color coding:
    - Yellow: All cells
    - Blue: Kept non-cancer (CK negative) cells
    - Red: Kept cancer (CK positive) cells
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask, cmap='gray')

    # All cells
    ax.scatter(all_cells['x'], all_cells['y'],
               s=10, c='yellow', label='All Cells', alpha=0.3)

    # Kept cancer (CK positive) cells
    cancer_cells = kept_cells[kept_cells['ck'] == 1]
    ax.scatter(cancer_cells['x'], cancer_cells['y'],
               s=10, c='red', label='Kept CK+ (Cancer)', alpha=0.8)

    # Kept non-cancer (CK negative) cells
    non_cancer_cells = kept_cells[kept_cells['ck'] == 0]
    ax.scatter(non_cancer_cells['x'], non_cancer_cells['y'],
               s=10, c='blue', label='Kept CK-', alpha=0.8)

    ax.set_title(f"Segmentation + Cells: {image_id}", fontsize=14)
    ax.set_xlim([0, mask.shape[1]])
    ax.set_ylim([mask.shape[0], 0])  # Flip Y to match image
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"Saved visualization: {out_path}")




    
# ----------------- Main Script -----------------

# Load CSV
df = pd.read_csv(csv_path)
otsu_thresh = threshold_otsu(df["ck_cyto_mean_raw"].dropna().to_numpy())
df["CK"] = (df["ck_cyto_mean_raw"] >= otsu_thresh).astype(int)
print(f"Otsu threshold = {otsu_thresh}")

# Extract image identifier
df['ImageID'] = df['filename'].apply(extract_identifier)

# Load mask paths
mask_paths = load_mask_dict(masks_dir)

# Process each image group
filtered_groups = []

for image_id, group in df.groupby('ImageID'):
    if image_id not in mask_paths:
        print(f"Mask not found for {image_id}. Skipping...")
        continue

    mask_path = mask_paths[image_id]
    print(image_id)
    try:
        mask = tifffile.imread(mask_path)

        mask = mask[3,:,:]
    except Exception as e:
        print(f"Failed to read {mask_path}: {e}")
        continue

    keep_rows = []

    for _, row in group.iterrows():
        x = int(round(row['x']))
        y = int(round(row['y']))

        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            if mask[y, x] != 0:
                keep_rows.append(row)
        else:
            keep_rows.append(row)  # Keep out-of-bounds cells

    filtered_group = pd.DataFrame(keep_rows)
    filtered_groups.append(filtered_group)

    # Visualization
    vis_path = os.path.join(vis_output_dir, f"vis_{image_id}.png")
    #visualize_mask_and_cells(mask, group, filtered_group, image_id, vis_path)

# Combine and save filtered cells
result_df = pd.concat(filtered_groups, ignore_index=True)
result_df.to_csv(output_csv, index=False)
print(f"\n✅ Saved filtered cells to:\n{output_csv}")
