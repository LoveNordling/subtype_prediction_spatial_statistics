import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Read the CSV files
cell_df = pd.read_csv("cellprofiler_extracted_cells_filtered_necrosis.csv")
samples_df = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/samples_labels.csv")
test_df = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/test.csv")

# Ensure 'ID' is string for merging consistency
samples_df['ID'] = samples_df['ID'].astype(str)
test_df['ID'] = test_df['ID'].astype(str)

# Prepare a mapping: patient ID -> sample_names
patient_samples = samples_df[samples_df['ID'].isin(test_df['ID'])].groupby('ID')['sample_name'].apply(list)

# Color map for CK status
ck_colors = {0: "grey", 1: "red"}

# Create output directory
output_dir = "patient_figures_cellprofiler"
os.makedirs(output_dir, exist_ok=True)

for patient_id, sample_list in tqdm(patient_samples.items()):
    n_samples = len(sample_list)
    if n_samples == 0:
        continue  # no samples for this patient

    fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 6), squeeze=False)
    axes = axes[0]  # flatten in case only 1 row

    for i, sample_name in enumerate(sample_list):
        # Select cells for this sample
        sample_cells = cell_df[cell_df['ImageID'] == sample_name]
        if sample_cells.empty:
            axes[i].set_title(f"{sample_name}\n(No data)")
            axes[i].axis("off")
            continue

        # Scatter plot, coloring by CK status
        for ck_val in [0, 1]:
            ck_cells = sample_cells[sample_cells['Classify_CK'] == ck_val]
            axes[i].scatter(
                ck_cells['Location_Center_X'], 
                ck_cells['Location_Center_Y'], 
                s=1, 
                label=f"CK {'+' if ck_val == 1 else '-'}",
                alpha=0.7,
                color=ck_colors[ck_val]
            )

        axes[i].set_title(f"Sample: {sample_name}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].legend()

    fig.suptitle(f"Patient ID: {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, f"{patient_id}.png")
    plt.savefig(out_path)
    plt.close()

print("Done! All figures are saved in the 'patient_figures' folder.")
