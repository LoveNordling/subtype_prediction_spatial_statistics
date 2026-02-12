
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import os
import random

from scipy.stats import pearsonr, spearmanr, chisquare, gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler

import warnings
from collections import Counter
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from skimage.filters import threshold_otsu
import time
import copy

from spatial_metrics import *

from imblearn.over_sampling import SMOTE

from scipy.spatial import cKDTree

import re
# Add near the top of spatial_statistics.py (after imports is fine)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as _mp


from sklearn.exceptions import ConvergenceWarning
import warnings



def _passes_count_qc(sample_cells_df: pd.DataFrame, min_total: int, min_cancer: int, min_stroma: int):
    """
    Return (pass_bool, qc_dict) where qc_dict contains counts for logging/debugging.

    Expects sample_cells_df to have a 'Cancer' column with 0/1.
    """
    if sample_cells_df is None or sample_cells_df.empty:
        return False, {"n_total": 0, "n_cancer": 0, "n_stroma": 0}

    ct = sample_cells_df["Cancer"].to_numpy()
    # Be strict: enforce binary
    ct = np.asarray(ct, dtype=int)

    n_total = int(ct.size)
    n_cancer = int((ct == 1).sum())
    n_stroma = int((ct == 0).sum())

    ok = (n_total >= int(min_total)) and (n_cancer >= int(min_cancer)) and (n_stroma >= int(min_stroma))
    return ok, {"n_total": n_total, "n_cancer": n_cancer, "n_stroma": n_stroma}



def _compute_one_sample(ID, sample_name, label, sample_cells_df):
    """Worker: compute metrics for a single sample and attach identifiers."""
    if sample_cells_df is None or len(sample_cells_df) < 10:
        return None
    metrics = calculate_spatial_metrics_for_sample(sample_cells_df)
    if not metrics:
        return None
    metrics['ID'] = ID
    metrics['sample_name'] = sample_name
    metrics['label'] = label
    return metrics




def load_and_preprocess_all(cellprofiler=False, cellpose=False):
    if cellprofiler:
        print("[INFO] Using cellprofiler file input...")
        cells_data = pd.read_csv("./cellprofiler_extracted_cells_filtered_necrosis.csv")
        cells_data = preprocess_cellproflier_data(cells_data)
    elif cellpose:
        print("[INFO] Using cellpose file input...")
        #cells_data = pd.read_csv("cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1.csv")#
        cells_data = pd.read_csv("./cellpose_extracted_cells_fitlered_necrosis.csv")
        #cells_data = pd.read_csv("./cellpose_extracted_cells.csv")
        cells_data = preprocess_cellpose_data(cells_data)
    else:
        cells_data = pd.read_csv(cells_path)
        cells_data = preprocess_cells(cells_data)
    

    samples_data = pd.read_csv(samples_path)
    samples_data = preprocess_samples(samples_data)

    patient_data_train_processed = preprocess_patients(patients_train)
    patient_data_test_processed = preprocess_patients(patients_test)

    # Merge both train and test patient metadata
    patient_data_all = pd.concat([patient_data_train_processed, patient_data_test_processed], ignore_index=True)

    return cells_data, samples_data, patient_data_all, patient_data_train_processed, patient_data_test_processed


def preprocess_cellpose_data(data):
    """
    Preprocess cells extracted from component analysis file format.
    Standardizes to match structure required by spatial pipeline.
    """

    # Extract sample name from FileName_CK, e.g.:
    # 'BOMI2_TIL_2_Core[1,10,A]_[5091,35249]_component_data_CK.tiff' -> 'BOMI2_TIL_2_[10,A]'
    def extract_sample_name(filename):
        match = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", filename)
        if match:
            return f"{match.group(1)}_[{match.group(2)}]"
        else:
            raise ValueError(f"Could not parse sample name from filename: {filename}")

    # Apply transformations
    data = data.rename(columns={
        "x": "x",
        "y": "y",
        "ck": "Cancer",
        "filename": "file_name"
    })

    data["sample_name"] = data["file_name"].apply(extract_sample_name)
    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["Cancer"] = data["Cancer"].astype(int)

    # Drop NAs
    data = data.dropna(subset=["x", "y", "Cancer", "sample_name"])

    print(f"[INFO] Loaded {len(data)} cells from component file.")
    return data[["x", "y", "Cancer", "sample_name", "ck_cyto_mean_raw"]]

def preprocess_external_cells(bomi1_cells: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize BOMI1_cells_all.csv to the columns expected by the spatial pipeline.
    - x, y: coordinates in microns (already correct to use)
    - Cancer: 1 if 'Class' == 'Neoplastic', else 0
    - sample_name: already present in your file
    """
    df = bomi1_cells.copy()
    if not {"CentroidX_um", "CentroidY_um", "Class", "sample_name"}.issubset(df.columns):
        missing = {"CentroidX_um", "CentroidY_um", "Class", "sample_name"} - set(df.columns)
        raise ValueError(f"External cells file missing columns: {missing}")

    df = df.rename(columns={"CentroidX_um": "x", "CentroidY_um": "y"})
    # Make robust to capitalization/whitespace
    df["Cancer"] = (df["Class"].astype(str).str.strip().str.lower() == "neoplastic").astype(int)

    # Keep only the required columns
    df = df[["x", "y", "Cancer", "sample_name"]].dropna()
    # Ensure numeric x,y
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    return df

def split_external_meta(meta_df: pd.DataFrame):
    """
    From clinical_data_and_cores_adenovsSQ.csv create:
      - patients_df: one row per patient with ['ID', 'label']
      - samples_df: mapping of samples to patients ['ID', 'sample_name']
    """
    needed = {"ID", "sample_name", "label"}
    if not needed.issubset(meta_df.columns):
        missing = needed - set(meta_df.columns)
        raise ValueError(f"External meta file missing columns: {missing}")

    # One label per patient
    patients_df = meta_df[["ID", "label"]].drop_duplicates(subset=["ID"]).copy()

    # Sample map has NO label (avoid merge collision)
    samples_df = meta_df[["ID", "sample_name"]].drop_duplicates(subset=["ID", "sample_name"]).copy()
    return patients_df, samples_df

def preprocess_cellproflier_data(data):
    """
    Preprocess cells extracted from component analysis file format.
    Standardizes to match structure required by spatial pipeline.
    """

    # Extract sample name from FileName_CK, e.g.:
    # 'BOMI2_TIL_2_Core[1,10,A]_[5091,35249]_component_data_CK.tiff' -> 'BOMI2_TIL_2_[10,A]'
    def extract_sample_name(filename):
        match = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", filename)
        if match:
            return f"{match.group(1)}_[{match.group(2)}]"
        else:
            raise ValueError(f"Could not parse sample name from filename: {filename}")

    # Apply transformations
    data = data.rename(columns={
        "Location_Center_X": "x",
        "Location_Center_Y": "y",
        "CK": "Cancer",
        "FileName_CK": "file_name"
    })

    data["sample_name"] = data["file_name"].apply(extract_sample_name)
    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data["Cancer"] = data["Cancer"].astype(int)

    # Drop NAs
    data = data.dropna(subset=["x", "y", "Cancer", "sample_name"])

    print(f"[INFO] Loaded {len(data)} cells from component file.")
    return data[["x", "y", "Cancer", "sample_name", "Intensity_MeanIntensity_CK"]]

def preprocess_cells(data):
    #data["ID"] = data["ID"].map(lambda x: int(x.replace("Lung # ", "")))
    #data = data.drop('Cell ID', axis=1)
    data["Tissue Category"] = data["Tissue Category"].map(lambda x: 1 if x == "Tumor" else 0)
    
    data["Cell X Position"] = data["Cell X Position"].map(lambda x: float(x) )
    data["Cell Y Position"] = data["Cell Y Position"].map(lambda x: float(x) )
    data = data.rename(columns={
                                #"Nucleus Opal 620 Mean (Normalized Counts, Total Weighting)": "FoxP3",
                                "Cytoplasm Opal 520 Mean (Normalized Counts, Total Weighting)": "CD4",
                                "Cytoplasm Opal 540 Mean (Normalized Counts, Total Weighting)": "CD20",
                                "Cytoplasm Opal 570 Mean (Normalized Counts, Total Weighting)": "CD8",
                                "Cytoplasm Opal 650 Mean (Normalized Counts, Total Weighting)": "CD45RO",
                                "Cytoplasm Opal 690 Mean (Normalized Counts, Total Weighting)": "PanCK",
                                "Nucleus DAPI Mean (Normalized Counts, Total Weighting)": "DAPI",
                                "Cytoplasm Autofluorescence Mean (Normalized Counts, Total Weighting)": "Autofluorescence",
                                "Nucleus Area (pixels)": "Nucleus Area"
                                })
    column_names = ["Cancer", "CD4", "CD20", "CD8", "PanCK", "DAPI", "Nucleus Area", "Nucleus Compactness",
                    "Nucleus Axis Ratio", "CK", "CD4_Single", "CD4_Treg", "CD8_Single", "CD8_Treg", "B_cells", "CKSingle", "Stroma_other"]

    print("unique panck values: ", data["PanCK"].unique())
    
    data.rename(columns = {"Sample Name": "sample_name", 'Tissue Category':'Cancer', 'Cell X Position': 'x', 'Cell Y Position': 'y'}, inplace = True)

    for column in tqdm(column_names):
        data[column] = data[column].astype(float)
        #data[column] = data[column].map(lambda x: float(x))

    

    data["CK_Single"] = ((data['CK'] == 1) & (data['CD4_Single'] == 0) & (data['CD4_Treg'] == 0)
                         & (data['CD8_Single'] == 0) & (data['CD8_Treg'] == 0)
                         & (data['B_cells'] == 0)
                         ).astype(int)
        
    data = data.drop('CD45RO', axis=1)
    data = data.drop('Lab ID', axis=1)
    data = data.drop('inForm 2.4.6781.17769', axis=1)
    print("num cells before dropping na", len(data))
    
    data = data[data.notna().all(axis=1)]
    print(data.isna().any())
    print("num cells after dropping na", len(data))
    print(data.columns)
    ### Use CK positivity as cancer marker
    
    data["Cancer"] = data["CK"]
    return data


def _prefix_ids(df: pd.DataFrame, prefix: str, id_col: str = "ID") -> pd.DataFrame:
    """
    Return a copy of df where ID is coerced to string and prefixed.
    Useful to avoid accidental ID collisions between cohorts.
    """
    if id_col not in df.columns:
        raise ValueError(f"Expected column '{id_col}' in dataframe. Columns: {list(df.columns)}")
    out = df.copy()
    out[id_col] = out[id_col].astype(str).map(lambda x: f"{prefix}{x}")
    return out


def _read_split_ids(csv_path: str) -> set:
    """
    Read a split CSV and return a set of patient IDs.

    Supports a few common column names to avoid brittle coupling.
    """
    df = pd.read_csv(csv_path)
    for col in ["ID", "ID or PAD_year", "ID_or_PAD_year", "patient_id", "Patient ID"]:
        if col in df.columns:
            return set(df[col].dropna().astype(str))
    raise ValueError(
        f"Could not find an ID column in {csv_path}. Columns: {list(df.columns)}"
    )


def _subset_by_ids(df: pd.DataFrame, ids: set) -> pd.DataFrame:
    """Subset df on df['ID'] using a string-based match for robustness."""
    if "ID" not in df.columns:
        raise ValueError(f"Expected column 'ID' in df. Columns: {list(df.columns)}")
    ids_str = {str(x) for x in ids}
    return df[df["ID"].astype(str).isin(ids_str)].copy()


def _save_predictions(path: str, test_df: pd.DataFrame, eval_dict: dict):
    out = pd.DataFrame({
        "ID": test_df["ID"].astype(str),
        "y_true": test_df["label"].values,
        "y_prob": eval_dict["y_probs"],
        "y_pred": eval_dict["y_pred"],
    })
    out.to_csv(path, index=False)
    print(f"Saved predictions to {path}")


def panck_min_valley(data: pd.DataFrame, column: str = "PanCK", grid: int = 2048):
    s = pd.to_numeric(data[column], errors="coerce")
    x = s.dropna().to_numpy(float)
    assert x.size >= 10, "Need at least 10 values"

    otsu = float(threshold_otsu(x))

    # KDE over all values
    xs = np.linspace(x.min(), x.max(), grid)
    kde_all = gaussian_kde(x)
    ys = kde_all(xs)

    # Peaks on each side of Otsu
    left_mask = xs < otsu
    right_mask = xs > otsu
    if left_mask.sum() < 3 or right_mask.sum() < 3:
        cut = otsu
    else:
        peak1 = xs[left_mask][np.argmax(ys[left_mask])]
        peak2 = xs[right_mask][np.argmax(ys[right_mask])]
        between = (xs > peak1) & (xs < peak2)
        cut = otsu if not between.any() else float(xs[between][np.argmin(ys[between])])

    binary = (s >= cut).fillna(0).astype("uint8")
    return cut, binary

def preprocess_patients(data):
    data = data.rename(columns={"ID or PAD_year":"ID", "Tumor_type":"Histology", "Sex":"Gender" , "Event_last_followup": "Dead/Alive"})
    t = {"Adenocarcinoma": "LUAD", "Squamous cell carcinoma": "LUSC", "Other":"Other"}
    data["Tumor_type_code"] = data["Histology"].map(lambda x: t[x])
    data["Smoking"] = data["Smoking"].map(lambda x: 0 if x == "Never-smoker" else 1)
    data["LUAD"] = (data["Tumor_type_code"] == "LUAD").astype(int)
    data["Gender"] = (data["Gender"] == "Male").astype(int)
    #data["Performance status (WHO)"] = data["Performance status (WHO)"].map(lambda x: 1 if x > 0 else 0)
    stage_dict = {"Ia": 0, "Ib": 1, "IIa": 2, "IIb": 3, "IIIa": 4, "IIIb": 5, "IV": 6}
    data["Stage"] = data["Stage (7th ed.)"].map(lambda x: stage_dict[x])
    
    return data[["ID", "LUAD", "Age", "Gender", "Smoking", "Stage", "Performance status (WHO)", "Follow-up (days)", "label", "Tumor_type_code"]]

def preprocess_samples(data):
    data["sample_name"] = data["sample_name"].map(lambda x: x[:x.rfind("_")])
    data["sample_name"] = data["sample_name"].map(lambda x: x.replace("Core[1,", "["))
    return data



def calculate_spatial_metrics_for_sample(cells_df):
    
    #Calculate all spatial metrics for a single sample
    
    if len(cells_df) < 10:  # Skip samples with too few cells
        return None
    
    coords = cells_df[['x', 'y']].values
    cell_types = cells_df['Cancer'].values
    
    metrics = {}
    metrics['cell_count'] = len(cells_df)
    
    # Calculate Ripley's K at different radii
    
    ripley_metrics = calculate_ripley_l(coords, cell_types)
    metrics.update(ripley_metrics)
    # Calculate proportions
    #metrics['cancer_proportion'] = np.mean(cell_types)
    
    
    # Add newly implemented metrics

    # 0. Carolinas suggestion - average minimal distance to other group
    mindist_metrics = calculate_bidirectional_min_distance(coords, cell_types)
    metrics.update(mindist_metrics)
    
    # 1. Newman's Assortativity
    metrics['newmans_assortativity'] = calculate_newmans_assortativity(coords, cell_types, radius=50)
    
    # 2. Centrality Scores
    centrality_metrics = calculate_centrality_scores(coords, cell_types)
    
    metrics.update(centrality_metrics)
    
    #Two types of objects
    # 3. Cluster Co-occurrence Ratio
    cluster_metrics = calculate_cluster_cooccurrence_ratio(coords, cell_types)
    metrics.update(cluster_metrics)
    
    # 4. Neighborhood Enrichment Test
    enrichment_metrics = calculate_neighborhood_enrichment_test(coords, cell_types)
    metrics.update(enrichment_metrics)
    
    # 5. Object-Object Correlation Analysis
    correlation_metrics = calculate_objectobject_correlation(coords, cell_types)
    metrics.update(correlation_metrics)
    
    
    return metrics



def get_or_compute_all_metrics(all_patients_df, samples_df, cells_df, recompute=False, cellprofiler=False, cellpose=False):
    #filename="spatial_metrics_cellpose.csv"):#filename="spatial_metrics_merged.csv"):#"spatial_metrics_cellprofiler.csv"):
    if cellprofiler:
        filename = "spatial_metrics_cellprofiler.csv"
    elif cellpose:
        filename = "spatial_metrics_cellpose.csv"
    else:
        filename = "spatial_metrics_merged.csv"
    
    if recompute or not os.path.exists(filename):
        print("Computing spatial metrics for all samples...")
        all_metrics = filter_and_analyze_data(all_patients_df, samples_df, cells_df)
        all_metrics.to_csv(filename, index=False)
    else:
        print(f"Loading precomputed spatial metrics from {filename}...")
        all_metrics = pd.read_csv(filename)
    return all_metrics


# Main analysis script

#def filter_and_analyze_data(patients_df, samples_df, cells_df, n_jobs=None):

def filter_and_analyze_data(
    patients_df,
    samples_df,
    cells_df,
    n_jobs=None,
    # NEW: fixed-count QC thresholds
    min_total_cells: int = 1500,
    min_cancer_cells: int = 30,
    min_stroma_cells: int = 30,
    # NEW: logging
    verbose_qc: bool = True,
):

    """
    Filter data to training set and analyze spatial patterns — parallel across samples.
    """
    # Default to all physical cores (sane default on Linux boxes)
    if n_jobs is None:
        n_jobs = _mp.cpu_count()

    # Merge patient info with samples
    patients_samples = pd.merge(patients_df, samples_df, on='ID')

    # Pre-split cell table by sample to avoid repeated filtering in workers
    cell_groups = {
        s: g[['x', 'y', 'Cancer']].copy()
        for s, g in cells_df.groupby('sample_name', sort=False)
    }

    # Build task list (skip samples with no cells)

    # Build task list with QC filtering
    tasks = []
    qc_rows = []  # optional QC report

    for _, row in patients_samples.iterrows():
        sname = row["sample_name"]
        if sname not in cell_groups:
            if verbose_qc:
                qc_rows.append({
                    "ID": row["ID"],
                    "sample_name": sname,
                    "label": row["label"],
                    "qc_pass": False,
                    "qc_reason": "no_cells_for_sample",
                    "n_total": 0,
                    "n_cancer": 0,
                    "n_stroma": 0,
                })
            continue

        sample_cells = cell_groups[sname]

        ok, counts = _passes_count_qc(
            sample_cells,
            min_total=min_total_cells,
            min_cancer=min_cancer_cells,
            min_stroma=min_stroma_cells,
        )

        if not ok:
            if verbose_qc:
                qc_rows.append({
                    "ID": row["ID"],
                    "sample_name": sname,
                    "label": row["label"],
                    "qc_pass": False,
                    "qc_reason": "below_cell_count_thresholds",
                    **counts,
                })
            continue

        if verbose_qc:
            qc_rows.append({
                "ID": row["ID"],
                "sample_name": sname,
                "label": row["label"],
                "qc_pass": True,
                "qc_reason": "",
                **counts,
            })

        tasks.append((row["ID"], sname, row["label"], sample_cells))

        
    results = []
    if not tasks:
        return pd.DataFrame()

    # Multiprocessing with a nice progress bar
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_compute_one_sample, *t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
            res = f.result()
            if res is not None:
                results.append(res)



    if verbose_qc:
        qc_df = pd.DataFrame(qc_rows)
        if not qc_df.empty:
            kept = int(qc_df["qc_pass"].sum())
            total = int(len(qc_df))
            print(f"[QC] Kept {kept}/{total} cores after count thresholds "
                  f"(min_total={min_total_cells}, min_cancer={min_cancer_cells}, min_stroma={min_stroma_cells})")

            # Optional: save QC table for reproducibility
            qc_df.to_csv("qc_core_counts.csv", index=False)
            print("[QC] Saved qc_core_counts.csv")

    return pd.DataFrame(results)



def run_correlation_analysis(results_df):
    """
    Run correlation analysis between spatial metrics and cancer type
    """
    # Prepare for correlation analysis

    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['ID', "label", 'cell_count', 'sample_name']]
    
    # Calculate correlations
    correlations = []
    for col in numeric_cols:

        try:
            corr, p_val = spearmanr(results_df[col], results_df['label'])
            correlations.append({
                'metric': col,
                'correlation': corr,
                'p_value': p_val
            })
        except:
            pass
    
    corr_df = pd.DataFrame(correlations)
    return corr_df.sort_values('p_value')


def analyze_feature_correlations(results_df, output_file="feature_correlations.png", 
                                csv_output="feature_correlations.csv"):
    """
    Analyze and visualize correlations between spatial metrics features.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing the spatial metrics results
    output_file : str
        Path to save the correlation heatmap visualization
    csv_output : str
        Path to save the correlation matrix as CSV
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Select only numeric columns, excluding IDs and target variables
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['ID', 'cell_count', 'sample_name']]
    
    # Remove any columns with NaN values
    data_for_corr = results_df[numeric_cols].dropna(axis=1)
    numeric_cols = data_for_corr.columns.tolist()
    
    # Calculate the correlation matrix
    corr_matrix = data_for_corr.corr(method='spearman').round(2)
    
    # Fix any asymmetry issues
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    
    # Create the visualization
    plt.figure(figsize=(20, 16))
    
    # Create standard heatmap
    g = sns.heatmap(
        corr_matrix, 
        cmap='coolwarm', 
        center=0,
        annot=(len(numeric_cols) < 30),  # Only show annotations if fewer than 30 features
        fmt='.2f', 
        linewidths=0.5,
        cbar_kws={"shrink": .8},
        vmin=-1, vmax=1
    )
    
    # Add title and labels
    plt.title('Correlation Matrix of Spatial Metrics', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Correlation matrix visualization saved to {output_file}")
    
    # Save correlation matrix to CSV
    corr_matrix.to_csv(csv_output)
    print(f"Correlation matrix saved to {csv_output}")
    
    return corr_matrix



def _pixel_area_to_mm2(area_pixels: pd.Series, pixel_size_um: float) -> pd.Series:
    pixel_size_mm = float(pixel_size_um) * 1e-3
    return pd.to_numeric(area_pixels, errors="coerce") * (pixel_size_mm * pixel_size_mm)


def _build_external_area_map(bomi1_area_csv: str, pixel_size_um: float = 0.5) -> dict:
    df = pd.read_csv(bomi1_area_csv)
    needed = {"sample_name", "tissue_area_pixels"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"External area CSV missing columns: {missing}. Columns: {list(df.columns)}")

    df = df.copy()
    df["sample_name"] = df["sample_name"].astype(str)
    df["area_mm2"] = _pixel_area_to_mm2(df["tissue_area_pixels"], pixel_size_um=pixel_size_um)
    

    # In case of duplicates, aggregate safely
    area_map = df.groupby("sample_name", sort=False)["area_mm2"].mean().to_dict()
    return area_map


def _build_internal_area_map(samples_df: pd.DataFrame, pixel_size_um: float = 0.5) -> dict:
    if "sample_name" not in samples_df.columns or "total_area" not in samples_df.columns:
        raise ValueError(f"Internal samples_df missing sample_name/total_area. Columns: {list(samples_df.columns)}")

    df = samples_df.copy()
    df["sample_name"] = df["sample_name"].astype(str)
    df["total_area"] = pd.to_numeric(df["total_area"], errors="coerce")
    df["area_mm2"] =  _pixel_area_to_mm2(df["mask_area_pixels"], pixel_size_um=pixel_size_um)
    
    area_map = df.groupby("sample_name", sort=False)["area_mm2"].mean().to_dict()
    return area_map


def _filter_metrics_by_area(
    metrics_df: pd.DataFrame,
    area_map: dict,
    min_area_mm2: float,
    cohort_name: str,
    keep_missing: bool = True,
) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    if "sample_name" not in metrics_df.columns:
        raise ValueError(f"{cohort_name} metrics_df has no 'sample_name' column. Columns: {list(metrics_df.columns)}")

    out = metrics_df.copy()
    out["__area_mm2"] = out["sample_name"].astype(str).map(area_map)

    n_missing = int(out["__area_mm2"].isna().sum())
    if n_missing > 0:
        print(f"[WARN] {cohort_name}: {n_missing}/{len(out)} samples missing area. "
              f"{'Keeping' if keep_missing else 'Dropping'} missing-area rows.")

    if keep_missing:
        keep_mask = out["__area_mm2"].isna() | (out["__area_mm2"] >= float(min_area_mm2))
    else:
        keep_mask = (out["__area_mm2"] >= float(min_area_mm2))

    before = len(out)
    out = out.loc[keep_mask].copy()
    after = len(out)

    print(f"[INFO] {cohort_name}: area filter >= {min_area_mm2} mm^2 kept {after}/{before} cores/samples.")
    return out




def visualize_spatial_patterns(cells_df, sample_name, patient_id, metric_name=None, metric_value=None):
    """
    Create visualization of cancer vs non-cancer cells for a specific sample
    """
    # Filter for specific sample
    sample_cells = cells_df[(cells_df['ID'] == patient_id) & 
                            (cells_df['sample_name'] == sample_name)]
    
    if len(sample_cells) == 0:
        return None
    
    # Create plot
    plt.figure(figsize=(10, 10))
    
    # Plot cells
    cancer_cells = sample_cells[sample_cells['Cancer'] == 1]
    non_cancer_cells = sample_cells[sample_cells['Cancer'] == 0]
    
    plt.scatter(non_cancer_cells['x'], non_cancer_cells['y'], 
                c='lightblue', alpha=0.7, label='Non-cancer')
    plt.scatter(cancer_cells['x'], cancer_cells['y'], 
                c='salmon', alpha=0.7, label='Cancer')
    
    # Add title and labels
    title = f"Patient {patient_id}, Sample {sample_name}"
    if metric_name is not None and metric_value is not None:
        title += f"\n{metric_name}: {metric_value:.3f}"
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    
    return plt


def merge_samples(data):
    cols_to_average = [col for col in data.columns 
                   if col not in ['ID', 'cell_count', 'sample_name']]

    # Group by ID and calculate weighted average
    
    data = data.groupby('ID').apply(lambda group: pd.Series({
        # Preserve the first value for non-averaged columns
        'label': group['label'].iloc[0],
    
        # Calculate weighted average for specified columns
        **{col: (group[col] * group['cell_count']).sum() / group['cell_count'].sum() 
           for col in cols_to_average}
    })).reset_index()
    return data


def visualize_per_cell_centrality(sample_cells, radius=50):
    """
    Compute per-cell degree centrality and visualize it.
    """

    coords = sample_cells[['x', 'y']].values
    if len(coords) < 3:
        print("Not enough cells to compute centrality.")
        return

    # Build KDTree
    tree = cKDTree(coords)

    # Find neighbors within the radius
    neighbors_list = tree.query_ball_point(coords, r=radius)

    # Degree centrality: number of neighbors (excluding self)
    centrality_scores = np.array([len(neighs) - 1 for neighs in neighbors_list])

    # Normalize scores
    if centrality_scores.max() > 0:
        centrality_scores = centrality_scores / centrality_scores.max()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(sample_cells['x'], sample_cells['y'], c=centrality_scores, cmap='viridis', s=10)
    plt.colorbar(sc, label="Normalized Degree Centrality")
    plt.title(f"Per-Cell Centrality Map\nSample: {sample_cells['sample_name'].iloc[0]}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal')
    plt.show()


cells_path = "BOMI2_all_cells_TIL.csv"
#samples_path = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"
samples_path = "../BOMI2_TIL_masks/samples.csv"
samples_path = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1.csv"

patients_train = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/train_val.csv")
patients_test = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/test.csv")

"""
patients_train = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_survival_prediction/static_split/train_val.csv")
patients_test = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_survival_prediction/static_split/test.csv")
"""

#cells_data = pd.read_csv(cells_path)
#cells_data = preprocess_cells(cells_data)

#patient_data = preprocess_patients(patient_data)

"""
sample = cells_data[cells_data['sample_name'].str.contains("1_\[1,E")]
example_sample_name = sample["sample_name"].iloc[0]
sample_cells = cells_data[cells_data['sample_name'] == example_sample_name]

# Visualize per-cell centrality
visualize_per_cell_centrality(sample_cells)
"""


def run_classification_cv(results_df, n_folds=5, hyperparameter_tuning=True, return_model=True):
    """Run cross-validation classification with correct model-matched feature selection."""

    # Prepare data
    numeric_cols = [col for col in results_df.select_dtypes(include=[np.number]).columns if col not in ['ID', 'label']]
    X = results_df[numeric_cols].values
    y = results_df['label'].values
    print("class imbalance:", y.sum()/len(y))
    #smote = SMOTE(random_state=42)
    #X, y = smote.fit_resample(X, y)
    #print("class imbalance balanced:", y.sum()/len(y))
    # Set up cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    if hyperparameter_tuning:
        print("Performing hyperparameter tuning...")
        start_time = time.time()

        models = {
            'RandomForest': (RandomForestClassifier(random_state=42), {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 3, 5, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2', None]
            }),
            'GradientBoosting': (GradientBoostingClassifier(random_state=42), {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5],
                'model__min_samples_leaf': [1, 2]
            }),
            'SVM': (SVC(probability=True, random_state=42, max_iter=10000), {
                'model__C': [0.1, 0.1, 1, 10, 100],
                'model__gamma': ['scale', 'auto', 0.1, 0.01],
                'model__kernel': ['rbf', 'linear']
            }),
            'LogisticRegression': (LogisticRegression(max_iter=10000, random_state=42), {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l2'],
                'model__solver': ['saga']
            }),

        }

        best_pipeline, best_model_name, best_score, best_params = None, None, -np.inf, None
        use_randomized = len(X) > 200

        for name, (model, params) in models.items():
            print(f"Tuning {name}...")

            
            if name == 'RandomForest':
                selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
            elif name == 'GradientBoosting':
                selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
            elif name == 'SVM':
                selector = SelectFromModel(LinearSVC(penalty='l1', dual=False, C=1.0, max_iter=10000, random_state=42))
            elif name == 'LogisticRegression':
                selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', C=1.0))
            else:
                selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))  # fallback

            # Build pipeline
            pipeline = Pipeline([
                ('scaler', MinMaxScaler()),#StandardScaler()),#RobustScaler()),#),
                #('smote', SMOTE(random_state=42)),  # Add SMOTE here
                ('oversample', RandomOverSampler(random_state=42)),
                ('selector', selector),
                ('model', model)
            ])

            search = RandomizedSearchCV(pipeline, params, n_iter=20, scoring='roc_auc', cv=kf, n_jobs=-1, random_state=42) if use_randomized \
                else GridSearchCV(pipeline, params, scoring='roc_auc', cv=kf, n_jobs=-1)

            try:
                search.fit(X, y)
                if search.best_score_ > best_score:
                    best_pipeline, best_model_name, best_score, best_params = search.best_estimator_, name, search.best_score_, search.best_params_
                print(f"  Best {name} params: {search.best_params_} | AUC: {search.best_score_:.4f}")
            except Exception as e:
                print(f"  Skipping {name} due to error: {e}")

        if best_pipeline is None:
            print("Tuning failed for all models. Using default RandomForest.")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
                ('model', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            best_pipeline = pipeline
            best_model_name = 'RandomForest (default)'
            best_params = {'n_estimators': 100}

        print(f"Hyperparameter tuning completed in {time.time() - start_time:.2f} seconds.")
        print(f"Selected Model: {best_model_name}")
    else:
        print("Skipping hyperparameter tuning. Using default RandomForest.")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        best_pipeline = pipeline
        best_model_name = 'RandomForest (default)'
        best_params = {'n_estimators': 100}

    # Cross-validation evaluation
    aucs, accuracies = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_pipeline.fit(X_train, y_train)
        y_probs = best_pipeline.predict_proba(X_test)[:, 1]
        y_pred = best_pipeline.predict(X_test)

        aucs.append(roc_auc_score(y_test, y_probs))
        accuracies.append(np.mean(y_pred == y_test))

    # === Final feature importance calculation ===
    print("Calculating final feature importances using permutation importance...")

    # Refit on the full dataset (optional: or a fresh train/test split)
    train_idx, test_idx = next(kf.split(X))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_pipeline.fit(X_train, y_train)

    perm_importance = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')

    feature_importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)

    results = (np.mean(aucs), np.std(aucs), np.mean(accuracies), np.std(accuracies), feature_importance_df, best_model_name)

    if return_model:
        return (*results, best_pipeline, None, best_params)
    else:
        return results


def evaluate_on_test_set(results_df_train, results_df_test, pipeline, feature_names=None):
    """Evaluate pipeline on provided training and test results DataFrames."""

    # 1. Extract numeric feature columns (excluding ID and label)
    numeric_cols = [col for col in results_df_train.select_dtypes(include=[np.number]).columns if col not in ['ID', 'label']]
    
    X_train = results_df_train[numeric_cols].values
    y_train = results_df_train['label'].values

    X_test = results_df_test[numeric_cols].values
    y_test = results_df_test['label'].values

    # 2. Fit the full pipeline on the training set
    pipeline.fit(X_train, y_train)

    # 3. Predict on test set
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # 4. Metrics
    auc = roc_auc_score(y_test, y_probs)
    accuracy = np.mean(y_pred == y_test)

    # 5. Feature importance via permutation
    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')

    if feature_names is None:
        feature_names = numeric_cols

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': perm.importances_mean
    }).sort_values('importance', ascending=False)

    # 6. Return consistent results
    return {
        'auc': auc,
        'accuracy': accuracy,
        'feature_importance': feature_importance,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_probs': y_probs
    }


def evaluate_external_cohort(
    external_cells_path: str = "BOMI1_cells_all.csv",
    external_meta_path: str = "BOMI1_clinical_data_LUADvsSqCC.csv",
    external_split_dir: str = "/home/love/multiplex_dataset/lung_cancer_BOMI1_dataset/HE_dataset/binary_subtype_prediction_ACvsSqCC/static_split/",
    external_train_csv: str = None,
    external_test_csv: str = None,
    recompute_internal_metrics: bool = False,
    cellprofiler: bool = False,
    cellpose: bool = True,
    # NEW (area filtering)
    external_area_csv: str = "../BOMI1_TMA/BOMI1_tissue_area_per_core.csv",
    min_core_area_mm2: float = 0.5,
    pixel_size_um: float = 0.5,
    filter_internal_cores: bool = False,
    filter_external_cores: bool = False,
):
    """
    Runs symmetric internal/external experiments:

    1. CV on internal train
    2. Train internal train, test internal test
    3. Train internal train, test external test

    4. CV on external train
    5. Train external train, test external test
    6. Train external train, test internal test

    7. Train combined (internal train + external train), test internal test
    8. Train combined (internal train + external train), test external test
    """

    # Resolve external split files
    if external_train_csv is None:
        external_train_csv = os.path.join(external_split_dir, "train_val.csv")
    if external_test_csv is None:
        external_test_csv = os.path.join(external_split_dir, "test.csv")

    # === Load internal cohort and compute/merge metrics (unchanged) ===
    cells_data, samples_data, all_patients, patients_train, patients_test = load_and_preprocess_all(cellprofiler, cellpose)
    all_results = get_or_compute_all_metrics(all_patients, samples_data, cells_data, recompute_internal_metrics, cellprofiler, cellpose)


    if (min_core_area_mm2 is not None) and filter_internal_cores:
        internal_area_map = _build_internal_area_map(samples_data)
        
        all_results = _filter_metrics_by_area(
            all_results,
            internal_area_map,
            min_area_mm2=min_core_area_mm2,
            cohort_name="INTERNAL/BOMI2",
            keep_missing=True,
        )
        all_results.drop(columns=["__area_mm2"], inplace=True)

    
    all_results_merged = merge_samples(all_results)

    for col in ["cancer_ripley_L_0.0", "stroma_ripley_L_0.0"]:
        all_results_merged.drop(columns=col, errors="ignore", inplace=True)

    train_ids_internal = set(patients_train["ID"].astype(str))
    test_ids_internal = set(patients_test["ID"].astype(str))

    results_df_train_int = _subset_by_ids(all_results_merged, train_ids_internal)
    results_df_test_int  = _subset_by_ids(all_results_merged, test_ids_internal)

    print(len(train_ids_internal), len(test_ids_internal))
    print(len(all_results_merged))
    print(len(results_df_train_int), len(results_df_test_int))

    
    # === Load external cohort, compute/merge metrics (mostly unchanged) ===
    print("Loading external cohort...")
    ext_cells_raw = pd.read_csv(external_cells_path)
    ext_meta_raw  = pd.read_csv(external_meta_path)

    ext_cells = preprocess_external_cells(ext_cells_raw)
    ext_patients_df, ext_samples_df = split_external_meta(ext_meta_raw)

    ext_cells[["x", "y"]] = ext_cells[["x", "y"]] * 2
    
    metrics_file = "external_spatial_metrics_BOMI1.csv"
    if os.path.exists(metrics_file):
        print(f"Loading precomputed external metrics from {metrics_file}")
        ext_results_per_sample = pd.read_csv(metrics_file)
    else:
        print("Computing spatial metrics for external cohort...")
        ext_results_per_sample = filter_and_analyze_data(ext_patients_df, ext_samples_df, ext_cells)
        ext_results_per_sample.to_csv(metrics_file, index=False)
        print(f"Saved computed external metrics to {metrics_file}")

    if ext_results_per_sample.empty:
        raise RuntimeError("No external samples produced metrics. Check sample_name matching and Cancer mapping.")


    if (min_core_area_mm2 is not None) and filter_external_cores:
        print("Filtering minimum cores")
        print(min_core_area_mm2)
        external_area_map = _build_external_area_map(external_area_csv, pixel_size_um=pixel_size_um)
        ext_results_per_sample = _filter_metrics_by_area(
            ext_results_per_sample,
            external_area_map,
            min_area_mm2=min_core_area_mm2,
            cohort_name="EXTERNAL/BOMI1",
            keep_missing=True,
        )
        ext_results_per_sample.drop(columns=["__area_mm2"], inplace=True)

    
    ext_results_merged = merge_samples(ext_results_per_sample)
    for col in ["cancer_ripley_L_0.0", "stroma_ripley_L_0.0"]:
        ext_results_merged.drop(columns=col, errors="ignore", inplace=True)

    # === External static split ===
    ext_train_ids = _read_split_ids(external_train_csv)
    ext_test_ids  = _read_split_ids(external_test_csv)

    results_df_train_ext = _subset_by_ids(ext_results_merged, ext_train_ids)
    results_df_test_ext  = _subset_by_ids(ext_results_merged, ext_test_ids)
    print(len(ext_train_ids), len(ext_test_ids))
    print(len(ext_results_merged))
    print(len(results_df_train_ext), len(results_df_test_ext))
    
    if results_df_train_ext.empty or results_df_test_ext.empty:
        raise RuntimeError(
            "External split produced empty train or test set. "
            f"Train rows: {len(results_df_train_ext)}, Test rows: {len(results_df_test_ext)}. "
            "Verify IDs in split CSVs match IDs in external meta/metrics."
        )

    # === EXPERIMENTS ===

    # 1) CV on internal train
    print("\n[1] CV on INTERNAL train...")
    auc_i, std_auc_i, acc_i, std_acc_i, fi_i, model_name_i, pipeline_i, _, best_params_i = run_classification_cv(
        results_df_train_int, return_model=True
    )
    print(f"[INTERNAL CV] {model_name_i} AUC: {auc_i:.3f} ± {std_auc_i:.3f}, Acc: {acc_i:.3f} ± {std_acc_i:.3f}")

    # 2) Train internal train, test internal test
    print("\n[2] Train INTERNAL train, test INTERNAL test...")
    eval_i2i = evaluate_on_test_set(results_df_train_int, results_df_test_int, pipeline_i)
    print(f"[INTERNAL→INTERNAL] AUC: {eval_i2i['auc']:.3f}, Acc: {eval_i2i['accuracy']:.3f}")
    _save_predictions("pred_internal_train_to_internal_test.csv", results_df_test_int, eval_i2i)

    # 3) Train internal train, test external test
    print("\n[3] Train INTERNAL train, test EXTERNAL test...")
    eval_i2e = evaluate_on_test_set(results_df_train_int, results_df_test_ext, pipeline_i)
    print(f"[INTERNAL→EXTERNAL] AUC: {eval_i2e['auc']:.3f}, Acc: {eval_i2e['accuracy']:.3f}")
    _save_predictions("pred_internal_train_to_external_test.csv", results_df_test_ext, eval_i2e)

    # 4) CV on external train
    print("\n[4] CV on EXTERNAL train...")
    auc_e, std_auc_e, acc_e, std_acc_e, fi_e, model_name_e, pipeline_e, _, best_params_e = run_classification_cv(
        results_df_train_ext, return_model=True
    )
    print(f"[EXTERNAL CV] {model_name_e} AUC: {auc_e:.3f} ± {std_auc_e:.3f}, Acc: {acc_e:.3f} ± {std_acc_e:.3f}")

    # 5) Train external train, test external test
    print("\n[5] Train EXTERNAL train, test EXTERNAL test...")
    eval_e2e = evaluate_on_test_set(results_df_train_ext, results_df_test_ext, pipeline_e)
    print(f"[EXTERNAL→EXTERNAL] AUC: {eval_e2e['auc']:.3f}, Acc: {eval_e2e['accuracy']:.3f}")
    _save_predictions("pred_external_train_to_external_test.csv", results_df_test_ext, eval_e2e)

    # 6) Train external train, test internal test
    print("\n[6] Train EXTERNAL train, test INTERNAL test...")
    eval_e2i = evaluate_on_test_set(results_df_train_ext, results_df_test_int, pipeline_e)
    print(f"[EXTERNAL→INTERNAL] AUC: {eval_e2i['auc']:.3f}, Acc: {eval_e2i['accuracy']:.3f}")
    _save_predictions("pred_external_train_to_internal_test.csv", results_df_test_int, eval_e2i)

    # 7) Train combined train, test internal test
    print("\n[7] Train COMBINED train (internal+external), test INTERNAL test...")
    # Prefix external IDs to avoid any ambiguity/collisions in bookkeeping (ID isn't used as a feature).
    results_df_train_ext_pref = _prefix_ids(results_df_train_ext, prefix="BOMI1_")
    combined_train = pd.concat([results_df_train_int, results_df_train_ext_pref], ignore_index=True, sort=False)

    eval_c2i = evaluate_on_test_set(combined_train, results_df_test_int, pipeline_i)
    print(f"[COMBINED→INTERNAL] AUC: {eval_c2i['auc']:.3f}, Acc: {eval_c2i['accuracy']:.3f}")
    _save_predictions("pred_combined_train_to_internal_test.csv", results_df_test_int, eval_c2i)

    # 8) Train combined train, test external test
    print("\n[8] Train COMBINED train (internal+external), test EXTERNAL test...")
    eval_c2e = evaluate_on_test_set(combined_train, results_df_test_ext, pipeline_i)
    print(f"[COMBINED→EXTERNAL] AUC: {eval_c2e['auc']:.3f}, Acc: {eval_c2e['accuracy']:.3f}")
    _save_predictions("pred_combined_train_to_external_test.csv", results_df_test_ext, eval_c2e)

    print("\nAll external/internal symmetric experiments completed.")




def select_non_redundant_features(results_df, label_col='label', corr_threshold=0.9, top_n=20):
    """
    Select the top-N most correlated features with the label,
    and remove redundant ones based on inter-feature correlation.
    
    Parameters:
    ------------
    results_df : pd.DataFrame
        The dataset with numeric features and a label column.
    label_col : str
        The column name of the label.
    corr_threshold : float
        Maximum correlation allowed between selected features.
    top_n : int
        Number of top features to consider based on correlation with the label.
    
    Returns:
    --------
    List[str] of selected non-redundant feature names.
    """
    # Step 1: Compute correlation with label
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['ID', 'cell_count', 'sample_name', label_col]]
    
    correlations = []
    top_features = []
    for col in feature_cols:
        corr, p_value = spearmanr(results_df[col], results_df[label_col])
        #correlations.append((col, abs(corr)))
        if p_value < 0.5:
            correlations.append((col, abs(corr)))
    
    # Sort by absolute correlation
    sorted_features = sorted(correlations, key=lambda x: x[1], reverse=True)
    
    top_features = [feat for feat, _ in sorted_features]
    
    # Step 2: Remove redundancy
    corr_matrix = results_df[top_features].corr(method='spearman').abs()
    selected_features = []
    for feature in top_features:
        if all(corr_matrix.loc[feature, f] < corr_threshold for f in selected_features):
            selected_features.append(feature)
    
    return selected_features


    
def main(recompute_metrics=False, cellprofiler=False, cellpose=False):
    # Load and preprocess
    cells_data, samples_data, all_patients, patients_train, patients_test = load_and_preprocess_all(cellprofiler, cellpose)

    # Compute or load all metrics
    all_results = get_or_compute_all_metrics(all_patients, samples_data, cells_data, recompute_metrics, cellprofiler, cellpose)

    # Merge spatial statistics by patient
    all_results_merged = merge_samples(all_results)
    
    # Clean up known unnecessary features
    for col in ["cancer_ripley_L_0.0", "stroma_ripley_L_0.0"]:
        all_results_merged.drop(columns=col, errors='ignore', inplace=True)

    # Split into train/test using patient IDs
    train_ids = set(patients_train["ID"])
    test_ids = set(patients_test["ID"])

    results_df_train = all_results_merged[all_results_merged["ID"].isin(train_ids)].copy()
    results_df_test = all_results_merged[all_results_merged["ID"].isin(test_ids)].copy()

    # Analyze correlations
    analyze_feature_correlations(results_df_train)
    corr_df = run_correlation_analysis(results_df_train)
    print("Top correlated features:\n", corr_df.head(10))

    # Run classification
    print("Running model selection and cross-validation...")
    auc, std_auc, acc, std_acc, feature_imp, model_name, pipeline, _, best_params = run_classification_cv(
        results_df_train, return_model=True
    )

    print(f"{model_name} AUC: {auc:.3f} ± {std_auc:.3f}, Acc: {acc:.3f} ± {std_acc:.3f}")
    print("Top features:\n", feature_imp.head(10))

    # Test evaluation
    print("Evaluating on test set...")
    test_results = evaluate_on_test_set(results_df_train, results_df_test, pipeline)
    print(f"Test set AUC: {test_results['auc']:.3f}")
    print(f"Test set Accuracy: {test_results['accuracy']:.3f}")
    if cellprofiler:
        model_results_filename = "cellprofiler_spatial_metrics_model_labels.csv"
    elif cellpose:
        model_results_filename = "cellpose_spatial_metrics_model_labels.csv"
    else:
        model_results_filename = "inform_spatial_metrics_model_labels.csv"
    pd.DataFrame({
        "Patient ID": results_df_test["ID"],
        "Label": np.where(test_results["y_pred"] == 1, "Adeno carcinoma", "Squamous Cell Carcinoma")
    }).to_csv(os.path.join("subtype_annotator_labels", model_results_filename), index=False)

    print("All done!")
if __name__ == "__main__":
    #main()
    evaluate_external_cohort(
        external_cells_path="BOMI1_cells_all.csv",
        external_meta_path="BOMI1_clinical_data_LUADvsSqCC.csv",
        recompute_internal_metrics=False,  # change to True if you want to recompute internal metrics
        cellprofiler=False,
        cellpose=True
    )
