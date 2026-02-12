"""
spatial_statistics.py

Readable, reproducible pipeline for:
- computing per-core spatial point-cloud metrics
- caching metrics to CSV (internal + external)
- attaching per-core cell counts (total/cancer/stroma) even when loading cached metrics
- filtering cores by fixed cell-count thresholds (cohort-agnostic)
- aggregating cores to patient-level features
- training / cross-validating on train sets and evaluating cross-cohort generalization

Key design choices:
- Counts are used for QC + weighting, but DROPPED from model features right before training.
- QC happens AFTER load/compute of cached metrics, so caching doesn't bypass filtering.

NOTE:
- This script assumes spatial_metrics.py is importable and provides:
  calculate_ripley_l, calculate_bidirectional_min_distance, calculate_newmans_assortativity,
  calculate_centrality_scores, calculate_cluster_cooccurrence_ratio,
  calculate_neighborhood_enrichment_test, calculate_objectobject_correlation
"""

from __future__ import annotations

import os
import re
import time
import random
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# Local feature functions
from spatial_metrics import (
    calculate_ripley_l,
    calculate_bidirectional_min_distance,
    calculate_newmans_assortativity,
    calculate_centrality_scores,
    calculate_cluster_cooccurrence_ratio,
    calculate_neighborhood_enrichment_test,
    calculate_objectobject_correlation,
)

RANDOM_STATE = 42


# ----------------------------
# Configuration
# ----------------------------

@dataclass(frozen=True)
class CountQC:
    min_total: int = 150
    min_cancer: int = 30
    min_stroma: int = 30


@dataclass(frozen=True)
class InternalPaths:
    # inForm / multiplex IF (default in your script)
    inform_cells_path: str = "BOMI2_all_cells_TIL.csv"
    inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"

    # Alternative: samples with computed mask areas (you had this)
    # inform_samples_path: str = "../BOMI2_TIL_masks/samples.csv"

    # Cellpose derived cells
    cellpose_cells_path: str = "./cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1.csv"
    cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1.csv"

    # CellProfiler derived cells
    cellprofiler_cells_path: str = "./cellprofiler_extracted_cells_filtered_necrosis.csv"
    cellprofiler_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"

    # Splits (internal)
    internal_train_csv: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/train_val.csv"
    internal_test_csv: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/test.csv"


@dataclass(frozen=True)
class ExternalPaths:
    external_cells_path: str = "BOMI1_cells_all.csv"
    external_meta_path: str = "BOMI1_clinical_data_LUADvsSqCC.csv"
    external_split_dir: str = "/home/love/multiplex_dataset/lung_cancer_BOMI1_dataset/HE_dataset/binary_subtype_prediction_ACvsSqCC/static_split/"
    external_train_csv: Optional[str] = None
    external_test_csv: Optional[str] = None


# ----------------------------
# Utility helpers
# ----------------------------

def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_split_ids(csv_path: str) -> set:
    df = pd.read_csv(csv_path)
    for col in ["ID", "ID or PAD_year", "ID_or_PAD_year", "patient_id", "Patient ID"]:
        if col in df.columns:
            return set(df[col].dropna().astype(str))
    raise ValueError(f"Could not find an ID column in {csv_path}. Columns: {list(df.columns)}")


def _subset_by_ids(df: pd.DataFrame, ids: set) -> pd.DataFrame:
    ids_str = {str(x) for x in ids}
    if "ID" not in df.columns:
        raise ValueError(f"Expected column 'ID'. Columns: {list(df.columns)}")
    return df[df["ID"].astype(str).isin(ids_str)].copy()


def _prefix_ids(df: pd.DataFrame, prefix: str, id_col: str = "ID") -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"Expected column '{id_col}'. Columns: {list(df.columns)}")
    out = df.copy()
    out[id_col] = out[id_col].astype(str).map(lambda x: f"{prefix}{x}")
    return out


def _save_predictions(path: str, test_df: pd.DataFrame, eval_dict: dict) -> None:
    out = pd.DataFrame({
        "ID": test_df["ID"].astype(str),
        "y_true": eval_dict["y_true"],
        "y_prob": eval_dict["y_probs"],
        "y_pred": eval_dict["y_pred"],
    })
    out.to_csv(path, index=False)
    print(f"[INFO] Saved predictions to {path}")


# ----------------------------
# Preprocessing (cells + metadata)
# ----------------------------

def preprocess_samples(samples_df: pd.DataFrame) -> pd.DataFrame:
    df = samples_df.copy()
    df["sample_name"] = df["sample_name"].astype(str)
    # Your legacy transforms
    df["sample_name"] = df["sample_name"].map(lambda x: x[:x.rfind("_")] if "_" in x else x)
    df["sample_name"] = df["sample_name"].map(lambda x: x.replace("Core[1,", "["))
    return df


def preprocess_patients(patients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the same mapping you had.
    """
    df = patients_df.copy()
    df = df.rename(columns={"ID or PAD_year": "ID", "Tumor_type": "Histology", "Sex": "Gender", "Event_last_followup": "Dead/Alive"})
    t = {"Adenocarcinoma": "LUAD", "Squamous cell carcinoma": "LUSC", "Other": "Other"}
    df["Tumor_type_code"] = df["Histology"].map(lambda x: t[x])
    df["Smoking"] = df["Smoking"].map(lambda x: 0 if x == "Never-smoker" else 1)
    df["LUAD"] = (df["Tumor_type_code"] == "LUAD").astype(int)
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    stage_dict = {"Ia": 0, "Ib": 1, "IIa": 2, "IIb": 3, "IIIa": 4, "IIIb": 5, "IV": 6}
    df["Stage"] = df["Stage (7th ed.)"].map(lambda x: stage_dict[x])

    keep_cols = ["ID", "LUAD", "Age", "Gender", "Smoking", "Stage",
                 "Performance status (WHO)", "Follow-up (days)", "label", "Tumor_type_code"]
    return df[keep_cols].copy()


def preprocess_cells_inform(cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    inForm multiplex IF cells preprocessing.
    Produces columns required for metrics: x,y,Cancer,sample_name
    """
    df = cells_df.copy()

    df["Tissue Category"] = df["Tissue Category"].map(lambda x: 1 if x == "Tumor" else 0)
    df["Cell X Position"] = df["Cell X Position"].map(lambda x: float(x))
    df["Cell Y Position"] = df["Cell Y Position"].map(lambda x: float(x))

    df.rename(columns={
        "Sample Name": "sample_name",
        "Tissue Category": "Cancer",
        "Cell X Position": "x",
        "Cell Y Position": "y",
        "Cytoplasm Opal 690 Mean (Normalized Counts, Total Weighting)": "PanCK",
    }, inplace=True)

    # Use CK positivity as cancer marker (your original logic uses CK column)
    if "CK" in df.columns:
        df["Cancer"] = df["CK"]

    # Ensure numeric
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["Cancer"] = pd.to_numeric(df["Cancer"], errors="coerce").fillna(0).astype(int)
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def preprocess_cellpose_data(cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cellpose component CSV -> standardized format.
    Expected columns in your file: x,y,ck,filename (based on your script).
    """
    df = cells_df.copy()

    def extract_sample_name(filename: str) -> str:
        match = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", str(filename))
        if match:
            return f"{match.group(1)}_[{match.group(2)}]"
        raise ValueError(f"Could not parse sample name from filename: {filename}")

    df = df.rename(columns={"ck": "Cancer", "filename": "file_name"})
    df["sample_name"] = df["file_name"].apply(extract_sample_name)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["Cancer"] = pd.to_numeric(df["Cancer"], errors="coerce").fillna(0).astype(int)
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def preprocess_cellprofiler_data(cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    CellProfiler -> standardized format.
    """
    df = cells_df.copy()

    def extract_sample_name(filename: str) -> str:
        match = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", str(filename))
        if match:
            return f"{match.group(1)}_[{match.group(2)}]"
        raise ValueError(f"Could not parse sample name from filename: {filename}")

    df = df.rename(columns={
        "Location_Center_X": "x",
        "Location_Center_Y": "y",
        "CK": "Cancer",
        "FileName_CK": "file_name"
    })
    df["sample_name"] = df["file_name"].apply(extract_sample_name)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["Cancer"] = pd.to_numeric(df["Cancer"], errors="coerce").fillna(0).astype(int)
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def preprocess_external_cells(bomi1_cells: pd.DataFrame) -> pd.DataFrame:
    """
    BOMI1 external cells standardization.
    Expects: CentroidX_um, CentroidY_um, Class, sample_name
    """
    df = bomi1_cells.copy()
    needed = {"CentroidX_um", "CentroidY_um", "Class", "sample_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"External cells file missing columns: {missing}. Columns: {list(df.columns)}")

    df = df.rename(columns={"CentroidX_um": "x", "CentroidY_um": "y"})
    df["Cancer"] = (df["Class"].astype(str).str.strip().str.lower() == "neoplastic").astype(int)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def split_external_meta(meta_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From clinical_data file create:
      - patients_df: one row per patient with ['ID', 'label']
      - samples_df: mapping of samples to patients ['ID', 'sample_name']
    """
    needed = {"ID", "sample_name", "label"}
    missing = needed - set(meta_df.columns)
    if missing:
        raise ValueError(f"External meta file missing columns: {missing}. Columns: {list(meta_df.columns)}")

    patients_df = meta_df[["ID", "label"]].drop_duplicates(subset=["ID"]).copy()
    samples_df = meta_df[["ID", "sample_name"]].drop_duplicates(subset=["ID", "sample_name"]).copy()
    return patients_df, samples_df


# ----------------------------
# Spatial metrics per sample
# ----------------------------

def calculate_spatial_metrics_for_sample(cells_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Compute spatial metrics for one sample.
    Returns a dict that INCLUDES counts for downstream QC/weighting.
    """
    if cells_df is None or len(cells_df) < 10:
        return None

    coords = cells_df[["x", "y"]].values
    cell_types = cells_df["Cancer"].values.astype(int)

    metrics: Dict[str, float] = {}

    # --- counts (always included in per-sample output) ---
    metrics["cell_count"] = int(len(cells_df))
    metrics["cancer_count"] = int((cell_types == 1).sum())
    metrics["stroma_count"] = int((cell_types == 0).sum())

    # --- spatial metrics (same as your current set) ---
    metrics.update(calculate_ripley_l(coords, cell_types))

    metrics.update(calculate_bidirectional_min_distance(coords, cell_types))

    metrics["newmans_assortativity"] = float(
        calculate_newmans_assortativity(coords, cell_types, radius=50)
    )

    metrics.update(calculate_centrality_scores(coords, cell_types))

    metrics.update(calculate_cluster_cooccurrence_ratio(coords, cell_types))

    metrics.update(calculate_neighborhood_enrichment_test(coords, cell_types))

    metrics.update(calculate_objectobject_correlation(coords, cell_types))

    return metrics


def _compute_one_sample_worker(ID: str, sample_name: str, label: int, sample_cells_df: pd.DataFrame) -> Optional[Dict]:
    """
    Worker: compute metrics for a single sample and attach identifiers.
    """
    metrics = calculate_spatial_metrics_for_sample(sample_cells_df)
    if metrics is None:
        return None
    metrics["ID"] = ID
    metrics["sample_name"] = sample_name
    metrics["label"] = int(label)
    return metrics


def compute_metrics_for_cohort(
    patients_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute per-sample metrics in parallel.
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    patients_samples = pd.merge(patients_df, samples_df, on="ID", how="inner")
    
    # Pre-group cells by sample_name (fast lookups)
    cell_groups = {s: g[["x", "y", "Cancer"]].copy()
                   for s, g in cells_df.groupby("sample_name", sort=False)}
    
    tasks: List[Tuple[str, str, int, pd.DataFrame]] = []
    for _, row in patients_samples.iterrows():
        sname = row["sample_name"]
        if sname not in cell_groups:
            continue
        tasks.append((str(row["ID"]), str(sname), int(row["label"]), cell_groups[sname]))
    print(len(patients_samples))
    if not tasks:
        return pd.DataFrame()

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_compute_one_sample_worker, *t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
            res = f.result()  # fail loudly if something is broken
            if res is not None:
                results.append(res)

    return pd.DataFrame(results)


# ----------------------------
# Counts attachment + filtering (works post-load too)
# ----------------------------

def attach_counts_from_cells(metrics_df: pd.DataFrame, cells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure metrics_df has cell_count/cancer_count/stroma_count derived from cells_df grouped by sample_name.
    Useful when loading old cached metrics that may be missing cancer/stroma counts.

    If metrics_df already has these columns, missing values are filled from cells_df.
    """
    if metrics_df is None or metrics_df.empty:
        return metrics_df

    needed_cols = {"sample_name"}
    if not needed_cols.issubset(metrics_df.columns):
        raise ValueError(f"metrics_df missing required columns {needed_cols}. Columns: {list(metrics_df.columns)}")

    counts = (
        cells_df.groupby("sample_name", sort=False)["Cancer"]
        .agg(
            cell_count="size",
            cancer_count=lambda x: int((x.astype(int) == 1).sum()),
            stroma_count=lambda x: int((x.astype(int) == 0).sum()),
        )
        .reset_index()
    )

    out = metrics_df.merge(counts, on="sample_name", how="left", suffixes=("", "__from_cells"))

    # If existing columns present, keep them but fill NaNs from derived values
    for col in ["cell_count", "cancer_count", "stroma_count"]:
        from_cells = f"{col}__from_cells"
        if col in out.columns and from_cells in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[from_cells] = pd.to_numeric(out[from_cells], errors="coerce")
            out[col] = out[col].fillna(out[from_cells])
            out.drop(columns=[from_cells], inplace=True)
        elif from_cells in out.columns and col not in out.columns:
            out.rename(columns={from_cells: col}, inplace=True)

    return out


def filter_metrics_by_counts(metrics_df: pd.DataFrame, qc: CountQC, cohort_name: str) -> pd.DataFrame:
    """
    Filter per-sample metrics by fixed thresholds.
    """
    if metrics_df is None or metrics_df.empty:
        return metrics_df

    for c in ["cell_count", "cancer_count", "stroma_count"]:
        if c not in metrics_df.columns:
            raise ValueError(f"{cohort_name}: missing '{c}' in metrics_df. Columns: {list(metrics_df.columns)}")

    df = metrics_df.copy()
    df["cell_count"] = pd.to_numeric(df["cell_count"], errors="coerce")
    df["cancer_count"] = pd.to_numeric(df["cancer_count"], errors="coerce")
    df["stroma_count"] = pd.to_numeric(df["stroma_count"], errors="coerce")

    keep = (
        (df["cell_count"] >= qc.min_total) &
        (df["cancer_count"] >= qc.min_cancer) &
        (df["stroma_count"] >= qc.min_stroma)
    )
    before = len(df)
    df = df.loc[keep].copy()
    after = len(df)
    print(f"[QC] {cohort_name}: kept {after}/{before} cores "
          f"(min_total={qc.min_total}, min_cancer={qc.min_cancer}, min_stroma={qc.min_stroma})")
    return df


# ----------------------------
# Aggregation to patient level
# ----------------------------

def aggregate_cores_to_patient(metrics_per_core: pd.DataFrame) -> pd.DataFrame:
    """
    Patient-level aggregation:
    - label: first
    - counts: summed across cores (kept for QC/reporting; dropped before training)
    - all other numeric features: weighted average by cell_count (same as your prior logic)
    """
    if metrics_per_core is None or metrics_per_core.empty:
        return pd.DataFrame()

    required = {"ID", "label", "cell_count"}
    missing = required - set(metrics_per_core.columns)
    if missing:
        raise ValueError(f"metrics_per_core missing {missing}. Columns: {list(metrics_per_core.columns)}")

    df = metrics_per_core.copy()

    # Columns to aggregate (features)
    exclude = {"ID", "label", "sample_name", "cell_count", "cancer_count", "stroma_count"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    def _agg_one(group: pd.DataFrame) -> pd.Series:
        weights = pd.to_numeric(group["cell_count"], errors="coerce").fillna(0).to_numpy(float)
        wsum = float(weights.sum()) if weights.size else 0.0

        out = {
            "label": int(group["label"].iloc[0]),
            "cell_count": int(pd.to_numeric(group["cell_count"], errors="coerce").fillna(0).sum()),
        }

        if "cancer_count" in group.columns:
            out["cancer_count"] = int(pd.to_numeric(group["cancer_count"], errors="coerce").fillna(0).sum())
        if "stroma_count" in group.columns:
            out["stroma_count"] = int(pd.to_numeric(group["stroma_count"], errors="coerce").fillna(0).sum())

        for col in feature_cols:
            x = pd.to_numeric(group[col], errors="coerce").to_numpy(float)
            if wsum > 0:
                out[col] = float(np.nansum(x * weights) / wsum)
            else:
                out[col] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

        return pd.Series(out)

    patient_df = df.groupby("ID", sort=False).apply(_agg_one).reset_index()
    return patient_df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Model feature columns (numeric) excluding identifiers and counts.
    """
    drop = {"ID", "label", "cell_count", "cancer_count", "stroma_count"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
    return numeric_cols


# ----------------------------
# Modeling
# ----------------------------

def run_classification_cv(
    train_df: pd.DataFrame,
    n_folds: int = 5,
    hyperparameter_tuning: bool = True,
    return_model: bool = True,
) -> Tuple[float, float, float, float, pd.DataFrame, str, Pipeline, dict]:
    """
    Keep behavior similar to your current code:
    - optionally tune model family/hyperparams (not nested CV)
    - evaluate CV AUC/Acc using the selected pipeline
    - compute permutation importance on one fold split
    """
    if train_df is None or train_df.empty:
        raise ValueError("train_df is empty")

    feat_cols = get_feature_columns(train_df)
    if not feat_cols:
        raise ValueError("No feature columns found after dropping counts/IDs")

    X = train_df[feat_cols].to_numpy(float)
    y = train_df["label"].to_numpy(int)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    best_pipeline: Optional[Pipeline] = None
    best_model_name = None
    best_params: dict = {}
    best_score = -np.inf

    if hyperparameter_tuning:
        print("[INFO] Hyperparameter tuning...")
        start = time.time()

        models = {
            "RandomForest": (
                RandomForestClassifier(random_state=RANDOM_STATE),
                {
                    "model__n_estimators": [50, 100, 200],
                    "model__max_depth": [None, 3, 5, 10, 20, 30],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2", None],
                },
                SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ),
            "GradientBoosting": (
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                {
                    "model__n_estimators": [50, 100, 200],
                    "model__learning_rate": [0.01, 0.1, 0.2],
                    "model__max_depth": [3, 5, 7],
                    "model__min_samples_split": [2, 5],
                    "model__min_samples_leaf": [1, 2],
                },
                SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ),
            "SVM": (
                SVC(probability=True, random_state=RANDOM_STATE, max_iter=10000),
                {
                    "model__C": [0.1, 1, 10, 100],
                    "model__gamma": ["scale", "auto", 0.1, 0.01],
                    "model__kernel": ["rbf", "linear"],
                },
                SelectFromModel(LinearSVC(penalty="l1", dual=False, C=1.0, max_iter=10000, random_state=RANDOM_STATE)),
            ),
            "LogisticRegression": (
                LogisticRegression(max_iter=10000, random_state=RANDOM_STATE),
                {
                    "model__C": [0.01, 0.1, 1, 10, 100],
                    "model__penalty": ["l2"],
                    "model__solver": ["saga"],
                },
                SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=RANDOM_STATE)),
            ),
        }

        use_randomized = len(X) > 200

        for name, (model, params, selector) in models.items():
            print(f"  Tuning {name}...")

            pipeline = Pipeline([
                ("scaler", MinMaxScaler()),
                ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
                ("selector", selector),
                ("model", model),
            ])

            search = (
                RandomizedSearchCV(
                    pipeline, params, n_iter=20, scoring="roc_auc", cv=kf, n_jobs=-1, random_state=RANDOM_STATE
                )
                if use_randomized
                else GridSearchCV(pipeline, params, scoring="roc_auc", cv=kf, n_jobs=-1)
            )

            # Keep your previous "skip on failure" behavior, but do not hide the error.
            try:
                search.fit(X, y)
            except Exception as e:
                print(f"[WARN] {name} failed during search.fit(): {e}")
                continue

            if search.best_score_ > best_score:
                best_score = float(search.best_score_)
                best_pipeline = search.best_estimator_
                best_model_name = name
                best_params = dict(search.best_params_)

            print(f"    Best {name}: AUC={search.best_score_:.4f} params={search.best_params_}")

        if best_pipeline is None:
            raise RuntimeError("All model searches failed; cannot proceed.")

        print(f"[INFO] Selected model: {best_model_name} (tuning time {time.time() - start:.1f}s)")
    else:
        best_model_name = "RandomForest (default)"
        best_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
            ("selector", SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE))),
            ("model", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ])

    # CV evaluation using the chosen pipeline
    aucs, accs = [], []
    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        best_pipeline.fit(X_tr, y_tr)
        y_prob = best_pipeline.predict_proba(X_te)[:, 1]
        y_pred = best_pipeline.predict(X_te)

        aucs.append(roc_auc_score(y_te, y_prob))
        accs.append(float(np.mean(y_pred == y_te)))

    # Permutation importance on one fold split (same spirit as your code)
    tr_idx, te_idx = next(kf.split(X))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    best_pipeline.fit(X_tr, y_tr)

    perm = permutation_importance(best_pipeline, X_te, y_te, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc")
    fi = pd.DataFrame({"feature": feat_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False)

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs)), fi, str(best_model_name), best_pipeline, best_params


def evaluate_on_test_set(train_df: pd.DataFrame, test_df: pd.DataFrame, pipeline: Pipeline) -> Dict:
    """
    Train on train_df, evaluate on test_df. Counts are excluded automatically.
    """
    feat_cols = get_feature_columns(train_df)
    X_tr = train_df[feat_cols].to_numpy(float)
    y_tr = train_df["label"].to_numpy(int)

    X_te = test_df[feat_cols].to_numpy(float)
    y_te = test_df["label"].to_numpy(int)

    pipeline.fit(X_tr, y_tr)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    y_pred = pipeline.predict(X_te)

    auc = float(roc_auc_score(y_te, y_prob))
    acc = float(np.mean(y_pred == y_te))

    perm = permutation_importance(pipeline, X_te, y_te, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc")
    fi = pd.DataFrame({"feature": feat_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False)

    return {
        "auc": auc,
        "accuracy": acc,
        "feature_importance": fi,
        "y_true": y_te,
        "y_pred": y_pred,
        "y_probs": y_prob,
    }


# ----------------------------
# Main experiment driver
# ----------------------------

def load_internal_dataset(
    paths: InternalPaths,
    source: str = "cellpose",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load internal cohort (BOMI2) for a given cell source.

    source:
      - "inform"
      - "cellpose"
      - "cellprofiler"
    """
    if source not in {"inform", "cellpose", "cellprofiler"}:
        raise ValueError(f"Unknown source: {source}")

    if source == "inform":
        cells_raw = pd.read_csv(paths.inform_cells_path)
        cells_df = preprocess_cells_inform(cells_raw)
        samples_raw = pd.read_csv(paths.inform_samples_path)
        samples_df = preprocess_samples(samples_raw)
    elif source == "cellpose":
        cells_raw = pd.read_csv(paths.cellpose_cells_path)
        cells_df = preprocess_cellpose_data(cells_raw)
        samples_raw = pd.read_csv(paths.cellpose_samples_path)
        samples_df = preprocess_samples(samples_raw)
    else:
        cells_raw = pd.read_csv(paths.cellprofiler_cells_path)
        cells_df = preprocess_cellprofiler_data(cells_raw)
        samples_raw = pd.read_csv(paths.cellprofiler_samples_path)
        samples_df = preprocess_samples(samples_raw)

    patients_train_raw = pd.read_csv(paths.internal_train_csv)
    patients_test_raw = pd.read_csv(paths.internal_test_csv)

    patients_train = preprocess_patients(patients_train_raw)
    patients_test = preprocess_patients(patients_test_raw)
    all_patients = pd.concat([patients_train, patients_test], ignore_index=True)

    return cells_df, samples_df, all_patients, patients_train, patients_test


def get_or_compute_metrics(
    cohort_name: str,
    metrics_cache_path: str,
    patients_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    recompute: bool = False,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute metrics if missing or recompute=True; otherwise load cached.
    Always attach counts from cells_df (fills missing cols).
    """
    if recompute or (not os.path.exists(metrics_cache_path)):
        print(f"[INFO] Computing metrics for {cohort_name} -> {metrics_cache_path}")
        m = compute_metrics_for_cohort(patients_df, samples_df, cells_df, n_jobs=n_jobs)
        m.to_csv(metrics_cache_path, index=False)
        print(f"[INFO] Saved metrics: {metrics_cache_path}")
    else:
        print(f"[INFO] Loading cached metrics for {cohort_name}: {metrics_cache_path}")
        m = pd.read_csv(metrics_cache_path)

    # Ensure counts are present (even if cache is old)
    m = attach_counts_from_cells(m, cells_df)
    return m


def evaluate_external_cohort(
    internal_source: str = "cellpose",
    internal_paths: InternalPaths = InternalPaths(),
    external_paths: ExternalPaths = ExternalPaths(),
    qc: CountQC = CountQC(),
    recompute_internal_metrics: bool = False,
    recompute_external_metrics: bool = False,
    n_jobs: Optional[int] = None,
    external_xy_scale: float = 2.0,
    hyperparameter_tuning: bool = True,
) -> None:
    """
    Symmetric internal/external experiments:

    1) CV on internal train
    2) Train internal train -> test internal test
    3) Train internal train -> test external test

    4) CV on external train
    5) Train external train -> test external test
    6) Train external train -> test internal test

    7) Train combined (internal train + external train) -> test internal test
    8) Train combined (internal train + external train) -> test external test
    """
    set_global_seed(RANDOM_STATE)

    # Resolve external split paths
    ext_train_csv = external_paths.external_train_csv or os.path.join(external_paths.external_split_dir, "train_val.csv")
    ext_test_csv = external_paths.external_test_csv or os.path.join(external_paths.external_split_dir, "test.csv")

    # -----------------------
    # Internal: load + metrics
    # -----------------------
    print(f"\n[LOAD] INTERNAL (BOMI2) source={internal_source}")
    int_cells, int_samples, int_patients_all, int_patients_train, int_patients_test = load_internal_dataset(
        internal_paths, source=internal_source
    )

    internal_metrics_path = f"spatial_metrics_{internal_source}.csv"
    int_metrics_per_core = get_or_compute_metrics(
        cohort_name=f"INTERNAL/{internal_source}",
        metrics_cache_path=internal_metrics_path,
        patients_df=int_patients_all,
        samples_df=int_samples,
        cells_df=int_cells,
        recompute=recompute_internal_metrics,
        n_jobs=n_jobs,
    )

    # QC filter (per-core) - applies even when loading cached metrics
    int_metrics_per_core = filter_metrics_by_counts(int_metrics_per_core, qc=qc, cohort_name="INTERNAL/BOMI2")

    # Aggregate cores -> patient
    int_patient_df = aggregate_cores_to_patient(int_metrics_per_core)

    # Optional: drop ripley at 0.0 (you did this)
    for col in ["cancer_ripley_L_0.0", "stroma_ripley_L_0.0"]:
        int_patient_df.drop(columns=[col], errors="ignore", inplace=True)

    # Split internal train/test by patient IDs
    train_ids_int = set(int_patients_train["ID"].astype(str))
    test_ids_int = set(int_patients_test["ID"].astype(str))
    df_train_int = _subset_by_ids(int_patient_df, train_ids_int)
    df_test_int = _subset_by_ids(int_patient_df, test_ids_int)

    print(f"[INFO] INTERNAL patients: train={len(df_train_int)} test={len(df_test_int)} (after core QC + aggregation)")

    # -----------------------
    # External: load + metrics
    # -----------------------
    print(f"\n[LOAD] EXTERNAL (BOMI1) H&E")
    ext_cells_raw = pd.read_csv(external_paths.external_cells_path)
    ext_meta_raw = pd.read_csv(external_paths.external_meta_path)

    ext_cells = preprocess_external_cells(ext_cells_raw)
    ext_patients_df, ext_samples_df = split_external_meta(ext_meta_raw)

    # your current scaling
    ext_cells[["x", "y"]] = ext_cells[["x", "y"]] * float(external_xy_scale)

    external_metrics_path = "spatial_metrics_EXTERNAL_BOMI1.csv"
    ext_metrics_per_core = get_or_compute_metrics(
        cohort_name="EXTERNAL/BOMI1",
        metrics_cache_path=external_metrics_path,
        patients_df=ext_patients_df,
        samples_df=ext_samples_df,
        cells_df=ext_cells,
        recompute=recompute_external_metrics,
        n_jobs=n_jobs,
    )

    # QC filter (per-core)
    ext_metrics_per_core = filter_metrics_by_counts(ext_metrics_per_core, qc=qc, cohort_name="EXTERNAL/BOMI1")

    # Aggregate -> patient
    ext_patient_df = aggregate_cores_to_patient(ext_metrics_per_core)
    for col in ["cancer_ripley_L_0.0", "stroma_ripley_L_0.0"]:
        ext_patient_df.drop(columns=[col], errors="ignore", inplace=True)

    # External static splits
    ext_train_ids = _read_split_ids(ext_train_csv)
    ext_test_ids = _read_split_ids(ext_test_csv)
    df_train_ext = _subset_by_ids(ext_patient_df, ext_train_ids)
    df_test_ext = _subset_by_ids(ext_patient_df, ext_test_ids)

    print(f"[INFO] EXTERNAL patients: train={len(df_train_ext)} test={len(df_test_ext)} (after core QC + aggregation)")

    if df_train_int.empty or df_test_int.empty or df_train_ext.empty or df_test_ext.empty:
        raise RuntimeError("One of the train/test splits is empty after QC. Adjust CountQC thresholds.")

    # -----------------------
    # Experiments
    # -----------------------

    print("\n[1] CV on INTERNAL train...")
    auc_i, std_auc_i, acc_i, std_acc_i, fi_i, model_name_i, pipe_i, best_params_i = run_classification_cv(
        df_train_int, n_folds=5, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[INTERNAL CV] {model_name_i} AUC: {auc_i:.3f} ± {std_auc_i:.3f}, Acc: {acc_i:.3f} ± {std_acc_i:.3f}")

    print("\n[2] Train INTERNAL train, test INTERNAL test...")
    eval_i2i = evaluate_on_test_set(df_train_int, df_test_int, pipe_i)
    print(f"[INTERNAL→INTERNAL] AUC: {eval_i2i['auc']:.3f}, Acc: {eval_i2i['accuracy']:.3f}")
    _save_predictions("pred_internal_train_to_internal_test.csv", df_test_int, eval_i2i)

    print("\n[3] Train INTERNAL train, test EXTERNAL test...")
    eval_i2e = evaluate_on_test_set(df_train_int, df_test_ext, pipe_i)
    print(f"[INTERNAL→EXTERNAL] AUC: {eval_i2e['auc']:.3f}, Acc: {eval_i2e['accuracy']:.3f}")
    _save_predictions("pred_internal_train_to_external_test.csv", df_test_ext, eval_i2e)

    print("\n[4] CV on EXTERNAL train...")
    auc_e, std_auc_e, acc_e, std_acc_e, fi_e, model_name_e, pipe_e, best_params_e = run_classification_cv(
        df_train_ext, n_folds=5, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[EXTERNAL CV] {model_name_e} AUC: {auc_e:.3f} ± {std_auc_e:.3f}, Acc: {acc_e:.3f} ± {std_acc_e:.3f}")

    print("\n[5] Train EXTERNAL train, test EXTERNAL test...")
    eval_e2e = evaluate_on_test_set(df_train_ext, df_test_ext, pipe_e)
    print(f"[EXTERNAL→EXTERNAL] AUC: {eval_e2e['auc']:.3f}, Acc: {eval_e2e['accuracy']:.3f}")
    _save_predictions("pred_external_train_to_external_test.csv", df_test_ext, eval_e2e)

    print("\n[6] Train EXTERNAL train, test INTERNAL test...")
    eval_e2i = evaluate_on_test_set(df_train_ext, df_test_int, pipe_e)
    print(f"[EXTERNAL→INTERNAL] AUC: {eval_e2i['auc']:.3f}, Acc: {eval_e2i['accuracy']:.3f}")
    _save_predictions("pred_external_train_to_internal_test.csv", df_test_int, eval_e2i)

    print("\n[7] Train COMBINED train (internal+external), test INTERNAL test...")
    df_train_ext_pref = _prefix_ids(df_train_ext, prefix="BOMI1_")
    combined_train = pd.concat([df_train_int, df_train_ext_pref], ignore_index=True, sort=False)

    eval_c2i = evaluate_on_test_set(combined_train, df_test_int, pipe_i)
    print(f"[COMBINED→INTERNAL] AUC: {eval_c2i['auc']:.3f}, Acc: {eval_c2i['accuracy']:.3f}")
    _save_predictions("pred_combined_train_to_internal_test.csv", df_test_int, eval_c2i)

    print("\n[8] Train COMBINED train (internal+external), test EXTERNAL test...")
    eval_c2e = evaluate_on_test_set(combined_train, df_test_ext, pipe_i)
    print(f"[COMBINED→EXTERNAL] AUC: {eval_c2e['auc']:.3f}, Acc: {eval_c2e['accuracy']:.3f}")
    _save_predictions("pred_combined_train_to_external_test.csv", df_test_ext, eval_c2e)

    print("\n[INFO] All experiments completed.")


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    # Choose internal cell source:
    #   internal_source="inform"
    #   internal_source="cellpose"
    #   internal_source="cellprofiler"
    evaluate_external_cohort(
        internal_source="cellpose",
        qc=CountQC(min_total=100, min_cancer=30, min_stroma=30),
        recompute_internal_metrics=False,
        recompute_external_metrics=False,
        hyperparameter_tuning=True,
        external_xy_scale=2.0,
    )
