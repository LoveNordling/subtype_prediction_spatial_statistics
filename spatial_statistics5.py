"""spatial_statistics3.py

Spatial point-cloud tissue statistics for NSCLC subtype prediction.

This script:
- Computes (or loads cached) per-core spatial metrics for:
    * INTERNAL cohort (BOMI2 multiplex IF) from one of: inform/cellpose/cellprofiler
    * EXTERNAL cohort (BOMI1 H&E / CellViT-derived) from pre-extracted cells + meta mapping
- Attaches per-core counts (total/cancer/stroma) even when loading cached metrics.
- Applies fixed, cohort-agnostic QC thresholds on cell counts.
- Aggregates cores to patient-level features via cell_count-weighted averaging.
- Runs symmetric base experiments:
    1) CV on each cohort's training set
    2) Within-cohort test
    3) Cross-cohort transfer tests
    4) Combined training tests
- Learning-curve / cohort-mixing analysis (symmetric):
    Measures benefit of adding the entire secondary training cohort as a function of
    primary training set size.

Learning-curve evaluation modes:
- fixed: evaluate on the (static) primary test split
- cv:    evaluate via StratifiedKFold on the primary training pool (recommended when
         the static test split is small/noisy)

Notes:
- Cell counts are used for QC only and are excluded from the feature set.
- The model selection stage performs a small hyperparameter search across
  RF / GB / SVM / LR and selects the best mean ROC-AUC.

"""

from __future__ import annotations

import os
import time
import random
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

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
from spatial_utils import (
    CountQC,
    aggregate_cores_to_patient,
    attach_counts_from_cells,
    ensure_columns,
    filter_metrics_by_counts,
    get_feature_columns,
    preprocess_cellpose_data,
    preprocess_cellprofiler_data,
    preprocess_cells_inform,
    preprocess_external_cells,
    preprocess_patients,
    preprocess_samples,
    split_external_meta,
)


RANDOM_STATE = 42


# ----------------------------
# Configuration
# ----------------------------

@dataclass(frozen=True)
class InternalPaths:
    # inForm / multiplex IF
    inform_cells_path: str = "data/raw/BOMI2_all_cells_TIL.csv"
    inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"

    #inform_cells_path: str = "BOMI2_all_cells_TIL__matchedBOMI1.csv"
    #inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__inform_matchedBOMI1.csv"

    #inform_cells_path: str = "BOMI2_all_cells_TIL__matchedBOMI1_stroma.csv"
    #inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__inform_matchedBOMI1_stroma.csv"

    #inform_cells_path: str = "BOMI2_all_cells_TIL__matchedBOMI1_separate.csv"
    #inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__inform_matchedBOMI1_separate.csv"

    

    # Cellpose derived cells (matched to external cohort coordinates)
    #cellpose_cells_path: str = "./cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1.csv"
    #cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1.csv"

    cellpose_cells_path: str = "data/interim/cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1_separate.csv"
    cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1_separate.csv"
    
    # Stroma matched
    #cellpose_cells_path: str = "cellpose_extracted_cells_fitlered_necrosis__matchedstromaBOMI1.csv"
    #cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1stroma.csv"

    #cellpose_cells_path: str = "cellpose_extracted_cells_fitlered_necrosis.csv"
    #cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"
    
    # CellProfiler derived cells
    cellprofiler_cells_path: str = "data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv"
    cellprofiler_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"

    # Splits (internal)
    internal_train_csv: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/train_val.csv"
    internal_test_csv: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/test.csv"


@dataclass(frozen=True)
class ExternalPaths:
    external_cells_path: str = "data/raw/BOMI1_cells_all.csv"
    external_meta_path: str = "data/reference/BOMI1_clinical_data_LUADvsSqCC.csv"
    external_split_dir: str = "/home/love/multiplex_dataset/lung_cancer_BOMI1_dataset/HE_dataset/binary_subtype_prediction_ACvsSqCC/static_split/"
    external_train_csv: Optional[str] = None
    external_test_csv: Optional[str] = None


# ----------------------------
# Utility helpers
# ----------------------------

def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sem(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def _safe_json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


def _extract_selected_features(pipeline, feat_cols: List[str]) -> Optional[List[str]]:
    # If the pipeline contains a selector step (SelectFromModel), record which features survived selection.
    if hasattr(pipeline, "named_steps") and "selector" in getattr(pipeline, "named_steps", {}):
        selector = pipeline.named_steps["selector"]
        if hasattr(selector, "get_support"):
            support = selector.get_support()
            if support is not None and len(support) == len(feat_cols):
                return [f for f, keep in zip(feat_cols, support) if bool(keep)]
    return None


def save_feature_importance_bundle(
    out_dir: str,
    *,
    cohort: str,
    regime: str,
    model_name: str,
    best_params: dict,
    n_patients_train: int,
    n_patients_eval: int,
    feature_columns: List[str],
    fi_df: pd.DataFrame,
    selected_features: Optional[List[str]] = None,
    metric: str = "roc_auc",
    importance_method: str = "permutation_importance",
    notes: str = "",
) -> Dict[str, str]:
    """Save feature importance as CSV + JSON with explicit context (LLM-readable)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fi_sorted = fi_df.copy()
    fi_sorted["feature"] = fi_sorted["feature"].astype(str)
    fi_sorted["importance"] = pd.to_numeric(fi_sorted["importance"], errors="coerce")
    fi_sorted = fi_sorted.sort_values("importance", ascending=False).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"feature_importance__{cohort}__{regime}__{model_name.replace(' ', '_')}__{ts}"

    csv_path = str(Path(out_dir) / f"{stem}.csv")
    fi_sorted.to_csv(csv_path, index=False)

    top_k = 30
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cohort": cohort,
        "regime": regime,
        "model_name": model_name,
        "best_params": best_params if isinstance(best_params, dict) else {},
        "importance_method": importance_method,
        "importance_metric": metric,
        "n_patients_train": int(n_patients_train),
        "n_patients_eval": int(n_patients_eval),
        "n_features_total": int(len(feature_columns)),
        "feature_columns": list(map(str, feature_columns)),
        "n_selected_features": int(len(selected_features)) if selected_features is not None else None,
        "selected_features": selected_features,
        "top_features": fi_sorted.head(top_k).to_dict(orient="records"),
        "all_features": fi_sorted.to_dict(orient="records"),
        "notes": notes,
    }

    json_path = str(Path(out_dir) / f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_safe_json_default)

    print(f"[INFO] Saved feature importance CSV: {csv_path}")
    print(f"[INFO] Saved feature importance JSON: {json_path}")
    return {"csv": csv_path, "json": json_path}


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
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[INFO] Saved predictions to {path}")


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified subsample by label, attempting to preserve class ratio."""
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column")

    rng = np.random.default_rng(seed)
    y = df["label"].astype(int).to_numpy()
    idx = np.arange(len(df))

    idx0 = idx[y == 0]
    idx1 = idx[y == 1]
    if len(idx0) == 0 or len(idx1) == 0:
        take = rng.choice(idx, size=n, replace=False)
        return df.iloc[take].copy().reset_index(drop=True)

    p1 = len(idx1) / len(df)
    n1 = int(round(n * p1))
    n1 = max(1, min(n - 1, n1))
    n0 = n - n1

    n0 = min(n0, len(idx0))
    n1 = min(n1, len(idx1))

    remainder = n - (n0 + n1)
    if remainder > 0:
        headroom0 = len(idx0) - n0
        headroom1 = len(idx1) - n1
        if headroom1 >= headroom0 and headroom1 > 0:
            add1 = min(remainder, headroom1)
            n1 += add1
            remainder -= add1
        if remainder > 0 and headroom0 > 0:
            add0 = min(remainder, headroom0)
            n0 += add0
            remainder -= add0

    if n0 + n1 < n:
        take = rng.choice(idx, size=n, replace=False)
        return df.iloc[take].copy().reset_index(drop=True)

    take0 = rng.choice(idx0, size=n0, replace=False)
    take1 = rng.choice(idx1, size=n1, replace=False)
    take = np.concatenate([take0, take1])
    rng.shuffle(take)
    return df.iloc[take].copy().reset_index(drop=True)


# ----------------------------
# Spatial metrics per sample
# ----------------------------

def calculate_spatial_metrics_for_sample(cells_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if cells_df is None or len(cells_df) < 10:
        return None

    coords = cells_df[["x", "y"]].values
    cell_types = cells_df["Cancer"].values.astype(int)

    metrics: Dict[str, float] = {}
    metrics["cell_count"] = int(len(cells_df))
    metrics["cancer_count"] = int((cell_types == 1).sum())
    metrics["stroma_count"] = int((cell_types == 0).sum())

    metrics.update(calculate_ripley_l(coords, cell_types))
    metrics.update(calculate_bidirectional_min_distance(coords, cell_types))
    metrics["newmans_assortativity"] = float(calculate_newmans_assortativity(coords, cell_types, radius=50))
    metrics.update(calculate_centrality_scores(coords, cell_types))
    metrics.update(calculate_cluster_cooccurrence_ratio(coords, cell_types))
    metrics.update(calculate_neighborhood_enrichment_test(coords, cell_types))
    metrics.update(calculate_objectobject_correlation(coords, cell_types))

    return metrics


def _compute_one_sample_worker(ID: str, sample_name: str, label: int, sample_cells_df: pd.DataFrame) -> Optional[Dict]:
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
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    patients_samples = pd.merge(patients_df, samples_df, on="ID", how="inner")
    cell_groups = {s: g[["x", "y", "Cancer"]].copy() for s, g in cells_df.groupby("sample_name", sort=False)}

    tasks: List[Tuple[str, str, int, pd.DataFrame]] = []
    for _, row in patients_samples.iterrows():
        sname = str(row["sample_name"])
        if sname not in cell_groups:
            continue
        tasks.append((str(row["ID"]), sname, int(row["label"]), cell_groups[sname]))

    if not tasks:
        return pd.DataFrame()

    results: List[Dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_compute_one_sample_worker, *t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Samples"):
            res = f.result()  # fail loudly
            if res is not None:
                results.append(res)

    return pd.DataFrame(results)


# ----------------------------
# Modeling
# ----------------------------

def _build_pipeline(model, selector) -> Pipeline:
    return Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
            ("selector", selector),
            ("model", model),
        ]
    )


def _tune_best_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    use_randomized: bool,
) -> Tuple[str, Pipeline, dict, float]:
    """Tune RF/GB/SVM/LR, return (best_name, best_estimator, best_params, best_score)."""

    models = {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 3, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
            SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
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
            SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE, max_iter=50000),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", "auto", 0.1, 0.01],
                "model__kernel": ["rbf", "linear"],
            },
            SelectFromModel(LinearSVC(penalty="l1", dual=False, C=1.0, max_iter=50000, random_state=RANDOM_STATE)),
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=50000, random_state=RANDOM_STATE),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__penalty": ["l2"],
                "model__solver": ["saga"],
            },
            SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=1.0, random_state=RANDOM_STATE)),
        ),
    }

    best_score = -np.inf
    best_name: Optional[str] = None
    best_pipeline: Optional[Pipeline] = None
    best_params: dict = {}

    for name, (model, params, selector) in models.items():
        print(f"  Tuning {name}...")
        pipe = _build_pipeline(model, selector)

        search = (
            RandomizedSearchCV(
                pipe,
                params,
                n_iter=20,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            if use_randomized
            else GridSearchCV(pipe, params, scoring="roc_auc", cv=cv, n_jobs=-1)
        )

        search.fit(X, y)

        score = float(search.best_score_)
        print(f"    Best {name}: AUC={score:.4f} params={search.best_params_}")

        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = search.best_estimator_
            best_params = dict(search.best_params_)

    if best_pipeline is None or best_name is None:
        raise RuntimeError("All model searches failed; cannot proceed.")

    return best_name, best_pipeline, best_params, best_score


def run_classification_cv(
    train_df: pd.DataFrame,
    n_folds: int = 5,
    hyperparameter_tuning: bool = True,
    return_model: bool = True,
) -> Tuple[float, float, float, float, pd.DataFrame, str, Pipeline, dict]:
    if train_df is None or train_df.empty:
        raise ValueError("train_df is empty")

    feat_cols = get_feature_columns(train_df)
    if not feat_cols:
        raise ValueError("No feature columns found after dropping counts/IDs")

    X = train_df[feat_cols].to_numpy(float)
    y = train_df["label"].to_numpy(int)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    best_pipeline: Pipeline
    best_model_name: str
    best_params: dict

    if hyperparameter_tuning:
        print("[INFO] Hyperparameter tuning...")
        start = time.time()

        use_randomized = len(X) > 200
        best_model_name, best_pipeline, best_params, best_score = _tune_best_pipeline(X, y, cv=cv, use_randomized=use_randomized)

        print(f"[INFO] Selected model: {best_model_name} (tuning time {time.time() - start:.1f}s)")
    else:
        best_model_name = "RandomForest (default)"
        best_params = {}
        best_pipeline = _build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        )

    # CV evaluation using the chosen pipeline
    aucs, accs = [], []
    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        best_pipeline.fit(X_tr, y_tr)
        y_prob = best_pipeline.predict_proba(X_te)[:, 1]
        y_pred = best_pipeline.predict(X_te)
        auc_fold = float(roc_auc_score(y_te, y_prob))
        acc_fold = float(np.mean(y_pred == y_te))
        aucs.append(auc_fold)
        accs.append(acc_fold)


    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    print(f"[CV] {n_folds}-fold: AUC={mean_auc:.3f} ± {std_auc:.3f}, Acc={mean_acc:.3f} ± {std_acc:.3f}")

    # Permutation importance on one split
    tr_idx, te_idx = next(cv.split(X, y))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    best_pipeline.fit(X_tr, y_tr)
    perm = permutation_importance(best_pipeline, X_te, y_te, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc")
    fi = pd.DataFrame({"feature": feat_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False)

    return (
        float(np.mean(aucs)),
        float(np.std(aucs)),
        float(np.mean(accs)),
        float(np.std(accs)),
        fi,
        str(best_model_name),
        best_pipeline,
        best_params,
    )



def run_combined_cv_target_cohort(
    df_train_int: pd.DataFrame,
    df_train_ext_pref: pd.DataFrame,
    pipeline: Pipeline,
    target: str,
    n_folds: int = 5,
) -> Tuple[float, float, float, float]:
    """Anchored CV for combined training, evaluated on one cohort at a time.

    For each fold on the target cohort:
      - Train on: (target train folds) + (all samples from the other cohort)
      - Test on: target held-out fold

    Returns: mean_auc, std_auc, mean_acc, std_acc
    """
    if target not in {"INTERNAL", "EXTERNAL"}:
        raise ValueError(f"target must be INTERNAL or EXTERNAL, got: {target}")

    if target == "INTERNAL":
        df_target = df_train_int.reset_index(drop=True)
        df_other = df_train_ext_pref.reset_index(drop=True)
    else:
        df_target = df_train_ext_pref.reset_index(drop=True)
        df_other = df_train_int.reset_index(drop=True)

    combined_for_cols = pd.concat([df_train_int, df_train_ext_pref], ignore_index=True, sort=False)
    feat_cols = get_feature_columns(combined_for_cols)

    df_target = ensure_columns(df_target, feat_cols)
    df_other = ensure_columns(df_other, feat_cols)

    X_target = df_target[feat_cols].to_numpy(float)
    y_target = df_target["label"].to_numpy(int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    aucs: List[float] = []
    accs: List[float] = []

    for tr_idx, te_idx in skf.split(X_target, y_target):
        fold_train_target = df_target.iloc[tr_idx].copy()
        fold_test_target = df_target.iloc[te_idx].copy()

        fold_train = pd.concat([fold_train_target, df_other], ignore_index=True, sort=False)

        res = evaluate_fast(fold_train, fold_test_target, clone(pipeline))
        aucs.append(float(res["auc"]))
        accs.append(float(res["accuracy"]))

    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(accs)), float(np.std(accs))


def _make_summary_row(
    *,
    bomi2_in_train: bool,
    bomi1_in_train: bool,
    bomi2_train_acc: float,
    bomi2_train_auc: float,
    bomi1_train_acc: float,
    bomi1_train_auc: float,
    bomi2_test_acc: float,
    bomi2_test_auc: float,
    bomi1_test_acc: float,
    bomi1_test_auc: float,
) -> Dict[str, object]:
    return {
        "BOMI2 Train": "x" if bomi2_in_train else "",
        "BOMI1 Train": "x" if bomi1_in_train else "",
        "BOMI2 Train\nAccuracy": float(bomi2_train_acc),
        "BOMI2 Train\nAUC": float(bomi2_train_auc),
        "BOMI1 Train\nAccuracy": float(bomi1_train_acc),
        "BOMI1 Train\nAUC": float(bomi1_train_auc),
        "BOMI2 Test\nAccuracy": float(bomi2_test_acc),
        "BOMI2 Test\nAUC": float(bomi2_test_auc),
        "BOMI1 Test\nAccuracy": float(bomi1_test_acc),
        "BOMI1 Test\nAUC": float(bomi1_test_auc),
    }


def evaluate_on_test_set(train_df: pd.DataFrame, test_df: pd.DataFrame, pipeline: Pipeline) -> Dict:
    feat_cols = get_feature_columns(train_df)
    train_df2 = ensure_columns(train_df, feat_cols)
    test_df2 = ensure_columns(test_df, feat_cols)

    X_tr = train_df2[feat_cols].to_numpy(float)
    y_tr = train_df2["label"].to_numpy(int)
    X_te = test_df2[feat_cols].to_numpy(float)
    y_te = test_df2["label"].to_numpy(int)

    pipeline.fit(X_tr, y_tr)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    y_pred = pipeline.predict(X_te)

    auc = float(roc_auc_score(y_te, y_prob))
    acc = float(np.mean(y_pred == y_te))

    # Permutation importance on the evaluation set (ΔAUC when permuted).
    perm = permutation_importance(
        pipeline,
        X_te,
        y_te,
        n_repeats=25,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
    )
    fi = pd.DataFrame({"feature": feat_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False)

    return {
        "auc": auc,
        "accuracy": acc,
        "y_true": y_te,
        "y_pred": y_pred,
        "y_probs": y_prob,
        "feature_importance": fi,
    }



# ----------------------------
# Fast evaluation (no permutation importance)
# ----------------------------

def _limit_threading_for_workers() -> None:
    """Avoid BLAS/OpenMP oversubscription when using ProcessPoolExecutor."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def evaluate_fast(train_df: pd.DataFrame, test_df: pd.DataFrame, pipeline: Pipeline) -> Dict[str, float]:
    """Evaluate AUC/accuracy only. Intended for learning-curve loops."""
    feat_cols = get_feature_columns(train_df)
    train_df2 = ensure_columns(train_df, feat_cols)
    test_df2 = ensure_columns(test_df, feat_cols)

    X_tr = train_df2[feat_cols].to_numpy(float)
    y_tr = train_df2["label"].to_numpy(int)
    X_te = test_df2[feat_cols].to_numpy(float)
    y_te = test_df2["label"].to_numpy(int)

    pipeline.fit(X_tr, y_tr)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    y_pred = pipeline.predict(X_te)

    return {
        "auc": float(roc_auc_score(y_te, y_prob)),
        "accuracy": float(np.mean(y_pred == y_te)),
    }


def _learning_worker_fixed(task: dict) -> list[dict]:
    _limit_threading_for_workers()

    df_sub = task["df_sub"]
    primary_test = task["primary_test"]
    secondary_train_pref = task["secondary_train_pref"]
    pipe_primary = task["pipe_primary"]
    pipe_mixed = task["pipe_mixed"]

    eval_p = evaluate_fast(df_sub, primary_test, clone(pipe_primary))

    train_mixed = pd.concat([df_sub, secondary_train_pref], ignore_index=True, sort=False)
    eval_m = evaluate_fast(train_mixed, primary_test, clone(pipe_mixed))

    base = {
        "eval_mode": "fixed",
        "primary": task["primary_name"],
        "secondary": task["secondary_name"],
        "train_size": int(task["train_size"]),
        "repeat": int(task["repeat"]),
        "outer_fold": -1,
    }

    return [
        {**base, "regime": "primary_only", "auc": float(eval_p["auc"]), "accuracy": float(eval_p["accuracy"])},
        {**base, "regime": "mixed", "auc": float(eval_m["auc"]), "accuracy": float(eval_m["accuracy"])},
    ]


def _learning_worker_cv(task: dict) -> list[dict]:
    _limit_threading_for_workers()

    df_sub = task["df_sub"]
    df_fold_val = task["df_fold_val"]
    secondary_train_pref = task["secondary_train_pref"]
    pipe_primary = task["pipe_primary"]
    pipe_mixed = task["pipe_mixed"]

    eval_p = evaluate_fast(df_sub, df_fold_val, clone(pipe_primary))

    train_mixed = pd.concat([df_sub, secondary_train_pref], ignore_index=True, sort=False)
    eval_m = evaluate_fast(train_mixed, df_fold_val, clone(pipe_mixed))

    base = {
        "eval_mode": "cv",
        "primary": task["primary_name"],
        "secondary": task["secondary_name"],
        "train_size": int(task["train_size"]),
        "repeat": int(task["repeat"]),
        "outer_fold": int(task["outer_fold"]),
    }

    return [
        {**base, "regime": "primary_only", "auc": float(eval_p["auc"]), "accuracy": float(eval_p["accuracy"])},
        {**base, "regime": "mixed", "auc": float(eval_m["auc"]), "accuracy": float(eval_m["accuracy"])},
    ]

# ----------------------------
# Dataset loading + caching
# ----------------------------

def load_internal_dataset(paths: InternalPaths, source: str = "cellpose") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    if recompute or (not os.path.exists(metrics_cache_path)):
        print(f"[INFO] Computing metrics for {cohort_name} -> {metrics_cache_path}")
        m = compute_metrics_for_cohort(patients_df, samples_df, cells_df, n_jobs=n_jobs)
        if m is None or m.empty:
            raise RuntimeError(
                f"Computed 0 metric rows for {cohort_name}. "
                "This usually means sample_name mismatch between samples_df and cells_df."
            )
        metrics_cache_dir = os.path.dirname(metrics_cache_path)
        if metrics_cache_dir:
            os.makedirs(metrics_cache_dir, exist_ok=True)
        m.to_csv(metrics_cache_path, index=False)
        print(f"[INFO] Saved metrics: {metrics_cache_path}")
    else:
        if os.path.getsize(metrics_cache_path) < 10:
            raise RuntimeError(
                f"Cached metrics file is empty: {metrics_cache_path}. "
                "Delete it or pass --recompute-*-metrics."
            )
        print(f"[INFO] Loading cached metrics for {cohort_name}: {metrics_cache_path}")
        m = pd.read_csv(metrics_cache_path)

    m = attach_counts_from_cells(m, cells_df)
    return m


def prepare_patient_level_tables(
    internal_source: str,
    internal_paths: InternalPaths,
    external_paths: ExternalPaths,
    qc: CountQC,
    recompute_internal_metrics: bool,
    recompute_external_metrics: bool,
    n_jobs: Optional[int],
    external_xy_scale: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return patient-level train/test dataframes for internal and external."""

    # External split paths
    ext_train_csv = external_paths.external_train_csv or os.path.join(external_paths.external_split_dir, "train_val.csv")
    ext_test_csv = external_paths.external_test_csv or os.path.join(external_paths.external_split_dir, "test.csv")

    # INTERNAL
    print(f"\n[LOAD] INTERNAL (BOMI2) source={internal_source}")
    int_cells, int_samples, int_patients_all, int_patients_train, int_patients_test = load_internal_dataset(
        internal_paths, source=internal_source
    )

    internal_metrics_path = os.path.join("outputs", "metrics", f"spatial_metrics_{internal_source}.csv")
    int_metrics_per_core = get_or_compute_metrics(
        cohort_name=f"INTERNAL/{internal_source}",
        metrics_cache_path=internal_metrics_path,
        patients_df=int_patients_all,
        samples_df=int_samples,
        cells_df=int_cells,
        recompute=recompute_internal_metrics,
        n_jobs=n_jobs,
    )

    int_metrics_per_core = filter_metrics_by_counts(int_metrics_per_core, qc=qc, cohort_name="INTERNAL/BOMI2")
    int_patient_df = aggregate_cores_to_patient(int_metrics_per_core)
    for col in ["cancer_ripley_L_0.0", "stroma_ripley_L_0.0"]:
        int_patient_df.drop(columns=[col], errors="ignore", inplace=True)

    train_ids_int = set(int_patients_train["ID"].astype(str))
    test_ids_int = set(int_patients_test["ID"].astype(str))
    df_train_int = _subset_by_ids(int_patient_df, train_ids_int)
    df_test_int = _subset_by_ids(int_patient_df, test_ids_int)

    print(f"[INFO] INTERNAL patients: train={len(df_train_int)} test={len(df_test_int)} (after core QC + aggregation)")

    # EXTERNAL
    print("\n[LOAD] EXTERNAL (BOMI1) H&E")
    ext_cells_raw = pd.read_csv(external_paths.external_cells_path)
    ext_meta_raw = pd.read_csv(external_paths.external_meta_path)
    ext_cells = preprocess_external_cells(ext_cells_raw)
    ext_patients_df, ext_samples_df = split_external_meta(ext_meta_raw)

    ext_cells[["x", "y"]] = ext_cells[["x", "y"]] * float(external_xy_scale)

    external_metrics_path = os.path.join("outputs", "metrics", "spatial_metrics_EXTERNAL_BOMI1.csv")
    ext_metrics_per_core = get_or_compute_metrics(
        cohort_name="EXTERNAL/BOMI1",
        metrics_cache_path=external_metrics_path,
        patients_df=ext_patients_df,
        samples_df=ext_samples_df,
        cells_df=ext_cells,
        recompute=recompute_external_metrics,
        n_jobs=n_jobs,
    )

    ext_metrics_per_core = filter_metrics_by_counts(ext_metrics_per_core, qc=qc, cohort_name="EXTERNAL/BOMI1")
    ext_patient_df = aggregate_cores_to_patient(ext_metrics_per_core)
    for col in ["cancer_ripley_L_0.0", "stroma_ripley_L_0.0"]:
        ext_patient_df.drop(columns=[col], errors="ignore", inplace=True)

    ext_train_ids = _read_split_ids(ext_train_csv)
    ext_test_ids = _read_split_ids(ext_test_csv)
    df_train_ext = _subset_by_ids(ext_patient_df, ext_train_ids)
    df_test_ext = _subset_by_ids(ext_patient_df, ext_test_ids)

    print(f"[INFO] EXTERNAL patients: train={len(df_train_ext)} test={len(df_test_ext)} (after core QC + aggregation)")

    if df_train_int.empty or df_test_int.empty or df_train_ext.empty or df_test_ext.empty:
        raise RuntimeError("One of the train/test splits is empty after QC. Adjust CountQC thresholds.")

    return df_train_int, df_test_int, df_train_ext, df_test_ext


# ----------------------------
# Base experiments
# ----------------------------

def run_base_experiments(
    df_train_int: pd.DataFrame,
    df_test_int: pd.DataFrame,
    df_train_ext: pd.DataFrame,
    df_test_ext: pd.DataFrame,
    hyperparameter_tuning: bool = True,
    tune_combined: bool = True,
    fi_dir: str = os.path.join("outputs", "feature_importance", "feature_importance_bundles"),
) -> None:

    # Collect summary rows for a compact results CSV requested for the paper.
    summary_rows: List[Dict[str, object]] = []

    print("\n[1] CV on INTERNAL train (BOMI2)...")
    auc_i, std_auc_i, acc_i, std_acc_i, fi_i, model_name_i, pipe_i, params_i = run_classification_cv(
        df_train_int, n_folds=5, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[BOMI2 CV] {model_name_i} AUC: {auc_i:.3f} ± {std_auc_i:.3f}, Acc: {acc_i:.3f} ± {std_acc_i:.3f}")

    save_feature_importance_bundle(
        fi_dir,
        cohort="BOMI2",
        regime="cv_train",
        model_name=model_name_i,
        best_params=params_i,
        n_patients_train=len(df_train_int),
        n_patients_eval=len(df_train_int),
        feature_columns=get_feature_columns(df_train_int),
        fi_df=fi_i,
        selected_features=_extract_selected_features(pipe_i, get_feature_columns(df_train_int)),
        notes="Permutation importance computed on one CV split (ΔAUC when permuted).",
    )

    print("\n[2] Train BOMI2 train, test BOMI2 test...")
    eval_i2i = evaluate_on_test_set(df_train_int, df_test_int, clone(pipe_i))
    print(f"[BOMI2→BOMI2 test] AUC: {eval_i2i['auc']:.3f}, Acc: {eval_i2i['accuracy']:.3f}")
    save_feature_importance_bundle(
        fi_dir,
        cohort="BOMI2",
        regime="train_internal__test_internal",
        model_name=model_name_i,
        best_params=params_i,
        n_patients_train=len(df_train_int),
        n_patients_eval=len(df_test_int),
        feature_columns=get_feature_columns(df_train_int),
        fi_df=eval_i2i["feature_importance"],
        notes="Permutation importance computed on BOMI2 test set using model trained on BOMI2 train.",
    )
    _save_predictions(os.path.join("outputs", "predictions", "pred_internal_train_to_internal_test.csv"), df_test_int, eval_i2i)

    print("\n[3] Train BOMI2 train, test BOMI1 test...")
    eval_i2e = evaluate_on_test_set(df_train_int, df_test_ext, clone(pipe_i))
    print(f"[BOMI2→BOMI1 test] AUC: {eval_i2e['auc']:.3f}, Acc: {eval_i2e['accuracy']:.3f}")
    save_feature_importance_bundle(
        fi_dir,
        cohort="BOMI2",
        regime="train_internal__test_external",
        model_name=model_name_i,
        best_params=params_i,
        n_patients_train=len(df_train_int),
        n_patients_eval=len(df_test_ext),
        feature_columns=get_feature_columns(df_train_int),
        fi_df=eval_i2e["feature_importance"],
        notes="Permutation importance computed on BOMI1 test set using model trained on BOMI2 train.",
    )
    _save_predictions(os.path.join("outputs", "predictions", "pred_internal_train_to_external_test.csv"), df_test_ext, eval_i2e)

    # If BOMI1 is not in train, report BOMI1 "train" metrics by evaluating the BOMI2-trained model on the full BOMI1 train set.
    eval_i_on_ext_train = evaluate_fast(df_train_int, df_train_ext, clone(pipe_i))

    summary_rows.append(
        _make_summary_row(
            bomi2_in_train=True,
            bomi1_in_train=False,
            bomi2_train_acc=acc_i,
            bomi2_train_auc=auc_i,
            bomi1_train_acc=eval_i_on_ext_train["accuracy"],
            bomi1_train_auc=eval_i_on_ext_train["auc"],
            bomi2_test_acc=eval_i2i["accuracy"],
            bomi2_test_auc=eval_i2i["auc"],
            bomi1_test_acc=eval_i2e["accuracy"],
            bomi1_test_auc=eval_i2e["auc"],
        )
    )

    print("\n[4] CV on EXTERNAL train (BOMI1)...")
    auc_e, std_auc_e, acc_e, std_acc_e, fi_e, model_name_e, pipe_e, params_e = run_classification_cv(
        df_train_ext, n_folds=5, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[BOMI1 CV] {model_name_e} AUC: {auc_e:.3f} ± {std_auc_e:.3f}, Acc: {acc_e:.3f} ± {std_acc_e:.3f}")

    save_feature_importance_bundle(
        fi_dir,
        cohort="BOMI1",
        regime="cv_train",
        model_name=model_name_e,
        best_params=params_e,
        n_patients_train=len(df_train_ext),
        n_patients_eval=len(df_train_ext),
        feature_columns=get_feature_columns(df_train_ext),
        fi_df=fi_e,
        selected_features=_extract_selected_features(pipe_e, get_feature_columns(df_train_ext)),
        notes="Permutation importance computed on one CV split (ΔAUC when permuted).",
    )

    print("\n[5] Train BOMI1 train, test BOMI1 test...")
    eval_e2e = evaluate_on_test_set(df_train_ext, df_test_ext, clone(pipe_e))
    print(f"[BOMI1→BOMI1 test] AUC: {eval_e2e['auc']:.3f}, Acc: {eval_e2e['accuracy']:.3f}")
    save_feature_importance_bundle(
        fi_dir,
        cohort="BOMI1",
        regime="train_external__test_external",
        model_name=model_name_e,
        best_params=params_e,
        n_patients_train=len(df_train_ext),
        n_patients_eval=len(df_test_ext),
        feature_columns=get_feature_columns(df_train_ext),
        fi_df=eval_e2e["feature_importance"],
        notes="Permutation importance computed on BOMI1 test set using model trained on BOMI1 train.",
    )
    _save_predictions(os.path.join("outputs", "predictions", "pred_external_train_to_external_test.csv"), df_test_ext, eval_e2e)

    print("\n[6] Train BOMI1 train, test BOMI2 test...")
    eval_e2i = evaluate_on_test_set(df_train_ext, df_test_int, clone(pipe_e))
    print(f"[BOMI1→BOMI2 test] AUC: {eval_e2i['auc']:.3f}, Acc: {eval_e2i['accuracy']:.3f}")
    save_feature_importance_bundle(
        fi_dir,
        cohort="BOMI1",
        regime="train_external__test_internal",
        model_name=model_name_e,
        best_params=params_e,
        n_patients_train=len(df_train_ext),
        n_patients_eval=len(df_test_int),
        feature_columns=get_feature_columns(df_train_ext),
        fi_df=eval_e2i["feature_importance"],
        notes="Permutation importance computed on BOMI2 test set using model trained on BOMI1 train.",
    )
    _save_predictions(os.path.join("outputs", "predictions", "pred_external_train_to_internal_test.csv"), df_test_int, eval_e2i)

    # If BOMI2 is not in train, report BOMI2 "train" metrics by evaluating the BOMI1-trained model on the full BOMI2 train set.
    eval_e_on_int_train = evaluate_fast(df_train_ext, df_train_int, clone(pipe_e))

    summary_rows.append(
        _make_summary_row(
            bomi2_in_train=False,
            bomi1_in_train=True,
            bomi2_train_acc=eval_e_on_int_train["accuracy"],
            bomi2_train_auc=eval_e_on_int_train["auc"],
            bomi1_train_acc=acc_e,
            bomi1_train_auc=auc_e,
            bomi2_test_acc=eval_e2i["accuracy"],
            bomi2_test_auc=eval_e2i["auc"],
            bomi1_test_acc=eval_e2e["accuracy"],
            bomi1_test_auc=eval_e2e["auc"],
        )
    )

    # Combined train
    print("\n[7] COMBINED train (BOMI2+BOMI1): tune/select model on combined pool...")
    df_train_ext_pref = _prefix_ids(df_train_ext, prefix="BOMI1_")
    combined_train = pd.concat([df_train_int, df_train_ext_pref], ignore_index=True, sort=False)

    if tune_combined:
        print("[INFO] Tuning model on COMBINED train...")
        _, _, _, _, fi_c, model_name_c, pipe_c, params_c = run_classification_cv(
            combined_train, n_folds=5, hyperparameter_tuning=hyperparameter_tuning, return_model=True
        )
        print(f"[INFO] Selected COMBINED model: {model_name_c}")

        if fi_c is not None:
            save_feature_importance_bundle(
                fi_dir,
                cohort="COMBINED",
                regime="cv_train",
                model_name=model_name_c,
                best_params=params_c,
                n_patients_train=len(combined_train),
                n_patients_eval=len(combined_train),
                feature_columns=get_feature_columns(combined_train),
                fi_df=fi_c,
                selected_features=_extract_selected_features(pipe_c, get_feature_columns(combined_train)),
                notes="Permutation importance computed on one CV split (ΔAUC when permuted) for the COMBINED training pool.",
            )
    else:
        pipe_c = pipe_i
        model_name_c = model_name_i
        params_c = params_i

    # Cohort-pure CV metrics for the combined pool (two CV scores: one per cohort)
    auc_ci, std_auc_ci, acc_ci, std_acc_ci = run_combined_cv_target_cohort(
        df_train_int=df_train_int,
        df_train_ext_pref=df_train_ext_pref,
        pipeline=pipe_c,
        target="INTERNAL",
        n_folds=5,
    )
    print(f"[COMBINED CV | test BOMI2-only] AUC: {auc_ci:.3f} ± {std_auc_ci:.3f}, Acc: {acc_ci:.3f} ± {std_acc_ci:.3f}")

    auc_ce, std_auc_ce, acc_ce, std_acc_ce = run_combined_cv_target_cohort(
        df_train_int=df_train_int,
        df_train_ext_pref=df_train_ext_pref,
        pipeline=pipe_c,
        target="EXTERNAL",
        n_folds=5,
    )
    print(f"[COMBINED CV | test BOMI1-only] AUC: {auc_ce:.3f} ± {std_auc_ce:.3f}, Acc: {acc_ce:.3f} ± {std_acc_ce:.3f}")

    print("\n[8] Train COMBINED train, test BOMI2 test...")
    eval_c2i = evaluate_on_test_set(combined_train, df_test_int, clone(pipe_c))
    print(f"[COMBINED→BOMI2 test] AUC: {eval_c2i['auc']:.3f}, Acc: {eval_c2i['accuracy']:.3f}")
    save_feature_importance_bundle(
        fi_dir,
        cohort="COMBINED",
        regime="train_combined__test_internal",
        model_name=model_name_c,
        best_params=params_c,
        n_patients_train=len(combined_train),
        n_patients_eval=len(df_test_int),
        feature_columns=get_feature_columns(combined_train),
        fi_df=eval_c2i["feature_importance"],
        notes="Permutation importance computed on BOMI2 test set using model trained on COMBINED train.",
    )
    _save_predictions(os.path.join("outputs", "predictions", "pred_combined_train_to_internal_test.csv"), df_test_int, eval_c2i)

    print("\n[9] Train COMBINED train, test BOMI1 test...")
    eval_c2e = evaluate_on_test_set(combined_train, df_test_ext, clone(pipe_c))
    print(f"[COMBINED→BOMI1 test] AUC: {eval_c2e['auc']:.3f}, Acc: {eval_c2e['accuracy']:.3f}")
    save_feature_importance_bundle(
        fi_dir,
        cohort="COMBINED",
        regime="train_combined__test_external",
        model_name=model_name_c,
        best_params=params_c,
        n_patients_train=len(combined_train),
        n_patients_eval=len(df_test_ext),
        feature_columns=get_feature_columns(combined_train),
        fi_df=eval_c2e["feature_importance"],
        notes="Permutation importance computed on BOMI1 test set using model trained on COMBINED train.",
    )
    _save_predictions(os.path.join("outputs", "predictions", "pred_combined_train_to_external_test.csv"), df_test_ext, eval_c2e)

    summary_rows.append(
        _make_summary_row(
            bomi2_in_train=True,
            bomi1_in_train=True,
            bomi2_train_acc=acc_ci,
            bomi2_train_auc=auc_ci,
            bomi1_train_acc=acc_ce,
            bomi1_train_auc=auc_ce,
            bomi2_test_acc=eval_c2i["accuracy"],
            bomi2_test_auc=eval_c2i["auc"],
            bomi1_test_acc=eval_c2e["accuracy"],
            bomi1_test_auc=eval_c2e["auc"],
        )
    )

    # Write the requested compact summary CSV
    summary_df = pd.DataFrame(summary_rows, columns=[
        "BOMI2 Train",
        "BOMI1 Train",
        "BOMI2 Train\nAccuracy",
        "BOMI2 Train\nAUC",
        "BOMI1 Train\nAccuracy",
        "BOMI1 Train\nAUC",
        "BOMI2 Test\nAccuracy",
        "BOMI2 Test\nAUC",
        "BOMI1 Test\nAccuracy",
        "BOMI1 Test\nAUC",
    ])

    out_csv = os.path.join("outputs", "metrics", "cohort_mixing_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[SUMMARY] Saved cohort mixing summary to {out_csv}")

    print("\n[INFO] Base experiments completed.")


# ----------------------------
# Learning-curve analysis
# ----------------------------

def parse_train_sizes(s: str, n_train: int) -> List[int]:
    """Parse comma-separated list of ints or fractions in (0,1]."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return []

    vals = []
    for p in parts:
        if "." in p:
            frac = float(p)
            if not (0 < frac <= 1.0):
                raise ValueError(f"Fractional train size must be in (0,1]. Got: {p}")
            vals.append(int(max(4, round(frac * n_train))))
        else:
            vals.append(int(p))

    vals = sorted(set([v for v in vals if v >= 4 and v <= n_train]))
    if n_train not in vals:
        vals.append(n_train)
    return vals


def default_train_sizes(n_train: int) -> List[int]:
    candidates = [10, 20, 30, 50, 80, 120, 160, 200]
    vals = [v for v in candidates if v < n_train]
    vals.append(n_train)
    vals = sorted(set([v for v in vals if v >= 4]))
    return vals




def _compute_learning_deltas(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mixed-minus-primary deltas per (primary, secondary, train_size, repeat, outer_fold)."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    key_cols = ["eval_mode", "primary", "secondary", "train_size", "repeat", "outer_fold"]
    needed = set(key_cols + ["regime", "auc", "accuracy"])
    missing = needed - set(raw_df.columns)
    if missing:
        raise ValueError(f"raw_df missing columns for delta computation: {missing}")

    pivot = (
        raw_df[key_cols + ["regime", "auc", "accuracy"]]
        .pivot_table(index=key_cols, columns="regime", values=["auc", "accuracy"], aggfunc="mean")
    )

    for metric in ["auc", "accuracy"]:
        if (metric, "mixed") not in pivot.columns or (metric, "primary_only") not in pivot.columns:
            raise ValueError(f"Delta computation requires both regimes for metric '{metric}'.")

    delta = pd.DataFrame(index=pivot.index).reset_index()
    delta["auc_delta"] = (pivot[("auc", "mixed")] - pivot[("auc", "primary_only")]).to_numpy(float)
    delta["acc_delta"] = (pivot[("accuracy", "mixed")] - pivot[("accuracy", "primary_only")]).to_numpy(float)
    return delta


def _summarize_learning_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df is None or delta_df.empty:
        return pd.DataFrame()
    return (
        delta_df.groupby(["eval_mode", "primary", "secondary", "train_size"], sort=False)
        .agg(
            auc_delta_mean=("auc_delta", "mean"),
            auc_delta_sem=("auc_delta", lambda x: sem(list(x))),
            acc_delta_mean=("acc_delta", "mean"),
            acc_delta_sem=("acc_delta", lambda x: sem(list(x))),
            n=("auc_delta", "size"),
        )
        .reset_index()
    )

def _summarize_learning(raw_df: pd.DataFrame) -> pd.DataFrame:
    return (
        raw_df.groupby(["eval_mode", "primary", "secondary", "regime", "train_size"], sort=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=("auc", lambda x: sem(list(x))),
            acc_mean=("accuracy", "mean"),
            acc_sem=("accuracy", lambda x: sem(list(x))),
            n=("auc", "size"),
        )
        .reset_index()
    )


def _plot_learning(summary_df: pd.DataFrame, out_dir: str, suffix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _plot(metric: str, ylabel: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        for (primary, secondary, regime), g in summary_df.groupby(["primary", "secondary", "regime"], sort=False):
            g = g.sort_values("train_size")
            if metric == "auc":
                y = g["auc_mean"].to_numpy(float)
                yerr = g["auc_sem"].to_numpy(float)
            else:
                y = g["acc_mean"].to_numpy(float)
                yerr = g["acc_sem"].to_numpy(float)

            label = f"{primary} only" if regime == "primary_only" else f"{primary}+{secondary}"
            linestyle = "-" if regime == "primary_only" else "--"
            ax.errorbar(
                g["train_size"].to_numpy(int),
                y,
                yerr=yerr,
                marker="o",
                linestyle=linestyle,
                capsize=3,
                label=label,
            )

        ax.set_xlabel("Primary training set size (patients)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[LEARNING] Saved plot to {out_path}")

    _plot("auc", "AUC", f"learning_curve_auc_{suffix}.png")
    _plot("acc", "Accuracy", f"learning_curve_accuracy_{suffix}.png")



def run_learning_curve_one_direction_fixed(
    primary_name: str,
    secondary_name: str,
    primary_train: pd.DataFrame,
    primary_test: pd.DataFrame,
    secondary_train: pd.DataFrame,
    train_sizes: List[int],
    n_repeats: int,
    tune_folds: int,
    hyperparameter_tuning: bool,
    out_dir: str,
    seed: int = RANDOM_STATE,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Fixed-test learning curve (evaluated on primary_test).

    Speedups:
    - Uses evaluate_fast (no permutation importance) inside the learning-curve loop.
    - Parallelizes (train_size, repeat) tasks with ProcessPoolExecutor.
    """

    os.makedirs(out_dir, exist_ok=True)

    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1 (or -1). Got: {n_jobs}")

    secondary_train_pref = _prefix_ids(secondary_train, prefix=f"{secondary_name}_")

    print(f"\n[LEARNING-fixed] Tuning primary-only model for {primary_name}...")
    _, _, _, _, _, model_p, pipe_primary, params_p = run_classification_cv(
        primary_train, n_folds=tune_folds, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[LEARNING-fixed] {primary_name} primary-only selected: {model_p} params={params_p}")

    print(f"\n[LEARNING-fixed] Tuning mixed model for {primary_name}+{secondary_name}...")
    mixed_full = pd.concat([primary_train, secondary_train_pref], ignore_index=True, sort=False)
    _, _, _, _, _, model_m, pipe_mixed, params_m = run_classification_cv(
        mixed_full, n_folds=tune_folds, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[LEARNING-fixed] {primary_name}+{secondary_name} mixed selected: {model_m} params={params_m}")

    tasks = []
    for n in train_sizes:
        for r in range(n_repeats):
            df_sub = stratified_subsample(primary_train, n=n, seed=seed + 1000 * n + r)
            tasks.append(
                dict(
                    primary_name=primary_name,
                    secondary_name=secondary_name,
                    train_size=int(n),
                    repeat=int(r),
                    df_sub=df_sub,
                    primary_test=primary_test,
                    secondary_train_pref=secondary_train_pref,
                    pipe_primary=pipe_primary,
                    pipe_mixed=pipe_mixed,
                )
            )

    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        futures = [ex.submit(_learning_worker_fixed, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"LEARNING-fixed {primary_name}"):
            out_rows = f.result()  # fail loudly
            rows.extend(out_rows)

    raw_df = pd.DataFrame(rows).sort_values(["train_size", "repeat", "regime"]).reset_index(drop=True)

    raw_path = os.path.join(out_dir, f"learning_curve_raw_fixed_{primary_name}_test_{primary_name}.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[LEARNING-fixed] Saved raw results to {raw_path}")

    summary = _summarize_learning(raw_df)
    summary_path = os.path.join(out_dir, f"learning_curve_summary_fixed_{primary_name}_test_{primary_name}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[LEARNING-fixed] Saved summary to {summary_path}")

    # Deltas: mixed - primary_only
    delta_df = _compute_learning_deltas(raw_df)
    delta_path = os.path.join(out_dir, f"learning_curve_delta_raw_fixed_{primary_name}_test_{primary_name}.csv")
    delta_df.to_csv(delta_path, index=False)
    delta_summary = _summarize_learning_deltas(delta_df)
    delta_summary_path = os.path.join(out_dir, f"learning_curve_delta_summary_fixed_{primary_name}_test_{primary_name}.csv")
    delta_summary.to_csv(delta_summary_path, index=False)
    print(f"[LEARNING-fixed] Saved delta summary to {delta_summary_path}")

    if not delta_df.empty:
        print(
        f"[LEARNING-fixed] Average gain ({primary_name}+{secondary_name} vs {primary_name} only): "
        f"ΔAUC={delta_df['auc_delta'].mean():.3f}, ΔAcc={delta_df['acc_delta'].mean():.3f} "
        f"(n={len(delta_df)})"
        )

    return raw_df



def run_learning_curve_one_direction_cv(
    primary_name: str,
    secondary_name: str,
    primary_train_pool: pd.DataFrame,
    secondary_train_pool: pd.DataFrame,
    train_sizes: List[int],
    n_repeats: int,
    tune_folds: int,
    outer_folds: int,
    hyperparameter_tuning: bool,
    out_dir: str,
    seed: int = RANDOM_STATE,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Cross-validated learning curve evaluated on folds of the primary training pool.

    Speedups:
    - Uses evaluate_fast (no permutation importance) inside the learning-curve loop.
    - Parallelizes (outer_fold, train_size, repeat) tasks with ProcessPoolExecutor.
    """

    os.makedirs(out_dir, exist_ok=True)

    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1 (or -1). Got: {n_jobs}")

    secondary_train_pref = _prefix_ids(secondary_train_pool, prefix=f"{secondary_name}_")

    # Tune once on full pools (keeps original behavior)
    print(f"\n[LEARNING-cv] Tuning primary-only model for {primary_name} (on full primary train pool)...")
    _, _, _, _, _, model_p, pipe_primary, params_p = run_classification_cv(
        primary_train_pool, n_folds=tune_folds, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[LEARNING-cv] {primary_name} primary-only selected: {model_p} params={params_p}")

    print(f"\n[LEARNING-cv] Tuning mixed model for {primary_name}+{secondary_name} (on full combined train)...")
    mixed_full = pd.concat([primary_train_pool, secondary_train_pref], ignore_index=True, sort=False)
    _, _, _, _, _, model_m, pipe_mixed, params_m = run_classification_cv(
        mixed_full, n_folds=tune_folds, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[LEARNING-cv] {primary_name}+{secondary_name} mixed selected: {model_m} params={params_m}")

    y = primary_train_pool["label"].astype(int).to_numpy()
    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    tasks = []
    for fold_idx, (idx_tr, idx_va) in enumerate(outer.split(np.zeros_like(y), y), start=0):
        df_fold_train = primary_train_pool.iloc[idx_tr].reset_index(drop=True)
        df_fold_val = primary_train_pool.iloc[idx_va].reset_index(drop=True)

        if df_fold_val["label"].nunique() < 2:
            raise RuntimeError(f"Outer fold {fold_idx} validation has <2 classes; cannot compute AUC.")

        for n in train_sizes:
            if n > len(df_fold_train):
                continue
            for r in range(n_repeats):
                df_sub = stratified_subsample(df_fold_train, n=n, seed=seed + 100000 * fold_idx + 1000 * n + r)
                tasks.append(
                    dict(
                        primary_name=primary_name,
                        secondary_name=secondary_name,
                        outer_fold=int(fold_idx),
                        train_size=int(n),
                        repeat=int(r),
                        df_sub=df_sub,
                        df_fold_val=df_fold_val,
                        secondary_train_pref=secondary_train_pref,
                        pipe_primary=pipe_primary,
                        pipe_mixed=pipe_mixed,
                    )
                )

    rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        futures = [ex.submit(_learning_worker_cv, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"LEARNING-cv {primary_name}"):
            out_rows = f.result()  # fail loudly
            rows.extend(out_rows)

    raw_df = pd.DataFrame(rows).sort_values(["outer_fold", "train_size", "repeat", "regime"]).reset_index(drop=True)

    raw_path = os.path.join(out_dir, f"learning_curve_raw_cv_{primary_name}_valfolds.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[LEARNING-cv] Saved raw results to {raw_path}")

    summary = _summarize_learning(raw_df)
    summary_path = os.path.join(out_dir, f"learning_curve_summary_cv_{primary_name}_valfolds.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[LEARNING-cv] Saved summary to {summary_path}")

    # Deltas: mixed - primary_only
    delta_df = _compute_learning_deltas(raw_df)
    delta_path = os.path.join(out_dir, f"learning_curve_delta_raw_cv_{primary_name}_valfolds.csv")
    delta_df.to_csv(delta_path, index=False)
    delta_summary = _summarize_learning_deltas(delta_df)
    delta_summary_path = os.path.join(out_dir, f"learning_curve_delta_summary_cv_{primary_name}_valfolds.csv")
    delta_summary.to_csv(delta_summary_path, index=False)
    print(f"[LEARNING-cv] Saved delta summary to {delta_summary_path}")

    if not delta_df.empty:
        print(
        f"[LEARNING-cv] Average gain ({primary_name}+{secondary_name} vs {primary_name} only): "
        f"ΔAUC={delta_df['auc_delta'].mean():.3f}, ΔAcc={delta_df['acc_delta'].mean():.3f} "
        f"(n={len(delta_df)})"
        )

    return raw_df


def run_learning_curve_analysis(
    df_train_int: pd.DataFrame,
    df_test_int: pd.DataFrame,
    df_train_ext: pd.DataFrame,
    df_test_ext: pd.DataFrame,
    train_sizes_internal: List[int],
    train_sizes_external: List[int],
    n_repeats: int,
    tune_folds: int,
    outer_folds: int,
    hyperparameter_tuning: bool,
    out_dir: str,
    eval_mode: str,
    n_jobs: Optional[int] = None,
) -> None:
    if eval_mode not in {"fixed", "cv"}:
        raise ValueError("eval_mode must be 'fixed' or 'cv'")

    all_raw = []

    if eval_mode == "fixed":
        raw_i = run_learning_curve_one_direction_fixed(
            primary_name="BOMI2",
            secondary_name="BOMI1",
            primary_train=df_train_int,
            primary_test=df_test_int,
            secondary_train=df_train_ext,
            train_sizes=train_sizes_internal,
            n_repeats=n_repeats,
            tune_folds=tune_folds,
            hyperparameter_tuning=hyperparameter_tuning,
            out_dir=out_dir,
            n_jobs=n_jobs,
        )
        raw_e = run_learning_curve_one_direction_fixed(
            primary_name="BOMI1",
            secondary_name="BOMI2",
            primary_train=df_train_ext,
            primary_test=df_test_ext,
            secondary_train=df_train_int,
            train_sizes=train_sizes_external,
            n_repeats=n_repeats,
            tune_folds=tune_folds,
            hyperparameter_tuning=hyperparameter_tuning,
            out_dir=out_dir,
            n_jobs=n_jobs,
        )
        all_raw = [raw_i, raw_e]

    else:
        raw_i = run_learning_curve_one_direction_cv(
            primary_name="BOMI2",
            secondary_name="BOMI1",
            primary_train_pool=df_train_int,
            secondary_train_pool=df_train_ext,
            train_sizes=train_sizes_internal,
            n_repeats=n_repeats,
            tune_folds=tune_folds,
            outer_folds=outer_folds,
            hyperparameter_tuning=hyperparameter_tuning,
            out_dir=out_dir,
            n_jobs=n_jobs,
        )
        raw_e = run_learning_curve_one_direction_cv(
            primary_name="BOMI1",
            secondary_name="BOMI2",
            primary_train_pool=df_train_ext,
            secondary_train_pool=df_train_int,
            train_sizes=train_sizes_external,
            n_repeats=n_repeats,
            tune_folds=tune_folds,
            outer_folds=outer_folds,
            hyperparameter_tuning=hyperparameter_tuning,
            out_dir=out_dir,
            n_jobs=n_jobs,
        )
        all_raw = [raw_i, raw_e]

    raw_all = pd.concat(all_raw, ignore_index=True)
    raw_all_path = os.path.join(out_dir, f"learning_curve_raw_ALL_{eval_mode}.csv")
    raw_all.to_csv(raw_all_path, index=False)
    print(f"[LEARNING] Saved combined raw to {raw_all_path}")

    summary_all = _summarize_learning(raw_all)
    summary_all_path = os.path.join(out_dir, f"learning_curve_summary_ALL_{eval_mode}.csv")
    summary_all.to_csv(summary_all_path, index=False)
    print(f"[LEARNING] Saved combined summary to {summary_all_path}")

    _plot_learning(summary_all, out_dir=out_dir, suffix=eval_mode)

    # Also compute and save deltas (mixed - primary_only) aggregated across both directions
    delta_all = _compute_learning_deltas(raw_all)
    delta_all_path = os.path.join(out_dir, f"learning_curve_delta_raw_ALL_{eval_mode}.csv")
    delta_all.to_csv(delta_all_path, index=False)
    print(f"[LEARNING] Saved combined delta raw to {delta_all_path}")

    delta_summary_all = _summarize_learning_deltas(delta_all)
    delta_summary_all_path = os.path.join(out_dir, f"learning_curve_delta_summary_ALL_{eval_mode}.csv")
    delta_summary_all.to_csv(delta_summary_all_path, index=False)
    print(f"[LEARNING] Saved combined delta summary to {delta_summary_all_path}")

    if not delta_all.empty:
        by_dir = delta_all.groupby(["primary", "secondary"], sort=False).agg(
            auc_delta_mean=("auc_delta", "mean"),
            acc_delta_mean=("acc_delta", "mean"),
            n=("auc_delta", "size"),
        ).reset_index()

        print("[LEARNING] Average gain by direction (Δ = mixed - primary_only):")
        for _, row in by_dir.iterrows():
            primary = row["primary"]
            secondary = row["secondary"]
            print(
                f"  Primary={primary}, add Secondary={secondary} "
                f"(mixed={primary}+{secondary} vs primary_only={primary}): "
                f"ΔAUC={row['auc_delta_mean']:.3f}, ΔAcc={row['acc_delta_mean']:.3f} (n={int(row['n'])})"
            )

# ----------------------------
# Entrypoint
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spatial metrics + cohort mixing analysis")

    p.add_argument("--internal-source", type=str, default="cellpose", choices=["inform", "cellpose", "cellprofiler"])
    p.add_argument("--recompute-internal-metrics", action="store_true")
    p.add_argument("--recompute-external-metrics", action="store_true")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--external-xy-scale", type=float, default=2.0)

    # QC
    p.add_argument("--qc-min-total", type=int, default=100)
    p.add_argument("--qc-min-cancer", type=int, default=30)
    p.add_argument("--qc-min-stroma", type=int, default=30)

    # Modeling
    p.add_argument("--no-hyperparameter-tuning", action="store_true")
    p.add_argument("--no-tune-combined", action="store_true")
    p.add_argument(
        "--fi-dir",
        type=str,
        default=os.path.join("outputs", "feature_importance", "feature_importance_bundles"),
        help="Output directory for feature-importance bundles (CSV+JSON).",
    )

    # Learning curve
    p.add_argument("--run-learning-curve", action="store_true")
    p.add_argument("--learning-eval", type=str, default="fixed", choices=["fixed", "cv"],
                   help="Learning-curve evaluation: fixed test split or CV on training pool")
    p.add_argument("--learning-outer-folds", type=int, default=5, help="Outer folds for learning-eval=cv")
    p.add_argument("--learning-repeats", type=int, default=30)
    p.add_argument("--learning-tune-folds", type=int, default=5)
    p.add_argument("--learning-train-sizes-internal", type=str, default="")
    p.add_argument("--learning-train-sizes-external", type=str, default="")
    p.add_argument("--learning-out-dir", type=str, default=os.path.join("outputs", "learning_curves", "learning_curve_out"))

    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_global_seed(RANDOM_STATE)

    qc = CountQC(min_total=args.qc_min_total, min_cancer=args.qc_min_cancer, min_stroma=args.qc_min_stroma)
    hyperparameter_tuning = (not args.no_hyperparameter_tuning)
    tune_combined = (not args.no_tune_combined)

    df_train_int, df_test_int, df_train_ext, df_test_ext = prepare_patient_level_tables(
        internal_source=args.internal_source,
        internal_paths=InternalPaths(),
        external_paths=ExternalPaths(),
        qc=qc,
        recompute_internal_metrics=args.recompute_internal_metrics,
        recompute_external_metrics=args.recompute_external_metrics,
        n_jobs=args.n_jobs,
        external_xy_scale=args.external_xy_scale,
    )

    run_base_experiments(
        df_train_int=df_train_int,
        df_test_int=df_test_int,
        df_train_ext=df_train_ext,
        df_test_ext=df_test_ext,
        hyperparameter_tuning=hyperparameter_tuning,
        tune_combined=tune_combined,
        fi_dir=args.fi_dir,
    )

    if args.run_learning_curve:
        if args.learning_train_sizes_internal.strip():
            sizes_int = parse_train_sizes(args.learning_train_sizes_internal, n_train=len(df_train_int))
        else:
            sizes_int = default_train_sizes(len(df_train_int))

        if args.learning_train_sizes_external.strip():
            sizes_ext = parse_train_sizes(args.learning_train_sizes_external, n_train=len(df_train_ext))
        else:
            sizes_ext = default_train_sizes(len(df_train_ext))

        print(f"\n[LEARNING] eval_mode={args.learning_eval} outer_folds={args.learning_outer_folds} repeats={args.learning_repeats}")
        print(f"[LEARNING] Internal train sizes: {sizes_int}")
        print(f"[LEARNING] External train sizes: {sizes_ext}")

        run_learning_curve_analysis(
            df_train_int=df_train_int,
            df_test_int=df_test_int,
            df_train_ext=df_train_ext,
            df_test_ext=df_test_ext,
            train_sizes_internal=sizes_int,
            train_sizes_external=sizes_ext,
            n_repeats=args.learning_repeats,
            tune_folds=args.learning_tune_folds,
            outer_folds=args.learning_outer_folds,
            hyperparameter_tuning=hyperparameter_tuning,
            out_dir=args.learning_out_dir,
            eval_mode=args.learning_eval,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
