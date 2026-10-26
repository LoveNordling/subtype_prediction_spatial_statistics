"""spatial_statistics5.py

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
import json
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

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

from spatial_artifact_utils import (
    extract_selected_features as _extract_selected_features,
    make_summary_row as _make_summary_row,
    prefix_ids as _prefix_ids,
    sem as _sem,
    save_feature_importance_bundle,
    save_predictions as _save_predictions,
)
from spatial_data_pipeline import ExternalPaths, InternalPaths, prepare_patient_level_tables
from spatial_learning_curves import (
    default_train_sizes,
    evaluate_fast,
    parse_train_sizes,
    run_learning_curve_analysis,
)
from spatial_metrics import BEHANOVA_FEATURE_COLUMNS
from spatial_utils import (
    CountQC,
    ensure_columns,
    get_feature_columns as _get_feature_columns,
)


RANDOM_STATE = 42

MODEL_FEATURE_COLUMNS = list(BEHANOVA_FEATURE_COLUMNS)
FEATURE_DIRECTION_POLICIES = ("none", "keep-consistent", "invert-discordant")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the locked manuscript feature panel when present.

    This prevents stale cached aliases or exploratory numeric columns from
    silently entering the model feature matrix.
    """

    present = [
        col
        for col in MODEL_FEATURE_COLUMNS
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    return present if present else _get_feature_columns(df)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)

# ----------------------------
# Modeling
# ----------------------------

def _build_pipeline(model, selector=None) -> Pipeline:
    steps = [
        ("scaler", MinMaxScaler()),
        ("oversample", RandomOverSampler(random_state=RANDOM_STATE)),
    ]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", model))
    return Pipeline(steps)


def _clone_with_seed(estimator, seed: int):
    cloned = clone(estimator)
    updates = {
        name: seed
        for name in cloned.get_params(deep=True)
        if name.endswith("random_state")
    }
    if updates:
        cloned = cloned.set_params(**updates)
    return cloned


def _tune_best_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    use_randomized: bool,
    use_feature_selection: bool,
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
        pipe = _build_pipeline(model, selector if use_feature_selection else None)

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
    use_feature_selection: bool = True,
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
        best_model_name, best_pipeline, best_params, best_score = _tune_best_pipeline(
            X,
            y,
            cv=cv,
            use_randomized=use_randomized,
            use_feature_selection=use_feature_selection,
        )

        print(f"[INFO] Selected model: {best_model_name} (tuning time {time.time() - start:.1f}s)")
    else:
        best_model_name = "RandomForest (default)"
        best_params = {}
        best_pipeline = _build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
            if use_feature_selection
            else None,
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


def _validated_stratified_splits(
    y: np.ndarray,
    requested_splits: int,
    *,
    label: str,
    random_state: int,
) -> Tuple[StratifiedKFold, int]:
    """Return a StratifiedKFold that is feasible for the smallest class."""

    requested_splits = int(requested_splits)
    if requested_splits < 2:
        raise ValueError(f"{label} must use at least 2 folds, got {requested_splits}")

    counts = pd.Series(y).value_counts(dropna=False)
    min_class_count = int(counts.min()) if not counts.empty else 0
    if min_class_count < 2:
        raise ValueError(
            f"{label} requires at least 2 samples in each class; "
            f"class counts were {counts.to_dict()}."
        )

    n_splits = min(requested_splits, min_class_count)
    if n_splits < requested_splits:
        print(
            f"[WARN] Reducing {label} from {requested_splits} to {n_splits} folds "
            f"because the smallest class has {min_class_count} samples."
        )

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_splits


def run_nested_classification_cv(
    train_df: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    hyperparameter_tuning: bool = True,
    use_feature_selection: bool = False,
) -> Dict[str, object]:
    """Estimate one classifier with nested CV for segmentation-method comparison.

    The outer split is used only for evaluation. The inner split tunes a small
    hyperparameter grid for a single fixed classifier family so the result is
    fast enough to compare across segmentation pipelines.
    """

    if train_df is None or train_df.empty:
        raise ValueError("train_df is empty")

    feat_cols = get_feature_columns(train_df)
    if not feat_cols:
        raise ValueError("No feature columns found after dropping counts/IDs")

    train_df = train_df.reset_index(drop=True).copy()
    X = train_df[feat_cols].to_numpy(float)
    y = train_df["label"].to_numpy(int)

    outer_cv, outer_n_splits = _validated_stratified_splits(
        y,
        outer_folds,
        label="nested outer CV",
        random_state=RANDOM_STATE,
    )

    fold_rows: List[Dict[str, object]] = []
    pred_rows: List[pd.DataFrame] = []

    print(
        f"[NESTED CV] Running {outer_n_splits} outer folds with "
        f"{inner_folds} requested inner folds on {len(train_df)} patients."
    )

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"[NESTED CV] Outer fold {fold_idx}/{outer_n_splits}...")

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if hyperparameter_tuning:
            inner_cv, inner_n_splits = _validated_stratified_splits(
                y_tr,
                inner_folds,
                label=f"nested inner CV fold {fold_idx}",
                random_state=RANDOM_STATE + fold_idx,
            )
            pipeline = _build_pipeline(
                LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, solver="liblinear"),
                SelectFromModel(
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=1.0,
                        random_state=RANDOM_STATE,
                    )
                )
                if use_feature_selection
                else None,
            )
            param_grid = {
                "model__C": [0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["liblinear"],
            }
            search = GridSearchCV(
                pipeline,
                param_grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=1,
            )
            search.fit(X_tr, y_tr)
            pipeline = search.best_estimator_
            best_params = dict(search.best_params_)
            model_name = "LogisticRegression"
            inner_best_auc = float(search.best_score_)
        else:
            inner_n_splits = 0
            model_name = "LogisticRegression (default)"
            best_params = {"model__C": 1.0, "model__penalty": "l2", "model__solver": "liblinear"}
            inner_best_auc = float("nan")
            pipeline = _build_pipeline(
                LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, solver="liblinear"),
                SelectFromModel(
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        C=1.0,
                        random_state=RANDOM_STATE,
                    )
                )
                if use_feature_selection
                else None,
            )

        pipeline.fit(X_tr, y_tr)
        y_prob = pipeline.predict_proba(X_te)[:, 1]
        y_pred = pipeline.predict(X_te)

        auc_fold = float(roc_auc_score(y_te, y_prob))
        acc_fold = float(np.mean(y_pred == y_te))

        print(
            f"[NESTED CV] Fold {fold_idx}: {model_name}, "
            f"inner AUC={inner_best_auc:.3f}, outer AUC={auc_fold:.3f}, Acc={acc_fold:.3f}"
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(tr_idx)),
                "n_eval": int(len(te_idx)),
                "positive_train": int(np.sum(y_tr == 1)),
                "positive_eval": int(np.sum(y_te == 1)),
                "inner_folds": int(inner_n_splits),
                "model_name": str(model_name),
                "inner_best_auc": float(inner_best_auc) if np.isfinite(inner_best_auc) else np.nan,
                "outer_auc": auc_fold,
                "outer_accuracy": acc_fold,
                "best_params_json": json.dumps(best_params, sort_keys=True),
            }
        )

        pred_rows.append(
            pd.DataFrame(
                {
                    "ID": train_df.iloc[te_idx]["ID"].astype(str).to_numpy(),
                    "fold": fold_idx,
                    "y_true": y_te,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                    "model_name": str(model_name),
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    aucs = fold_metrics["outer_auc"].astype(float).tolist()
    accs = fold_metrics["outer_accuracy"].astype(float).tolist()

    return {
        "outer_folds": int(outer_n_splits),
        "requested_outer_folds": int(outer_folds),
        "requested_inner_folds": int(inner_folds),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "auc_sem": 0.0 if len(aucs) <= 1 else float(_sem(aucs)),
        "accuracy_mean": float(np.nanmean(accs)),
        "accuracy_std": float(np.nanstd(accs)),
        "accuracy_sem": 0.0 if len(accs) <= 1 else float(_sem(accs)),
        "fold_metrics": fold_metrics,
        "predictions": predictions,
        "feature_columns": feat_cols,
    }


def save_nested_cv_outputs(
    nested_cv: Dict[str, object],
    *,
    cohort: str,
    out_dir: str,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    stem = f"{cohort.lower()}_nested_cv"
    fold_metrics_path = os.path.join(out_dir, f"{stem}__fold_metrics.csv")
    predictions_path = os.path.join(out_dir, f"{stem}__predictions.csv")
    summary_path = os.path.join(out_dir, f"{stem}__summary.csv")

    fold_metrics = nested_cv["fold_metrics"]
    predictions = nested_cv["predictions"]
    if not isinstance(fold_metrics, pd.DataFrame) or not isinstance(predictions, pd.DataFrame):
        raise TypeError("nested_cv must contain DataFrame values for fold_metrics and predictions")

    fold_metrics.to_csv(fold_metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "outer_folds": nested_cv["outer_folds"],
                "requested_outer_folds": nested_cv["requested_outer_folds"],
                "requested_inner_folds": nested_cv["requested_inner_folds"],
                "auc_mean": nested_cv["auc_mean"],
                "auc_std": nested_cv["auc_std"],
                "auc_sem": nested_cv["auc_sem"],
                "accuracy_mean": nested_cv["accuracy_mean"],
                "accuracy_std": nested_cv["accuracy_std"],
                "accuracy_sem": nested_cv["accuracy_sem"],
                "n_patients": int(len(predictions)),
                "n_features": int(len(nested_cv["feature_columns"])),
                "feature_columns": json.dumps(nested_cv["feature_columns"]),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)

    print(f"[NESTED CV] Saved fold metrics to {fold_metrics_path}")
    print(f"[NESTED CV] Saved out-of-fold predictions to {predictions_path}")
    print(f"[NESTED CV] Saved summary to {summary_path}")
    return {
        "fold_metrics": fold_metrics_path,
        "predictions": predictions_path,
        "summary": summary_path,
    }



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


def _save_cv_feature_bundle(
    fi_dir: str,
    *,
    cohort: str,
    model_name: str,
    best_params: dict,
    train_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    pipeline: Pipeline,
    notes: str,
) -> None:
    save_feature_importance_bundle(
        fi_dir,
        cohort=cohort,
        regime="cv_train",
        model_name=model_name,
        best_params=best_params,
        n_patients_train=len(train_df),
        n_patients_eval=len(train_df),
        feature_columns=get_feature_columns(train_df),
        fi_df=fi_df,
        selected_features=_extract_selected_features(pipeline, get_feature_columns(train_df)),
        notes=notes,
    )


def _evaluate_and_save_test_run(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline: Pipeline,
    fi_dir: str,
    cohort: str,
    regime: str,
    model_name: str,
    best_params: dict,
    prediction_path: str,
    notes: str,
    repeat_test_seeds: int = 1,
    base_seed: int = RANDOM_STATE,
) -> Dict:
    eval_dict = evaluate_on_test_set_repeated(
        train_df,
        test_df,
        clone(pipeline),
        repeat_test_seeds=repeat_test_seeds,
        base_seed=base_seed,
    )
    save_feature_importance_bundle(
        fi_dir,
        cohort=cohort,
        regime=regime,
        model_name=model_name,
        best_params=best_params,
        n_patients_train=len(train_df),
        n_patients_eval=len(test_df),
        feature_columns=get_feature_columns(train_df),
        fi_df=eval_dict["feature_importance"],
        notes=(
            f"{notes} "
            f"Holdout evaluation repeated across {eval_dict['n_repeats']} seed(s) "
            f"starting at seed={eval_dict['base_seed']}; feature importance is averaged over repeats."
        ),
    )
    _save_predictions(prediction_path, test_df, eval_dict["base_eval"])

    _save_repeated_holdout_outputs(
        eval_dict=eval_dict,
        cohort=cohort,
        regime=regime,
        model_name=model_name,
    )
    return eval_dict


def evaluate_on_test_set(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline: Pipeline,
    seed: int = RANDOM_STATE,
) -> Dict:
    feat_cols = get_feature_columns(train_df)
    train_df2 = ensure_columns(train_df, feat_cols)
    test_df2 = ensure_columns(test_df, feat_cols)

    X_tr = train_df2[feat_cols].to_numpy(float)
    y_tr = train_df2["label"].to_numpy(int)
    X_te = test_df2[feat_cols].to_numpy(float)
    y_te = test_df2["label"].to_numpy(int)

    set_global_seed(seed)
    pipeline = _clone_with_seed(pipeline, seed)
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
        random_state=seed,
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


def evaluate_on_test_set_repeated(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline: Pipeline,
    repeat_test_seeds: int = 1,
    base_seed: int = RANDOM_STATE,
) -> Dict:
    repeat_test_seeds = max(int(repeat_test_seeds), 1)
    seeds = [int(base_seed) + idx for idx in range(repeat_test_seeds)]
    evals = [evaluate_on_test_set(train_df, test_df, pipeline, seed=seed) for seed in seeds]

    repeat_metrics_df = pd.DataFrame(
        {
            "seed": seeds,
            "auc": [float(ed["auc"]) for ed in evals],
            "accuracy": [float(ed["accuracy"]) for ed in evals],
        }
    )

    fi_concat = pd.concat(
        [
            ed["feature_importance"][["feature", "importance"]].assign(seed=seed)
            for seed, ed in zip(seeds, evals)
        ],
        ignore_index=True,
    )
    fi_summary = (
        fi_concat.groupby("feature", as_index=False)["importance"]
        .agg(["mean", lambda x: 0.0 if len(x) <= 1 else _sem(list(x))])
        .reset_index()
        .rename(columns={"mean": "importance", "<lambda_0>": "importance_sem"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    base_eval = evals[0]
    return {
        "auc": float(repeat_metrics_df["auc"].mean()),
        "accuracy": float(repeat_metrics_df["accuracy"].mean()),
        "auc_sem": 0.0 if len(repeat_metrics_df) <= 1 else float(_sem(repeat_metrics_df["auc"].tolist())),
        "accuracy_sem": 0.0 if len(repeat_metrics_df) <= 1 else float(_sem(repeat_metrics_df["accuracy"].tolist())),
        "n_repeats": int(repeat_test_seeds),
        "base_seed": int(base_seed),
        "seeds": seeds,
        "repeat_metrics": repeat_metrics_df,
        "feature_importance": fi_summary,
        "base_eval": base_eval,
        "y_true": base_eval["y_true"],
        "y_pred": base_eval["y_pred"],
        "y_probs": base_eval["y_probs"],
    }


def _save_repeated_holdout_outputs(
    *,
    eval_dict: Dict,
    cohort: str,
    regime: str,
    model_name: str,
) -> None:
    if int(eval_dict.get("n_repeats", 1)) <= 1:
        return

    repeat_dir = os.path.join("outputs", "metrics", "test_seed_repeats")
    os.makedirs(repeat_dir, exist_ok=True)
    repeat_stem = f"{cohort}__{regime}"
    repeat_metrics_path = os.path.join(repeat_dir, f"{repeat_stem}__per_seed.csv")
    repeat_summary_path = os.path.join(repeat_dir, f"{repeat_stem}__summary.csv")
    eval_dict["repeat_metrics"].to_csv(repeat_metrics_path, index=False)
    pd.DataFrame(
        [
            {
                "cohort": cohort,
                "regime": regime,
                "model_name": model_name,
                "n_repeats": eval_dict["n_repeats"],
                "base_seed": eval_dict["base_seed"],
                "auc_mean": eval_dict["auc"],
                "auc_sem": eval_dict["auc_sem"],
                "accuracy_mean": eval_dict["accuracy"],
                "accuracy_sem": eval_dict["accuracy_sem"],
            }
        ]
    ).to_csv(repeat_summary_path, index=False)
    print(f"[INFO] Saved per-seed holdout metrics to {repeat_metrics_path}")
    print(f"[INFO] Saved repeated holdout summary to {repeat_summary_path}")


def fit_feature_alignment_meanstd(
    source_train: pd.DataFrame,
    target_train: pd.DataFrame,
    feat_cols: List[str],
) -> Dict[str, np.ndarray]:
    src = source_train[feat_cols].to_numpy(float)
    tgt = target_train[feat_cols].to_numpy(float)

    src_mean = np.nanmean(src, axis=0)
    tgt_mean = np.nanmean(tgt, axis=0)
    src_std = np.nanstd(src, axis=0)
    tgt_std = np.nanstd(tgt, axis=0)

    src_std = np.where(np.isfinite(src_std) & (src_std > 0), src_std, 1.0)
    tgt_std = np.where(np.isfinite(tgt_std) & (tgt_std > 0), tgt_std, 1.0)

    return {
        "source_mean": src_mean,
        "source_std": src_std,
        "target_mean": tgt_mean,
        "target_std": tgt_std,
    }


def apply_feature_alignment_meanstd(
    df: pd.DataFrame,
    feat_cols: List[str],
    alignment: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = df.copy()
    vals = out[feat_cols].to_numpy(float)
    vals = ((vals - alignment["source_mean"]) / alignment["source_std"]) * alignment["target_std"] + alignment["target_mean"]
    out.loc[:, feat_cols] = vals
    return out


def fit_feature_alignment_median_iqr(
    source_train: pd.DataFrame,
    target_train: pd.DataFrame,
    feat_cols: List[str],
) -> Dict[str, np.ndarray]:
    src = source_train[feat_cols].to_numpy(float)
    tgt = target_train[feat_cols].to_numpy(float)

    src_median = np.nanmedian(src, axis=0)
    tgt_median = np.nanmedian(tgt, axis=0)
    src_iqr = np.nanpercentile(src, 75, axis=0) - np.nanpercentile(src, 25, axis=0)
    tgt_iqr = np.nanpercentile(tgt, 75, axis=0) - np.nanpercentile(tgt, 25, axis=0)

    src_iqr = np.where(np.isfinite(src_iqr) & (src_iqr > 0), src_iqr, 1.0)
    tgt_iqr = np.where(np.isfinite(tgt_iqr) & (tgt_iqr > 0), tgt_iqr, 1.0)

    return {
        "source_center": src_median,
        "source_scale": src_iqr,
        "target_center": tgt_median,
        "target_scale": tgt_iqr,
    }


def apply_feature_alignment_median_iqr(
    df: pd.DataFrame,
    feat_cols: List[str],
    alignment: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = df.copy()
    vals = out[feat_cols].to_numpy(float)
    vals = ((vals - alignment["source_center"]) / alignment["source_scale"]) * alignment["target_scale"] + alignment["target_center"]
    out.loc[:, feat_cols] = vals
    return out


def fit_feature_alignment(
    source_train: pd.DataFrame,
    target_train: pd.DataFrame,
    feat_cols: List[str],
    method: str,
) -> Dict[str, np.ndarray]:
    if method == "meanstd":
        return fit_feature_alignment_meanstd(source_train, target_train, feat_cols)
    if method == "median_iqr":
        return fit_feature_alignment_median_iqr(source_train, target_train, feat_cols)
    raise ValueError(f"Unknown feature alignment method: {method}")


def apply_feature_alignment(
    df: pd.DataFrame,
    feat_cols: List[str],
    alignment: Dict[str, np.ndarray],
    method: str,
) -> pd.DataFrame:
    if method == "meanstd":
        return apply_feature_alignment_meanstd(df, feat_cols, alignment)
    if method == "median_iqr":
        return apply_feature_alignment_median_iqr(df, feat_cols, alignment)
    raise ValueError(f"Unknown feature alignment method: {method}")


def _safe_feature_direction(values: pd.Series, labels: pd.Series) -> Dict[str, float]:
    work = pd.DataFrame({"value": values, "label": labels}).copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work = work.dropna(subset=["value", "label"])
    work = work[work["label"].isin([0, 1])]

    label0 = work.loc[work["label"] == 0, "value"].to_numpy(float)
    label1 = work.loc[work["label"] == 1, "value"].to_numpy(float)
    mean0 = float(np.mean(label0)) if label0.size else np.nan
    mean1 = float(np.mean(label1)) if label1.size else np.nan
    delta = mean1 - mean0 if np.isfinite(mean0) and np.isfinite(mean1) else np.nan

    if np.isfinite(delta) and delta > 0:
        sign = 1.0
    elif np.isfinite(delta) and delta < 0:
        sign = -1.0
    elif np.isfinite(delta):
        sign = 0.0
    else:
        sign = np.nan

    auc = np.nan
    if label0.size > 0 and label1.size > 0 and work["value"].nunique(dropna=True) > 1:
        try:
            auc = float(roc_auc_score(work["label"].to_numpy(int), work["value"].to_numpy(float)))
        except ValueError:
            auc = np.nan

    return {
        "n_lusc": int(label0.size),
        "n_luad": int(label1.size),
        "mean_lusc": mean0,
        "mean_luad": mean1,
        "delta_luad_minus_lusc": delta,
        "direction_sign": sign,
        "auc_luad_raw_feature": auc,
        "signed_auc_delta": auc - 0.5 if np.isfinite(auc) else np.nan,
    }


def build_feature_direction_consistency_report(
    df_train_int: pd.DataFrame,
    df_train_ext: pd.DataFrame,
    feat_cols: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feat in feat_cols:
        if feat not in df_train_int.columns or feat not in df_train_ext.columns:
            continue
        int_stats = _safe_feature_direction(df_train_int[feat], df_train_int["label"])
        ext_stats = _safe_feature_direction(df_train_ext[feat], df_train_ext["label"])

        int_sign = int_stats["direction_sign"]
        ext_sign = ext_stats["direction_sign"]
        comparable = np.isfinite(int_sign) and np.isfinite(ext_sign) and int_sign != 0 and ext_sign != 0
        direction_match = bool(comparable and int_sign == ext_sign)
        discordant = bool(comparable and int_sign != ext_sign)

        rows.append(
            {
                "feature": feat,
                "internal_n_lusc": int_stats["n_lusc"],
                "internal_n_luad": int_stats["n_luad"],
                "internal_mean_lusc": int_stats["mean_lusc"],
                "internal_mean_luad": int_stats["mean_luad"],
                "internal_delta_luad_minus_lusc": int_stats["delta_luad_minus_lusc"],
                "internal_direction_sign": int_sign,
                "internal_auc_luad_raw_feature": int_stats["auc_luad_raw_feature"],
                "internal_signed_auc_delta": int_stats["signed_auc_delta"],
                "external_n_lusc": ext_stats["n_lusc"],
                "external_n_luad": ext_stats["n_luad"],
                "external_mean_lusc": ext_stats["mean_lusc"],
                "external_mean_luad": ext_stats["mean_luad"],
                "external_delta_luad_minus_lusc": ext_stats["delta_luad_minus_lusc"],
                "external_direction_sign": ext_sign,
                "external_auc_luad_raw_feature": ext_stats["auc_luad_raw_feature"],
                "external_signed_auc_delta": ext_stats["signed_auc_delta"],
                "direction_match": direction_match,
                "discordant": discordant,
            }
        )
    return pd.DataFrame(rows)


def apply_feature_direction_policy(
    df_train_int: pd.DataFrame,
    df_test_int: pd.DataFrame,
    df_train_ext: pd.DataFrame,
    df_test_ext: pd.DataFrame,
    policy: str,
    *,
    invert_cohort: str = "internal",
    out_dir: str = os.path.join("outputs", "metrics"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if policy not in FEATURE_DIRECTION_POLICIES:
        raise ValueError(f"Unknown feature direction policy: {policy}")
    if invert_cohort not in {"internal", "external"}:
        raise ValueError(f"invert_cohort must be 'internal' or 'external', got {invert_cohort!r}")

    feat_cols = get_feature_columns(pd.concat([df_train_int, df_train_ext], ignore_index=True, sort=False))
    report = build_feature_direction_consistency_report(df_train_int, df_train_ext, feat_cols)
    if report.empty:
        return df_train_int, df_test_int, df_train_ext, df_test_ext, report

    report["policy"] = policy
    report["invert_cohort"] = invert_cohort if policy == "invert-discordant" else ""
    report["action"] = "keep"

    out_train_int = df_train_int.copy()
    out_test_int = df_test_int.copy()
    out_train_ext = df_train_ext.copy()
    out_test_ext = df_test_ext.copy()

    discordant_features = report.loc[report["discordant"], "feature"].astype(str).tolist()

    if policy == "keep-consistent":
        report.loc[report["discordant"], "action"] = "drop"
        drop_cols = [feat for feat in discordant_features if feat in out_train_int.columns or feat in out_train_ext.columns]
        out_train_int = out_train_int.drop(columns=drop_cols, errors="ignore")
        out_test_int = out_test_int.drop(columns=drop_cols, errors="ignore")
        out_train_ext = out_train_ext.drop(columns=drop_cols, errors="ignore")
        out_test_ext = out_test_ext.drop(columns=drop_cols, errors="ignore")
        print(f"[FEATURE DIRECTION] Dropped {len(drop_cols)} discordant feature(s): {drop_cols}")
    elif policy == "invert-discordant":
        report.loc[report["discordant"], "action"] = f"invert_{invert_cohort}"
        targets = (
            (out_train_int, out_test_int)
            if invert_cohort == "internal"
            else (out_train_ext, out_test_ext)
        )
        for df in targets:
            for feat in discordant_features:
                if feat in df.columns:
                    df.loc[:, feat] = -pd.to_numeric(df[feat], errors="coerce")
        print(f"[FEATURE DIRECTION] Inverted {len(discordant_features)} discordant feature(s) in {invert_cohort}: {discordant_features}")
    else:
        print("[FEATURE DIRECTION] Policy 'none': report only; no feature values changed.")

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"feature_direction_consistency__{policy}.csv")
    report.to_csv(report_path, index=False)
    print(f"[FEATURE DIRECTION] Saved train-set feature direction report to {report_path}")
    return out_train_int, out_test_int, out_train_ext, out_test_ext, report


def _add_patient_table_label(row: Dict[str, object], patient_table: str) -> Dict[str, object]:
    return {"Patient table": patient_table, **row}


def _get_single_summary_row(summary_df: pd.DataFrame, patient_table: str, *, bomi2_in_train: bool, bomi1_in_train: bool) -> pd.Series:
    mask = (
        (summary_df["Patient table"] == patient_table)
        & (summary_df["BOMI2 Train"].eq("x") == bool(bomi2_in_train))
        & (summary_df["BOMI1 Train"].eq("x") == bool(bomi1_in_train))
    )
    matches = summary_df.loc[mask]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one summary row for patient_table={patient_table}, "
            f"BOMI2 Train={bomi2_in_train}, BOMI1 Train={bomi1_in_train}; found {len(matches)}."
        )
    return matches.iloc[0]


def build_final_report_table(summary_df: pd.DataFrame, patient_table: str) -> pd.DataFrame:
    row_i = _get_single_summary_row(summary_df, patient_table, bomi2_in_train=True, bomi1_in_train=False)
    row_e = _get_single_summary_row(summary_df, patient_table, bomi2_in_train=False, bomi1_in_train=True)
    row_c = _get_single_summary_row(summary_df, patient_table, bomi2_in_train=True, bomi1_in_train=True)

    within_mif_auc = float(row_i["BOMI2 Test\nAUC"])
    within_he_auc = float(row_e["BOMI1 Test\nAUC"])

    rows = [
        {
            "Training cohort": "mIF",
            "Test cohort": "mIF",
            "Setting": "Within-cohort",
            "ROC AUC": within_mif_auc,
            "ΔAUC vs within": 0.0,
        },
        {
            "Training cohort": "H&E",
            "Test cohort": "H&E",
            "Setting": "Within-cohort",
            "ROC AUC": within_he_auc,
            "ΔAUC vs within": 0.0,
        },
        {
            "Training cohort": "mIF",
            "Test cohort": "H&E",
            "Setting": "Cross-cohort",
            "ROC AUC": float(row_i["BOMI1 Test\nAUC"]),
            "ΔAUC vs within": float(row_i["BOMI1 Test\nAUC"]) - within_he_auc,
        },
        {
            "Training cohort": "H&E",
            "Test cohort": "mIF",
            "Setting": "Cross-cohort",
            "ROC AUC": float(row_e["BOMI2 Test\nAUC"]),
            "ΔAUC vs within": float(row_e["BOMI2 Test\nAUC"]) - within_mif_auc,
        },
        {
            "Training cohort": "mIF + H&E",
            "Test cohort": "mIF",
            "Setting": "Combined",
            "ROC AUC": float(row_c["BOMI2 Test\nAUC"]),
            "ΔAUC vs within": float(row_c["BOMI2 Test\nAUC"]) - within_mif_auc,
        },
        {
            "Training cohort": "mIF + H&E",
            "Test cohort": "H&E",
            "Setting": "Combined",
            "ROC AUC": float(row_c["BOMI1 Test\nAUC"]),
            "ΔAUC vs within": float(row_c["BOMI1 Test\nAUC"]) - within_he_auc,
        },
    ]
    return pd.DataFrame(rows)


def _format_markdown_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row[col]) for col in df.columns) + " |")
    return "\n".join(lines) + "\n"


def save_final_report_tables(summary_df: pd.DataFrame, out_dir: str = os.path.join("outputs", "metrics")) -> None:
    os.makedirs(out_dir, exist_ok=True)

    report_tables_all: List[pd.DataFrame] = []
    for patient_table in summary_df["Patient table"].astype(str).drop_duplicates().tolist():
        report_df = build_final_report_table(summary_df, patient_table)
        report_tables_all.append(report_df.assign(**{"Patient table": patient_table}))

        stem = os.path.join(out_dir, f"final_report_table__{patient_table}")
        report_df.to_csv(f"{stem}.csv", index=False)
        with open(f"{stem}.md", "w", encoding="utf-8") as f:
            f.write(_dataframe_to_markdown_table(report_df))
        print(f"[SUMMARY] Saved final report table to {stem}.csv")
        print(f"[SUMMARY] Saved final report table markdown to {stem}.md")

    if report_tables_all:
        combined_df = pd.concat(report_tables_all, ignore_index=True)
        combined_csv = os.path.join(out_dir, "final_report_tables_all.csv")
        combined_md = os.path.join(out_dir, "final_report_tables_all.md")
        combined_df.to_csv(combined_csv, index=False)
        with open(combined_md, "w", encoding="utf-8") as f:
            f.write(_dataframe_to_markdown_table(combined_df))
        print(f"[SUMMARY] Saved all final report tables to {combined_csv}")
        print(f"[SUMMARY] Saved all final report tables markdown to {combined_md}")


def _cohort_domain_auc(df_a: pd.DataFrame, df_b: pd.DataFrame, feat_cols: List[str], n_folds: int = 5) -> float:
    a = df_a.copy()
    b = df_b.copy()
    a["domain_label"] = 0
    b["domain_label"] = 1
    both = pd.concat([a, b], ignore_index=True, sort=False)
    both = ensure_columns(both, feat_cols)

    X = both[feat_cols].to_numpy(float)
    y = both["domain_label"].to_numpy(int)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    aucs: List[float] = []
    for tr_idx, te_idx in cv.split(X, y):
        pipe = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("model", LogisticRegression(max_iter=50000, random_state=RANDOM_STATE)),
            ]
        )
        pipe.fit(X[tr_idx], y[tr_idx])
        prob = pipe.predict_proba(X[te_idx])[:, 1]
        aucs.append(float(roc_auc_score(y[te_idx], prob)))
    return float(np.mean(aucs))


def _feature_shift_rows(
    *,
    patient_table: str,
    source_name: str,
    target_name: str,
    source_before: pd.DataFrame,
    source_after: pd.DataFrame,
    target_df: pd.DataFrame,
    feat_cols: List[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for idx, feat in enumerate(feat_cols):
        src_before = source_before[feat].to_numpy(float)
        src_after = source_after[feat].to_numpy(float)
        tgt = target_df[feat].to_numpy(float)

        before_delta = float(abs(np.nanmean(src_before) - np.nanmean(tgt)))
        after_delta = float(abs(np.nanmean(src_after) - np.nanmean(tgt)))

        pooled_before = np.sqrt((np.nanvar(src_before) + np.nanvar(tgt)) / 2.0)
        pooled_after = np.sqrt((np.nanvar(src_after) + np.nanvar(tgt)) / 2.0)
        pooled_before = pooled_before if np.isfinite(pooled_before) and pooled_before > 0 else 1.0
        pooled_after = pooled_after if np.isfinite(pooled_after) and pooled_after > 0 else 1.0

        rows.append(
            {
                "Patient table": patient_table,
                "source": source_name,
                "target": target_name,
                "feature": feat,
                "abs_mean_delta_before": before_delta,
                "abs_mean_delta_after": after_delta,
                "abs_smd_before": float(abs((np.nanmean(src_before) - np.nanmean(tgt)) / pooled_before)),
                "abs_smd_after": float(abs((np.nanmean(src_after) - np.nanmean(tgt)) / pooled_after)),
            }
        )
    return rows

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
    use_feature_selection: bool = False,
    run_feature_alignment: bool = False,
    feature_alignment_method: str = "meanstd",
    repeat_test_seeds: int = 1,
) -> None:
    summary_rows: List[Dict[str, object]] = []
    repeated_holdout_rows: List[Dict[str, object]] = []
    alignment_diag_rows: List[Dict[str, object]] = []
    alignment_shift_rows: List[Dict[str, object]] = []
    feat_cols = get_feature_columns(df_train_int)

    def _record_repeated_holdout(patient_table: str, train_label: str, test_label: str, model_name: str, eval_dict: Dict) -> None:
        if int(eval_dict.get("n_repeats", 1)) <= 1:
            return
        repeated_holdout_rows.append(
            {
                "Patient table": patient_table,
                "Train cohort": train_label,
                "Test cohort": test_label,
                "Model": model_name,
                "n_repeats": int(eval_dict["n_repeats"]),
                "base_seed": int(eval_dict["base_seed"]),
                "AUC mean": float(eval_dict["auc"]),
                "AUC SEM": float(eval_dict["auc_sem"]),
                "Accuracy mean": float(eval_dict["accuracy"]),
                "Accuracy SEM": float(eval_dict["accuracy_sem"]),
            }
        )

    print("\n[1] CV on INTERNAL train (BOMI2)...")
    auc_i, std_auc_i, acc_i, std_acc_i, fi_i, model_name_i, pipe_i, params_i = run_classification_cv(
        df_train_int,
        n_folds=5,
        hyperparameter_tuning=hyperparameter_tuning,
        return_model=True,
        use_feature_selection=use_feature_selection,
    )
    print(f"[BOMI2 CV] {model_name_i} AUC: {auc_i:.3f} ± {std_auc_i:.3f}, Acc: {acc_i:.3f} ± {std_acc_i:.3f}")
    _save_cv_feature_bundle(
        fi_dir,
        cohort="BOMI2",
        model_name=model_name_i,
        best_params=params_i,
        train_df=df_train_int,
        fi_df=fi_i,
        pipeline=pipe_i,
        notes="Permutation importance computed on one CV split (ΔAUC when permuted).",
    )

    print("\n[2] Train BOMI2 train, test BOMI2 test...")
    eval_i2i = _evaluate_and_save_test_run(
        train_df=df_train_int,
        test_df=df_test_int,
        pipeline=pipe_i,
        fi_dir=fi_dir,
        cohort="BOMI2",
        regime="train_internal__test_internal",
        model_name=model_name_i,
        best_params=params_i,
        prediction_path=os.path.join("outputs", "predictions", "pred_internal_train_to_internal_test.csv"),
        notes="Permutation importance computed on BOMI2 test set using model trained on BOMI2 train.",
        repeat_test_seeds=repeat_test_seeds,
    )
    print(
        f"[BOMI2→BOMI2 test] AUC: {eval_i2i['auc']:.3f}"
        + (f" ± {eval_i2i['auc_sem']:.3f}" if repeat_test_seeds > 1 else "")
        + f", Acc: {eval_i2i['accuracy']:.3f}"
        + (f" ± {eval_i2i['accuracy_sem']:.3f}" if repeat_test_seeds > 1 else "")
    )
    _record_repeated_holdout("original", "BOMI2", "BOMI2", model_name_i, eval_i2i)

    print("\n[3] Train BOMI2 train, test BOMI1 test...")
    eval_i2e = _evaluate_and_save_test_run(
        train_df=df_train_int,
        test_df=df_test_ext,
        pipeline=pipe_i,
        fi_dir=fi_dir,
        cohort="BOMI2",
        regime="train_internal__test_external",
        model_name=model_name_i,
        best_params=params_i,
        prediction_path=os.path.join("outputs", "predictions", "pred_internal_train_to_external_test.csv"),
        notes="Permutation importance computed on BOMI1 test set using model trained on BOMI2 train.",
        repeat_test_seeds=repeat_test_seeds,
    )
    print(
        f"[BOMI2→BOMI1 test] AUC: {eval_i2e['auc']:.3f}"
        + (f" ± {eval_i2e['auc_sem']:.3f}" if repeat_test_seeds > 1 else "")
        + f", Acc: {eval_i2e['accuracy']:.3f}"
        + (f" ± {eval_i2e['accuracy_sem']:.3f}" if repeat_test_seeds > 1 else "")
    )
    _record_repeated_holdout("original", "BOMI2", "BOMI1", model_name_i, eval_i2e)
    eval_i_on_ext_train = evaluate_fast(df_train_int, df_train_ext, clone(pipe_i))
    summary_rows.append(
        _add_patient_table_label(
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
            ),
            patient_table="original",
        )
    )

    print("\n[4] CV on EXTERNAL train (BOMI1)...")
    auc_e, std_auc_e, acc_e, std_acc_e, fi_e, model_name_e, pipe_e, params_e = run_classification_cv(
        df_train_ext,
        n_folds=5,
        hyperparameter_tuning=hyperparameter_tuning,
        return_model=True,
        use_feature_selection=use_feature_selection,
    )
    print(f"[BOMI1 CV] {model_name_e} AUC: {auc_e:.3f} ± {std_auc_e:.3f}, Acc: {acc_e:.3f} ± {std_acc_e:.3f}")
    _save_cv_feature_bundle(
        fi_dir,
        cohort="BOMI1",
        model_name=model_name_e,
        best_params=params_e,
        train_df=df_train_ext,
        fi_df=fi_e,
        pipeline=pipe_e,
        notes="Permutation importance computed on one CV split (ΔAUC when permuted).",
    )

    print("\n[5] Train BOMI1 train, test BOMI1 test...")
    eval_e2e = _evaluate_and_save_test_run(
        train_df=df_train_ext,
        test_df=df_test_ext,
        pipeline=pipe_e,
        fi_dir=fi_dir,
        cohort="BOMI1",
        regime="train_external__test_external",
        model_name=model_name_e,
        best_params=params_e,
        prediction_path=os.path.join("outputs", "predictions", "pred_external_train_to_external_test.csv"),
        notes="Permutation importance computed on BOMI1 test set using model trained on BOMI1 train.",
        repeat_test_seeds=repeat_test_seeds,
    )
    print(
        f"[BOMI1→BOMI1 test] AUC: {eval_e2e['auc']:.3f}"
        + (f" ± {eval_e2e['auc_sem']:.3f}" if repeat_test_seeds > 1 else "")
        + f", Acc: {eval_e2e['accuracy']:.3f}"
        + (f" ± {eval_e2e['accuracy_sem']:.3f}" if repeat_test_seeds > 1 else "")
    )
    _record_repeated_holdout("original", "BOMI1", "BOMI1", model_name_e, eval_e2e)

    print("\n[6] Train BOMI1 train, test BOMI2 test...")
    eval_e2i = _evaluate_and_save_test_run(
        train_df=df_train_ext,
        test_df=df_test_int,
        pipeline=pipe_e,
        fi_dir=fi_dir,
        cohort="BOMI1",
        regime="train_external__test_internal",
        model_name=model_name_e,
        best_params=params_e,
        prediction_path=os.path.join("outputs", "predictions", "pred_external_train_to_internal_test.csv"),
        notes="Permutation importance computed on BOMI2 test set using model trained on BOMI1 train.",
        repeat_test_seeds=repeat_test_seeds,
    )
    print(
        f"[BOMI1→BOMI2 test] AUC: {eval_e2i['auc']:.3f}"
        + (f" ± {eval_e2i['auc_sem']:.3f}" if repeat_test_seeds > 1 else "")
        + f", Acc: {eval_e2i['accuracy']:.3f}"
        + (f" ± {eval_e2i['accuracy_sem']:.3f}" if repeat_test_seeds > 1 else "")
    )
    _record_repeated_holdout("original", "BOMI1", "BOMI2", model_name_e, eval_e2i)
    eval_e_on_int_train = evaluate_fast(df_train_ext, df_train_int, clone(pipe_e))
    summary_rows.append(
        _add_patient_table_label(
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
            ),
            patient_table="original",
        )
    )

    print("\n[7] COMBINED train (BOMI2+BOMI1): tune/select model on combined pool...")
    df_train_ext_pref = _prefix_ids(df_train_ext, prefix="BOMI1_")
    combined_train = pd.concat([df_train_int, df_train_ext_pref], ignore_index=True, sort=False)
    if tune_combined:
        print("[INFO] Tuning model on COMBINED train...")
        _, _, _, _, fi_c, model_name_c, pipe_c, params_c = run_classification_cv(
            combined_train,
            n_folds=5,
            hyperparameter_tuning=hyperparameter_tuning,
            return_model=True,
            use_feature_selection=use_feature_selection,
        )
        print(f"[INFO] Selected COMBINED model: {model_name_c}")
        if fi_c is not None:
            _save_cv_feature_bundle(
                fi_dir,
                cohort="COMBINED",
                model_name=model_name_c,
                best_params=params_c,
                train_df=combined_train,
                fi_df=fi_c,
                pipeline=pipe_c,
                notes="Permutation importance computed on one CV split (ΔAUC when permuted) for the COMBINED training pool.",
            )
    else:
        pipe_c = pipe_i
        model_name_c = model_name_i
        params_c = params_i

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
    eval_c2i = _evaluate_and_save_test_run(
        train_df=combined_train,
        test_df=df_test_int,
        pipeline=pipe_c,
        fi_dir=fi_dir,
        cohort="COMBINED",
        regime="train_combined__test_internal",
        model_name=model_name_c,
        best_params=params_c,
        prediction_path=os.path.join("outputs", "predictions", "pred_combined_train_to_internal_test.csv"),
        notes="Permutation importance computed on BOMI2 test set using model trained on COMBINED train.",
        repeat_test_seeds=repeat_test_seeds,
    )
    print(
        f"[COMBINED→BOMI2 test] AUC: {eval_c2i['auc']:.3f}"
        + (f" ± {eval_c2i['auc_sem']:.3f}" if repeat_test_seeds > 1 else "")
        + f", Acc: {eval_c2i['accuracy']:.3f}"
        + (f" ± {eval_c2i['accuracy_sem']:.3f}" if repeat_test_seeds > 1 else "")
    )
    _record_repeated_holdout("original", "COMBINED", "BOMI2", model_name_c, eval_c2i)

    print("\n[9] Train COMBINED train, test BOMI1 test...")
    eval_c2e = _evaluate_and_save_test_run(
        train_df=combined_train,
        test_df=df_test_ext,
        pipeline=pipe_c,
        fi_dir=fi_dir,
        cohort="COMBINED",
        regime="train_combined__test_external",
        model_name=model_name_c,
        best_params=params_c,
        prediction_path=os.path.join("outputs", "predictions", "pred_combined_train_to_external_test.csv"),
        notes="Permutation importance computed on BOMI1 test set using model trained on COMBINED train.",
        repeat_test_seeds=repeat_test_seeds,
    )
    print(
        f"[COMBINED→BOMI1 test] AUC: {eval_c2e['auc']:.3f}"
        + (f" ± {eval_c2e['auc_sem']:.3f}" if repeat_test_seeds > 1 else "")
        + f", Acc: {eval_c2e['accuracy']:.3f}"
        + (f" ± {eval_c2e['accuracy_sem']:.3f}" if repeat_test_seeds > 1 else "")
    )
    _record_repeated_holdout("original", "COMBINED", "BOMI1", model_name_c, eval_c2e)
    summary_rows.append(
        _add_patient_table_label(
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
            ),
            patient_table="original",
        )
    )

    if run_feature_alignment:
        print(f"\n[ALIGN] Running train-only patient-level feature alignment experiments with method={feature_alignment_method}...")

        label_int_to_ext = f"aligned_BOMI2_to_BOMI1_{feature_alignment_method}"

        align_int_to_ext = fit_feature_alignment(
            df_train_int,
            df_train_ext,
            feat_cols,
            method=feature_alignment_method,
        )
        df_train_int_aligned = apply_feature_alignment(df_train_int, feat_cols, align_int_to_ext, method=feature_alignment_method)
        df_test_int_aligned = apply_feature_alignment(df_test_int, feat_cols, align_int_to_ext, method=feature_alignment_method)
        alignment_diag_rows.append(
            {
                "Patient table": label_int_to_ext,
                "source": "BOMI2",
                "target": "BOMI1",
                "method": feature_alignment_method,
                "domain_auc_train_before": _cohort_domain_auc(df_train_int, df_train_ext, feat_cols),
                "domain_auc_train_after": _cohort_domain_auc(df_train_int_aligned, df_train_ext, feat_cols),
            }
        )
        alignment_shift_rows.extend(
            _feature_shift_rows(
                patient_table=label_int_to_ext,
                source_name="BOMI2",
                target_name="BOMI1",
                source_before=df_train_int,
                source_after=df_train_int_aligned,
                target_df=df_train_ext,
                feat_cols=feat_cols,
            )
        )
        auc_i_align, _, acc_i_align, _, _, model_name_i_align, pipe_i_align, _ = run_classification_cv(
            df_train_int_aligned,
            n_folds=5,
            hyperparameter_tuning=hyperparameter_tuning,
            return_model=True,
            use_feature_selection=use_feature_selection,
        )
        eval_i_align_on_ext_train = evaluate_fast(df_train_int_aligned, df_train_ext, clone(pipe_i_align))
        eval_i_align_on_int_test = evaluate_on_test_set_repeated(
            df_train_int_aligned, df_test_int_aligned, clone(pipe_i_align), repeat_test_seeds=repeat_test_seeds
        )
        eval_i_align_on_ext_test = evaluate_on_test_set_repeated(
            df_train_int_aligned, df_test_ext, clone(pipe_i_align), repeat_test_seeds=repeat_test_seeds
        )
        _save_repeated_holdout_outputs(
            eval_dict=eval_i_align_on_int_test,
            cohort="BOMI2",
            regime=f"{label_int_to_ext}__train_internal__test_internal",
            model_name=str(model_name_i_align),
        )
        _save_repeated_holdout_outputs(
            eval_dict=eval_i_align_on_ext_test,
            cohort="BOMI2",
            regime=f"{label_int_to_ext}__train_internal__test_external",
            model_name=str(model_name_i_align),
        )
        _record_repeated_holdout(label_int_to_ext, "BOMI2", "BOMI2", model_name_i_align, eval_i_align_on_int_test)
        _record_repeated_holdout(label_int_to_ext, "BOMI2", "BOMI1", model_name_i_align, eval_i_align_on_ext_test)
        summary_rows.append(
            _add_patient_table_label(
                _make_summary_row(
                    bomi2_in_train=True,
                    bomi1_in_train=False,
                    bomi2_train_acc=acc_i_align,
                    bomi2_train_auc=auc_i_align,
                    bomi1_train_acc=eval_i_align_on_ext_train["accuracy"],
                    bomi1_train_auc=eval_i_align_on_ext_train["auc"],
                    bomi2_test_acc=eval_i_align_on_int_test["accuracy"],
                    bomi2_test_auc=eval_i_align_on_int_test["auc"],
                    bomi1_test_acc=eval_i_align_on_ext_test["accuracy"],
                    bomi1_test_auc=eval_i_align_on_ext_test["auc"],
                ),
                patient_table=label_int_to_ext,
            )
        )

        eval_e_on_int_train_aligned = evaluate_fast(df_train_ext, df_train_int_aligned, clone(pipe_e))
        eval_e_on_int_test_aligned = evaluate_on_test_set_repeated(
            df_train_ext, df_test_int_aligned, clone(pipe_e), repeat_test_seeds=repeat_test_seeds
        )
        _save_repeated_holdout_outputs(
            eval_dict=eval_e_on_int_test_aligned,
            cohort="BOMI1",
            regime=f"{label_int_to_ext}__train_external__test_internal",
            model_name=str(model_name_e),
        )
        _record_repeated_holdout(label_int_to_ext, "BOMI1", "BOMI2", model_name_e, eval_e_on_int_test_aligned)
        summary_rows.append(
            _add_patient_table_label(
                _make_summary_row(
                    bomi2_in_train=False,
                    bomi1_in_train=True,
                    bomi2_train_acc=eval_e_on_int_train_aligned["accuracy"],
                    bomi2_train_auc=eval_e_on_int_train_aligned["auc"],
                    bomi1_train_acc=acc_e,
                    bomi1_train_auc=auc_e,
                    bomi2_test_acc=eval_e_on_int_test_aligned["accuracy"],
                    bomi2_test_auc=eval_e_on_int_test_aligned["auc"],
                    bomi1_test_acc=eval_e2e["accuracy"],
                    bomi1_test_auc=eval_e2e["auc"],
                ),
                patient_table=label_int_to_ext,
            )
        )

        combined_train_i_align = pd.concat([df_train_int_aligned, df_train_ext_pref], ignore_index=True, sort=False)
        if tune_combined:
            _, _, _, _, _, model_name_ci_align, pipe_ci_align, _ = run_classification_cv(
                combined_train_i_align,
                n_folds=5,
                hyperparameter_tuning=hyperparameter_tuning,
                return_model=True,
                use_feature_selection=use_feature_selection,
            )
        else:
            pipe_ci_align = pipe_i_align
            model_name_ci_align = model_name_i_align
        auc_ci_align, _, acc_ci_align, _ = run_combined_cv_target_cohort(
            df_train_int=df_train_int_aligned,
            df_train_ext_pref=df_train_ext_pref,
            pipeline=pipe_ci_align,
            target="INTERNAL",
            n_folds=5,
        )
        auc_ce_align, _, acc_ce_align, _ = run_combined_cv_target_cohort(
            df_train_int=df_train_int_aligned,
            df_train_ext_pref=df_train_ext_pref,
            pipeline=pipe_ci_align,
            target="EXTERNAL",
            n_folds=5,
        )
        eval_ci_align_on_int_test = evaluate_on_test_set_repeated(
            combined_train_i_align, df_test_int_aligned, clone(pipe_ci_align), repeat_test_seeds=repeat_test_seeds
        )
        eval_ci_align_on_ext_test = evaluate_on_test_set_repeated(
            combined_train_i_align, df_test_ext, clone(pipe_ci_align), repeat_test_seeds=repeat_test_seeds
        )
        _save_repeated_holdout_outputs(
            eval_dict=eval_ci_align_on_int_test,
            cohort="COMBINED",
            regime=f"{label_int_to_ext}__train_combined__test_internal",
            model_name=str(model_name_ci_align),
        )
        _save_repeated_holdout_outputs(
            eval_dict=eval_ci_align_on_ext_test,
            cohort="COMBINED",
            regime=f"{label_int_to_ext}__train_combined__test_external",
            model_name=str(model_name_ci_align),
        )
        _record_repeated_holdout(label_int_to_ext, "COMBINED", "BOMI2", model_name_ci_align, eval_ci_align_on_int_test)
        _record_repeated_holdout(label_int_to_ext, "COMBINED", "BOMI1", model_name_ci_align, eval_ci_align_on_ext_test)
        summary_rows.append(
            _add_patient_table_label(
                _make_summary_row(
                    bomi2_in_train=True,
                    bomi1_in_train=True,
                    bomi2_train_acc=acc_ci_align,
                    bomi2_train_auc=auc_ci_align,
                    bomi1_train_acc=acc_ce_align,
                    bomi1_train_auc=auc_ce_align,
                    bomi2_test_acc=eval_ci_align_on_int_test["accuracy"],
                    bomi2_test_auc=eval_ci_align_on_int_test["auc"],
                    bomi1_test_acc=eval_ci_align_on_ext_test["accuracy"],
                    bomi1_test_auc=eval_ci_align_on_ext_test["auc"],
                ),
                patient_table=label_int_to_ext,
            )
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "Patient table",
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
        ],
    )
    out_csv = os.path.join("outputs", "metrics", "cohort_mixing_summary.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[SUMMARY] Saved cohort mixing summary to {out_csv}")
    save_final_report_tables(summary_df)

    if run_feature_alignment and alignment_diag_rows:
        diag_path = os.path.join("outputs", "metrics", "feature_alignment_diagnostics.csv")
        pd.DataFrame(alignment_diag_rows).to_csv(diag_path, index=False)
        print(f"[ALIGN] Saved diagnostics to {diag_path}")

    if run_feature_alignment and alignment_shift_rows:
        shift_path = os.path.join("outputs", "metrics", "feature_alignment_shift_summary.csv")
        pd.DataFrame(alignment_shift_rows).to_csv(shift_path, index=False)
        print(f"[ALIGN] Saved feature-shift summary to {shift_path}")

    if repeated_holdout_rows:
        repeated_holdout_path = os.path.join("outputs", "metrics", "cohort_mixing_summary_repeated_holdout.csv")
        pd.DataFrame(repeated_holdout_rows).to_csv(repeated_holdout_path, index=False)
        print(f"[SUMMARY] Saved repeated holdout summary to {repeated_holdout_path}")

    print("\n[INFO] Base experiments completed.")

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
    p.add_argument("--no-feature-selection", action="store_true")
    p.add_argument(
        "--repeat-test-seeds",
        type=int,
        default=1,
        help="Repeat each fixed holdout test evaluation across N consecutive seeds and report mean ± SEM.",
    )
    p.add_argument(
        "--nested-cv-only",
        action="store_true",
        help="Run the INTERNAL/mIF nested CV and skip the rest of the base experiments.",
    )
    p.add_argument("--nested-outer-folds", type=int, default=5)
    p.add_argument("--nested-inner-folds", type=int, default=3)
    p.add_argument(
        "--nested-cv-out-dir",
        type=str,
        default=os.path.join("outputs", "metrics", "nested_cv"),
        help="Output directory for nested-CV summary, fold metrics, and out-of-fold predictions.",
    )
    p.add_argument("--run-feature-alignment", action="store_true")
    p.add_argument("--feature-alignment-method", type=str, default="meanstd", choices=["meanstd", "median_iqr"])
    p.add_argument(
        "--feature-direction-policy",
        type=str,
        default="none",
        choices=FEATURE_DIRECTION_POLICIES,
        help=(
            "Train-set-only handling for features whose LUAD-vs-LUSC direction differs "
            "between internal and external cohorts."
        ),
    )
    p.add_argument(
        "--feature-direction-invert-cohort",
        type=str,
        default="internal",
        choices=["internal", "external"],
        help=(
            "Cohort to multiply by -1 for discordant features when "
            "--feature-direction-policy=invert-discordant."
        ),
    )
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
    use_feature_selection = (not args.no_feature_selection)
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

    df_train_int, df_test_int, df_train_ext, df_test_ext, _ = apply_feature_direction_policy(
        df_train_int,
        df_test_int,
        df_train_ext,
        df_test_ext,
        args.feature_direction_policy,
        invert_cohort=args.feature_direction_invert_cohort,
    )

    if args.nested_cv_only:
        print("\n[NESTED CV] INTERNAL/mIF model-selection estimate...")
        nested_cv = run_nested_classification_cv(
            df_train_int,
            outer_folds=args.nested_outer_folds,
            inner_folds=args.nested_inner_folds,
            hyperparameter_tuning=hyperparameter_tuning,
            use_feature_selection=use_feature_selection,
        )
        save_nested_cv_outputs(nested_cv, cohort="BOMI2", out_dir=args.nested_cv_out_dir)
        print(
            "[NESTED CV] BOMI2 train estimate: "
            f"AUC={nested_cv['auc_mean']:.3f} ± {nested_cv['auc_std']:.3f} SD "
            f"(SEM={nested_cv['auc_sem']:.3f}), "
            f"Acc={nested_cv['accuracy_mean']:.3f} ± {nested_cv['accuracy_std']:.3f} SD"
        )
        if args.nested_cv_only:
            print("[INFO] --nested-cv-only set; skipping base experiments.")
            return

    run_base_experiments(
        df_train_int=df_train_int,
        df_test_int=df_test_int,
        df_train_ext=df_train_ext,
        df_test_ext=df_test_ext,
        hyperparameter_tuning=hyperparameter_tuning,
        tune_combined=tune_combined,
        fi_dir=args.fi_dir,
        use_feature_selection=use_feature_selection,
        run_feature_alignment=args.run_feature_alignment,
        feature_alignment_method=args.feature_alignment_method,
        repeat_test_seeds=args.repeat_test_seeds,
    )

    if args.run_learning_curve:
        learning_fit_alignment_fn = None
        learning_apply_alignment_fn = None
        learning_alignment_method = None

        if args.run_feature_alignment:
            if args.feature_alignment_method not in {"meanstd", "median_iqr"}:
                raise ValueError(
                    "--run-learning-curve currently supports --run-feature-alignment only for "
                    "'meanstd' and 'median_iqr'."
                )
            learning_fit_alignment_fn = fit_feature_alignment
            learning_apply_alignment_fn = apply_feature_alignment
            learning_alignment_method = args.feature_alignment_method

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
            run_classification_cv_fn=lambda *a, **kw: run_classification_cv(
                *a,
                **kw,
                use_feature_selection=use_feature_selection,
            ),
            prefix_ids_fn=_prefix_ids,
            fit_feature_alignment_fn=learning_fit_alignment_fn,
            apply_feature_alignment_fn=learning_apply_alignment_fn,
            feature_alignment_method=learning_alignment_method,
            feature_alignment_kwargs=None,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
