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
from spatial_utils import (
    CountQC,
    ensure_columns,
    get_feature_columns,
)


RANDOM_STATE = 42

def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)

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
) -> Dict:
    eval_dict = evaluate_on_test_set(train_df, test_df, clone(pipeline))
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
        notes=notes,
    )
    _save_predictions(prediction_path, test_df, eval_dict)
    return eval_dict


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
    )
    print(f"[BOMI2→BOMI2 test] AUC: {eval_i2i['auc']:.3f}, Acc: {eval_i2i['accuracy']:.3f}")

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
    )
    print(f"[BOMI2→BOMI1 test] AUC: {eval_i2e['auc']:.3f}, Acc: {eval_i2e['accuracy']:.3f}")

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
    )
    print(f"[BOMI1→BOMI1 test] AUC: {eval_e2e['auc']:.3f}, Acc: {eval_e2e['accuracy']:.3f}")

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
    )
    print(f"[BOMI1→BOMI2 test] AUC: {eval_e2i['auc']:.3f}, Acc: {eval_e2i['accuracy']:.3f}")

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
    )
    print(f"[COMBINED→BOMI2 test] AUC: {eval_c2i['auc']:.3f}, Acc: {eval_c2i['accuracy']:.3f}")

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
    )
    print(f"[COMBINED→BOMI1 test] AUC: {eval_c2e['auc']:.3f}, Acc: {eval_c2e['accuracy']:.3f}")

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
            run_classification_cv_fn=run_classification_cv,
            prefix_ids_fn=_prefix_ids,
            n_jobs=args.n_jobs,
        )


if __name__ == "__main__":
    main()
