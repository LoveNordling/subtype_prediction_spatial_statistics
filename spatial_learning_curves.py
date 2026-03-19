from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from spatial_artifact_utils import sem
from spatial_utils import ensure_columns, get_feature_columns


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


def _limit_threading_for_workers() -> None:
    """Avoid BLAS/OpenMP oversubscription when using ProcessPoolExecutor."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def evaluate_fast(train_df: pd.DataFrame, test_df: pd.DataFrame, pipeline) -> Dict[str, float]:
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
    run_classification_cv_fn: Callable,
    prefix_ids_fn: Callable,
    seed: int = 42,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1 (or -1). Got: {n_jobs}")

    secondary_train_pref = prefix_ids_fn(secondary_train, prefix=f"{secondary_name}_")

    print(f"\n[LEARNING-fixed] Tuning primary-only model for {primary_name}...")
    _, _, _, _, _, model_p, pipe_primary, params_p = run_classification_cv_fn(
        primary_train, n_folds=tune_folds, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[LEARNING-fixed] {primary_name} primary-only selected: {model_p} params={params_p}")

    print(f"\n[LEARNING-fixed] Tuning mixed model for {primary_name}+{secondary_name}...")
    mixed_full = pd.concat([primary_train, secondary_train_pref], ignore_index=True, sort=False)
    _, _, _, _, _, model_m, pipe_mixed, params_m = run_classification_cv_fn(
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
            out_rows = f.result()
            rows.extend(out_rows)

    raw_df = pd.DataFrame(rows).sort_values(["train_size", "repeat", "regime"]).reset_index(drop=True)

    raw_path = os.path.join(out_dir, f"learning_curve_raw_fixed_{primary_name}_test_{primary_name}.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[LEARNING-fixed] Saved raw results to {raw_path}")

    summary = _summarize_learning(raw_df)
    summary_path = os.path.join(out_dir, f"learning_curve_summary_fixed_{primary_name}_test_{primary_name}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[LEARNING-fixed] Saved summary to {summary_path}")

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
    run_classification_cv_fn: Callable,
    prefix_ids_fn: Callable,
    seed: int = 42,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    if n_jobs is None:
        n_jobs = mp.cpu_count()
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1 (or -1). Got: {n_jobs}")

    secondary_train_pref = prefix_ids_fn(secondary_train_pool, prefix=f"{secondary_name}_")

    print(f"\n[LEARNING-cv] Tuning primary-only model for {primary_name} (on full primary train pool)...")
    _, _, _, _, _, model_p, pipe_primary, params_p = run_classification_cv_fn(
        primary_train_pool, n_folds=tune_folds, hyperparameter_tuning=hyperparameter_tuning, return_model=True
    )
    print(f"[LEARNING-cv] {primary_name} primary-only selected: {model_p} params={params_p}")

    print(f"\n[LEARNING-cv] Tuning mixed model for {primary_name}+{secondary_name} (on full combined train)...")
    mixed_full = pd.concat([primary_train_pool, secondary_train_pref], ignore_index=True, sort=False)
    _, _, _, _, _, model_m, pipe_mixed, params_m = run_classification_cv_fn(
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
            out_rows = f.result()
            rows.extend(out_rows)

    raw_df = pd.DataFrame(rows).sort_values(["outer_fold", "train_size", "repeat", "regime"]).reset_index(drop=True)

    raw_path = os.path.join(out_dir, f"learning_curve_raw_cv_{primary_name}_valfolds.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"[LEARNING-cv] Saved raw results to {raw_path}")

    summary = _summarize_learning(raw_df)
    summary_path = os.path.join(out_dir, f"learning_curve_summary_cv_{primary_name}_valfolds.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[LEARNING-cv] Saved summary to {summary_path}")

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
    run_classification_cv_fn: Callable,
    prefix_ids_fn: Callable,
    n_jobs: Optional[int] = None,
) -> None:
    if eval_mode not in {"fixed", "cv"}:
        raise ValueError("eval_mode must be 'fixed' or 'cv'")

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
            run_classification_cv_fn=run_classification_cv_fn,
            prefix_ids_fn=prefix_ids_fn,
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
            run_classification_cv_fn=run_classification_cv_fn,
            prefix_ids_fn=prefix_ids_fn,
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
            run_classification_cv_fn=run_classification_cv_fn,
            prefix_ids_fn=prefix_ids_fn,
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
            run_classification_cv_fn=run_classification_cv_fn,
            prefix_ids_fn=prefix_ids_fn,
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
