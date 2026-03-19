from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def sem(x: List[float]) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size <= 1:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def _safe_json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


def extract_selected_features(pipeline, feat_cols: List[str]) -> Optional[List[str]]:
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
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fi_sorted = fi_df.copy()
    fi_sorted["feature"] = fi_df["feature"].astype(str)
    fi_sorted["importance"] = pd.to_numeric(fi_df["importance"], errors="coerce")
    fi_sorted = fi_sorted.sort_values("importance", ascending=False).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"feature_importance__{cohort}__{regime}__{model_name.replace(' ', '_')}__{ts}"

    csv_path = str(Path(out_dir) / f"{stem}.csv")
    fi_sorted.to_csv(csv_path, index=False)

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
        "top_features": fi_sorted.head(30).to_dict(orient="records"),
        "all_features": fi_sorted.to_dict(orient="records"),
        "notes": notes,
    }

    json_path = str(Path(out_dir) / f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_safe_json_default)

    print(f"[INFO] Saved feature importance CSV: {csv_path}")
    print(f"[INFO] Saved feature importance JSON: {json_path}")
    return {"csv": csv_path, "json": json_path}


def read_split_ids(csv_path: str) -> set:
    df = pd.read_csv(csv_path)
    for col in ["ID", "ID or PAD_year", "ID_or_PAD_year", "patient_id", "Patient ID"]:
        if col in df.columns:
            return set(df[col].dropna().astype(str))
    raise ValueError(f"Could not find an ID column in {csv_path}. Columns: {list(df.columns)}")


def subset_by_ids(df: pd.DataFrame, ids: set) -> pd.DataFrame:
    ids_str = {str(x) for x in ids}
    if "ID" not in df.columns:
        raise ValueError(f"Expected column 'ID'. Columns: {list(df.columns)}")
    return df[df["ID"].astype(str).isin(ids_str)].copy()


def prefix_ids(df: pd.DataFrame, prefix: str, id_col: str = "ID") -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"Expected column '{id_col}'. Columns: {list(df.columns)}")
    out = df.copy()
    out[id_col] = out[id_col].astype(str).map(lambda x: f"{prefix}{x}")
    return out


def save_predictions(path: str, test_df: pd.DataFrame, eval_dict: dict) -> None:
    out = pd.DataFrame(
        {
            "ID": test_df["ID"].astype(str),
            "y_true": eval_dict["y_true"],
            "y_prob": eval_dict["y_probs"],
            "y_pred": eval_dict["y_pred"],
        }
    )
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[INFO] Saved predictions to {path}")


def make_summary_row(
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
