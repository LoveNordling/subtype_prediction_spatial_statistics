#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, roc_auc_score


SESSION_ROOT = Path("data/reference/subtype_annotator_labels/sessions")
OUTPUT_ROOT = Path("outputs/qc/annotation_analysis")
DATASET_LABELS = {
    "bomi1_he": "BOMI1 H&E",
    "bomi2_inform": "BOMI2 inForm",
}
LABEL_TEXT_TO_INT = {
    "Adeno carcinoma": 1,
    "Squamous Cell Carcinoma": 0,
}
CONFIDENCE_LEVEL_LABELS = {
    1: "Def. Sq",
    2: "Mod. Sq",
    3: "Slight Sq",
    4: "Slight Ad",
    5: "Mod. Ad",
    6: "Def. Ad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze session-based pathologist subtype annotations."
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=SESSION_ROOT,
        help="Root directory containing session subdirectories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where per-dataset outputs will be written.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *sorted(DATASET_LABELS)],
        default="all",
        help="Dataset to analyze.",
    )
    parser.add_argument(
        "--debug-names",
        action="store_true",
        help="Show actual reviewer names instead of anonymized labels in figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    return parser.parse_args()


def load_latest_sessions(session_root: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for summary_path in sorted(session_root.glob("*/*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        annotations_path = Path(summary["annotations_path"])
        if not annotations_path.is_absolute():
            annotations_path = Path.cwd() / annotations_path
        records.append(
            {
                "summary_path": summary_path,
                "annotations_path": annotations_path,
                "session_id": str(summary["session_id"]),
                "dataset_key": str(summary["dataset_key"]),
                "pathologist_name": str(summary["pathologist_name"]),
                "pathologist_key": str(summary["pathologist_name"]).strip().lower(),
                "started_at": pd.to_datetime(summary["started_at"], utc=True),
                "last_updated_at": pd.to_datetime(summary["last_updated_at"], utc=True),
                "n_total": int(summary["n_total"]),
                "n_completed": int(summary["n_completed"]),
                "is_complete": int(summary["n_completed"]) == int(summary["n_total"]),
            }
        )

    if not records:
        raise FileNotFoundError(f"No session summaries found under {session_root}")

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.loc[summary_df["is_complete"]].copy()
    summary_df = summary_df.sort_values(
        ["dataset_key", "pathologist_key", "started_at", "last_updated_at"],
        ascending=[True, True, False, False],
    )
    latest = summary_df.groupby(["dataset_key", "pathologist_key"], as_index=False).head(1).copy()
    return latest.reset_index(drop=True)


def confidence_strength_from_score(series: pd.Series) -> pd.Series:
    score = pd.to_numeric(series, errors="coerce")
    return np.maximum(score, 1.0 - score)


def load_annotations(session_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in session_df.itertuples(index=False):
        annotations = pd.read_csv(row.annotations_path)
        annotations["dataset_key"] = annotations["dataset_key"].astype(str)
        annotations["pathologist_name"] = annotations["pathologist_name"].astype(str)
        annotations["pathologist_key"] = str(row.pathologist_key)
        annotations["session_id"] = annotations["session_id"].astype(str)
        annotations["patient_id"] = annotations["patient_id"].astype(str)
        annotations["true_label"] = pd.to_numeric(annotations["true_label"], errors="coerce").astype(int)
        annotations["pred_label"] = pd.to_numeric(annotations["pred_label"], errors="coerce").astype(int)
        if "pred_score" not in annotations.columns:
            annotations["pred_score"] = annotations["pred_label"].astype(float)
        else:
            annotations["pred_score"] = pd.to_numeric(annotations["pred_score"], errors="coerce")
            annotations["pred_score"] = annotations["pred_score"].fillna(annotations["pred_label"].astype(float))
        if "confidence_level" not in annotations.columns:
            annotations["confidence_level"] = pd.NA
        else:
            annotations["confidence_level"] = pd.to_numeric(
                annotations["confidence_level"], errors="coerce"
            ).astype("Int64")
        annotations["confidence_strength"] = confidence_strength_from_score(annotations["pred_score"])
        annotations["is_correct"] = (
            annotations["pred_label"].astype(int) == annotations["true_label"].astype(int)
        ).astype(int)
        records.extend(annotations.to_dict(orient="records"))

    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def compute_auc(y_true: pd.Series, pred_score: pd.Series) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, pred_score))


def build_summary_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for reviewer, group in dataset_df.groupby("pathologist_key", sort=False):
        y_true = group["true_label"].astype(int)
        y_pred = group["pred_label"].astype(int)
        pred_score = pd.to_numeric(group["pred_score"], errors="coerce").fillna(y_pred.astype(float))
        confidence_levels = pd.to_numeric(group["confidence_level"], errors="coerce")
        rows.append(
            {
                "pathologist_key": reviewer,
                "pathologist_name": group["pathologist_name"].iloc[0],
                "session_id": group["session_id"].iloc[0],
                "n_cases": int(len(group)),
                "accuracy": float((y_true == y_pred).mean()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "auc": compute_auc(y_true, pred_score),
                "mean_confidence_strength": float(group["confidence_strength"].mean()),
                "mean_confidence_level": float(confidence_levels.dropna().mean()) if confidence_levels.notna().any() else np.nan,
                "confidence_coverage": float(confidence_levels.notna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def build_display_labels(summary_df: pd.DataFrame, debug_names: bool) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, row in enumerate(summary_df.itertuples(index=False), start=1):
        if debug_names:
            labels[row.pathologist_key] = str(row.pathologist_name)
        else:
            labels[row.pathologist_key] = f"Pathologist {index}"
    return labels


def build_kappa_matrix(
    dataset_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    display_labels: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviewers = list(summary_df["pathologist_key"])
    label_lookup = {"ground_truth": "Ground Truth", **display_labels}
    entities = ["ground_truth", *reviewers]
    kappa = pd.DataFrame(index=entities, columns=entities, dtype=float)
    counts = pd.DataFrame(index=entities, columns=entities, dtype=int)

    truth_by_case = (
        dataset_df[["patient_id", "true_label"]]
        .drop_duplicates()
        .set_index("patient_id")["true_label"]
    )
    reviewer_predictions = {
        reviewer: group.set_index("patient_id")["pred_label"]
        for reviewer, group in dataset_df.groupby("pathologist_key", sort=False)
    }

    for left in entities:
        for right in entities:
            if left == "ground_truth":
                left_series = truth_by_case
            else:
                left_series = reviewer_predictions[left]
            if right == "ground_truth":
                right_series = truth_by_case
            else:
                right_series = reviewer_predictions[right]

            merged = pd.concat([left_series, right_series], axis=1, join="inner").dropna()
            counts.loc[left, right] = int(len(merged))
            if merged.empty:
                kappa.loc[left, right] = np.nan
            elif left == right:
                kappa.loc[left, right] = 1.0
            else:
                kappa.loc[left, right] = float(cohen_kappa_score(merged.iloc[:, 0], merged.iloc[:, 1]))

    kappa.index = [label_lookup[key] for key in kappa.index]
    kappa.columns = [label_lookup[key] for key in kappa.columns]
    counts.index = kappa.index
    counts.columns = kappa.columns
    return kappa, counts


def plot_kappa_heatmap(
    kappa_df: pd.DataFrame,
    dataset_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(1.2 * len(kappa_df.columns) + 3, 1.0 * len(kappa_df.index) + 2))
    sns.heatmap(kappa_df, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Kappa"})
    plt.title(f"{dataset_label}: Cohen's Kappa")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_label_matrix(
    dataset_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    display_labels: dict[str, str],
) -> pd.DataFrame:
    patient_truth = (
        dataset_df[["patient_id", "true_label"]]
        .drop_duplicates()
        .set_index("patient_id")["true_label"]
    )
    matrix = pd.DataFrame({"Ground Truth": patient_truth})

    for row in summary_df.itertuples(index=False):
        reviewer_cases = (
            dataset_df.loc[dataset_df["pathologist_key"] == row.pathologist_key, ["patient_id", "pred_label"]]
            .drop_duplicates()
            .set_index("patient_id")["pred_label"]
        )
        row_label = f"{display_labels[row.pathologist_key]} | AUC {row.auc:.2f}"
        matrix[row_label] = reviewer_cases

    matrix["n_correct"] = (
        matrix.drop(columns=["Ground Truth"])
        .eq(matrix["Ground Truth"], axis=0)
        .sum(axis=1)
    )
    matrix = matrix.sort_values(["Ground Truth", "n_correct"], ascending=[False, False])

    sq_mask = matrix["Ground Truth"] == 0
    matrix.loc[sq_mask] = matrix.loc[sq_mask].sort_values("n_correct", ascending=True).values
    matrix = matrix.drop(columns=["n_correct"])
    return matrix.T


def plot_label_matrix(
    matrix: pd.DataFrame,
    output_path: Path,
    debug_names: bool,
) -> None:
    cmap = ListedColormap(["#2f6db3", "#e57f49"])
    plt.figure(figsize=(max(12, 0.4 * matrix.shape[1] + 4), 0.75 * matrix.shape[0] + 2))
    ax = sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.25,
        linecolor="white",
    )
    if debug_names:
        ax.set_xlabel("Patient ID")
    else:
        ax.set_xlabel("Cases")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    if debug_names:
        ax.set_xticklabels(matrix.columns, rotation=90, ha="center", fontsize=8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.set_yticklabels(matrix.index, rotation=0)
    legend_handles = [
        Patch(facecolor="#e57f49", edgecolor="none", label="LUAD"),
        Patch(facecolor="#2f6db3", edgecolor="none", label="LUSC"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_confidence_accuracy_by_level(dataset_df: pd.DataFrame) -> pd.DataFrame:
    subset = dataset_df.dropna(subset=["confidence_level"]).copy()
    if subset.empty:
        return pd.DataFrame()
    grouped = (
        subset.groupby("confidence_level", sort=True)
        .agg(
            n_cases=("patient_id", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence_strength=("confidence_strength", "mean"),
        )
        .reset_index()
    )
    grouped["level_label"] = grouped["confidence_level"].map(CONFIDENCE_LEVEL_LABELS)
    return grouped


def plot_confidence_summary(
    dataset_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    display_labels: dict[str, str],
    dataset_label: str,
    output_path: Path,
) -> None:
    confidence_rows = dataset_df.dropna(subset=["confidence_level"]).copy()
    if confidence_rows.empty:
        return

    confidence_rows["reviewer_label"] = confidence_rows["pathologist_key"].map(display_labels)
    confidence_rows["Correctness"] = confidence_rows["is_correct"].map({1: "Correct", 0: "Incorrect"})
    confidence_rows["Confidence Strength"] = confidence_rows["confidence_strength"]

    order = [display_labels[key] for key in summary_df["pathologist_key"] if key in display_labels]
    level_stats = build_confidence_accuracy_by_level(dataset_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, 0.55 * len(order) + 2.5)))
    sns.barplot(
        data=confidence_rows,
        x="Confidence Strength",
        y="reviewer_label",
        hue="Correctness",
        order=order,
        orient="h",
        ax=axes[0],
        palette={"Correct": "#4c9a6a", "Incorrect": "#c84a1b"},
    )
    axes[0].set_xlim(0.45, 1.0)
    axes[0].set_xlabel("Mean Confidence Strength")
    axes[0].set_ylabel("")
    axes[0].set_title("Confidence By Reviewer")
    axes[0].legend(title="")

    sns.barplot(
        data=level_stats,
        x="level_label",
        y="accuracy",
        color="#6fa3d8",
        ax=axes[1],
    )
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel("Confidence Level")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy By Confidence Level")
    for idx, row in level_stats.iterrows():
        axes[1].text(idx, min(row["accuracy"] + 0.03, 0.98), f"n={int(row['n_cases'])}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"{dataset_label}: Confidence Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def export_case_difficulty(dataset_df: pd.DataFrame, output_path: Path) -> None:
    per_case = (
        dataset_df.groupby(["patient_id", "true_label", "true_label_text"], dropna=False)
        .agg(
            n_reviewers=("pathologist_key", "nunique"),
            mean_accuracy=("is_correct", "mean"),
            mean_confidence_strength=("confidence_strength", "mean"),
            pred_labels=("pred_label", lambda values: "|".join(str(int(v)) for v in values)),
            reviewers=("pathologist_name", lambda values: "|".join(str(v) for v in values)),
        )
        .reset_index()
    )
    per_case["disagreement_rate"] = 1.0 - per_case["mean_accuracy"]
    per_case = per_case.sort_values(
        ["disagreement_rate", "mean_confidence_strength", "patient_id"],
        ascending=[False, True, True],
    )
    per_case.to_csv(output_path, index=False)


def analyze_dataset(
    dataset_key: str,
    dataset_df: pd.DataFrame,
    output_root: Path,
    debug_names: bool,
) -> None:
    dataset_dir = output_root / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_label = DATASET_LABELS.get(dataset_key, dataset_key)

    summary_df = build_summary_table(dataset_df)
    summary_df.to_csv(dataset_dir / "reviewer_summary.csv", index=False)

    display_labels = build_display_labels(summary_df, debug_names=debug_names)
    kappa_df, counts_df = build_kappa_matrix(dataset_df, summary_df, display_labels)
    kappa_df.to_csv(dataset_dir / "kappa_matrix.csv")
    counts_df.to_csv(dataset_dir / "kappa_shared_case_counts.csv")
    plot_kappa_heatmap(kappa_df, dataset_label, dataset_dir / "kappa_heatmap.png")

    label_matrix = build_label_matrix(dataset_df, summary_df, display_labels)
    label_matrix.to_csv(dataset_dir / "label_matrix.csv")
    plot_label_matrix(label_matrix, dataset_dir / "label_matrix.png", debug_names=debug_names)

    plot_confidence_summary(
        dataset_df,
        summary_df,
        display_labels,
        dataset_label,
        dataset_dir / "confidence_summary.png",
    )
    export_case_difficulty(dataset_df, dataset_dir / "case_difficulty.csv")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="white")
    latest_sessions = load_latest_sessions(args.session_root)
    annotations = load_annotations(latest_sessions)

    dataset_keys = sorted(annotations["dataset_key"].unique())
    if args.dataset != "all":
        dataset_keys = [key for key in dataset_keys if key == args.dataset]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset_key in dataset_keys:
        dataset_df = annotations.loc[annotations["dataset_key"] == dataset_key].copy()
        analyze_dataset(dataset_key, dataset_df, args.out_dir, debug_names=args.debug_names)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
