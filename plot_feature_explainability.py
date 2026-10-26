"""Create report-ready feature explainability figures.

Outputs:
- side-by-side mIF and H&E in-domain permutation importance plot
- train-set LUAD-vs-LUSC signed direction heatmap
- CSV tables backing both figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spatial_metrics import BEHANOVA_FEATURE_COLUMNS
from spatial_utils import aggregate_cores_to_patient


FEATURE_LABELS = {
    "cancer_ripley_L_auc": "Cancer Ripley's L AUC",
    "stroma_ripley_L_auc": "Stroma Ripley's L AUC",
    "newmans_assortativity": "Cancer/stroma assortativity",
    "cancer_degree_centrality": "Cancer group degree centrality",
    "stroma_degree_centrality": "Stroma group degree centrality",
    "cancer_closeness_centrality": "Cancer group closeness centrality",
    "stroma_closeness_centrality": "Stroma group closeness centrality",
    "cancer_average_clustering": "Cancer local clustering",
    "stroma_average_clustering": "Stroma local clustering",
    "cancer_stroma_cooccurrence_auc": "Cancer-stroma co-occurrence ratio AUC",
    "cancer_stroma_cooccurrence_deviation_auc": "Cancer-stroma co-occurrence deviation",
    "cancer_cancer_neighborhood_enrichment_zscore": "Cancer-cancer enrichment",
    "cancer_stroma_neighborhood_enrichment_zscore": "Cancer-stroma enrichment",
    "stroma_stroma_neighborhood_enrichment_zscore": "Stroma-stroma enrichment",
    "cancer_stroma_object_object_pearson_correlation": "Cancer-stroma local correlation",
}


FEATURE_FAMILIES = {
    "cancer_ripley_L_auc": "Within-compartment clustering",
    "stroma_ripley_L_auc": "Within-compartment clustering",
    "newmans_assortativity": "Compartment segregation",
    "cancer_degree_centrality": "Graph interface/centrality",
    "stroma_degree_centrality": "Graph interface/centrality",
    "cancer_closeness_centrality": "Graph interface/centrality",
    "stroma_closeness_centrality": "Graph interface/centrality",
    "cancer_average_clustering": "Within-compartment graph structure",
    "stroma_average_clustering": "Within-compartment graph structure",
    "cancer_stroma_cooccurrence_auc": "Cancer-stroma interface",
    "cancer_stroma_cooccurrence_deviation_auc": "Deviation from random mixing",
    "cancer_cancer_neighborhood_enrichment_zscore": "Same-type enrichment",
    "cancer_stroma_neighborhood_enrichment_zscore": "Cancer-stroma interface",
    "stroma_stroma_neighborhood_enrichment_zscore": "Same-type enrichment",
    "cancer_stroma_object_object_pearson_correlation": "Cancer-stroma interface",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance-dir", default="feature_importance_bundles")
    parser.add_argument("--mif-importance-csv", default="")
    parser.add_argument("--he-importance-csv", default="")
    parser.add_argument("--out-dir", default=str(Path("outputs") / "figures" / "explainability"))
    parser.add_argument("--internal-metrics", default=str(Path("outputs") / "metrics" / "spatial_metrics_cellpose.csv"))
    parser.add_argument("--external-metrics", default=str(Path("outputs") / "metrics" / "spatial_metrics_EXTERNAL_BOMI1.csv"))
    parser.add_argument(
        "--internal-train-csv",
        default="../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/train_val.csv",
    )
    parser.add_argument(
        "--external-train-csv",
        default="/home/love/multiplex_dataset/lung_cancer_BOMI1_dataset/HE_dataset/binary_subtype_prediction_ACvsSqCC/static_split/train_val.csv",
    )
    return parser.parse_args()


def latest_matching_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern!r} under {root}")
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def load_importance_table(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"feature", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    out = df[["feature", "importance"]].copy()
    out["feature"] = out["feature"].astype(str)
    out["importance"] = pd.to_numeric(out["importance"], errors="coerce")
    out["source"] = source_label
    out["feature_label"] = out["feature"].map(FEATURE_LABELS).fillna(out["feature"])
    out["feature_family"] = out["feature"].map(FEATURE_FAMILIES).fillna("Other")
    return out


def _direction_stats(df: pd.DataFrame, feature: str) -> dict[str, float | str]:
    work = df[[feature, "label"]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work = work.dropna(subset=[feature, "label"])
    work = work[work["label"].isin([0, 1])]

    values0 = work.loc[work["label"] == 0, feature].to_numpy(float)
    values1 = work.loc[work["label"] == 1, feature].to_numpy(float)
    mean0 = float(np.mean(values0)) if values0.size else np.nan
    mean1 = float(np.mean(values1)) if values1.size else np.nan
    delta = mean1 - mean0 if np.isfinite(mean0) and np.isfinite(mean1) else np.nan
    auc = np.nan
    if values0.size > 0 and values1.size > 0 and work[feature].nunique(dropna=True) > 1:
        auc = float(roc_auc_score(work["label"].to_numpy(int), work[feature].to_numpy(float)))
    signed_auc_delta = auc - 0.5 if np.isfinite(auc) else np.nan
    direction = "LUAD higher" if delta > 0 else ("LUSC higher" if delta < 0 else "Flat")
    return {
        "mean_lusc": mean0,
        "mean_luad": mean1,
        "delta_luad_minus_lusc": delta,
        "auc_luad_raw_feature": auc,
        "signed_auc_delta": signed_auc_delta,
        "direction": direction,
    }


def patient_train_table(metrics_path: Path, split_path: Path) -> pd.DataFrame:
    patient = aggregate_cores_to_patient(pd.read_csv(metrics_path))
    patient["ID"] = patient["ID"].astype(str)
    train_ids = set(pd.read_csv(split_path)["ID"].astype(str))
    return patient[patient["ID"].isin(train_ids)].copy()


def build_direction_table(
    internal_metrics: Path,
    external_metrics: Path,
    internal_train_csv: Path,
    external_train_csv: Path,
    features: list[str],
) -> pd.DataFrame:
    internal = patient_train_table(internal_metrics, internal_train_csv)
    external = patient_train_table(external_metrics, external_train_csv)
    rows = []
    for source, df in [("mIF train", internal), ("H&E train", external)]:
        for feature in features:
            if feature not in df.columns:
                continue
            stats = _direction_stats(df, feature)
            rows.append(
                {
                    "source": source,
                    "feature": feature,
                    "feature_label": FEATURE_LABELS.get(feature, feature),
                    "feature_family": FEATURE_FAMILIES.get(feature, "Other"),
                    **stats,
                }
            )
    out = pd.DataFrame(rows)
    pivot = out.pivot(index="feature", columns="source", values="signed_auc_delta")
    if {"mIF train", "H&E train"}.issubset(pivot.columns):
        discordant = (np.sign(pivot["mIF train"]) * np.sign(pivot["H&E train"])) < 0
        out["discordant"] = out["feature"].map(discordant.to_dict()).fillna(False).astype(bool)
    else:
        out["discordant"] = False
    return out


def plot_importance(importance: pd.DataFrame, out_dir: Path) -> None:
    wide = (
        importance.pivot_table(index="feature", columns="source", values="importance", aggfunc="mean")
        .reindex(columns=["mIF", "H&E"])
        .fillna(0.0)
    )
    order = wide.max(axis=1).sort_values(ascending=True).index.tolist()
    y = np.arange(len(order))
    labels = [FEATURE_LABELS.get(feature, feature) for feature in order]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 7.0), sharey=True)
    colors = {"mIF": "#31688e", "H&E": "#b45f06"}
    x_min = min(float(wide.min().min()) * 1.12, 0.0)
    x_max = max(float(wide.max().max()) * 1.12, 0.01)
    for ax, source in zip(axes, ["mIF", "H&E"]):
        vals = wide.loc[order, source]
        ax.barh(y, vals, color=colors[source], height=0.68)
        ax.set_title(f"{source} in-domain model", fontsize=12, fontweight="bold")
        ax.set_xlabel("Permutation importance (ΔROC AUC)")
        ax.set_xlim(x_min, x_max)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.suptitle("Spatial Feature Importance by Training Modality", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"feature_importance_mif_he.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_direction_heatmap(direction: pd.DataFrame, importance: pd.DataFrame, out_dir: Path) -> None:
    imp_order = (
        importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    matrix = (
        direction.pivot_table(index="feature", columns="source", values="signed_auc_delta", aggfunc="mean")
        .reindex(index=imp_order, columns=["mIF train", "H&E train"])
    )
    matrix = matrix.dropna(how="all")
    discordant_by_feature = direction.groupby("feature")["discordant"].any().to_dict()
    labels = [
        f"{FEATURE_LABELS.get(feature, feature)}*" if discordant_by_feature.get(feature, False) else FEATURE_LABELS.get(feature, feature)
        for feature in matrix.index
    ]

    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    vmax = max(0.05, float(np.nanmax(np.abs(matrix.to_numpy(float)))))
    image = ax.imshow(matrix.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(labels)
    ax.set_title("Train-Set Feature Direction by Cohort", fontsize=14, fontweight="bold")

    for i, feature in enumerate(matrix.index):
        for j, source in enumerate(matrix.columns):
            value = matrix.loc[feature, source]
            if not np.isfinite(value):
                text = ""
            else:
                text = f"{value:+.2f}"
            color = "white" if np.isfinite(value) and abs(value) > vmax * 0.55 else "#222222"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_xlabel("Cohort train split")
    ax.set_ylabel("")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Signed AUC delta: LUAD higher (+), LUSC higher (-)")
    fig.text(0.02, 0.015, "* Discordant train-set direction", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_dir / f"feature_direction_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    importance_dir = Path(args.importance_dir)

    mif_path = (
        Path(args.mif_importance_csv)
        if args.mif_importance_csv
        else latest_matching_file(
            importance_dir,
            "feature_importance__BOMI2__train_internal__test_internal__*.csv",
        )
    )
    he_path = (
        Path(args.he_importance_csv)
        if args.he_importance_csv
        else latest_matching_file(
            importance_dir,
            "feature_importance__BOMI1__train_external__test_external__*.csv",
        )
    )
    importance = pd.concat(
        [
            load_importance_table(mif_path, "mIF"),
            load_importance_table(he_path, "H&E"),
        ],
        ignore_index=True,
    )
    importance.to_csv(out_dir / "feature_importance_mif_he.csv", index=False)

    features = [feature for feature in BEHANOVA_FEATURE_COLUMNS if feature in set(importance["feature"])]
    direction = build_direction_table(
        Path(args.internal_metrics),
        Path(args.external_metrics),
        Path(args.internal_train_csv),
        Path(args.external_train_csv),
        features,
    )
    direction.to_csv(out_dir / "feature_direction_train_mif_he.csv", index=False)

    plot_importance(importance, out_dir)
    plot_direction_heatmap(direction, importance, out_dir)

    print(f"[INFO] mIF importance source: {mif_path}")
    print(f"[INFO] H&E importance source: {he_path}")
    print(f"[INFO] Saved explainability figures and CSVs to {out_dir}")


if __name__ == "__main__":
    main()
