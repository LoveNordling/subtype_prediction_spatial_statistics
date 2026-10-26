"""Analyze transfer-oriented univariate feature discriminativity.

Primary diagnostic:
- source-cohort train versus opposite-cohort held-out test, matching the
  actual cross-cohort transfer experiments.

Secondary sensitivity analyses:
- train versus train
- test versus test

Note:
- The target test sets contain only 30 patients, so target-test univariate
  AUCs are noisy. These diagnostics should be interpreted as
  explanatory/hypothesis-generating, not definitive statistical proof.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from spatial_data_pipeline import ExternalPaths, InternalPaths, prepare_patient_level_tables
from spatial_utils import CountQC, get_feature_columns


NOISY_TARGET_NOTE = (
    "The target test sets contain only 30 patients, so target-test univariate "
    "AUCs are noisy. These diagnostics should be interpreted as "
    "explanatory/hypothesis-generating, not definitive statistical proof."
)
EXCLUDED_FEATURES = {"cell_count", "cancer_count", "stroma_count"}
KEY_TRANSFER_FEATURES = [
    "cancer_stroma_object_object_pearson_correlation",
    "cancer_stroma_cooccurrence_auc",
    "cancer_stroma_cooccurrence_deviation_auc",
    "stroma_ripley_L_auc",
    "cancer_ripley_L_auc",
    "cancer_stroma_neighborhood_enrichment_zscore",
    "cancer_degree_centrality",
]
KEY_INTERACTION_FEATURE = "cancer_stroma_object_object_pearson_correlation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(Path("outputs") / "comparisons" / "feature_discriminativity_transfer"),
        help="Directory for CSV outputs and figures.",
    )
    parser.add_argument(
        "--internal-source",
        default="cellpose",
        choices=["inform", "cellpose", "cellprofiler"],
        help="Internal BOMI2 source used to rebuild patient-level tables.",
    )
    parser.add_argument(
        "--mif-importance-csv",
        default=None,
        help="Optional override for the mIF/BOMI2 train_internal->test_external feature importance CSV.",
    )
    parser.add_argument(
        "--he-importance-csv",
        default=None,
        help="Optional override for the H&E/BOMI1 train_external->test_internal feature importance CSV.",
    )
    parser.add_argument(
        "--external-xy-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to BOMI1 x/y before metric loading/recomputation.",
    )
    parser.add_argument(
        "--recompute-internal-metrics",
        action="store_true",
        help="Recompute internal cached metrics instead of loading them.",
    )
    parser.add_argument(
        "--recompute-external-metrics",
        action="store_true",
        help="Recompute external cached metrics instead of loading them.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for metric recomputation when enabled.",
    )
    return parser.parse_args()


def latest_matching_file(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern {pattern!r} under {root}")
    return max(matches, key=lambda path: (path.stat().st_mtime, path.name))


def load_importance_table(path: Path, importance_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"feature", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}. Found: {list(df.columns)}")
    out = df[["feature", "importance"]].copy()
    out["feature"] = out["feature"].astype(str)
    out[importance_col_name] = pd.to_numeric(out["importance"], errors="coerce")
    out = out.drop(columns=["importance"]).sort_values(importance_col_name, ascending=False).reset_index(drop=True)
    return out


def collect_feature_columns(frames: Iterable[pd.DataFrame]) -> list[str]:
    feature_cols: set[str] = set()
    for df in frames:
        feature_cols.update(get_feature_columns(df))
    return sorted(feature_cols - EXCLUDED_FEATURES)


def _safe_sign(value: float) -> float:
    if pd.isna(value):
        return np.nan
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def compute_feature_discriminativity(df: pd.DataFrame, feature: str) -> dict[str, float]:
    if feature not in df.columns:
        raise ValueError(f"Feature {feature!r} not found in columns.")
    if "label" not in df.columns:
        raise ValueError("Expected column 'label' in dataframe.")

    work = df[[feature, "label"]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work["label"] = pd.to_numeric(work["label"], errors="coerce")
    work = work.dropna(subset=["label"])
    work = work[work["label"].isin([0, 1])]
    work = work.dropna(subset=[feature])

    values0 = work.loc[work["label"] == 0, feature].to_numpy(dtype=float)
    values1 = work.loc[work["label"] == 1, feature].to_numpy(dtype=float)

    mean0 = float(np.mean(values0)) if values0.size else np.nan
    mean1 = float(np.mean(values1)) if values1.size else np.nan
    median0 = float(np.median(values0)) if values0.size else np.nan
    median1 = float(np.median(values1)) if values1.size else np.nan
    std0 = float(np.std(values0, ddof=1)) if values0.size >= 2 else np.nan
    std1 = float(np.std(values1, ddof=1)) if values1.size >= 2 else np.nan
    delta_mean = mean1 - mean0 if np.isfinite(mean0) and np.isfinite(mean1) else np.nan

    result = {
        "n_total": int(values0.size + values1.size),
        "n_label0": int(values0.size),
        "n_label1": int(values1.size),
        "mean_label0": mean0,
        "mean_label1": mean1,
        "median_label0": median0,
        "median_label1": median1,
        "std_label0": std0,
        "std_label1": std1,
        "delta_mean": delta_mean,
        "cohen_d": np.nan,
        "auc": np.nan,
        "auc_strength": np.nan,
        "signed_auc_delta": np.nan,
        "abs_auc_delta": np.nan,
    }

    if work.empty or values0.size < 2 or values1.size < 2:
        return result

    feature_values = work[feature].to_numpy(dtype=float)
    labels = work["label"].to_numpy(dtype=int)
    if not np.isfinite(feature_values).any():
        return result
    if np.nanstd(feature_values) == 0:
        return result

    n0 = values0.size
    n1 = values1.size
    var0 = np.var(values0, ddof=1)
    var1 = np.var(values1, ddof=1)
    pooled_var = ((n0 - 1) * var0 + (n1 - 1) * var1) / (n0 + n1 - 2)
    pooled_std = np.sqrt(pooled_var) if pooled_var >= 0 else np.nan
    if np.isfinite(pooled_std) and pooled_std > 0:
        result["cohen_d"] = float((mean1 - mean0) / pooled_std)

    auc = float(roc_auc_score(labels, feature_values))
    result["auc"] = auc
    result["auc_strength"] = float(max(auc, 1.0 - auc))
    result["signed_auc_delta"] = float(auc - 0.5)
    result["abs_auc_delta"] = float(abs(auc - 0.5))
    return result


def build_discriminativity_table(table_name: str, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in tqdm(feature_cols, desc=f"{table_name} features"):
        row = {"table_name": table_name, "feature": feature}
        row.update(compute_feature_discriminativity(df, feature))
        rows.append(row)
    return pd.DataFrame(rows)


def rename_stats_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {
        "auc": f"{prefix}_auc",
        "auc_strength": f"{prefix}_auc_strength",
        "signed_auc_delta": f"{prefix}_signed_auc_delta",
        "abs_auc_delta": f"{prefix}_abs_auc_delta",
        "cohen_d": f"{prefix}_cohen_d",
    }
    return df[["feature", *rename_map.keys()]].rename(columns=rename_map)


def add_transfer_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    source_sign = out["source_signed_auc_delta"].map(_safe_sign)
    target_sign = out["target_signed_auc_delta"].map(_safe_sign)
    out["direction_match"] = np.where(
        source_sign.isna() | target_sign.isna(),
        np.nan,
        source_sign == target_sign,
    )
    out["signal_loss"] = out["source_abs_auc_delta"] - out["target_abs_auc_delta"]
    out["abs_cohen_d_loss"] = out["source_cohen_d"].abs() - out["target_cohen_d"].abs()
    return out


def build_comparison_table(
    left_df: pd.DataFrame,
    left_name: str,
    right_df: pd.DataFrame,
    right_name: str,
) -> pd.DataFrame:
    cols = [
        "feature",
        "auc_strength",
        "signed_auc_delta",
        "abs_auc_delta",
        "cohen_d",
    ]
    merged = left_df[cols].merge(right_df[cols], on="feature", suffixes=(f"_{left_name}", f"_{right_name}"))
    merged[f"abs_auc_delta_diff_{left_name}_minus_{right_name}"] = (
        merged[f"abs_auc_delta_{left_name}"] - merged[f"abs_auc_delta_{right_name}"]
    )
    merged[f"abs_cohen_d_diff_{left_name}_minus_{right_name}"] = (
        merged[f"cohen_d_{left_name}"].abs() - merged[f"cohen_d_{right_name}"].abs()
    )
    return merged.sort_values(f"abs_auc_delta_diff_{left_name}_minus_{right_name}", ascending=False).reset_index(drop=True)


def annotate_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    importance_col: str,
    y_col: str,
    always_label: Iterable[str] = (),
) -> None:
    labels_to_show = set(df.nlargest(10, importance_col)["feature"].astype(str).tolist())
    labels_to_show.update(always_label)
    for _, row in df.iterrows():
        feature = str(row["feature"])
        if feature not in labels_to_show:
            continue
        x = row[importance_col]
        y = row[y_col]
        if pd.isna(x) or pd.isna(y):
            continue
        ax.annotate(feature, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)


def save_importance_signal_scatter(
    df: pd.DataFrame,
    *,
    importance_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    always_label: Iterable[str] = (),
) -> None:
    plot_df = df.dropna(subset=[importance_col, y_col]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(plot_df[importance_col], plot_df[y_col], alpha=0.8, color="#1f77b4")
    annotate_points(ax, plot_df, importance_col, y_col, always_label=always_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_source_target_bars(
    df: pd.DataFrame,
    *,
    importance_col: str,
    title: str,
    out_path: Path,
) -> None:
    top_df = df.nlargest(10, importance_col).copy().iloc[::-1]
    y = np.arange(len(top_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y - width / 2, top_df["source_auc_strength"], height=width, label="Source train", color="#4c72b0")
    ax.barh(y + width / 2, top_df["target_auc_strength"], height=width, label="Target test", color="#dd8452")
    ax.set_yticks(y)
    ax.set_yticklabels(top_df["feature"])
    ax.set_xlabel("AUC strength")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _distribution_groups(df: pd.DataFrame, feature: str) -> tuple[np.ndarray, np.ndarray]:
    luad = pd.to_numeric(df.loc[df["label"] == 0, feature], errors="coerce").dropna().to_numpy(dtype=float)
    lusc = pd.to_numeric(df.loc[df["label"] == 1, feature], errors="coerce").dropna().to_numpy(dtype=float)
    return luad, lusc


def save_key_feature_distributions(
    *,
    mif_train: pd.DataFrame,
    he_test: pd.DataFrame,
    he_train: pd.DataFrame,
    mif_test: pd.DataFrame,
    feature_cols: list[str],
    out_path: Path,
) -> None:
    features = [feature for feature in KEY_TRANSFER_FEATURES if feature in feature_cols]
    if not features:
        return

    fig, axes = plt.subplots(len(features), 2, figsize=(14, max(4, 3.2 * len(features))), squeeze=False)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    left_labels = ["mIF train\nLUAD", "mIF train\nLUSC", "H&E test\nLUAD", "H&E test\nLUSC"]
    right_labels = ["H&E train\nLUAD", "H&E train\nLUSC", "mIF test\nLUAD", "mIF test\nLUSC"]

    for row_idx, feature in enumerate(features):
        left_data = [*_distribution_groups(mif_train, feature), *_distribution_groups(he_test, feature)]
        right_data = [*_distribution_groups(he_train, feature), *_distribution_groups(mif_test, feature)]

        for col_idx, (ax, data, labels, title) in enumerate(
            [
                (axes[row_idx, 0], left_data, left_labels, "mIF train vs H&E test"),
                (axes[row_idx, 1], right_data, right_labels, "H&E train vs mIF test"),
            ]
        ):
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel(feature)
            if row_idx == 0:
                ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Key feature subtype distributions across transfer directions", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_direction_summary(df: pd.DataFrame, importance_col: str, label: str) -> None:
    cols = [
        "feature",
        importance_col,
        "source_auc_strength",
        "target_auc_strength",
        "signal_loss",
        "direction_match",
    ]
    top_df = df.nlargest(10, importance_col)[cols].copy()
    print(f"\n{label}")
    print(top_df.to_string(index=False))


def print_interpretation(mif_diag: pd.DataFrame, he_diag: pd.DataFrame) -> None:
    print("\nInterpretation")
    print(NOISY_TARGET_NOTE)

    interaction_row = mif_diag.loc[mif_diag["feature"] == KEY_INTERACTION_FEATURE]
    interaction_supports = False
    if not interaction_row.empty:
        row = interaction_row.iloc[0]
        importance_rank = int((mif_diag["mif_importance"].rank(method="min", ascending=False)[interaction_row.index[0]]))
        interaction_supports = (
            importance_rank <= 10
            and pd.notna(row["source_auc_strength"])
            and pd.notna(row["target_auc_strength"])
            and row["source_auc_strength"] >= 0.65
            and (
                row["target_auc_strength"] <= 0.60
                or (
                    pd.notna(row["direction_match"])
                    and bool(row["direction_match"]) is False
                )
            )
        )

    he_top = he_diag.nlargest(10, "he_importance")
    he_retains = (
        not he_top.empty
        and he_top["direction_match"].fillna(False).mean() >= 0.7
        and he_top["target_auc_strength"].dropna().median() >= 0.60
    )

    mif_top = mif_diag.nlargest(10, "mif_importance")
    mif_loss = mif_top["signal_loss"].dropna().median() if not mif_top.empty else np.nan
    he_loss = he_top["signal_loss"].dropna().median() if not he_top.empty else np.nan
    similar_loss = (
        pd.notna(mif_loss)
        and pd.notna(he_loss)
        and abs(mif_loss - he_loss) < 0.03
    )

    if interaction_supports:
        print(
            f"{KEY_INTERACTION_FEATURE} shows high mIF importance, strong mIF-train discriminativity, and weak or reversed "
            "H&E-test signal. This supports the hypothesis that mIF→H&E transfer fails partly because an "
            "mIF-learned cancer-stroma association signal is not preserved in the H&E test representation."
        )
    if he_retains:
        print(
            "Top H&E-important features largely retain direction and target-test discriminativity in mIF test. "
            "This supports the hypothesis that H&E→mIF transfer succeeds because the H&E model relies on broader "
            "spatial features that remain discriminative in mIF."
        )
    if similar_loss or (not interaction_supports and not he_retains):
        print(
            "Both directions show similar univariate signal loss, or the retained-versus-lost pattern is not clean. "
            "This analysis does not explain the observed asymmetry by itself; the cause may be multivariate feature "
            "interactions, calibration, or broader distribution shift rather than simple univariate discriminativity loss."
        )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(NOISY_TARGET_NOTE)

    mif_importance_path = (
        Path(args.mif_importance_csv)
        if args.mif_importance_csv
        else latest_matching_file("feature_importance__BOMI2__train_internal__test_external__*.csv", Path("feature_importance_bundles"))
    )
    he_importance_path = (
        Path(args.he_importance_csv)
        if args.he_importance_csv
        else latest_matching_file("feature_importance__BOMI1__train_external__test_internal__*.csv", Path("feature_importance_bundles"))
    )
    print(f"[INFO] Using mIF importance CSV: {mif_importance_path}")
    print(f"[INFO] Using H&E importance CSV: {he_importance_path}")

    qc = CountQC()
    df_train_mif, df_test_mif, df_train_he, df_test_he = prepare_patient_level_tables(
        internal_source=args.internal_source,
        internal_paths=InternalPaths(),
        external_paths=ExternalPaths(),
        qc=qc,
        recompute_internal_metrics=args.recompute_internal_metrics,
        recompute_external_metrics=args.recompute_external_metrics,
        n_jobs=args.n_jobs,
        external_xy_scale=args.external_xy_scale,
    )
    df_all_mif = pd.concat([df_train_mif, df_test_mif], ignore_index=True)
    df_all_he = pd.concat([df_train_he, df_test_he], ignore_index=True)

    feature_cols = collect_feature_columns([df_train_mif, df_test_mif, df_all_mif, df_train_he, df_test_he, df_all_he])

    table_map = {
        "mif_train": df_train_mif,
        "mif_test": df_test_mif,
        "mif_all": df_all_mif,
        "he_train": df_train_he,
        "he_test": df_test_he,
        "he_all": df_all_he,
    }

    by_table = pd.concat(
        [build_discriminativity_table(table_name, df, feature_cols) for table_name, df in table_map.items()],
        ignore_index=True,
    )
    by_table.to_csv(out_dir / "feature_discriminativity_by_table.csv", index=False)

    stats_by_name = {
        name: by_table.loc[by_table["table_name"] == name].reset_index(drop=True)
        for name in table_map
    }

    mif_importance = load_importance_table(mif_importance_path, "mif_importance")
    he_importance = load_importance_table(he_importance_path, "he_importance")

    mif_diag = mif_importance.merge(rename_stats_columns(stats_by_name["mif_train"], "source"), on="feature", how="left")
    mif_diag = mif_diag.merge(rename_stats_columns(stats_by_name["he_test"], "target"), on="feature", how="left")
    mif_diag = add_transfer_columns(mif_diag).sort_values("mif_importance", ascending=False).reset_index(drop=True)
    mif_diag.to_csv(out_dir / "transfer_diagnostic_mif_train_to_he_test.csv", index=False)

    he_diag = he_importance.merge(rename_stats_columns(stats_by_name["he_train"], "source"), on="feature", how="left")
    he_diag = he_diag.merge(rename_stats_columns(stats_by_name["mif_test"], "target"), on="feature", how="left")
    he_diag = add_transfer_columns(he_diag).sort_values("he_importance", ascending=False).reset_index(drop=True)
    he_diag.to_csv(out_dir / "transfer_diagnostic_he_train_to_mif_test.csv", index=False)

    top_mif_signal_loss = (
        mif_diag.nlargest(10, "mif_importance").sort_values("signal_loss", ascending=False).reset_index(drop=True)
    )
    top_mif_signal_loss.to_csv(out_dir / "top_mif_features_transfer_signal_loss.csv", index=False)

    top_he_signal_loss = (
        he_diag.nlargest(10, "he_importance").sort_values("signal_loss", ascending=False).reset_index(drop=True)
    )
    top_he_signal_loss.to_csv(out_dir / "top_he_features_transfer_signal_loss.csv", index=False)

    train_vs_train = build_comparison_table(stats_by_name["mif_train"], "mif_train", stats_by_name["he_train"], "he_train")
    train_vs_train.to_csv(out_dir / "train_vs_train_discriminativity_comparison.csv", index=False)

    test_vs_test = build_comparison_table(stats_by_name["mif_test"], "mif_test", stats_by_name["he_test"], "he_test")
    test_vs_test.to_csv(out_dir / "test_vs_test_discriminativity_comparison.csv", index=False)

    save_importance_signal_scatter(
        mif_diag,
        importance_col="mif_importance",
        y_col="target_abs_auc_delta",
        title="mIF importance vs H&E test signal",
        x_label="mIF permutation importance",
        y_label="H&E test abs(AUC - 0.5)",
        out_path=figures_dir / "scatter_mif_importance_vs_he_test_signal.png",
        always_label=(
            "cancer_stroma_object_object_pearson_correlation",
            "cancer_stroma_cooccurrence_auc",
        ),
    )
    save_importance_signal_scatter(
        he_diag,
        importance_col="he_importance",
        y_col="target_abs_auc_delta",
        title="H&E importance vs mIF test signal",
        x_label="H&E permutation importance",
        y_label="mIF test abs(AUC - 0.5)",
        out_path=figures_dir / "scatter_he_importance_vs_mif_test_signal.png",
    )
    save_source_target_bars(
        mif_diag,
        importance_col="mif_importance",
        title="Top mIF-important features: source train vs target test AUC strength",
        out_path=figures_dir / "top_mif_features_source_vs_target_auc_strength.png",
    )
    save_source_target_bars(
        he_diag,
        importance_col="he_importance",
        title="Top H&E-important features: source train vs target test AUC strength",
        out_path=figures_dir / "top_he_features_source_vs_target_auc_strength.png",
    )
    save_key_feature_distributions(
        mif_train=df_train_mif,
        he_test=df_test_he,
        he_train=df_train_he,
        mif_test=df_test_mif,
        feature_cols=feature_cols,
        out_path=figures_dir / "key_feature_transfer_distributions.png",
    )

    print_direction_summary(mif_diag, "mif_importance", "mIF → H&E: top 10 mIF-important features")
    print_direction_summary(he_diag, "he_importance", "H&E → mIF: top 10 H&E-important features")

    interaction_row = mif_diag.loc[mif_diag["feature"] == KEY_INTERACTION_FEATURE]
    if not interaction_row.empty:
        print(f"\nmIF → H&E {KEY_INTERACTION_FEATURE} row")
        print(interaction_row.to_string(index=False))

    print_interpretation(mif_diag, he_diag)


if __name__ == "__main__":
    main()
