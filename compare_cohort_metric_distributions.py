#!/usr/bin/env python3
"""
compare_cohort_metric_distributions.py

Compare spatial metric distributions between two cohorts.

Typical usage:
  python compare_cohort_metric_distributions.py \
    --a outputs/metrics/spatial_metrics_cellpose.csv --a-name BOMI2_cellpose \
    --b outputs/metrics/spatial_metrics_EXTERNAL_BOMI1.csv --b-name BOMI1_CellViT \
    --out-dir outputs/comparisons/cohort_metric_compare

Outputs:
  - out-dir/hists/*.png                     Overlay histograms per metric
  - out-dir/summary_metrics.csv             Per-metric summary stats + distribution distances
  - out-dir/summary_metrics_top.md          LLM-friendly narrative summary (ranked differences)
  - out-dir/cohort_separability.txt         Cohort-classifier AUC + top separating features

Notes:
- This is intended for diagnosing cohort/domain shift.
- Uses KS test, Wasserstein distance, Cohen's d, Cliff's delta.
- Also fits a cohort classifier (logistic regression) to quantify multivariate separability.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


META_COLS_DEFAULT = ["ID", "sample_name", "label"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_csv_checked(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if os.path.getsize(path) < 10:
        raise RuntimeError(f"CSV seems empty (size < 10 bytes): {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Loaded empty dataframe from: {path}")
    return df


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    nx, ny = x.size, y.size
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    denom = nx + ny - 2
    if denom <= 0:
        return float("nan")
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / denom)
    if not np.isfinite(sp) or sp == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / sp)


def cliffs_delta(x: np.ndarray, y: np.ndarray, max_samples: int = 5000, seed: int = 0) -> float:
    """
    Nonparametric effect size in [-1, 1].
    Subsamples for speed if arrays are large.
    """
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    if x.size > max_samples:
        x = rng.choice(x, size=max_samples, replace=False)
    if y.size > max_samples:
        y = rng.choice(y, size=max_samples, replace=False)

    gt = 0
    lt = 0
    for xv in x:
        gt += int(np.sum(xv > y))
        lt += int(np.sum(xv < y))
    n = x.size * y.size
    return float((gt - lt) / n) if n > 0 else float("nan")


@dataclass
class MetricSummary:
    metric: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    median_a: float
    median_b: float
    std_a: float
    std_b: float
    q05_a: float
    q95_a: float
    q05_b: float
    q95_b: float
    ks_stat: float
    ks_p: float
    wdist: float
    cohens_d: float
    cliffs_delta: float
    frac_nan_a: float
    frac_nan_b: float
    frac_inf_a: float
    frac_inf_b: float


def safe_quantile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def summarize_metric(metric: str, a: pd.Series, b: pd.Series, seed: int = 0) -> MetricSummary:
    a_raw = a.to_numpy()
    b_raw = b.to_numpy()

    frac_nan_a = float(np.mean(np.isnan(a_raw)))
    frac_nan_b = float(np.mean(np.isnan(b_raw)))
    frac_inf_a = float(np.mean(np.isinf(a_raw)))
    frac_inf_b = float(np.mean(np.isinf(b_raw)))

    a2 = a_raw.copy()
    b2 = b_raw.copy()
    a2[np.isinf(a2)] = np.nan
    b2[np.isinf(b2)] = np.nan

    a_vals = a2[np.isfinite(a2)]
    b_vals = b2[np.isfinite(b2)]

    if a_vals.size == 0 or b_vals.size == 0:
        return MetricSummary(
            metric=metric,
            n_a=int(a_vals.size), n_b=int(b_vals.size),
            mean_a=float("nan"), mean_b=float("nan"),
            median_a=float("nan"), median_b=float("nan"),
            std_a=float("nan"), std_b=float("nan"),
            q05_a=float("nan"), q95_a=float("nan"),
            q05_b=float("nan"), q95_b=float("nan"),
            ks_stat=float("nan"), ks_p=float("nan"),
            wdist=float("nan"),
            cohens_d=float("nan"),
            cliffs_delta=float("nan"),
            frac_nan_a=frac_nan_a, frac_nan_b=frac_nan_b,
            frac_inf_a=frac_inf_a, frac_inf_b=frac_inf_b,
        )

    ks = ks_2samp(a_vals, b_vals, alternative="two-sided", mode="auto")
    wdist = wasserstein_distance(a_vals, b_vals)

    return MetricSummary(
        metric=metric,
        n_a=int(a_vals.size),
        n_b=int(b_vals.size),
        mean_a=float(np.mean(a_vals)),
        mean_b=float(np.mean(b_vals)),
        median_a=float(np.median(a_vals)),
        median_b=float(np.median(b_vals)),
        std_a=float(np.std(a_vals, ddof=1)) if a_vals.size > 1 else float("nan"),
        std_b=float(np.std(b_vals, ddof=1)) if b_vals.size > 1 else float("nan"),
        q05_a=safe_quantile(a_vals, 0.05),
        q95_a=safe_quantile(a_vals, 0.95),
        q05_b=safe_quantile(b_vals, 0.05),
        q95_b=safe_quantile(b_vals, 0.95),
        ks_stat=float(ks.statistic),
        ks_p=float(ks.pvalue),
        wdist=float(wdist),
        cohens_d=cohens_d(a_vals, b_vals),
        cliffs_delta=cliffs_delta(a_vals, b_vals, seed=seed),
        frac_nan_a=frac_nan_a,
        frac_nan_b=frac_nan_b,
        frac_inf_a=frac_inf_a,
        frac_inf_b=frac_inf_b,
    )


def overlay_hist(metric: str,
                 a_vals: np.ndarray,
                 b_vals: np.ndarray,
                 a_name: str,
                 b_name: str,
                 out_path: str,
                 bins: int,
                 clip_qlo: float,
                 clip_qhi: float) -> None:
    a_vals = a_vals[np.isfinite(a_vals)]
    b_vals = b_vals[np.isfinite(b_vals)]
    if a_vals.size == 0 or b_vals.size == 0:
        return

    combined = np.concatenate([a_vals, b_vals])
    lo = np.quantile(combined, clip_qlo)
    hi = np.quantile(combined, clip_qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.min(combined)), float(np.max(combined))
        if lo >= hi:
            return

    # clip for display only
    a_plot = np.clip(a_vals, lo, hi)
    b_plot = np.clip(b_vals, lo, hi)

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(a_plot, bins=bins, density=True, alpha=0.55, label=a_name)
    plt.hist(b_plot, bins=bins, density=True, alpha=0.55, label=b_name)
    plt.title(metric)
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def patient_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-core metrics to per-patient via cell_count-weighted averaging
    (mirrors your training aggregation logic).

    Requires: ID, label, cell_count.
    """
    required = {"ID", "label", "cell_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Patient aggregation requires columns: {sorted(missing)}")

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]
    feat_cols = [c for c in num_cols if c not in {"cell_count", "cancer_count", "stroma_count"}]

    def agg_one(g: pd.DataFrame) -> Dict[str, float]:
        w = pd.to_numeric(g["cell_count"], errors="coerce").fillna(0).to_numpy(float)
        wsum = float(np.sum(w))
        out: Dict[str, float] = {
            "label": int(g["label"].iloc[0]),
            "cell_count": float(np.nansum(w)),
        }
        for c in ["cancer_count", "stroma_count"]:
            if c in g.columns:
                out[c] = float(pd.to_numeric(g[c], errors="coerce").fillna(0).sum())

        for c in feat_cols:
            x = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
            x[np.isinf(x)] = np.nan
            if wsum > 0:
                out[c] = float(np.nansum(x * w) / wsum)
            else:
                out[c] = float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")
        return out

    rows = []
    for patient_id, g in df.groupby("ID", sort=False):
        row = {"ID": patient_id}
        row.update(agg_one(g))
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_separability(df_a: pd.DataFrame,
                        df_b: pd.DataFrame,
                        feat_cols: List[str],
                        a_name: str,
                        b_name: str,
                        out_path: str,
                        seed: int = 0) -> None:
    """
    Train a cohort classifier (A vs B) to quantify multivariate shift.
    Writes CV AUC and top coefficients.
    """
    X = pd.concat([df_a[feat_cols], df_b[feat_cols]], ignore_index=True)
    y = np.array([0] * len(df_a) + [1] * len(df_b), dtype=int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    pipe.fit(X, y)
    coef = pipe.named_steps["clf"].coef_.ravel()

    idx = np.argsort(-np.abs(coef))[:25]
    top = [(feat_cols[i], float(coef[i])) for i in idx]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Cohort separability (multivariate shift)\n")
        f.write(f"Classes: 0={a_name}, 1={b_name}\n\n")
        f.write(f"5-fold CV AUC: mean={float(np.mean(aucs)):.4f}, std={float(np.std(aucs, ddof=1)):.4f}\n\n")
        f.write("Top 25 separating features (logistic regression coefficients):\n")
        f.write("  Positive => higher in B, Negative => higher in A\n\n")
        for feat, c in top:
            f.write(f"{feat:40s} {c:+.6f}\n")


def write_llm_summary_md(summary_df: pd.DataFrame, a_name: str, b_name: str, out_path: str) -> None:
    """
    Compact LLM-friendly summary. Ranks by KS, |Cohen's d|, |Cliff's delta|.
    """
    s = summary_df.copy()

    s_ks = s.sort_values("ks_stat", ascending=False).head(25)
    s_cd = s.assign(abs_d=np.abs(s["cohens_d"])).sort_values("abs_d", ascending=False).head(25)
    s_cl = s.assign(abs_cl=np.abs(s["cliffs_delta"])).sort_values("abs_cl", ascending=False).head(25)

    def fmt(r) -> str:
        return (
            f"- **{r['metric']}**: "
            f"KS={r['ks_stat']:.3f}, p={r['ks_p']:.1e}, "
            f"mean({a_name})={r['mean_a']:.4g}, mean({b_name})={r['mean_b']:.4g}, "
            f"d={r['cohens_d']:.3g}, cliff={r['cliffs_delta']:.3g}, "
            f"NaN% A={100*r['frac_nan_a']:.2f}, B={100*r['frac_nan_b']:.2f}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Cohort metric distribution comparison\n\n")
        f.write(f"Compared cohorts:\n- A: {a_name}\n- B: {b_name}\n\n")

        f.write("## Interpretation guide\n")
        f.write("- KS statistic close to 0 means similar distributions; close to 1 means strong separation.\n")
        f.write("- Cohen's d is standardized mean difference (positive means higher in A).\n")
        f.write("- Cliff's delta is nonparametric effect size in [-1, 1].\n")
        f.write("- With large N, KS p-values often go to ~0. Focus on effect sizes + plots.\n\n")

        f.write("## Top metrics by KS distance\n")
        for _, r in s_ks.iterrows():
            f.write(fmt(r) + "\n")

        f.write("\n## Top metrics by |Cohen's d|\n")
        for _, r in s_cd.iterrows():
            f.write(fmt(r) + "\n")

        f.write("\n## Top metrics by |Cliff's delta|\n")
        for _, r in s_cl.iterrows():
            f.write(fmt(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Cohort A metrics CSV")
    ap.add_argument("--b", required=True, help="Cohort B metrics CSV")
    ap.add_argument("--a-name", default="CohortA", help="Label for cohort A")
    ap.add_argument("--b-name", default="CohortB", help="Label for cohort B")
    ap.add_argument("--out-dir", default="outputs/comparisons/cohort_metric_compare", help="Output directory")
    ap.add_argument("--bins", type=int, default=50, help="Histogram bins")
    ap.add_argument("--clip-qlo", type=float, default=0.01, help="Lower quantile for plot clipping")
    ap.add_argument("--clip-qhi", type=float, default=0.99, help="Upper quantile for plot clipping")
    ap.add_argument("--aggregate", choices=["core", "patient"], default="core",
                    help="Compare per-core or per-patient (cell_count-weighted) metrics")
    ap.add_argument("--include-counts", action="store_true",
                    help="Include cell_count/cancer_count/stroma_count in comparisons (default excluded)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    hist_dir = os.path.join(args.out_dir, "hists")
    ensure_dir(hist_dir)

    df_a = read_csv_checked(args.a)
    df_b = read_csv_checked(args.b)

    if args.aggregate == "patient":
        df_a = patient_aggregate(df_a)
        df_b = patient_aggregate(df_b)

    # Determine comparable numeric metrics (intersection)
    drop_cols = set([c for c in META_COLS_DEFAULT if c in df_a.columns or c in df_b.columns])
    if not args.include_counts:
        drop_cols |= {"cell_count", "cancer_count", "stroma_count"}

    num_a = set(df_a.select_dtypes(include=[np.number]).columns)
    num_b = set(df_b.select_dtypes(include=[np.number]).columns)
    metrics = sorted(list((num_a & num_b) - drop_cols))
    if not metrics:
        raise RuntimeError(
            f"No common numeric metric columns found after dropping: {sorted(drop_cols)}. "
            "Try --include-counts or inspect CSV columns."
        )

    # Robust numeric conversion
    df_a = to_numeric(df_a, metrics)
    df_b = to_numeric(df_b, metrics)

    # Summaries + plots
    rows = []
    for m in metrics:
        summ = summarize_metric(m, df_a[m], df_b[m], seed=args.seed)
        rows.append(summ.__dict__)

        out_png = os.path.join(hist_dir, f"hist_{m}.png")
        a_vals = df_a[m].to_numpy()
        b_vals = df_b[m].to_numpy()
        a_vals = np.where(np.isinf(a_vals), np.nan, a_vals)
        b_vals = np.where(np.isinf(b_vals), np.nan, b_vals)
        overlay_hist(
            metric=m,
            a_vals=a_vals,
            b_vals=b_vals,
            a_name=args.a_name,
            b_name=args.b_name,
            out_path=out_png,
            bins=args.bins,
            clip_qlo=args.clip_qlo,
            clip_qhi=args.clip_qhi,
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(args.out_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_md = os.path.join(args.out_dir, "summary_metrics_top.md")
    write_llm_summary_md(summary_df, args.a_name, args.b_name, summary_md)

    sep_txt = os.path.join(args.out_dir, "cohort_separability.txt")
    cohort_separability(df_a, df_b, metrics, args.a_name, args.b_name, sep_txt, seed=args.seed)

    print(f"[OK] Compared {len(metrics)} shared numeric metrics ({args.aggregate}-level).")
    print(f"[OK] Histograms saved to: {hist_dir}")
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] LLM summary: {summary_md}")
    print(f"[OK] Cohort separability: {sep_txt}")


if __name__ == "__main__":
    main()
