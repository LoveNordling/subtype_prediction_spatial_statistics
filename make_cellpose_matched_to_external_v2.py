#!/usr/bin/env python3
"""
Create a matched BOMI2 internal dataset whose per-core nearest-neighbor distance (NND)
distribution matches the external BOMI1 cohort.

Supported internal sources:
- cellpose
- inform

Supported matching compartments:
- all:      thin all cells to match the external all-cell NND distribution
- stroma:   keep cancer cells unchanged and thin only stroma cells
- separate: thin cancer and stroma independently against the corresponding
            external cancer and stroma NND distributions

The script writes:
- a new internal cells CSV with replicated / thinned cores
- a duplicated samples.csv aligned to those new cores
- QC histograms showing the original vs matched NND distributions
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from spatial_utils import preprocess_external_cells


_CORE_PAT = re.compile(r"(.+?)_Core\[1,(\d+,[A-Z])\]")


def extract_sample_name_from_filename(filename: str) -> str:
    match = _CORE_PAT.match(str(filename))
    if not match:
        raise ValueError(f"Could not parse sample name from filename: {filename}")
    return f"{match.group(1)}_[{match.group(2)}]"


def insert_suffix_before_core(value: str, suffix: str) -> str:
    text = str(value)
    if "_Core[1," not in text:
        raise ValueError(f"Expected '_Core[1,' in value, got: {text}")
    return text.replace("_Core[1,", f"{suffix}_Core[1,", 1)


def insert_suffix_in_canonical_sample_name(sample_name: str, suffix: str) -> str:
    text = str(sample_name)
    token = "_["
    if token not in text:
        raise ValueError(f"Expected canonical sample like '<prefix>_[<pos>]', got: {text}")
    prefix, rest = text.split(token, 1)
    return f"{prefix}{suffix}{token}{rest}"


def canonicalize_samples_csv_name(value: str) -> str:
    text = str(value)
    if "_" not in text:
        return text
    text = text[: text.rfind("_")]
    return text.replace("Core[1,", "[")


def remove_isolated_points(coords: np.ndarray, k: int = 5, max_percentile: float = 99.0) -> np.ndarray:
    if coords.shape[0] <= k:
        return np.ones(coords.shape[0], dtype=bool)
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k + 1)
    kth = dists[:, k]
    return kth <= np.percentile(kth, max_percentile)


def _median_nnd(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return float("nan")
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return float(np.median(dists[:, 1]))


def compute_sample_median_nnd(
    df: pd.DataFrame,
    *,
    min_cells: int,
    compartment: str,
) -> pd.DataFrame:
    rows = []
    for sample_name, group in df.groupby("sample_name", sort=False):
        if compartment == "all":
            selected = group
            metric_col = "median_nnd_all"
            count_col = "n_cells"
        elif compartment == "stroma":
            selected = group[group["CK"] == 0]
            metric_col = "median_nnd_stroma"
            count_col = "n_cells_stroma"
        else:
            raise ValueError(f"Unknown compartment: {compartment}")

        if len(selected) < min_cells:
            continue

        coords = selected[["x", "y"]].to_numpy(float, copy=False)
        keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
        coords_f = coords[keep]
        if coords_f.shape[0] < min_cells:
            continue

        rows.append(
            {
                "sample_name": sample_name,
                metric_col: _median_nnd(coords_f),
                count_col: int(coords_f.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _binary_search_keep_mask(
    coords_filtered: np.ndarray,
    *,
    target_nnd: float,
    min_cells: int,
    rng: np.random.Generator,
    p_init: float,
    n_binsrch: int,
    tol_rel: float,
) -> np.ndarray | None:
    if len(coords_filtered) < min_cells:
        return None

    p_min = min_cells / len(coords_filtered)
    lo = max(0.0, p_min)
    hi = min(1.0, max(p_init, p_min))
    uniform = rng.random(len(coords_filtered))

    best_keep = None
    best_err = float("inf")
    for _ in range(n_binsrch):
        p_trial = 0.5 * (lo + hi)
        keep_mask = uniform < p_trial
        if keep_mask.sum() < min_cells:
            lo = p_trial
            continue

        nnd_now = _median_nnd(coords_filtered[keep_mask])
        err = abs(nnd_now - target_nnd) / target_nnd
        if err < best_err:
            best_err = err
            best_keep = keep_mask.copy()

        if nnd_now < target_nnd:
            hi = p_trial
        else:
            lo = p_trial

        if best_err <= tol_rel:
            break

    return best_keep


def thin_internal_to_match_external_nnd(
    df_internal: pd.DataFrame,
    df_external: pd.DataFrame,
    *,
    compartment: str,
    n_subsamples: int,
    min_cells: int,
    seed: int,
    n_binsrch: int = 8,
    tol_rel: float = 0.03,
) -> pd.DataFrame:
    if compartment == "separate":
        src_stats_c = compute_sample_median_nnd(df_internal[df_internal["CK"] == 1].assign(CK=1), min_cells=min_cells, compartment="all")
        ref_stats_c = compute_sample_median_nnd(df_external[df_external["CK"] == 1].assign(CK=1), min_cells=min_cells, compartment="all")
        src_stats_s = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="stroma")
        ref_stats_s = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="stroma")

        if src_stats_c.empty or ref_stats_c.empty or src_stats_s.empty or ref_stats_s.empty:
            raise ValueError("No valid per-sample cancer/stroma NND stats for separate matching.")

        src_c_vals = src_stats_c["median_nnd_all"].dropna().to_numpy(float)
        ref_c_vals = ref_stats_c["median_nnd_all"].dropna().to_numpy(float)
        src_s_vals = src_stats_s["median_nnd_stroma"].dropna().to_numpy(float)
        ref_s_vals = ref_stats_s["median_nnd_stroma"].dropna().to_numpy(float)
        if min(len(src_c_vals), len(ref_c_vals), len(src_s_vals), len(ref_s_vals)) == 0:
            raise ValueError("No finite cancer/stroma NND values for separate matching.")

        src_c_sorted = np.sort(src_c_vals)
        src_s_sorted = np.sort(src_s_vals)
        src_stats_c = src_stats_c.set_index("sample_name")
        src_stats_s = src_stats_s.set_index("sample_name")
    else:
        src_stats = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment=compartment)
        ref_stats = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment=compartment)

        metric_col = "median_nnd_all" if compartment == "all" else "median_nnd_stroma"

        if src_stats.empty or ref_stats.empty:
            raise ValueError(
                f"No valid per-sample {compartment} NND stats for source or reference cohort. "
                "Check the chosen input and min-cells threshold."
            )

        src_vals = src_stats[metric_col].dropna().to_numpy(float)
        ref_vals = ref_stats[metric_col].dropna().to_numpy(float)
        if src_vals.size == 0 or ref_vals.size == 0:
            raise ValueError(f"No finite {compartment} NND values for source or reference.")

        src_sorted = np.sort(src_vals)
    groups = {sample_name: group for sample_name, group in df_internal.groupby("sample_name", sort=False)}

    out_parts = []
    for rep in range(1, n_subsamples + 1):
        rep_rng = np.random.default_rng(seed + rep)

        iterable = groups.keys() if compartment == "separate" else [row.sample_name for row in src_stats.itertuples(index=False)]
        for sample_name in tqdm(iterable, total=len(iterable), desc=f"Matching rep {rep}/{n_subsamples}"):
            group = groups.get(sample_name)
            if group is None:
                continue

            if compartment == "all":
                row = src_stats[src_stats["sample_name"] == sample_name].iloc[0]
                nnd_src = float(getattr(row, "median_nnd_all"))
                if not (np.isfinite(nnd_src) and nnd_src > 0):
                    continue
                rank = np.searchsorted(src_sorted, nnd_src, side="left")
                quantile = (rank + 0.5) / len(src_sorted)
                nnd_tgt = float(np.quantile(ref_vals, quantile))
                if not (np.isfinite(nnd_tgt) and nnd_tgt > 0):
                    continue
                p_keep = float(min(1.0, max(0.0, (nnd_src / nnd_tgt) ** 2)))

                base = group
                coords = base[["x", "y"]].to_numpy(float, copy=False)
                keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
                base_filtered = base.loc[keep]
                if len(base_filtered) < min_cells:
                    continue
                coords_filtered = base_filtered[["x", "y"]].to_numpy(float, copy=False)
                keep_mask = _binary_search_keep_mask(
                    coords_filtered,
                    target_nnd=nnd_tgt,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                if keep_mask is None:
                    continue
                matched = base_filtered.iloc[keep_mask].copy()
            elif compartment == "stroma":
                row = src_stats[src_stats["sample_name"] == sample_name].iloc[0]
                nnd_src = float(getattr(row, "median_nnd_stroma"))
                if not (np.isfinite(nnd_src) and nnd_src > 0):
                    continue
                rank = np.searchsorted(src_sorted, nnd_src, side="left")
                quantile = (rank + 0.5) / len(src_sorted)
                nnd_tgt = float(np.quantile(ref_vals, quantile))
                if not (np.isfinite(nnd_tgt) and nnd_tgt > 0):
                    continue
                p_keep = float(min(1.0, max(0.0, (nnd_src / nnd_tgt) ** 2)))

                cancer = group[group["CK"] == 1]
                stroma = group[group["CK"] == 0]
                if len(stroma) < min_cells:
                    continue

                coords = stroma[["x", "y"]].to_numpy(float, copy=False)
                keep = remove_isolated_points(coords, k=5, max_percentile=99.0)
                stroma_filtered = stroma.loc[keep]
                if len(stroma_filtered) < min_cells:
                    continue
                coords_filtered = stroma_filtered[["x", "y"]].to_numpy(float, copy=False)
                keep_mask = _binary_search_keep_mask(
                    coords_filtered,
                    target_nnd=nnd_tgt,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                if keep_mask is None:
                    continue
                stroma_sub = stroma_filtered.iloc[keep_mask].copy()
                matched = pd.concat([cancer, stroma_sub], axis=0)
            else:
                cancer = group[group["CK"] == 1]
                stroma = group[group["CK"] == 0]
                if len(cancer) < min_cells or len(stroma) < min_cells:
                    continue

                if sample_name not in src_stats_c.index or sample_name not in src_stats_s.index:
                    continue

                nnd_src_c = float(src_stats_c.loc[sample_name, "median_nnd_all"])
                nnd_src_s = float(src_stats_s.loc[sample_name, "median_nnd_stroma"])
                if not (np.isfinite(nnd_src_c) and nnd_src_c > 0 and np.isfinite(nnd_src_s) and nnd_src_s > 0):
                    continue

                rank_c = np.searchsorted(src_c_sorted, nnd_src_c, side="left")
                rank_s = np.searchsorted(src_s_sorted, nnd_src_s, side="left")
                quantile_c = (rank_c + 0.5) / len(src_c_sorted)
                quantile_s = (rank_s + 0.5) / len(src_s_sorted)
                nnd_tgt_c = float(np.quantile(ref_c_vals, quantile_c))
                nnd_tgt_s = float(np.quantile(ref_s_vals, quantile_s))
                if not (np.isfinite(nnd_tgt_c) and nnd_tgt_c > 0 and np.isfinite(nnd_tgt_s) and nnd_tgt_s > 0):
                    continue

                p_keep_c = float(min(1.0, max(0.0, (nnd_src_c / nnd_tgt_c) ** 2)))
                p_keep_s = float(min(1.0, max(0.0, (nnd_src_s / nnd_tgt_s) ** 2)))

                coords_c = cancer[["x", "y"]].to_numpy(float, copy=False)
                coords_s = stroma[["x", "y"]].to_numpy(float, copy=False)
                keep_c0 = remove_isolated_points(coords_c, k=5, max_percentile=99.0)
                keep_s0 = remove_isolated_points(coords_s, k=5, max_percentile=99.0)
                cancer_filtered = cancer.loc[keep_c0]
                stroma_filtered = stroma.loc[keep_s0]
                if len(cancer_filtered) < min_cells or len(stroma_filtered) < min_cells:
                    continue

                keep_c = _binary_search_keep_mask(
                    cancer_filtered[["x", "y"]].to_numpy(float, copy=False),
                    target_nnd=nnd_tgt_c,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep_c,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                keep_s = _binary_search_keep_mask(
                    stroma_filtered[["x", "y"]].to_numpy(float, copy=False),
                    target_nnd=nnd_tgt_s,
                    min_cells=min_cells,
                    rng=rep_rng,
                    p_init=p_keep_s,
                    n_binsrch=n_binsrch,
                    tol_rel=tol_rel,
                )
                if keep_c is None or keep_s is None:
                    continue

                cancer_sub = cancer_filtered.iloc[keep_c].copy()
                stroma_sub = stroma_filtered.iloc[keep_s].copy()
                matched = pd.concat([cancer_sub, stroma_sub], axis=0)

            suffix = f"__thin{rep}"
            matched["orig_sample_name"] = sample_name
            matched["rep"] = rep
            matched["sample_name_new"] = insert_suffix_in_canonical_sample_name(sample_name, suffix)
            out_parts.append(matched)

    if not out_parts:
        raise RuntimeError("Matching produced no cores. Check the selected compartment and thresholds.")

    return pd.concat(out_parts, axis=0)


def load_internal_cells(path: Path, method: str, keep_all_columns: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "cellpose":
        if keep_all_columns:
            raw = pd.read_csv(path, low_memory=False)
        else:
            needed = ["filename", "x", "y", "ck", "ck_cyto_mean_raw"]
            head = pd.read_csv(path, nrows=1)
            missing = [col for col in needed if col not in head.columns]
            if missing:
                raise ValueError(f"Cellpose cells file missing required columns {missing}. Found: {list(head.columns)}")
            raw = pd.read_csv(path, usecols=needed, low_memory=False)

        minimal = raw.copy()
        minimal["x"] = pd.to_numeric(minimal["x"], errors="coerce")
        minimal["y"] = pd.to_numeric(minimal["y"], errors="coerce")
        minimal["CK"] = pd.to_numeric(minimal["ck"], errors="coerce").fillna(0).astype(int)
        minimal["sample_name"] = minimal["filename"].map(extract_sample_name_from_filename)
        minimal = minimal.dropna(subset=["x", "y", "sample_name"])
        return raw, minimal[["sample_name", "x", "y", "CK"]]

    if method == "inform":
        raw = pd.read_csv(path, low_memory=False)
        required = ["Sample Name", "Cell X Position", "Cell Y Position", "Cancer"]
        missing = [col for col in required if col not in raw.columns]
        if missing:
            raise ValueError(f"inForm cells file missing required columns {missing}. Found: {list(raw.columns)}")

        minimal = raw.copy()
        minimal["sample_name"] = minimal["Sample Name"].astype(str)
        minimal["x"] = pd.to_numeric(minimal["Cell X Position"], errors="coerce")
        minimal["y"] = pd.to_numeric(minimal["Cell Y Position"], errors="coerce")
        minimal["CK"] = pd.to_numeric(minimal["Cancer"], errors="coerce").fillna(0).astype(int)
        minimal = minimal.dropna(subset=["x", "y", "sample_name"])
        return raw, minimal[["sample_name", "x", "y", "CK"]]

    raise ValueError(f"Unsupported internal method: {method}")


def load_allowed_internal_sample_names(samples_path: Path) -> set[str]:
    samples_df = pd.read_csv(samples_path, low_memory=False)
    if "sample_name" not in samples_df.columns:
        raise ValueError(f"Expected 'sample_name' column in {samples_path}, got: {list(samples_df.columns)}")
    return set(samples_df["sample_name"].astype(str).map(canonicalize_samples_csv_name))


def filter_internal_cells_to_samples(
    df_internal_raw: pd.DataFrame,
    df_internal: pd.DataFrame,
    *,
    allowed_sample_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keep = df_internal["sample_name"].astype(str).isin(allowed_sample_names)
    df_internal_f = df_internal.loc[keep].copy()
    df_internal_raw_f = df_internal_raw.loc[df_internal_f.index].copy()
    return df_internal_raw_f, df_internal_f


def apply_suffix_to_internal_rows(df_out_cells: pd.DataFrame, matched_df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df_out_cells.copy()
    suffixes = np.array([f"__thin{rep}" for rep in matched_df["rep"].to_numpy(int)], dtype=object)

    if method == "cellpose":
        new_values = []
        old_values = out["filename"].astype(str).to_numpy()
        for value, suffix in tqdm(zip(old_values, suffixes), total=len(old_values), desc="Renaming filenames"):
            new_values.append(insert_suffix_before_core(value, suffix))
        out["filename"] = new_values
    elif method == "inform":
        old_values = matched_df["sample_name_new"].astype(str).to_numpy()
        out["Sample Name"] = old_values
    else:
        raise ValueError(f"Unsupported internal method: {method}")

    out["orig_sample_name"] = matched_df["orig_sample_name"].values
    out["sample_name_new"] = matched_df["sample_name_new"].values
    out["thin_rep"] = matched_df["rep"].values
    return out


def build_output_samples(
    samples_path: Path,
    matched_df: pd.DataFrame,
    *,
    n_subsamples: int,
    out_path: Path,
) -> None:
    samples_df = pd.read_csv(samples_path, low_memory=False)
    if "sample_name" not in samples_df.columns:
        raise ValueError(f"Expected 'sample_name' column in {samples_path}, got: {list(samples_df.columns)}")

    parts = []
    for rep in range(1, n_subsamples + 1):
        suffix = f"__thin{rep}"
        dup = samples_df.copy()
        dup["sample_name"] = dup["sample_name"].astype(str).map(
            lambda value: insert_suffix_before_core(value, suffix) if "_Core[1," in str(value) else value
        )
        dup["thin_rep"] = rep
        dup["orig_sample_name"] = samples_df["sample_name"].astype(str).values
        parts.append(dup)

    samples_out = pd.concat(parts, ignore_index=True)
    present = set(matched_df["sample_name_new"].astype(str).unique())
    canon = samples_out["sample_name"].astype(str).map(canonicalize_samples_csv_name)
    samples_out = samples_out.loc[canon.isin(present)].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples_out.to_csv(out_path, index=True)


def _plot_log_histogram(values: np.ndarray, title: str, path: Path) -> None:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return
    plt.figure()
    plt.hist(np.log10(x), bins=60)
    plt.xlabel("log10(median NND)")
    plt.ylabel("cores")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_overlay(series_list: list[pd.Series], labels: list[str], title: str, path: Path) -> None:
    arrays = [np.log10(s.dropna().to_numpy(float)) for s in series_list]
    combined = np.concatenate(arrays)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return

    bins = np.linspace(combined.min(), combined.max(), 80)
    plt.figure()
    for arr, label in zip(arrays, labels):
        plt.hist(arr[np.isfinite(arr)], bins=bins, alpha=0.5, label=label)
    plt.xlabel("log10(median NND)")
    plt.ylabel("cores")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_qc_histograms(
    df_external: pd.DataFrame,
    df_internal: pd.DataFrame,
    df_matched: pd.DataFrame,
    *,
    compartment: str,
    outdir: Path,
    internal_label: str,
    min_cells: int,
    min_cells_cancer_qc: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    tmp = df_matched[["sample_name_new", "x", "y", "CK"]].rename(columns={"sample_name_new": "sample_name"}).copy()

    if compartment == "all":
        ext_stats = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="all")
        int_stats = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="all")
        match_stats = compute_sample_median_nnd(tmp, min_cells=min_cells, compartment="all")

        _plot_log_histogram(ext_stats["median_nnd_all"].values, "External all cells: per-core median NND", outdir / "external_all_median_nnd_log10.png")
        _plot_log_histogram(int_stats["median_nnd_all"].values, f"{internal_label} original all cells: per-core median NND", outdir / "internal_orig_all_median_nnd_log10.png")
        _plot_log_histogram(match_stats["median_nnd_all"].values, f"{internal_label} matched all cells: per-core median NND", outdir / "internal_matched_all_median_nnd_log10.png")
        _plot_overlay(
            [ext_stats["median_nnd_all"], int_stats["median_nnd_all"], match_stats["median_nnd_all"]],
            ["external", f"{internal_label} orig", f"{internal_label} matched"],
            "All-cell per-core median NND distributions",
            outdir / "all_median_nnd_log10_overlay.png",
        )
        return

    ext_stroma = compute_sample_median_nnd(df_external, min_cells=min_cells, compartment="stroma")
    int_stroma = compute_sample_median_nnd(df_internal, min_cells=min_cells, compartment="stroma")
    match_stroma = compute_sample_median_nnd(tmp, min_cells=min_cells, compartment="stroma")

    _plot_log_histogram(ext_stroma["median_nnd_stroma"].values, "External stroma: per-core median NND", outdir / "external_stroma_median_nnd_log10.png")
    _plot_log_histogram(int_stroma["median_nnd_stroma"].values, f"{internal_label} original stroma: per-core median NND", outdir / "internal_orig_stroma_median_nnd_log10.png")
    _plot_log_histogram(match_stroma["median_nnd_stroma"].values, f"{internal_label} matched stroma: per-core median NND", outdir / "internal_matched_stroma_median_nnd_log10.png")
    _plot_overlay(
        [ext_stroma["median_nnd_stroma"], int_stroma["median_nnd_stroma"], match_stroma["median_nnd_stroma"]],
        ["external stroma", f"{internal_label} stroma orig", f"{internal_label} stroma matched"],
        "Stroma per-core median NND distributions",
        outdir / "stroma_median_nnd_log10_overlay.png",
    )

    ext_cancer = compute_sample_median_nnd(df_external[df_external["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
    int_cancer = compute_sample_median_nnd(df_internal[df_internal["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
    match_cancer = compute_sample_median_nnd(tmp[tmp["CK"] == 1].assign(CK=1), min_cells=min_cells_cancer_qc, compartment="all")
    _plot_overlay(
        [ext_cancer["median_nnd_all"], int_cancer["median_nnd_all"], match_cancer["median_nnd_all"]],
        ["external cancer", f"{internal_label} cancer orig", f"{internal_label} cancer matched"],
        "Cancer per-core median NND distributions",
        outdir / "cancer_median_nnd_log10_overlay.png",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-method", choices=["cellpose", "inform"], default="cellpose")
    parser.add_argument("--match-compartment", choices=["all", "stroma", "separate"], default="all")
    parser.add_argument("--internal-cells", type=Path)
    parser.add_argument("--internal-samples", type=Path)
    parser.add_argument("--cellpose-cells", dest="internal_cells_legacy", type=Path)
    parser.add_argument("--cellpose-samples", dest="internal_samples_legacy", type=Path)
    parser.add_argument("--external-cells", type=Path, required=True)
    parser.add_argument("--external-xy-scale", type=float, default=2.0)
    parser.add_argument("--n-subsamples", type=int, default=3)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--min-cells-cancer-qc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-binsrch", type=int, default=8)
    parser.add_argument("--tol-rel", type=float, default=0.03)
    parser.add_argument("--out-cells", type=Path, required=True)
    parser.add_argument("--out-samples", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, default=Path("outputs/qc/qc_matched_internal"))
    parser.add_argument("--keep-all-columns", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    internal_cells_path = args.internal_cells or args.internal_cells_legacy
    internal_samples_path = args.internal_samples or args.internal_samples_legacy

    if internal_cells_path is None or internal_samples_path is None:
        raise ValueError("Provide --internal-cells and --internal-samples.")
    if args.n_subsamples < 1:
        raise ValueError("--n-subsamples must be >= 1")

    print("[INFO] Loading external cells...")
    ext_raw = pd.read_csv(args.external_cells, low_memory=False)
    df_external = preprocess_external_cells(ext_raw).rename(columns={"Cancer": "CK"})
    df_external["x"] = pd.to_numeric(df_external["x"], errors="coerce") * float(args.external_xy_scale)
    df_external["y"] = pd.to_numeric(df_external["y"], errors="coerce") * float(args.external_xy_scale)
    df_external["CK"] = pd.to_numeric(df_external["CK"], errors="coerce").fillna(0).astype(int)
    df_external = df_external.dropna(subset=["x", "y", "sample_name"])
    print(f"[INFO] External cells: {len(df_external):,} rows, {df_external['sample_name'].nunique():,} samples")

    print(f"[INFO] Loading internal cells ({args.internal_method})...")
    df_internal_raw, df_internal = load_internal_cells(internal_cells_path, args.internal_method, args.keep_all_columns)
    allowed_sample_names = load_allowed_internal_sample_names(internal_samples_path)
    df_internal_raw, df_internal = filter_internal_cells_to_samples(
        df_internal_raw,
        df_internal,
        allowed_sample_names=allowed_sample_names,
    )
    print(f"[INFO] Internal cells: {len(df_internal):,} rows, {df_internal['sample_name'].nunique():,} samples")

    print(f"[INFO] Matching internal {args.match_compartment} NND distribution to external...")
    df_matched = thin_internal_to_match_external_nnd(
        df_internal,
        df_external,
        compartment=args.match_compartment,
        n_subsamples=args.n_subsamples,
        min_cells=args.min_cells,
        seed=args.seed,
        n_binsrch=args.n_binsrch,
        tol_rel=args.tol_rel,
    )
    print(f"[INFO] Matched selection: {len(df_matched):,} cells across {df_matched['sample_name_new'].nunique():,} subsampled cores")

    print("[INFO] Building output cells CSV...")
    df_out_cells = df_internal_raw.loc[df_matched.index].copy()
    df_out_cells = apply_suffix_to_internal_rows(df_out_cells, df_matched, args.internal_method)
    args.out_cells.parent.mkdir(parents=True, exist_ok=True)
    df_out_cells.to_csv(args.out_cells, index=False)
    print(f"[INFO] Wrote: {args.out_cells} ({len(df_out_cells):,} rows)")

    print("[INFO] Building output samples CSV...")
    build_output_samples(
        internal_samples_path,
        df_matched,
        n_subsamples=args.n_subsamples,
        out_path=args.out_samples,
    )
    print(f"[INFO] Wrote: {args.out_samples}")

    print("[INFO] Writing QC plots...")
    plot_qc_histograms(
        df_external,
        df_internal,
        df_matched,
        compartment=args.match_compartment,
        outdir=args.qc_dir,
        internal_label=f"{args.internal_method}_BOMI2",
        min_cells=args.min_cells,
        min_cells_cancer_qc=args.min_cells_cancer_qc,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
