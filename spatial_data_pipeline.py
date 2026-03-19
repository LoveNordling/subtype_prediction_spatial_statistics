from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from spatial_artifact_utils import read_split_ids, subset_by_ids
from spatial_utils import (
    CountQC,
    aggregate_cores_to_patient,
    attach_counts_from_cells,
    filter_metrics_by_counts,
    preprocess_cellpose_data,
    preprocess_cellprofiler_data,
    preprocess_cells_inform,
    preprocess_external_cells,
    preprocess_patients,
    preprocess_samples,
    split_external_meta,
)


def _get_spatial_metric_functions():
    # Import lazily so orchestration code can be imported without immediately
    # pulling in squidpy/datashader-dependent metric backends.
    from spatial_metrics import (
        calculate_bidirectional_min_distance,
        calculate_centrality_scores,
        calculate_cluster_cooccurrence_ratio,
        calculate_neighborhood_enrichment_test,
        calculate_newmans_assortativity,
        calculate_objectobject_correlation,
        calculate_ripley_l,
    )

    return {
        "calculate_bidirectional_min_distance": calculate_bidirectional_min_distance,
        "calculate_centrality_scores": calculate_centrality_scores,
        "calculate_cluster_cooccurrence_ratio": calculate_cluster_cooccurrence_ratio,
        "calculate_neighborhood_enrichment_test": calculate_neighborhood_enrichment_test,
        "calculate_newmans_assortativity": calculate_newmans_assortativity,
        "calculate_objectobject_correlation": calculate_objectobject_correlation,
        "calculate_ripley_l": calculate_ripley_l,
    }


@dataclass(frozen=True)
class InternalPaths:
    inform_cells_path: str = "data/raw/BOMI2_all_cells_TIL.csv"
    inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"

    #inform_cells_path: str = "BOMI2_all_cells_TIL__matchedBOMI1.csv"
    #inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__inform_matchedBOMI1.csv"
    #inform_cells_path: str = "BOMI2_all_cells_TIL__matchedBOMI1_stroma.csv"
    #inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__inform_matchedBOMI1_stroma.csv"
    #inform_cells_path: str = "BOMI2_all_cells_TIL__matchedBOMI1_separate.csv"
    #inform_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__inform_matchedBOMI1_separate.csv"

    #cellpose_cells_path: str = "./cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1.csv"
    #cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1.csv"
    cellpose_cells_path: str = "data/interim/cellpose_extracted_cells_fitlered_necrosis__matchedBOMI1_separate.csv"
    cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1_separate.csv"
    #cellpose_cells_path: str = "cellpose_extracted_cells_fitlered_necrosis__matchedstromaBOMI1.csv"
    #cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples__cellpose_matchedBOMI1stroma.csv"
    #cellpose_cells_path: str = "cellpose_extracted_cells_fitlered_necrosis.csv"
    #cellpose_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"

    cellprofiler_cells_path: str = "data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv"
    cellprofiler_samples_path: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"

    internal_train_csv: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/train_val.csv"
    internal_test_csv: str = "../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/test.csv"


@dataclass(frozen=True)
class ExternalPaths:
    external_cells_path: str = "data/raw/BOMI1_cells_all.csv"
    external_meta_path: str = "data/reference/BOMI1_clinical_data_LUADvsSqCC.csv"
    external_split_dir: str = "/home/love/multiplex_dataset/lung_cancer_BOMI1_dataset/HE_dataset/binary_subtype_prediction_ACvsSqCC/static_split/"
    external_train_csv: Optional[str] = None
    external_test_csv: Optional[str] = None


def calculate_spatial_metrics_for_sample(cells_df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if cells_df is None or len(cells_df) < 10:
        return None

    coords = cells_df[["x", "y"]].values
    cell_types = cells_df["Cancer"].values.astype(int)
    metric_fns = _get_spatial_metric_functions()

    metrics: Dict[str, float] = {}
    metrics["cell_count"] = int(len(cells_df))
    metrics["cancer_count"] = int((cell_types == 1).sum())
    metrics["stroma_count"] = int((cell_types == 0).sum())

    metrics.update(metric_fns["calculate_ripley_l"](coords, cell_types))
    metrics.update(metric_fns["calculate_bidirectional_min_distance"](coords, cell_types))
    metrics["newmans_assortativity"] = float(
        metric_fns["calculate_newmans_assortativity"](coords, cell_types, radius=50)
    )
    metrics.update(metric_fns["calculate_centrality_scores"](coords, cell_types))
    metrics.update(metric_fns["calculate_cluster_cooccurrence_ratio"](coords, cell_types))
    metrics.update(metric_fns["calculate_neighborhood_enrichment_test"](coords, cell_types))
    metrics.update(metric_fns["calculate_objectobject_correlation"](coords, cell_types))
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
            res = f.result()
            if res is not None:
                results.append(res)

    return pd.DataFrame(results)


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

    return attach_counts_from_cells(m, cells_df)


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
    ext_train_csv = external_paths.external_train_csv or os.path.join(external_paths.external_split_dir, "train_val.csv")
    ext_test_csv = external_paths.external_test_csv or os.path.join(external_paths.external_split_dir, "test.csv")

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
    df_train_int = subset_by_ids(int_patient_df, train_ids_int)
    df_test_int = subset_by_ids(int_patient_df, test_ids_int)
    print(f"[INFO] INTERNAL patients: train={len(df_train_int)} test={len(df_test_int)} (after core QC + aggregation)")

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
    ext_train_ids = read_split_ids(ext_train_csv)
    ext_test_ids = read_split_ids(ext_test_csv)
    df_train_ext = subset_by_ids(ext_patient_df, ext_train_ids)
    df_test_ext = subset_by_ids(ext_patient_df, ext_test_ids)
    print(f"[INFO] EXTERNAL patients: train={len(df_train_ext)} test={len(df_test_ext)} (after core QC + aggregation)")

    if df_train_int.empty or df_test_int.empty or df_train_ext.empty or df_test_ext.empty:
        raise RuntimeError("One of the train/test splits is empty after QC. Adjust CountQC thresholds.")

    return df_train_int, df_test_int, df_train_ext, df_test_ext
