from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CountQC:
    min_total: int = 150
    min_cancer: int = 30
    min_stroma: int = 30


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Ensure df has all cols (add missing as NaN), return copy."""
    out = df.copy()
    missing = [c for c in cols if c not in out.columns]
    for c in missing:
        out[c] = np.nan
    return out


def preprocess_samples(samples_df: pd.DataFrame) -> pd.DataFrame:
    df = samples_df.copy()
    df["sample_name"] = df["sample_name"].astype(str)
    df["sample_name"] = df["sample_name"].map(lambda x: x[: x.rfind("_")] if "_" in x else x)
    df["sample_name"] = df["sample_name"].map(lambda x: x.replace("Core[1,", "["))
    return df


def preprocess_patients(patients_df: pd.DataFrame) -> pd.DataFrame:
    df = patients_df.copy()
    df = df.rename(
        columns={
            "ID or PAD_year": "ID",
            "Tumor_type": "Histology",
            "Sex": "Gender",
            "Event_last_followup": "Dead/Alive",
        }
    )

    if "Histology" in df.columns:
        histology_map = {"Adenocarcinoma": "LUAD", "Squamous cell carcinoma": "LUSC", "Other": "Other"}
        df["Tumor_type_code"] = df["Histology"].map(lambda x: histology_map.get(x, str(x)))
        df["LUAD"] = (df["Tumor_type_code"] == "LUAD").astype(int)

    if "Smoking" in df.columns:
        df["Smoking"] = df["Smoking"].map(lambda x: 0 if x == "Never-smoker" else 1)

    if "Gender" in df.columns:
        df["Gender"] = (df["Gender"] == "Male").astype(int)

    stage_dict = {"Ia": 0, "Ib": 1, "IIa": 2, "IIb": 3, "IIIa": 4, "IIIb": 5, "IV": 6}
    if "Stage (7th ed.)" in df.columns:
        df["Stage"] = df["Stage (7th ed.)"].map(lambda x: stage_dict.get(x, np.nan))

    keep_cols = [
        "ID",
        "LUAD",
        "Age",
        "Gender",
        "Smoking",
        "Stage",
        "Performance status (WHO)",
        "Follow-up (days)",
        "label",
        "Tumor_type_code",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    if "ID" not in keep_cols:
        raise ValueError(f"Patient table missing ID after renaming. Columns: {list(df.columns)}")
    if "label" not in keep_cols:
        raise ValueError(f"Patient table missing label. Columns: {list(df.columns)}")

    return df[keep_cols].copy()


def preprocess_cells_inform(cells_df: pd.DataFrame) -> pd.DataFrame:
    df = cells_df.copy()

    if "CK" not in df.columns:
        raise ValueError(
            "inForm cells file missing required 'CK' column. "
            "Run prepare_spatial_inputs.py to generate the masked, CK-thresholded input schema."
        )

    if "Cancer" in df.columns:
        df = df.drop(columns=["Cancer"])

    df.rename(
        columns={
            "Sample Name": "sample_name",
            "Cell X Position": "x",
            "Cell Y Position": "y",
            "CK": "Cancer",
        },
        inplace=True,
    )

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["Cancer"] = pd.to_numeric(df["Cancer"], errors="coerce").fillna(0).astype(int)
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def extract_sample_name_core(filename: str) -> str:
    match = re.match(r"(.+?)_Core\[1,(\d+,[A-Z])\]", str(filename))
    if match:
        return f"{match.group(1)}_[{match.group(2)}]"
    raise ValueError(f"Could not parse sample name from filename: {filename}")


def preprocess_cellpose_data(cells_df: pd.DataFrame) -> pd.DataFrame:
    df = cells_df.copy()
    df = df.rename(columns={"ck": "Cancer", "filename": "file_name"})
    df["sample_name"] = df["file_name"].apply(extract_sample_name_core)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["Cancer"] = pd.to_numeric(df["Cancer"], errors="coerce").fillna(0).astype(int)
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def preprocess_cellprofiler_data(cells_df: pd.DataFrame) -> pd.DataFrame:
    df = cells_df.copy()
    df = df.rename(
        columns={
            "Location_Center_X": "x",
            "Location_Center_Y": "y",
            "CK": "Cancer",
            "FileName_CK": "file_name",
        }
    )
    df["sample_name"] = df["file_name"].apply(extract_sample_name_core)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["Cancer"] = pd.to_numeric(df["Cancer"], errors="coerce").fillna(0).astype(int)
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def preprocess_external_cells(bomi1_cells: pd.DataFrame) -> pd.DataFrame:
    df = bomi1_cells.copy()
    needed = {"CentroidX_um", "CentroidY_um", "Class", "sample_name"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"External cells file missing columns: {missing}. Columns: {list(df.columns)}")

    df = df.rename(columns={"CentroidX_um": "x", "CentroidY_um": "y"})
    df["Cancer"] = (df["Class"].astype(str).str.strip().str.lower() == "neoplastic").astype(int)

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["sample_name"] = df["sample_name"].astype(str)

    df = df.dropna(subset=["x", "y", "Cancer", "sample_name"])
    return df[["x", "y", "Cancer", "sample_name"]].copy()


def split_external_meta(meta_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    needed = {"ID", "sample_name", "label"}
    missing = needed - set(meta_df.columns)
    if missing:
        raise ValueError(f"External meta file missing columns: {missing}. Columns: {list(meta_df.columns)}")

    patients_df = meta_df[["ID", "label"]].drop_duplicates(subset=["ID"]).copy()
    samples_df = meta_df[["ID", "sample_name"]].drop_duplicates(subset=["ID", "sample_name"]).copy()
    return patients_df, samples_df


def attach_counts_from_cells(metrics_df: pd.DataFrame, cells_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return metrics_df

    if "sample_name" not in metrics_df.columns:
        raise ValueError(f"metrics_df missing required column 'sample_name'. Columns: {list(metrics_df.columns)}")

    counts = (
        cells_df.groupby("sample_name", sort=False)["Cancer"]
        .agg(
            cell_count="size",
            cancer_count=lambda x: int((x.astype(int) == 1).sum()),
            stroma_count=lambda x: int((x.astype(int) == 0).sum()),
        )
        .reset_index()
    )

    out = metrics_df.merge(counts, on="sample_name", how="left", suffixes=("", "__from_cells"))

    for col in ["cell_count", "cancer_count", "stroma_count"]:
        from_cells = f"{col}__from_cells"
        if col in out.columns and from_cells in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[from_cells] = pd.to_numeric(out[from_cells], errors="coerce")
            out[col] = out[col].fillna(out[from_cells])
            out.drop(columns=[from_cells], inplace=True)
        elif from_cells in out.columns and col not in out.columns:
            out.rename(columns={from_cells: col}, inplace=True)

    return out


def filter_metrics_by_counts(metrics_df: pd.DataFrame, qc: CountQC, cohort_name: str) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return metrics_df

    for col in ["cell_count", "cancer_count", "stroma_count"]:
        if col not in metrics_df.columns:
            raise ValueError(f"{cohort_name}: missing '{col}' in metrics_df. Columns: {list(metrics_df.columns)}")

    df = metrics_df.copy()
    df["cell_count"] = pd.to_numeric(df["cell_count"], errors="coerce")
    df["cancer_count"] = pd.to_numeric(df["cancer_count"], errors="coerce")
    df["stroma_count"] = pd.to_numeric(df["stroma_count"], errors="coerce")

    keep = (
        (df["cell_count"] >= qc.min_total)
        & (df["cancer_count"] >= qc.min_cancer)
        & (df["stroma_count"] >= qc.min_stroma)
    )
    before = len(df)
    df = df.loc[keep].copy()
    after = len(df)
    print(
        f"[QC] {cohort_name}: kept {after}/{before} cores "
        f"(min_total={qc.min_total}, min_cancer={qc.min_cancer}, min_stroma={qc.min_stroma})"
    )
    return df


def aggregate_cores_to_patient(metrics_per_core: pd.DataFrame) -> pd.DataFrame:
    if metrics_per_core is None or metrics_per_core.empty:
        return pd.DataFrame()

    required = {"ID", "label", "cell_count"}
    missing = required - set(metrics_per_core.columns)
    if missing:
        raise ValueError(f"metrics_per_core missing {missing}. Columns: {list(metrics_per_core.columns)}")

    df = metrics_per_core.copy()
    exclude = {"ID", "label", "sample_name", "cell_count", "cancer_count", "stroma_count"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    def _agg_one(group: pd.DataFrame) -> pd.Series:
        weights = pd.to_numeric(group["cell_count"], errors="coerce").fillna(0).to_numpy(float)
        wsum = float(weights.sum()) if weights.size else 0.0

        out = {
            "label": int(group["label"].iloc[0]),
            "cell_count": int(pd.to_numeric(group["cell_count"], errors="coerce").fillna(0).sum()),
        }
        if "cancer_count" in group.columns:
            out["cancer_count"] = int(pd.to_numeric(group["cancer_count"], errors="coerce").fillna(0).sum())
        if "stroma_count" in group.columns:
            out["stroma_count"] = int(pd.to_numeric(group["stroma_count"], errors="coerce").fillna(0).sum())

        for col in feature_cols:
            x = pd.to_numeric(group[col], errors="coerce").to_numpy(float)
            if wsum > 0:
                out[col] = float(np.nansum(x * weights) / wsum)
            else:
                out[col] = float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
        return pd.Series(out)

    return df.groupby("ID", sort=False).apply(_agg_one).reset_index()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {"ID", "label", "cell_count", "cancer_count", "stroma_count"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]
