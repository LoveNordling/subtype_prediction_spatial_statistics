import pandas as pd
import pytest


@pytest.fixture
def spatial_utils_module(load_script_module):
    return load_script_module(
        "spatial_utils.py",
        "spatial_utils_under_test",
    )


@pytest.fixture
def spatial_data_pipeline_module(load_script_module):
    return load_script_module(
        "spatial_data_pipeline.py",
        "spatial_data_pipeline_under_test",
    )


def test_preprocess_samples_and_patients(spatial_utils_module):
    samples = pd.DataFrame({"sample_name": ["BOMI2_TIL_1_Core[1,1,A]_1", "BOMI2_TIL_1_Core[1,2,B]_2"]})
    processed_samples = spatial_utils_module.preprocess_samples(samples)
    assert processed_samples["sample_name"].tolist() == ["BOMI2_TIL_1_[1,A]", "BOMI2_TIL_1_[2,B]"]

    patients = pd.DataFrame(
        {
            "ID or PAD_year": ["p1", "p2"],
            "Tumor_type": ["Adenocarcinoma", "Squamous cell carcinoma"],
            "Sex": ["Male", "Female"],
            "Smoking": ["Never-smoker", "Former smoker"],
            "Stage (7th ed.)": ["Ia", "IIIa"],
            "Age": [70, 65],
            "Performance status (WHO)": [0, 1],
            "Follow-up (days)": [1200, 900],
            "label": [1, 0],
        }
    )

    processed_patients = spatial_utils_module.preprocess_patients(patients)

    assert processed_patients["ID"].tolist() == ["p1", "p2"]
    assert processed_patients["Tumor_type_code"].tolist() == ["LUAD", "LUSC"]
    assert processed_patients["LUAD"].tolist() == [1, 0]
    assert processed_patients["Gender"].tolist() == [1, 0]
    assert processed_patients["Smoking"].tolist() == [0, 1]
    assert processed_patients["Stage"].tolist() == [0, 4]


def test_preprocess_cell_tables_and_external_meta(spatial_utils_module):
    inform = pd.DataFrame(
        {
            "Sample Name": ["BOMI2_TIL_1_[1,A]", "BOMI2_TIL_1_[1,A]"],
            "Cell X Position": ["10", "11"],
            "Cell Y Position": ["20", "21"],
            "CK": [1, 0],
            "Cancer": [99, 99],
        }
    )
    inform_processed = spatial_utils_module.preprocess_cells_inform(inform)
    assert list(inform_processed.columns) == ["x", "y", "Cancer", "sample_name"]
    assert inform_processed["Cancer"].tolist() == [1, 0]

    cellpose = pd.DataFrame(
        {
            "filename": ["BOMI2_TIL_1_Core[1,10,D]_[10854,50870]_component_data.tif"],
            "x": ["1.5"],
            "y": ["2.5"],
            "ck": [1],
        }
    )
    cellpose_processed = spatial_utils_module.preprocess_cellpose_data(cellpose)
    assert cellpose_processed.iloc[0].to_dict() == {
        "x": 1.5,
        "y": 2.5,
        "Cancer": 1,
        "sample_name": "BOMI2_TIL_1_[10,D]",
    }

    cellprofiler = pd.DataFrame(
        {
            "FileName_CK": ["BOMI2_TIL_1_Core[1,10,D]_[10854,50870]_component_data_CK.tiff"],
            "Location_Center_X": ["3.5"],
            "Location_Center_Y": ["4.5"],
            "CK": [0],
        }
    )
    cellprofiler_processed = spatial_utils_module.preprocess_cellprofiler_data(cellprofiler)
    assert cellprofiler_processed.iloc[0].to_dict() == {
        "x": 3.5,
        "y": 4.5,
        "Cancer": 0,
        "sample_name": "BOMI2_TIL_1_[10,D]",
    }

    external = pd.DataFrame(
        {
            "CentroidX_um": [100.0, 110.0],
            "CentroidY_um": [200.0, 210.0],
            "Class": ["neoplastic", "stroma"],
            "sample_name": ["ULunA1_A-1", "ULunA1_A-1"],
        }
    )
    external_processed = spatial_utils_module.preprocess_external_cells(external)
    assert external_processed["Cancer"].tolist() == [1, 0]

    meta = pd.DataFrame(
        {
            "ID": [257, 257, 300],
            "sample_name": ["ULunA1_A-1", "ULunA1_A-2", "ULunA2_B-1"],
            "label": [1, 1, 0],
        }
    )
    patients_df, samples_df = spatial_utils_module.split_external_meta(meta)
    assert patients_df.to_dict(orient="records") == [{"ID": 257, "label": 1}, {"ID": 300, "label": 0}]
    assert samples_df["sample_name"].tolist() == ["ULunA1_A-1", "ULunA1_A-2", "ULunA2_B-1"]


def test_attach_counts_filter_and_aggregate(spatial_utils_module):
    metrics = pd.DataFrame(
        {
            "ID": ["p1", "p1", "p2"],
            "label": [1, 1, 0],
            "sample_name": ["s1", "s2", "s3"],
            "cell_count": [None, 30, 8],
            "feat_a": [1.0, 3.0, 4.0],
        }
    )
    cells = pd.DataFrame(
        {
            "sample_name": ["s1"] * 3 + ["s2"] * 4 + ["s3"] * 2,
            "Cancer": [1, 0, 0, 1, 1, 0, 0, 1, 0],
        }
    )

    with_counts = spatial_utils_module.attach_counts_from_cells(metrics, cells)
    assert with_counts[["sample_name", "cell_count", "cancer_count", "stroma_count"]].to_dict(orient="records") == [
        {"sample_name": "s1", "cell_count": 3.0, "cancer_count": 1, "stroma_count": 2},
        {"sample_name": "s2", "cell_count": 30.0, "cancer_count": 2, "stroma_count": 2},
        {"sample_name": "s3", "cell_count": 8.0, "cancer_count": 1, "stroma_count": 1},
    ]

    qc = spatial_utils_module.CountQC(min_total=3, min_cancer=1, min_stroma=1)
    filtered = spatial_utils_module.filter_metrics_by_counts(with_counts, qc, "test")
    assert filtered["sample_name"].tolist() == ["s1", "s2", "s3"]

    strict_qc = spatial_utils_module.CountQC(min_total=4, min_cancer=2, min_stroma=2)
    strict_filtered = spatial_utils_module.filter_metrics_by_counts(with_counts, strict_qc, "test")
    assert strict_filtered["sample_name"].tolist() == ["s2"]

    aggregated = spatial_utils_module.aggregate_cores_to_patient(filtered.iloc[:2].copy())
    assert aggregated.to_dict(orient="records") == [
        {
            "ID": "p1",
            "label": 1,
            "cell_count": 33,
            "cancer_count": 3,
            "stroma_count": 4,
            "feat_a": pytest.approx((1.0 * 3 + 3.0 * 30) / 33),
        }
    ]


def test_spatial_data_pipeline_sorts_metrics_and_patients_deterministically(spatial_data_pipeline_module):
    per_core = pd.DataFrame(
        {
            "sample_name": ["s2", "s1", "s3"],
            "ID": ["p2", "p1", "p1"],
            "label": [0, 1, 1],
            "cell_count": [20, 10, 30],
            "feat_a": [2.0, 1.0, 3.0],
        }
    )

    sorted_per_core = spatial_data_pipeline_module._sort_metrics_rows(per_core)
    assert sorted_per_core[["ID", "sample_name"]].to_dict(orient="records") == [
        {"ID": "p1", "sample_name": "s1"},
        {"ID": "p1", "sample_name": "s3"},
        {"ID": "p2", "sample_name": "s2"},
    ]

    patient_df = pd.DataFrame(
        {
            "ID": ["p2", "p1"],
            "label": [0, 1],
            "feat_a": [2.0, 3.0],
        }
    )

    sorted_patients = spatial_data_pipeline_module._sort_patient_rows(patient_df)
    assert sorted_patients["ID"].tolist() == ["p1", "p2"]
