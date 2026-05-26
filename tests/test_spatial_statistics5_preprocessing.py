import numpy as np
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


@pytest.fixture
def spatial_statistics5_module(load_script_module):
    return load_script_module(
        "spatial_statistics5.py",
        "spatial_statistics5_under_test",
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
            "CentroidX_um": [100.0, 110.0, 120.0],
            "CentroidY_um": [200.0, 210.0, 220.0],
            "Class": ["neoplastic", "epithelial", "stroma"],
            "sample_name": ["ULunA1_A-1", "ULunA1_A-1", "ULunA1_A-1"],
        }
    )
    external_processed = spatial_utils_module.preprocess_external_cells(external)
    assert external_processed["Cancer"].tolist() == [1, 1, 0]

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


def test_feature_alignment_coral_handles_colinear_features(spatial_statistics5_module):
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(120, 2))
    source = np.column_stack(
        [
            latent[:, 0],
            2.0 * latent[:, 0] + 0.01 * rng.normal(size=120),
            latent[:, 1],
            latent[:, 0] + latent[:, 1] + 0.01 * rng.normal(size=120),
        ]
    )

    mixing = np.array(
        [
            [1.2, 0.3, 0.0, 0.4],
            [0.1, 0.8, 0.5, 0.2],
            [0.3, 0.0, 1.1, 0.2],
            [0.2, 0.1, 0.4, 1.0],
        ]
    )
    shift = np.array([1.5, -0.8, 0.7, 2.0])
    target = source @ mixing + shift + 0.02 * rng.normal(size=source.shape)

    feat_cols = ["f1", "f2", "f3", "f4"]
    source_df = pd.DataFrame(source, columns=feat_cols)
    target_df = pd.DataFrame(target, columns=feat_cols)

    alignment = spatial_statistics5_module.fit_feature_alignment(
        source_df,
        target_df,
        feat_cols,
        method="coral",
        coral_shrinkage=0.2,
        coral_eig_floor=1e-5,
    )
    aligned = spatial_statistics5_module.apply_feature_alignment(
        source_df,
        feat_cols,
        alignment,
        method="coral",
    )

    source_cov = np.cov(source_df[feat_cols].to_numpy(float), rowvar=False)
    target_cov = np.cov(target_df[feat_cols].to_numpy(float), rowvar=False)
    aligned_cov = np.cov(aligned[feat_cols].to_numpy(float), rowvar=False)

    before_mean_gap = np.linalg.norm(source_df[feat_cols].mean().to_numpy() - target_df[feat_cols].mean().to_numpy())
    after_mean_gap = np.linalg.norm(aligned[feat_cols].mean().to_numpy() - target_df[feat_cols].mean().to_numpy())
    before_cov_gap = np.linalg.norm(source_cov - target_cov, ord="fro")
    after_cov_gap = np.linalg.norm(aligned_cov - target_cov, ord="fro")

    assert np.isfinite(aligned[feat_cols].to_numpy()).all()
    assert after_mean_gap < before_mean_gap * 0.05
    assert after_cov_gap < before_cov_gap * 0.6


def test_feature_alignment_quantile_matches_marginals(spatial_statistics5_module):
    rng = np.random.default_rng(11)
    feat_cols = ["f1", "f2"]
    source = pd.DataFrame(
        {
            "f1": rng.normal(loc=0.0, scale=1.0, size=400),
            "f2": rng.exponential(scale=1.0, size=400),
        }
    )
    target = pd.DataFrame(
        {
            "f1": rng.normal(loc=2.5, scale=0.6, size=400),
            "f2": rng.gamma(shape=2.0, scale=1.5, size=400),
        }
    )

    alignment = spatial_statistics5_module.fit_feature_alignment(
        source,
        target,
        feat_cols,
        method="quantile",
        quantile_n_quantiles=101,
    )
    aligned = spatial_statistics5_module.apply_feature_alignment(
        source,
        feat_cols,
        alignment,
        method="quantile",
    )

    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    before_gap = 0.0
    after_gap = 0.0
    for feat in feat_cols:
        src_q = np.quantile(source[feat].to_numpy(float), quantiles)
        aligned_q = np.quantile(aligned[feat].to_numpy(float), quantiles)
        tgt_q = np.quantile(target[feat].to_numpy(float), quantiles)
        before_gap += float(np.mean(np.abs(src_q - tgt_q)))
        after_gap += float(np.mean(np.abs(aligned_q - tgt_q)))

    assert np.isfinite(aligned[feat_cols].to_numpy()).all()
    assert after_gap < before_gap * 0.2


def test_repeat_test_seed_helpers_return_summary(spatial_statistics5_module):
    mod = spatial_statistics5_module

    train_df = pd.DataFrame(
        {
            "ID": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "label": [0, 0, 0, 1, 1, 1],
            "cell_count": [10, 10, 10, 10, 10, 10],
            "cancer_count": [5, 5, 5, 5, 5, 5],
            "stroma_count": [5, 5, 5, 5, 5, 5],
            "feat_a": [0.0, 0.2, 0.3, 1.0, 1.1, 1.2],
            "feat_b": [0.1, 0.0, 0.2, 0.9, 1.0, 1.1],
        }
    )
    test_df = pd.DataFrame(
        {
            "ID": ["t1", "t2", "t3", "t4"],
            "label": [0, 0, 1, 1],
            "cell_count": [10, 10, 10, 10],
            "cancer_count": [5, 5, 5, 5],
            "stroma_count": [5, 5, 5, 5],
            "feat_a": [0.15, 0.25, 1.05, 1.15],
            "feat_b": [0.05, 0.18, 0.95, 1.05],
        }
    )

    pipeline = mod._build_pipeline(mod.LogisticRegression(max_iter=5000, random_state=mod.RANDOM_STATE))
    seeded = mod._clone_with_seed(pipeline, seed=123)
    assert seeded.get_params()["oversample__random_state"] == 123
    assert seeded.get_params()["model__random_state"] == 123

    repeated = mod.evaluate_on_test_set_repeated(
        train_df,
        test_df,
        pipeline,
        repeat_test_seeds=3,
        base_seed=100,
    )

    assert repeated["n_repeats"] == 3
    assert repeated["seeds"] == [100, 101, 102]
    assert list(repeated["repeat_metrics"].columns) == ["seed", "auc", "accuracy"]
    assert len(repeated["repeat_metrics"]) == 3
    assert {"feature", "importance", "importance_sem"}.issubset(repeated["feature_importance"].columns)


def test_build_final_report_table_matches_within_deltas(spatial_statistics5_module):
    mod = spatial_statistics5_module

    summary_df = pd.DataFrame(
        [
            {
                "Patient table": "original",
                "BOMI2 Train": "x",
                "BOMI1 Train": "",
                "BOMI2 Test\nAUC": 0.978,
                "BOMI1 Test\nAUC": 0.618,
            },
            {
                "Patient table": "original",
                "BOMI2 Train": "",
                "BOMI1 Train": "x",
                "BOMI2 Test\nAUC": 0.853,
                "BOMI1 Test\nAUC": 0.782,
            },
            {
                "Patient table": "original",
                "BOMI2 Train": "x",
                "BOMI1 Train": "x",
                "BOMI2 Test\nAUC": 0.880,
                "BOMI1 Test\nAUC": 0.782,
            },
            {
                "Patient table": "aligned_BOMI2_to_BOMI1_meanstd",
                "BOMI2 Train": "x",
                "BOMI1 Train": "",
                "BOMI2 Test\nAUC": 0.960,
                "BOMI1 Test\nAUC": 0.700,
            },
            {
                "Patient table": "aligned_BOMI2_to_BOMI1_meanstd",
                "BOMI2 Train": "",
                "BOMI1 Train": "x",
                "BOMI2 Test\nAUC": 0.840,
                "BOMI1 Test\nAUC": 0.782,
            },
            {
                "Patient table": "aligned_BOMI2_to_BOMI1_meanstd",
                "BOMI2 Train": "x",
                "BOMI1 Train": "x",
                "BOMI2 Test\nAUC": 0.910,
                "BOMI1 Test\nAUC": 0.790,
            },
        ]
    )

    report_df = mod.build_final_report_table(summary_df, "original")

    assert report_df["Training cohort"].tolist() == [
        "mIF",
        "H&E",
        "mIF",
        "H&E",
        "mIF + H&E",
        "mIF + H&E",
    ]
    assert report_df["Test cohort"].tolist() == ["mIF", "H&E", "H&E", "mIF", "mIF", "H&E"]
    assert report_df["Setting"].tolist() == [
        "Within-cohort",
        "Within-cohort",
        "Cross-cohort",
        "Cross-cohort",
        "Combined",
        "Combined",
    ]
    assert report_df["ROC AUC"].tolist() == pytest.approx([0.978, 0.782, 0.618, 0.853, 0.880, 0.782])
    assert report_df["ΔAUC vs within"].tolist() == pytest.approx([0.0, 0.0, -0.164, -0.125, -0.098, 0.0])

    aligned_report_df = mod.build_final_report_table(summary_df, "aligned_BOMI2_to_BOMI1_meanstd")
    assert aligned_report_df["ΔAUC vs within"].tolist() == pytest.approx([0.0, 0.0, -0.082, -0.120, -0.050, 0.008])
