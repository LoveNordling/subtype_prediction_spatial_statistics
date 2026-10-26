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


def test_feature_alignment_meanstd_matches_target_mean_and_std(spatial_statistics5_module):
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
        method="meanstd",
    )
    aligned = spatial_statistics5_module.apply_feature_alignment(
        source,
        feat_cols,
        alignment,
        method="meanstd",
    )

    assert np.isfinite(aligned[feat_cols].to_numpy()).all()
    assert np.allclose(aligned[feat_cols].mean().to_numpy(), target[feat_cols].mean().to_numpy())
    assert np.allclose(aligned[feat_cols].std(ddof=0).to_numpy(), target[feat_cols].std(ddof=0).to_numpy())


def test_feature_alignment_median_iqr_matches_target_median_and_iqr(spatial_statistics5_module):
    rng = np.random.default_rng(12)
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
        method="median_iqr",
    )
    aligned = spatial_statistics5_module.apply_feature_alignment(
        source,
        feat_cols,
        alignment,
        method="median_iqr",
    )

    aligned_median = np.median(aligned[feat_cols].to_numpy(float), axis=0)
    target_median = np.median(target[feat_cols].to_numpy(float), axis=0)
    aligned_iqr = np.percentile(aligned[feat_cols].to_numpy(float), 75, axis=0) - np.percentile(
        aligned[feat_cols].to_numpy(float), 25, axis=0
    )
    target_iqr = np.percentile(target[feat_cols].to_numpy(float), 75, axis=0) - np.percentile(
        target[feat_cols].to_numpy(float), 25, axis=0
    )

    assert np.isfinite(aligned[feat_cols].to_numpy()).all()
    assert np.allclose(aligned_median, target_median)
    assert np.allclose(aligned_iqr, target_iqr)


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


def test_nested_cv_returns_out_of_fold_estimate_and_outputs(spatial_statistics5_module, tmp_path):
    mod = spatial_statistics5_module

    train_df = pd.DataFrame(
        {
            "ID": [f"p{i}" for i in range(12)],
            "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "cell_count": [10] * 12,
            "cancer_count": [5] * 12,
            "stroma_count": [5] * 12,
            "feat_a": [0.0, 0.1, 0.2, 0.15, 0.05, 0.25, 1.0, 1.1, 1.2, 1.15, 1.05, 1.25],
            "feat_b": [0.2, 0.0, 0.1, 0.18, 0.08, 0.22, 0.9, 1.0, 1.1, 1.08, 0.98, 1.18],
        }
    )

    nested = mod.run_nested_classification_cv(
        train_df,
        outer_folds=3,
        inner_folds=2,
        hyperparameter_tuning=False,
        use_feature_selection=False,
    )

    assert nested["outer_folds"] == 3
    assert nested["requested_outer_folds"] == 3
    assert nested["requested_inner_folds"] == 2
    assert len(nested["fold_metrics"]) == 3
    assert len(nested["predictions"]) == 12
    assert set(nested["predictions"]["ID"]) == set(train_df["ID"])
    assert nested["fold_metrics"]["model_name"].eq("LogisticRegression (default)").all()
    assert 0.0 <= nested["auc_mean"] <= 1.0

    paths = mod.save_nested_cv_outputs(nested, cohort="BOMI2", out_dir=str(tmp_path))
    assert set(paths) == {"fold_metrics", "predictions", "summary"}
    summary = pd.read_csv(paths["summary"])
    assert summary.loc[0, "cohort"] == "BOMI2"
    assert int(summary.loc[0, "n_patients"]) == 12


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


def test_feature_direction_policy_drops_discordant_features(spatial_statistics5_module, tmp_path):
    mod = spatial_statistics5_module

    df_train_int = pd.DataFrame(
        {
            "ID": ["i0", "i1", "i2", "i3"],
            "label": [0, 0, 1, 1],
            "feature_same": [0.0, 0.2, 1.0, 1.2],
            "feature_flip": [1.0, 1.2, 0.0, 0.2],
        }
    )
    df_test_int = df_train_int.copy()
    df_train_ext = pd.DataFrame(
        {
            "ID": ["e0", "e1", "e2", "e3"],
            "label": [0, 0, 1, 1],
            "feature_same": [0.1, 0.3, 1.1, 1.3],
            "feature_flip": [0.1, 0.3, 1.1, 1.3],
        }
    )
    df_test_ext = df_train_ext.copy()

    out_i, out_it, out_e, out_et, report = mod.apply_feature_direction_policy(
        df_train_int,
        df_test_int,
        df_train_ext,
        df_test_ext,
        "keep-consistent",
        out_dir=str(tmp_path),
    )

    assert "feature_same" in out_i.columns
    assert "feature_flip" not in out_i.columns
    assert "feature_flip" not in out_it.columns
    assert "feature_flip" not in out_e.columns
    assert "feature_flip" not in out_et.columns
    flip = report.set_index("feature").loc["feature_flip"]
    assert bool(flip["discordant"])
    assert flip["action"] == "drop"
    assert (tmp_path / "feature_direction_consistency__keep-consistent.csv").exists()


def test_feature_direction_policy_inverts_discordant_internal_features(spatial_statistics5_module, tmp_path):
    mod = spatial_statistics5_module

    df_train_int = pd.DataFrame(
        {
            "ID": ["i0", "i1", "i2", "i3"],
            "label": [0, 0, 1, 1],
            "feature_same": [0.0, 0.2, 1.0, 1.2],
            "feature_flip": [1.0, 1.2, 0.0, 0.2],
        }
    )
    df_test_int = pd.DataFrame(
        {
            "ID": ["it0", "it1"],
            "label": [0, 1],
            "feature_same": [0.4, 0.8],
            "feature_flip": [2.0, 3.0],
        }
    )
    df_train_ext = pd.DataFrame(
        {
            "ID": ["e0", "e1", "e2", "e3"],
            "label": [0, 0, 1, 1],
            "feature_same": [0.1, 0.3, 1.1, 1.3],
            "feature_flip": [0.1, 0.3, 1.1, 1.3],
        }
    )
    df_test_ext = df_train_ext.copy()

    out_i, out_it, out_e, out_et, report = mod.apply_feature_direction_policy(
        df_train_int,
        df_test_int,
        df_train_ext,
        df_test_ext,
        "invert-discordant",
        invert_cohort="internal",
        out_dir=str(tmp_path),
    )

    assert out_i["feature_flip"].tolist() == [-1.0, -1.2, -0.0, -0.2]
    assert out_it["feature_flip"].tolist() == [-2.0, -3.0]
    assert out_e["feature_flip"].tolist() == df_train_ext["feature_flip"].tolist()
    assert out_et["feature_flip"].tolist() == df_test_ext["feature_flip"].tolist()
    flip = report.set_index("feature").loc["feature_flip"]
    assert bool(flip["discordant"])
    assert flip["action"] == "invert_internal"
    assert (tmp_path / "feature_direction_consistency__invert-discordant.csv").exists()
