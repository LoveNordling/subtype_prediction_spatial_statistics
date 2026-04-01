import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def compare_pointclouds_module(load_script_module, stub_spatial_metrics_module, stub_spatial_utils_module):
    return load_script_module(
        "compare_pointclouds.py",
        "compare_pointclouds_under_test",
        stub_modules={
            "spatial_metrics": stub_spatial_metrics_module,
            "spatial_utils": stub_spatial_utils_module,
        },
    )


def test_canonicalize_sample_names(compare_pointclouds_module):
    filename = "BOMI2_TIL_1_Core[1,10,A]_[5091,35249]_component_data_CK.tiff"
    inform_name = "BOMI2_TIL_1_[10,A]"

    assert compare_pointclouds_module.canonicalize_from_filename(filename) == "BOMI2_TIL_1_[10,A]"
    assert compare_pointclouds_module.canonicalize_inform_name(inform_name) == "BOMI2_TIL_1_[10,A]"


def test_extract_ripley_vectors_aligns_on_common_bins(compare_pointclouds_module):
    ripley = {
        "cancer_ripley_L_10": 1.0,
        "cancer_ripley_L_20": 2.0,
        "stroma_ripley_L_20": 3.0,
        "stroma_ripley_L_30": 4.0,
    }

    bins, cancer, stroma = compare_pointclouds_module.extract_ripley_vectors(ripley)

    assert np.array_equal(bins, np.array([20.0]))
    assert np.array_equal(cancer, np.array([2.0]))
    assert np.array_equal(stroma, np.array([3.0]))


def test_geometry_helpers_behave_on_small_examples(compare_pointclouds_module):
    a = np.array([[0.0, 0.0], [2.0, 0.0]])
    b = np.array([[0.0, 0.0], [1.0, 0.0]])

    assert compare_pointclouds_module.hd95(a, b) == pytest.approx(1.0)
    assert compare_pointclouds_module.greedy_f1_match(a, b, radius=1.1) == pytest.approx(1.0)

    df_a = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 0.0], "CK": [1, 0]})
    df_b = pd.DataFrame({"x": [0.2, 9.8], "y": [0.0, 0.0], "CK": [1, 0]})
    assert compare_pointclouds_module.type_aware_macro_f1(df_a, df_b, radius=1.0) == pytest.approx(1.0)


def test_filter_histogram_nnd_rows_applies_histogram_only_threshold(compare_pointclouds_module):
    df = pd.DataFrame(
        {
            "sample_name": ["s1", "s2", "s3"],
            "cohort": ["A", "A", "B"],
            "median_nnd_all": [20.0, 80.0, 40.0],
            "median_nnd_cancer": [10.0, 20.0, 90.0],
            "median_nnd_stroma": [15.0, 25.0, 30.0],
        }
    )

    filtered = compare_pointclouds_module.filter_histogram_nnd_rows(df, max_nnd=60.0)

    assert filtered["sample_name"].tolist() == ["s1"]
