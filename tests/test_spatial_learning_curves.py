import pytest


@pytest.fixture
def spatial_learning_curves_module(load_script_module):
    return load_script_module(
        "spatial_learning_curves.py",
        "spatial_learning_curves_under_test",
    )


def test_learning_regime_label_uses_display_cohort_names(spatial_learning_curves_module):
    assert spatial_learning_curves_module._learning_regime_label("BOMI2", "BOMI1", "primary_only") == "mIF only"
    assert spatial_learning_curves_module._learning_regime_label("BOMI2", "BOMI1", "mixed") == "mIF+H&E"
    assert spatial_learning_curves_module._learning_regime_label("BOMI1", "BOMI2", "primary_only") == "H&E only"
    assert spatial_learning_curves_module._learning_regime_label("BOMI1", "BOMI2", "mixed") == "H&E+mIF"
