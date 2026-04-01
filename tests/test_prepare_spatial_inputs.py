import numpy as np
import pandas as pd
import pytest
import tifffile


@pytest.fixture
def prepare_spatial_inputs_module(load_script_module):
    return load_script_module("prepare_spatial_inputs.py", "prepare_spatial_inputs_under_test")


def test_panck_min_valley_finds_separating_cut(prepare_spatial_inputs_module):
    values = np.concatenate([np.linspace(0.0, 0.9, 10), np.linspace(9.1, 10.0, 10)])
    df = pd.DataFrame({"PanCK": values})

    cut, binary = prepare_spatial_inputs_module.panck_min_valley(df, "PanCK")

    assert 1.0 < cut < 9.0
    assert binary.dtype == np.uint8
    assert binary.sum() == 10


def test_sample_name_and_image_id_parsing(prepare_spatial_inputs_module):
    raw = "BOMI2_TIL_1_Core[1,10,A]_[5091,35249].im3"

    assert prepare_spatial_inputs_module.canonicalize_inform_sample_name(raw) == "BOMI2_TIL_1_[10,A]"
    assert prepare_spatial_inputs_module.extract_image_id(raw) == "BOMI2_TIL_1_Core[1,10,A]_[5091,35249]"


def test_select_mask_plane_handles_common_layouts(prepare_spatial_inputs_module):
    mask_2d = np.arange(6).reshape(2, 3)
    plane, used = prepare_spatial_inputs_module.select_mask_plane(mask_2d, mask_plane=3)
    assert used == 0
    assert np.array_equal(plane, mask_2d)

    leading = np.stack([np.full((2, 2), fill_value=i) for i in range(3)])
    plane, used = prepare_spatial_inputs_module.select_mask_plane(leading, mask_plane=1)
    assert used == 1
    assert np.array_equal(plane, np.full((2, 2), 1))

    trailing = np.stack([np.full((2, 2), fill_value=i) for i in range(4)], axis=-1)
    trailing = np.repeat(trailing, repeats=9, axis=0)
    plane, used = prepare_spatial_inputs_module.select_mask_plane(trailing, mask_plane=2)
    assert used == 2
    assert np.array_equal(plane, trailing[:, :, 2])

    with pytest.raises(ValueError, match="Could not infer mask axis"):
        prepare_spatial_inputs_module.select_mask_plane(np.zeros((9, 9, 9)), mask_plane=0)


def test_filter_cells_by_mask_keeps_expected_rows_and_stats(prepare_spatial_inputs_module, tmp_path, monkeypatch):
    image_id = "BOMI2_TIL_1_Core[1,10,A]_[5091,35249]"
    mask_path = tmp_path / f"{image_id}_binary_seg_maps.tif"
    mask = np.array([[1, 0, 2], [2, 1, 0]], dtype=np.uint8)
    tifffile.imwrite(mask_path, mask)
    monkeypatch.setattr(prepare_spatial_inputs_module, "tqdm", lambda iterable, **kwargs: iterable)

    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5, 6],
            "ImageID": [image_id, image_id, image_id, image_id, "missing_mask", np.nan],
            "x": [0, 1, 2, 10, 0, 0],
            "y": [0, 0, 0, 10, 0, 0],
        }
    )

    filtered, stats = prepare_spatial_inputs_module.filter_cells_by_mask(
        df,
        image_col="ImageID",
        x_col="x",
        y_col="y",
        mask_paths={image_id: mask_path},
        mask_plane=0,
    )

    assert filtered["row_id"].tolist() == [2, 3, 4]
    assert stats == {
        "input_rows": 6,
        "kept_rows": 3,
        "dropped_rows": 2,
        "missing_mask_groups": 1,
        "missing_image_id_rows": 1,
        "mask_plane_min": 0,
        "mask_plane_max": 0,
    }
