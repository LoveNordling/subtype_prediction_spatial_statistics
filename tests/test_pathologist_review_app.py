from __future__ import annotations

import pandas as pd

from pathologist_review_app import (
    ReviewCase,
    build_annotation_rows,
    build_legacy_labels,
    compute_summary_metrics,
    order_cases_for_review,
    ReviewSession,
    slugify_name,
)


def test_slugify_name_normalizes_text() -> None:
    assert slugify_name("Dr. Jane Doe") == "dr_jane_doe"


def test_compute_summary_metrics_uses_luad_as_positive_class() -> None:
    rows = pd.DataFrame(
        {
            "true_label": [1, 1, 0, 0],
            "pred_label": [1, 0, 0, 0],
            "pred_score": [1.0, 0.4, 0.2, 0.0],
        }
    )

    metrics = compute_summary_metrics(rows)

    assert metrics["n_completed"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["auc"] == 1.0


def test_build_annotation_rows_and_legacy_export() -> None:
    cases = [
        ReviewCase(patient_id="101", true_label=1, sample_names=("A", "B")),
        ReviewCase(patient_id="202", true_label=0, sample_names=("C",)),
    ]
    annotations = {
        "101": {"pred_label": 1, "confidence_level": 5, "annotated_at": "2026-04-20T10:00:00+00:00"},
        "202": {"pred_label": 0, "confidence_level": 2, "annotated_at": "2026-04-20T10:01:00+00:00"},
    }

    rows = build_annotation_rows(
        cases,
        annotations,
        pathologist_name="Alice",
        dataset_key="bomi2_inform",
        session_id="session_1",
        started_at="2026-04-20T09:55:00+00:00",
    )
    legacy = build_legacy_labels(rows)

    assert list(rows["patient_id"]) == ["101", "202"]
    assert list(rows["n_cores"]) == [2, 1]
    assert list(rows["confidence_level"]) == [5, 2]
    assert list(rows["confidence_label"]) == ["Moderate Adeno", "Moderate Squamous"]
    assert list(rows["pred_score"]) == [0.8, 0.2]
    assert list(rows["pred_label_text"]) == ["Adeno carcinoma", "Squamous Cell Carcinoma"]
    assert list(legacy.columns) == ["Patient ID", "Label"]
    assert list(legacy["Patient ID"]) == ["101", "202"]


def test_order_cases_for_review_shuffles_all_cases_together() -> None:
    cases = [
        ReviewCase(patient_id="1", true_label=1, sample_names=("A",)),
        ReviewCase(patient_id="2", true_label=1, sample_names=("B",)),
        ReviewCase(patient_id="3", true_label=1, sample_names=("C",)),
        ReviewCase(patient_id="4", true_label=0, sample_names=("D",)),
        ReviewCase(patient_id="5", true_label=0, sample_names=("E",)),
        ReviewCase(patient_id="6", true_label=0, sample_names=("F",)),
    ]

    ordered = order_cases_for_review(cases, seed=123)

    assert [case.patient_id for case in ordered] == ["4", "5", "2", "6", "3", "1"]


def test_review_session_queue_does_not_write_true_labels(tmp_path) -> None:
    session_dir = tmp_path / "session"
    legacy_labels_path = tmp_path / "legacy.csv"
    cases = [
        ReviewCase(patient_id="101", true_label=1, sample_names=("A", "B")),
        ReviewCase(patient_id="202", true_label=0, sample_names=("C",)),
    ]
    session = ReviewSession(
        cases=cases,
        cell_groups={},
        dataset_key="bomi2_inform",
        pathologist_name="Alice",
        session_dir=session_dir,
        legacy_labels_path=legacy_labels_path,
        show_patient_id=False,
        review_order_seed=123,
    )

    queue = pd.read_csv(session.queue_path)

    assert list(queue.columns) == ["order_index", "patient_id", "n_cores", "sample_names"]
