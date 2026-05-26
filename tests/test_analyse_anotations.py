from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analyse_anotations import confidence_strength_from_score, load_annotations, load_latest_sessions


def _write_session(
    root: Path,
    *,
    dataset_key: str,
    session_id: str,
    pathologist_name: str,
    started_at: str,
    last_updated_at: str,
    n_total: int,
    n_completed: int,
    annotations: pd.DataFrame,
) -> None:
    session_dir = root / dataset_key / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = session_dir / "annotations.csv"
    annotations.to_csv(annotations_path, index=False)
    summary = {
        "session_id": session_id,
        "dataset_key": dataset_key,
        "pathologist_name": pathologist_name,
        "started_at": started_at,
        "last_updated_at": last_updated_at,
        "n_total": n_total,
        "n_completed": n_completed,
        "annotations_path": str(annotations_path),
    }
    (session_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_load_latest_sessions_keeps_latest_complete_session(tmp_path: Path) -> None:
    annotations = pd.DataFrame(
        {
            "session_id": ["s1"],
            "pathologist_name": ["Alice"],
            "dataset_key": ["bomi2_inform"],
            "patient_id": ["100"],
            "true_label": [1],
            "pred_label": [1],
        }
    )
    _write_session(
        tmp_path,
        dataset_key="bomi2_inform",
        session_id="20260424_100000_alice",
        pathologist_name="Alice",
        started_at="2026-04-24T10:00:00+00:00",
        last_updated_at="2026-04-24T10:10:00+00:00",
        n_total=1,
        n_completed=1,
        annotations=annotations,
    )
    _write_session(
        tmp_path,
        dataset_key="bomi2_inform",
        session_id="20260425_100000_alice",
        pathologist_name="Alice",
        started_at="2026-04-25T10:00:00+00:00",
        last_updated_at="2026-04-25T10:10:00+00:00",
        n_total=2,
        n_completed=1,
        annotations=annotations,
    )

    latest = load_latest_sessions(tmp_path)

    assert list(latest["session_id"]) == ["20260424_100000_alice"]


def test_load_annotations_falls_back_to_pred_label_score(tmp_path: Path) -> None:
    annotations = pd.DataFrame(
        {
            "session_id": ["20260424_100000_alice"],
            "pathologist_name": ["Alice"],
            "dataset_key": ["bomi1_he"],
            "patient_id": ["100"],
            "true_label": [0],
            "pred_label": [1],
        }
    )
    _write_session(
        tmp_path,
        dataset_key="bomi1_he",
        session_id="20260424_100000_alice",
        pathologist_name="Alice",
        started_at="2026-04-24T10:00:00+00:00",
        last_updated_at="2026-04-24T10:10:00+00:00",
        n_total=1,
        n_completed=1,
        annotations=annotations,
    )

    latest = load_latest_sessions(tmp_path)
    loaded = load_annotations(latest)

    assert list(loaded["pred_score"]) == [1.0]
    assert pd.isna(loaded["confidence_level"].iloc[0])
    assert list(loaded["confidence_strength"]) == [1.0]
    assert list(loaded["is_correct"]) == [0]


def test_confidence_strength_from_score_is_symmetric() -> None:
    scores = pd.Series([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    result = confidence_strength_from_score(scores)
    assert list(result) == [1.0, 0.8, 0.6, 0.6, 0.8, 1.0]
