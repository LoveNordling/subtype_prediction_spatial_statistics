#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from patient_scatter_plot_utils import SCATTER_SIZE, load_bomi1_cells, plot_scatter_panel
from spatial_artifact_utils import read_split_ids
from spatial_data_pipeline import ExternalPaths, InternalPaths
from spatial_utils import preprocess_cells_inform, preprocess_patients, preprocess_samples, split_external_meta


LABEL_TO_TEXT = {
    1: "Adeno carcinoma",
    0: "Squamous Cell Carcinoma",
}
DATASET_LABELS = {
    "bomi2_inform": "BOMI2 inForm",
    "bomi1_he": "BOMI1 H&E",
}
CONFIDENCE_OPTIONS = {
    1: {"pred_label": 0, "pred_score": 0.0, "label_text": "Definite Squamous"},
    2: {"pred_label": 0, "pred_score": 0.2, "label_text": "Moderate Squamous"},
    3: {"pred_label": 0, "pred_score": 0.4, "label_text": "Slight Squamous"},
    4: {"pred_label": 1, "pred_score": 0.6, "label_text": "Slight Adeno"},
    5: {"pred_label": 1, "pred_score": 0.8, "label_text": "Moderate Adeno"},
    6: {"pred_label": 1, "pred_score": 1.0, "label_text": "Definite Adeno"},
}
BUTTON_COLORS = {
    1: "#2f6db3",
    2: "#6fa3d8",
    3: "#bfd5ee",
    4: "#f4c2a1",
    5: "#e57f49",
    6: "#c84a1b",
}
BUTTON_HOVER_COLORS = {
    1: "#25588f",
    2: "#5b8abc",
    3: "#a7c1de",
    4: "#ecae84",
    5: "#d46d36",
    6: "#ab3f17",
}


@dataclass(frozen=True)
class ReviewCase:
    patient_id: str
    true_label: int
    sample_names: tuple[str, ...]


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Pathologist name must contain at least one letter or digit.")
    return slug


def prompt_pathologist_name() -> str:
    while True:
        name = input("Pathologist name: ").strip()
        if name:
            return name
        print("Please enter a non-empty name.")


def prompt_dataset_key() -> str:
    options = {
        "1": "bomi2_inform",
        "2": "bomi1_he",
        "bomi2_inform": "bomi2_inform",
        "bomi1_he": "bomi1_he",
    }
    print("Select review dataset:")
    print("  1. BOMI2 inForm")
    print("  2. BOMI1 H&E")
    while True:
        value = input("Choice: ").strip().lower()
        if value in options:
            return options[value]
        print("Enter 1, 2, bomi2_inform, or bomi1_he.")


def order_cases_for_review(cases: Iterable[ReviewCase], seed: int) -> list[ReviewCase]:
    case_list = list(cases)
    rng = random.Random(seed)
    rng.shuffle(case_list)
    return case_list


def ensure_ck_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "CK" not in out.columns:
        if "Cancer" not in out.columns:
            raise ValueError(f"Expected either 'CK' or 'Cancer' column. Columns: {list(out.columns)}")
        out = out.rename(columns={"Cancer": "CK"})
    return out[["sample_name", "x", "y", "CK"]].copy()


def build_bomi2_inform_cases(paths: InternalPaths) -> tuple[list[ReviewCase], dict[str, pd.DataFrame]]:
    cells = ensure_ck_schema(preprocess_cells_inform(pd.read_csv(paths.inform_cells_path, low_memory=False)))
    samples = preprocess_samples(pd.read_csv(paths.inform_samples_path, low_memory=False))
    patients_test = preprocess_patients(pd.read_csv(paths.internal_test_csv, low_memory=False))[["ID", "label"]].copy()

    cells["sample_name"] = cells["sample_name"].astype(str)
    samples["sample_name"] = samples["sample_name"].astype(str)
    patients_test["ID"] = patients_test["ID"].astype(str)
    samples["ID"] = samples["ID"].astype(str)

    sample_lookup = (
        samples.groupby("ID", sort=False)["sample_name"]
        .apply(lambda s: [str(v) for v in s])
        .to_dict()
    )
    cell_groups = {name: group.copy() for name, group in cells.groupby("sample_name", sort=False)}

    cases: list[ReviewCase] = []
    for row in patients_test.itertuples(index=False):
        patient_id = str(row.ID)
        sample_names = tuple(s for s in sample_lookup.get(patient_id, []) if s in cell_groups)
        if not sample_names:
            continue
        cases.append(ReviewCase(patient_id=patient_id, true_label=int(row.label), sample_names=sample_names))
    return cases, cell_groups


def build_bomi1_he_cases(paths: ExternalPaths) -> tuple[list[ReviewCase], dict[str, pd.DataFrame]]:
    cells = load_bomi1_cells(paths.external_cells_path, paths.external_meta_path)
    meta = pd.read_csv(paths.external_meta_path, low_memory=False)
    patients_df, samples_df = split_external_meta(meta)
    ext_test_csv = paths.external_test_csv or str(Path(paths.external_split_dir) / "test.csv")
    test_ids = read_split_ids(ext_test_csv)

    patients_df["ID"] = patients_df["ID"].astype(str)
    samples_df["ID"] = samples_df["ID"].astype(str)
    samples_df["sample_name"] = samples_df["sample_name"].astype(str)

    patients_test = patients_df.loc[patients_df["ID"].isin(test_ids), ["ID", "label"]].copy()
    cell_groups = {name: group.copy() for name, group in cells.groupby("sample_name", sort=False)}
    sample_lookup = (
        samples_df.groupby("ID", sort=False)["sample_name"]
        .apply(lambda s: [str(v) for v in s])
        .to_dict()
    )

    cases: list[ReviewCase] = []
    for patient_id in test_ids:
        matches = patients_test.loc[patients_test["ID"] == str(patient_id)]
        if matches.empty:
            continue
        label = int(matches["label"].iloc[0])
        sample_names = tuple(s for s in sample_lookup.get(str(patient_id), []) if s in cell_groups)
        if not sample_names:
            continue
        cases.append(ReviewCase(patient_id=str(patient_id), true_label=label, sample_names=sample_names))
    return cases, cell_groups


def build_cases(dataset_key: str) -> tuple[list[ReviewCase], dict[str, pd.DataFrame]]:
    if dataset_key == "bomi2_inform":
        return build_bomi2_inform_cases(InternalPaths())
    if dataset_key == "bomi1_he":
        return build_bomi1_he_cases(ExternalPaths())
    raise ValueError(f"Unknown dataset key: {dataset_key}")


def compute_summary_metrics(rows: pd.DataFrame) -> dict[str, float | int | None]:
    if rows.empty:
        return {
            "n_completed": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "auc": None,
        }

    y_true = rows["true_label"].astype(int)
    y_pred = rows["pred_label"].astype(int)
    auc = None
    if "pred_score" in rows.columns and y_true.nunique() >= 2:
        auc = float(roc_auc_score(y_true, rows["pred_score"].astype(float)))
    return {
        "n_completed": int(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


def build_annotation_rows(
    cases: Iterable[ReviewCase],
    annotations: dict[str, dict[str, object]],
    *,
    pathologist_name: str,
    dataset_key: str,
    session_id: str,
    started_at: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for order_index, case in enumerate(cases, start=1):
        record = annotations.get(case.patient_id)
        if record is None:
            continue
        confidence_level = int(record.get("confidence_level", 6 if int(record["pred_label"]) == 1 else 1))
        confidence = CONFIDENCE_OPTIONS[confidence_level]
        pred_label = int(confidence["pred_label"])
        rows.append(
            {
                "session_id": session_id,
                "pathologist_name": pathologist_name,
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS[dataset_key],
                "started_at": started_at,
                "annotated_at": str(record["annotated_at"]),
                "order_index": order_index,
                "patient_id": case.patient_id,
                "sample_names": "|".join(case.sample_names),
                "n_cores": len(case.sample_names),
                "true_label": int(case.true_label),
                "true_label_text": LABEL_TO_TEXT[int(case.true_label)],
                "confidence_level": confidence_level,
                "confidence_label": str(confidence["label_text"]),
                "pred_score": float(confidence["pred_score"]),
                "pred_label": pred_label,
                "pred_label_text": LABEL_TO_TEXT[pred_label],
                "is_correct": int(pred_label == int(case.true_label)),
            }
        )
    return pd.DataFrame(rows)


def build_legacy_labels(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["Patient ID", "Label"])
    return rows[["patient_id", "pred_label_text"]].rename(
        columns={"patient_id": "Patient ID", "pred_label_text": "Label"}
    )


class ReviewSession:
    def __init__(
        self,
        *,
        cases: list[ReviewCase],
        cell_groups: dict[str, pd.DataFrame],
        dataset_key: str,
        pathologist_name: str,
        session_dir: Path,
        legacy_labels_path: Path,
        show_patient_id: bool,
        review_order_seed: int,
    ) -> None:
        self.cases = cases
        self.cell_groups = cell_groups
        self.dataset_key = dataset_key
        self.pathologist_name = pathologist_name
        self.session_dir = session_dir
        self.legacy_labels_path = legacy_labels_path
        self.show_patient_id = show_patient_id
        self.review_order_seed = int(review_order_seed)

        self.session_id = session_dir.name
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.annotations: dict[str, dict[str, object]] = {}
        self.current_index = 0
        self._closed = False

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.queue_path = self.session_dir / "queue.csv"
        self.annotations_path = self.session_dir / "annotations.csv"
        self.summary_path = self.session_dir / "summary.json"
        self._write_queue()

        self.fig = None
        self.axes = []
        self.status_text = None
        self.confidence_buttons: list[Button] = []
        self.button_back = None
        self.button_quit = None
        self._is_rendering = False

    def _write_queue(self) -> None:
        rows = [
            {
                "order_index": idx,
                "patient_id": case.patient_id,
                "n_cores": len(case.sample_names),
                "sample_names": "|".join(case.sample_names),
            }
            for idx, case in enumerate(self.cases, start=1)
        ]
        pd.DataFrame(rows).to_csv(self.queue_path, index=False)

    def _save_state(self) -> None:
        rows = build_annotation_rows(
            self.cases,
            self.annotations,
            pathologist_name=self.pathologist_name,
            dataset_key=self.dataset_key,
            session_id=self.session_id,
            started_at=self.started_at,
        )
        rows.to_csv(self.annotations_path, index=False)
        build_legacy_labels(rows).to_csv(self.legacy_labels_path, index=False)

        metrics = compute_summary_metrics(rows)
        summary = {
            "session_id": self.session_id,
            "pathologist_name": self.pathologist_name,
            "dataset_key": self.dataset_key,
            "dataset_label": DATASET_LABELS[self.dataset_key],
            "started_at": self.started_at,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "n_total": len(self.cases),
            "n_completed": metrics["n_completed"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "auc": metrics["auc"],
            "review_order_seed": self.review_order_seed,
            "legacy_labels_path": str(self.legacy_labels_path),
            "annotations_path": str(self.annotations_path),
            "queue_path": str(self.queue_path),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _record_prediction(self, confidence_level: int) -> None:
        case = self.cases[self.current_index]
        self.annotations[case.patient_id] = {
            "pred_label": int(CONFIDENCE_OPTIONS[confidence_level]["pred_label"]),
            "confidence_level": int(confidence_level),
            "pred_score": float(CONFIDENCE_OPTIONS[confidence_level]["pred_score"]),
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_state()
        self.current_index += 1
        if self.current_index >= len(self.cases):
            self.finish()
            return
        self.render_current_case()

    def _go_back(self) -> None:
        if self.current_index == 0:
            return
        self.current_index -= 1
        self.render_current_case()

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key in {str(level) for level in CONFIDENCE_OPTIONS}:
            self._record_prediction(int(key))
        elif key in {"left", "backspace"}:
            self._go_back()
        elif key in {"q", "escape"}:
            self.finish()

    def _on_close(self, _event) -> None:
        if not self._closed:
            self._save_state()
            self._closed = True

    def _on_resize(self, _event) -> None:
        if self._closed or self.fig is None or self._is_rendering or not self.cases:
            return
        if self.current_index >= len(self.cases):
            return
        self.render_current_case()

    def _set_status_text(self, case: ReviewCase) -> None:
        existing = self.annotations.get(case.patient_id)
        if existing is None:
            message = "Shortcuts: 1-3 = squamous confidence, 4-6 = adeno confidence, backspace = previous, q = save and quit"
        else:
            confidence_level = int(existing.get("confidence_level", 6 if int(existing["pred_label"]) == 1 else 1))
            message = f"Existing selection for this case: {CONFIDENCE_OPTIONS[confidence_level]['label_text']}"
        self.status_text.set_text(message)

    def _setup_figure(self, n_cols: int) -> None:
        if self.fig is None:
            self.fig = plt.figure(figsize=(6 * n_cols, 6))
            self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
            self.fig.canvas.mpl_connect("close_event", self._on_close)
            self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        else:
            self.fig.clf()

        self.fig.subplots_adjust(bottom=0.24, top=0.86, wspace=0.28)
        gs = self.fig.add_gridspec(1, n_cols, left=0.06, right=0.98, bottom=0.28, top=0.84, wspace=0.28)
        self.axes = [self.fig.add_subplot(gs[0, idx]) for idx in range(n_cols)]

        self.confidence_buttons = []
        button_specs = [
            (1, "1\nDefinite\nSquamous"),
            (2, "2\nModerate\nSquamous"),
            (3, "3\nSlight\nSquamous"),
            (4, "4\nSlight\nAdeno"),
            (5, "5\nModerate\nAdeno"),
            (6, "6\nDefinite\nAdeno"),
        ]
        x_positions = [0.05, 0.19, 0.33, 0.49, 0.63, 0.77]
        for (confidence_level, label), x_pos in zip(button_specs, x_positions):
            ax_button = self.fig.add_axes([x_pos, 0.07, 0.12, 0.15])
            button = Button(
                ax_button,
                label,
                color=BUTTON_COLORS[confidence_level],
                hovercolor=BUTTON_HOVER_COLORS[confidence_level],
            )
            button.label.set_fontsize(10)
            button.label.set_fontweight("bold")
            button.on_clicked(lambda _event, level=confidence_level: self._record_prediction(level))
            self.confidence_buttons.append(button)

        ax_back = self.fig.add_axes([0.02, 0.90, 0.08, 0.06])
        ax_quit = self.fig.add_axes([0.90, 0.90, 0.08, 0.06])
        self.button_back = Button(ax_back, "Back", color="#dddddd", hovercolor="#c8c8c8")
        self.button_quit = Button(ax_quit, "Quit", color="#dddddd", hovercolor="#c8c8c8")
        self.button_back.label.set_fontsize(12)
        self.button_quit.label.set_fontsize(12)
        self.button_back.label.set_fontweight("bold")
        self.button_quit.label.set_fontweight("bold")

        self.button_back.on_clicked(lambda _event: self._go_back())
        self.button_quit.on_clicked(lambda _event: self.finish())
        self.fig.text(0.24, 0.225, "Squamous: darker = higher confidence", ha="center", va="center", fontsize=12, color="#25588f")
        self.fig.text(0.66, 0.225, "Adeno: darker = higher confidence", ha="center", va="center", fontsize=12, color="#ab3f17")
        self.status_text = self.fig.text(0.5, 0.255, "", ha="center", va="center", fontsize=12)

    def _get_marker_size(self, n_cols: int) -> float:
        if self.fig is None:
            return SCATTER_SIZE
        default_width = 6.0 * max(1, n_cols)
        default_height = 6.0
        width_scale = self.fig.get_figwidth() / default_width
        height_scale = self.fig.get_figheight() / default_height
        linear_scale = max(1.0, min(width_scale, height_scale))
        return SCATTER_SIZE * linear_scale * linear_scale

    def render_current_case(self) -> None:
        self._is_rendering = True
        try:
            case = self.cases[self.current_index]
            n_cols = len(case.sample_names)
            self._setup_figure(n_cols)

            x_max = 1.0
            y_max = 1.0
            for sample_name in case.sample_names:
                group = self.cell_groups[sample_name]
                x_max = max(x_max, float(group["x"].max()))
                y_max = max(y_max, float(group["y"].max()))
            xlim = (0.0, x_max)
            ylim = (0.0, y_max)
            marker_size = self._get_marker_size(n_cols)

            for ax, sample_name, core_index in zip(self.axes, case.sample_names, range(1, n_cols + 1)):
                group = self.cell_groups[sample_name]
                plot_scatter_panel(
                    ax,
                    group,
                    title=f"Core {core_index}\nn={len(group)}",
                    xlim=xlim,
                    ylim=ylim,
                    marker_size=marker_size,
                )

            title = f"{DATASET_LABELS[self.dataset_key]} | Case {self.current_index + 1}/{len(self.cases)} | cores={len(case.sample_names)}"
            if self.show_patient_id:
                title += f" | patient {case.patient_id}"
            self.fig.suptitle(title, fontsize=16)
            self._set_status_text(case)
            self.fig.canvas.draw_idle()
        finally:
            self._is_rendering = False

    def finish(self) -> None:
        self._save_state()
        self._closed = True
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)

    def run(self) -> None:
        if not self.cases:
            raise RuntimeError("No review cases found.")
        self.render_current_case()
        plt.show()
        self._save_state()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Interactive pathologist review tool for subtype labeling from scatter plots only."
    )
    ap.add_argument("--pathologist-name")
    ap.add_argument("--dataset", choices=sorted(DATASET_LABELS))
    ap.add_argument("--show-patient-id", action="store_true", help="Show patient IDs in the UI title.")
    ap.add_argument("--seed", type=int, help="Optional seed for deterministic review ordering.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pathologist_name = args.pathologist_name or prompt_pathologist_name()
    dataset_key = args.dataset or prompt_dataset_key()
    pathologist_slug = slugify_name(pathologist_name)

    print(f"[INFO] Loading data for {DATASET_LABELS[dataset_key]}...")
    cases, cell_groups = build_cases(dataset_key)
    print(f"[INFO] Loaded data for {len(cases)} patients. Preparing session...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("data/reference/subtype_annotator_labels/sessions") / dataset_key / f"{timestamp}_{pathologist_slug}"
    legacy_labels_path = Path("data/reference/subtype_annotator_labels") / f"{pathologist_slug}_{dataset_key}_labels.csv"
    review_order_seed = int(args.seed) if args.seed is not None else random.SystemRandom().randrange(1, 2**31)
    ordered_cases = order_cases_for_review(cases, seed=review_order_seed)

    print(
        f"[INFO] Starting review for {pathologist_name} on {DATASET_LABELS[dataset_key]} "
        f"with {len(cases)} patients."
    )
    print(f"[INFO] Session outputs: {session_dir}")
    print(f"[INFO] Legacy labels CSV: {legacy_labels_path}")
    print(f"[INFO] Review order seed: {review_order_seed}")

    session = ReviewSession(
        cases=ordered_cases,
        cell_groups=cell_groups,
        dataset_key=dataset_key,
        pathologist_name=pathologist_name,
        session_dir=session_dir,
        legacy_labels_path=legacy_labels_path,
        show_patient_id=args.show_patient_id,
        review_order_seed=review_order_seed,
    )
    session.run()

    summary = json.loads(session.summary_path.read_text(encoding="utf-8"))
    print(
        "[DONE] "
        f"completed {summary['n_completed']}/{summary['n_total']} | "
        f"accuracy={summary['accuracy']} | precision={summary['precision']} | "
        f"recall={summary['recall']} | auc={summary['auc']}"
    )


if __name__ == "__main__":
    main()
