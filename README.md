# Spatial statistics for lung cancer subtype prediction

Python workflows for extracting spatial features from cancer/stroma cell point
clouds and predicting lung adenocarcinoma versus squamous cell carcinoma.
Analyses compare BOMI2 (Cellpose, CellProfiler, or inForm) with BOMI1, including
cross-cohort transfer, cohort shift, cell thinning, and learning curves.

Spatial metrics are computed per core and aggregated to patient level for
subtype modeling. Cell classes are `0 = stroma` and `1 = cancer`. Raw cell counts
are used for QC and excluded from the model feature set.

## Environment and inputs

Run commands from this repository's root. The existing local environment is:

```bash
source ../rudbeck_intro/venv/bin/activate
```

The examples below invoke that environment directly. This repository does not
yet provide a pinned dependency file; a fresh checkout alone is insufficient to
recreate the environment. Record Python and package versions with each
reproducibility checkpoint.

Data must be supplied separately. Review `InternalPaths` and `ExternalPaths`
in `spatial_data_pipeline.py` before running: some sample metadata and fixed
train/test split files live outside this repository under `../multiplex_dataset/`.
Data-preparation scripts also have their own input path configuration.

Preserve these original cell tables:

- `data/raw/BOMI1_cells_all.csv`
- `data/raw/BOMI2_all_cells_TIL.csv`
- `data/raw/cellprofiler_extracted_cells.csv`
- `data/raw/cellpose_extracted_cells.csv`

Prepared cell tables belong in `data/interim/`; labels and reference metadata
belong in `data/reference/`. Large datasets should be stored separately from Git,
with checksums and documented storage locations. Labels, clinical metadata, and
patient split IDs require storage appropriate to their access restrictions.

## Run the workflow

Inspect available options:

```bash
../rudbeck_intro/venv/bin/python spatial_statistics5.py --help
```

Run the default subtype and cohort-mixing experiments:

```bash
../rudbeck_intro/venv/bin/python spatial_statistics5.py
```

Default settings include:

| Setting | Default |
| --- | --- |
| Internal source | BOMI2 Cellpose |
| External coordinate scale | `2.0` |
| Minimum cells per core | 100 total, 30 cancer, 30 stroma |
| Feature selection | On |
| Hyperparameter tuning | On |
| Feature alignment | Off |
| Feature direction policy | `none` |
| Learning curves | Off |

Compatible metric caches are reused. Missing caches or caches missing required
feature columns are recomputed. To explicitly recompute both cohorts:

```bash
../rudbeck_intro/venv/bin/python spatial_statistics5.py \
  --recompute-internal-metrics --recompute-external-metrics
```

Disable feature selection:

```bash
../rudbeck_intro/venv/bin/python spatial_statistics5.py --no-feature-selection
```

Run only nested cross-validation on the internal training cohort:

```bash
../rudbeck_intro/venv/bin/python spatial_statistics5.py --nested-cv-only
```

Add learning-curve evaluation to the main experiments:

```bash
../rudbeck_intro/venv/bin/python spatial_statistics5.py --run-learning-curve
```

To regenerate masked/filtered inputs after configuring source paths:

```bash
../rudbeck_intro/venv/bin/python prepare_spatial_inputs.py
```

For cell thinning and feature-panel matching, see
[the domain-shift workflow](docs/domain_shift_feature_panel.md).
Subtype-conditioned correction design must use training labels only; retain
held-out patients for evaluation.

## Code organization

| File | Purpose |
| --- | --- |
| `spatial_statistics5.py` | Experiment runner and model evaluation |
| `spatial_metrics.py` | Spatial feature implementations and active feature schema |
| `spatial_data_pipeline.py` | Input paths, metric caching, and patient tables |
| `spatial_utils.py` | Preprocessing, QC, aggregation, and feature selection helpers |
| `spatial_artifact_utils.py` | Predictions, summaries, and feature-importance artifacts |
| `spatial_learning_curves.py` | Subsampling, learning-curve evaluation, and plots |
| `prepare_spatial_inputs.py` | Cell-table preparation |
| `make_cellpose_matched_to_external_v2.py` | Cell thinning and cohort matching |
| `compare_pointclouds.py` | Pairwise segmentation point-cloud comparisons |
| `compare_cohort_metric_distributions.py` | Cohort feature-distribution comparisons |
| `tests/` | Automated helper and regression tests |

Additional `plot_*.py` and `make_*.py` scripts produce analysis and paper figures.
See [metric implementation notes](docs/spatial_metric_implementation_notes.md)
for estimator definitions and caveats. Older artifacts can use a different
feature schema; record the schema used for each experiment.

## Outputs and reproducibility

Generated artifacts are organized under `outputs/`:

- `metrics/`: cached core metrics and evaluation summaries, including
  `cohort_mixing_summary.csv` and nested-CV outputs.
- `predictions/`: prediction tables.
- `feature_importance/`: feature-importance bundles.
- `learning_curves/`: learning-curve tables and plots.
- `comparisons/`, `figures/`, and `qc/`: comparison results and visualizations.

Routine generated outputs are ignored by Git. For a reproducibility checkpoint,
preserve the source revision, exact commands, environment versions, random seeds,
model/QC settings, feature schema, input checksums, and patient split assignments.
Archive the exact metric tables, predictions, and summaries used for reported
results. Store large artifacts externally and record their versions and locations;
small selected checkpoints can be explicitly tracked in Git.

## Validation

Run the automated tests:

```bash
../rudbeck_intro/venv/bin/python -m pytest tests
```

Check that the runner imports:

```bash
../rudbeck_intro/venv/bin/python -c "import spatial_statistics5; print('import ok')"
```

Tests validate helper behavior; reproducing full experiments also requires the
matching data, environment, configuration, and splits.
