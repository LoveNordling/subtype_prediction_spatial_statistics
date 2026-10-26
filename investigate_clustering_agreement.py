#!/usr/bin/env python3
"""Descriptive clustering agreement audit; never changes metric caches or models.

Run from the repository root. Radius units are the input coordinate units.
"""
from itertools import combinations
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from tqdm import tqdm

from plot_segmentation_metric_correlations import (
    DEFAULT_METHOD_TO_PATH, load_allowed_bomi2_samples,
    compute_pairwise_correlations,
)
from spatial_metrics import _build_graph_from_edges
from spatial_utils import (
    aggregate_cores_to_patient, preprocess_cells_inform,
    preprocess_cellpose_data, preprocess_cellprofiler_data,
)

OUT = Path('outputs/comparisons/clustering_detective')
CLUSTER = ['cancer_average_clustering', 'stroma_average_clustering']


def local_clustering(coords, radius):
    """Exact unweighted NetworkX clustering, computed using sparse triangles."""
    edges = cKDTree(coords).query_pairs(radius, output_type='ndarray')
    n = len(coords)
    if not len(edges):
        return np.zeros(n), np.zeros(n)
    i, j = edges.T
    a = csr_matrix((np.ones(2 * len(i), dtype=np.float64),
                    (np.r_[i, j], np.r_[j, i])), shape=(n, n))
    degree = np.asarray(a.sum(axis=1)).ravel()
    closed = np.asarray(a.multiply(a @ a).sum(axis=1)).ravel()
    clustering = np.divide(closed, degree * (degree - 1),
                           out=np.zeros(n), where=degree > 1)
    return clustering, degree


def simulate_geometric_null(out_dir):
    """Check whether high clustering appears without biological organization."""
    rows = []
    for seed in tqdm(range(5), desc='Uniform-point graph controls'):
        xy = np.random.default_rng(seed).uniform(0, 1500, size=(5000, 2))
        for radius in [50, 75, 100]:
            clustering, degree = local_clustering(xy, radius)
            rows.append(dict(seed=seed, radius=radius, n_cells=len(xy),
                             mean_clustering=clustering.mean(),
                             mean_degree=degree.mean(), fraction_degree_lt2=(degree < 2).mean()))
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / 'uniform_point_control.csv', index=False)
    return result


def pair_stats(frames, metrics, level, radius=None):
    rows = []
    key = 'ID' if level == 'patient' else 'sample_name'
    for left, right in combinations(frames, 2):
        merged = frames[left].merge(frames[right], on=key, suffixes=('_l', '_r'))
        for metric in metrics:
            x, y = merged[metric + '_l'], merged[metric + '_r']
            r, rho = compute_pairwise_correlations(x, y)
            delta = y - x
            # Removes constant method bias, retaining patient-specific disagreement.
            spread = np.sqrt((x.var() + y.var()) / 2)
            centered_rmse = np.sqrt(np.mean((delta - delta.mean()) ** 2))
            # Influence check, rather than selecting a better-looking result.
            loo = [x.drop(k).corr(y.drop(k)) for k in x.index]
            rows.append(dict(level=level, radius=radius, left=left, right=right,
                             metric=metric, n=len(x), pearson=r, spearman=rho,
                             bias=delta.mean(), pooled_sd=spread,
                             centered_rmse=centered_rmse,
                             error_to_spread=centered_rmse / spread,
                             loo_min=min(loo), loo_max=max(loo)))
    return pd.DataFrame(rows)


def audit_sensitivity(out_dir):
    """Prespecified count QC and paired-patient bootstrap of radius changes."""
    core = pd.read_csv(out_dir / 'radius_core_metrics.csv', dtype={'ID': str})
    cached = {m: pd.read_csv(p) for m, p in DEFAULT_METHOD_TO_PATH.items()}
    good = set.intersection(*(set(f.loc[(f.cell_count >= 150) &
        (f.cancer_count >= 30) & (f.stroma_count >= 30), 'sample_name'])
        for f in cached.values()))
    output = []
    for name, subset in [('all', core), ('without_407', core[core.ID != '407']),
                         ('count_qc_all_methods', core[core.sample_name.isin(good)])]:
        for radius, group in subset.groupby('radius'):
            pat = {m: aggregate_cores_to_patient(g.drop(columns=['method', 'radius']))
                   for m, g in group.groupby('method', sort=False)}
            output.append(pair_stats(pat, CLUSTER, 'patient', radius).assign(subset=name))
    pd.concat(output).to_csv(out_dir / 'sensitivity_agreement.csv', index=False)
    patient = pd.read_csv(out_dir / 'radius_patient_metrics.csv', dtype={'ID': str})
    rows = []
    for metric in CLUSTER:
        for l, r in combinations(patient.method.unique(), 2):
            x = patient[patient.method == l].pivot(index='ID', columns='radius', values=metric)
            y = patient[patient.method == r].pivot(index='ID', columns='radius', values=metric).reindex(x.index)
            radii = list(x.columns)
            draws = np.random.default_rng(42).integers(0, len(x), (2000, len(x)))
            xx, yy = x.to_numpy()[draws], y.to_numpy()[draws]
            xx -= xx.mean(axis=1, keepdims=True)
            yy -= yy.mean(axis=1, keepdims=True)
            boot = (xx * yy).sum(axis=1) / np.sqrt((xx * xx).sum(axis=1) * (yy * yy).sum(axis=1))
            if 50 not in radii:
                continue
            for j, radius in enumerate(radii):
                change = boot[:, j] - boot[:, radii.index(50)]
                rows.append(dict(metric=metric, left=l, right=r, radius=radius,
                                 pearson_ci_low=np.quantile(boot[:, j], .025),
                                 pearson_ci_high=np.quantile(boot[:, j], .975),
                                 delta_from50_ci_low=np.quantile(change, .025),
                                 delta_from50_ci_high=np.quantile(change, .975)))
    pd.DataFrame(rows).to_csv(out_dir / 'radius_bootstrap.csv', index=False)
    sensitivity = pd.concat(output)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    colors = {('inform', 'cellpose'): '#1b9e77', ('inform', 'cellprofiler'): '#d95f02',
              ('cellpose', 'cellprofiler'): '#7570b3'}
    for i, subset in enumerate(['all', 'count_qc_all_methods']):
        for j, metric in enumerate(CLUSTER):
            part = sensitivity[(sensitivity.subset == subset) & (sensitivity.metric == metric)]
            for (left, right), g in part.groupby(['left', 'right']):
                axes[i, j].plot(g.radius, g.pearson, 'o-', color=colors[left, right], label=f'{left} / {right}')
            n = part.n.iloc[0]
            axes[i, j].set(title=f'{metric.split("_")[0].capitalize()} clustering: {"all matched" if i == 0 else "count QC"} ({n} patients)',
                           ylim=(0, 1.02), xticks=sorted(part.radius.unique()),
                           ylabel='Patient Pearson r', xlabel='Radius (input coordinate units)')
            axes[i, j].grid(alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Larger graph radii improve clustering agreement, including after count QC')
    fig.tight_layout()
    fig.savefig(out_dir / 'radius_sensitivity.png', dpi=180)
    plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--patients', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--radii', type=float, nargs='+', default=[20, 35, 50, 75, 100])
    ap.add_argument('--out-dir', type=Path, default=OUT)
    ap.add_argument('--skip-cache-summary', action='store_true')
    args = ap.parse_args()
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    simulate_geometric_null(OUT)
    allowed = load_allowed_bomi2_samples(
        '../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/samples_labels.csv')
    frames = {m: pd.read_csv(p, dtype={'ID': str}) for m, p in DEFAULT_METHOD_TO_PATH.items()}
    frames = {m: f[f.sample_name.isin(allowed)].copy() for m, f in frames.items()}
    common = sorted(set.intersection(*(set(f.sample_name) for f in frames.values())))
    # Same cores across methods; preserve the existing cell-count weighting.
    frames = {m: f[f.sample_name.isin(common)].copy() for m, f in frames.items()}
    assert all(not f.sample_name.duplicated().any() for f in frames.values())
    patients = {m: aggregate_cores_to_patient(f) for m, f in frames.items()}
    features = [c for c in frames['inform'] if c not in
                {'ID', 'sample_name', 'label', 'cell_count', 'cancer_count', 'stroma_count'}]
    if not args.skip_cache_summary:
        pd.concat([pair_stats(frames, features, 'core'),
                   pair_stats(patients, features, 'patient')]).to_csv(OUT / 'cached_agreement.csv', index=False)
    spread_rows = []
    for m, f in patients.items():
        for c in features:
            x = f[c]
            spread_rows.append(dict(method=m, metric=c, n=len(x), mean=x.mean(),
                                    sd=x.std(), cv=x.std()/abs(x.mean()),
                                    p05=x.quantile(.05), p95=x.quantile(.95)))
    pd.DataFrame(spread_rows).to_csv(OUT / 'patient_spread.csv', index=False)
    ids = sorted(set.intersection(*(set(f.ID) for f in frames.values())))
    selected = sorted(np.random.default_rng(args.seed).choice(ids, min(args.patients, len(ids)), replace=False))
    meta = frames['inform'].set_index('sample_name')[['ID', 'label']]
    samples = set(meta.index[meta.ID.isin(selected)])
    (OUT / 'selection.json').write_text(json.dumps(dict(seed=args.seed, patients=selected,
        samples=sorted(samples), radii=args.radii), indent=2))
    configs = {
        'inform': ('data/raw/BOMI2_all_cells_TIL.csv', ['Sample Name', 'Cell X Position', 'Cell Y Position', 'CK'], preprocess_cells_inform),
        'cellpose': ('data/interim/cellpose_extracted_cells_fitlered_necrosis.csv', ['filename', 'x', 'y', 'ck'], preprocess_cellpose_data),
        'cellprofiler': ('data/interim/cellprofiler_extracted_cells_filtered_necrosis.csv', ['FileName_CK', 'Location_Center_X', 'Location_Center_Y', 'CK'], preprocess_cellprofiler_data),
    }
    rows = []
    for method, (path, cols, prep) in configs.items():
        chunks = []
        for chunk in tqdm(pd.read_csv(path, usecols=cols, chunksize=200000), desc=f'Loading {method}', unit='chunk'):
            f = prep(chunk)
            chunks.append(f[f.sample_name.isin(samples)])
        cells = pd.concat(chunks, ignore_index=True)
        for sample, f in tqdm(list(cells.groupby('sample_name')), desc=f'Radius sweep {method}', unit='core'):
            xy, types = f[['x', 'y']].to_numpy(), f.Cancer.to_numpy()
            for radius in args.radii:
                cc, degree = local_clustering(xy, radius)
                row = dict(method=method, sample_name=sample, ID=meta.loc[sample, 'ID'],
                           label=int(meta.loc[sample, 'label']), cell_count=len(f), radius=radius)
                for code, name in [(1, 'cancer'), (0, 'stroma')]:
                    mask = types == code
                    row[f'{name}_average_clustering'] = cc[mask].mean() if mask.any() else 0
                    row[f'{name}_mean_degree'] = degree[mask].mean() if mask.any() else 0
                    row[f'{name}_fraction_degree_lt2'] = (degree[mask] < 2).mean() if mask.any() else 0
                if radius == 50:
                    cached = frames[method].set_index('sample_name').loc[sample]
                    assert int(cached.cell_count) == len(f), (method, sample, 'cell count mismatch')
                    for c in CLUSTER:
                        assert np.isclose(row[c], cached[c], atol=1e-10), (method, sample, c, row[c], cached[c])
                rows.append(row)
            # One real-core check against the original NetworkX definition per method.
            if sample == sorted(samples)[0]:
                e = cKDTree(xy).query_pairs(50, output_type='ndarray')
                graph = _build_graph_from_edges(types, e)
                cc50, _ = local_clustering(xy, 50)
                assert np.allclose(cc50, list(nx.clustering(graph).values()))
        pd.DataFrame(rows).to_csv(OUT / 'radius_core_metrics.csv', index=False)
    results = pd.DataFrame(rows)
    correlations = []
    patient_rows = []
    for radius, group in results.groupby('radius'):
        core = {m: g.drop(columns=['method', 'radius']) for m, g in group.groupby('method', sort=False)}
        pat = {m: aggregate_cores_to_patient(g) for m, g in core.items()}
        correlations.extend([pair_stats(core, CLUSTER, 'core', radius), pair_stats(pat, CLUSTER, 'patient', radius)])
        patient_rows.extend(g.assign(method=m, radius=radius) for m, g in pat.items())
    corr = pd.concat(correlations, ignore_index=True)
    corr.to_csv(OUT / 'radius_agreement.csv', index=False)
    pd.concat(patient_rows).to_csv(OUT / 'radius_patient_metrics.csv', index=False)
    audit_sensitivity(OUT)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, metric in enumerate(CLUSTER):
        subset = corr[(corr.level == 'patient') & (corr.metric == metric)]
        for (l, r), g in subset.groupby(['left', 'right']):
            axes[0, j].plot(g.radius, g.pearson, 'o-', label=f'{l} / {r}')
        axes[0, j].set(title=metric.replace('_', ' '), ylabel='Patient Pearson r',
                       ylim=(min(-.05, subset.pearson.min() - .05), 1.05),
                       xlabel='Radius (input coordinate units)')
        for m, f in patients.items():
            axes[1, j].hist(f[metric], bins=25, histtype='step', label=m, linewidth=1.6)
        axes[1, j].set(xlabel='Clustering at radius 50 (full cohort)', ylabel='Patients')
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(f'Clustering audit: radius sweep in {len(selected)} random patients / {len(samples)} cores')
    fig.tight_layout()
    fig.savefig(OUT / 'clustering_audit.png', dpi=180)
    print(corr[corr.level == 'patient'].to_string(index=False))


if __name__ == '__main__':
    main()
