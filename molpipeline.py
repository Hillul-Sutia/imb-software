"""Reproducible, held-out molecular classification experiments."""
import argparse
import hashlib
from functools import partial
import importlib.metadata as metadata
import itertools
import json
import logging
import tempfile
import os
from pathlib import Path
import platform
import re
import sys
import time

import joblib
import result_reporting as reporting
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

AXES = {'feature': ['morgan', 'chemberta'], 'split': ['random', 'scaffold'],
        'model': ['random_forest', 'catboost'], 'strategy': ['default', 'mtune'],
        'optimization': ['default']}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


LOG = logging.getLogger(__name__)


class LabelConflictError(ValueError):
    def __init__(self, rows):
        self.rows = rows
        super().__init__(f'Conflicting labels for the same canonical molecule: '
                         f'{rows.smiles.nunique()} groups, {len(rows)} source rows')


def load_data(spec, root):
    path = root / spec['path']
    df = pd.read_csv(path)
    sm, target = spec.get('smiles_column', 'smiles'), spec.get('target_column', 'label')

    if not sm or not target:
        raise ValueError(f'{path}: fill smiles_column and target_column in the config before running')
    if sm not in df or target not in df:
        raise ValueError(f'{path}: missing {sm!r} or {target!r}')
    
    if df[[sm, target]].isna().any().any():
        raise ValueError(f'{path}: missing SMILES or labels')
    
    if not df[target].isin([0, 1]).all() or df[target].nunique() != 2:
        raise ValueError('Labels must contain exactly binary 0 and 1')
    
    policy = spec.get('invalid_smiles', 'error')
    if policy not in ('error', 'skip'):
        raise ValueError('invalid_smiles must be error or skip')
    records, excluded = [], []
    for row, value in enumerate(df[sm]):
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(str(value))
        if mol is None or mol.GetNumAtoms() == 0:
            if policy == 'error':
                raise ValueError(f'Invalid SMILES at CSV data row {row + 1}: {value}. '
                                 'To exclude invalid structures with an audit report, set invalid_smiles=skip for this dataset.')
            excluded.append({'source_data_row': row + 1, 'smiles': str(value),
                             'label': int(df[target].iloc[row]), 'reason': 'RDKit rejected SMILES or empty molecule'})
            continue
        records.append({'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
                        'label': int(df[target].iloc[row]),
                        'scaffold': MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
                        'source_data_row': row + 1, 'original_smiles': str(value)})
    out = pd.DataFrame(records, columns=['smiles', 'label', 'scaffold', 'source_data_row', 'original_smiles'])
    conflicts = out.groupby('smiles').label.transform('nunique') > 1
    if conflicts.any():
        raise LabelConflictError(out.loc[conflicts].sort_values(['smiles', 'source_data_row']))
    out = out.drop(columns='original_smiles')
    cleaned = out.drop_duplicates('smiles').reset_index(drop=True)
    if cleaned.label.nunique() != 2:
        raise ValueError('Both binary classes must remain after invalid-SMILES filtering and deduplication')
    cleaned.attrs['excluded_rows'] = excluded
    cleaned.attrs['data_quality'] = {'input_rows': len(df), 'invalid_smiles_policy': policy,
        'invalid_smiles_removed': len(excluded), 'duplicate_rows_removed': len(out)-len(cleaned),
        'retained_rows': len(cleaned),
        'input_class_counts': {str(k): int(v) for k,v in df[target].value_counts().items()},
        'retained_class_counts': {str(k): int(v) for k,v in cleaned.label.value_counts().items()},
        'row_numbering': 'source_data_row is 1-based CSV data-record position, excluding header'}
    return cleaned, hashlib.sha256(path.read_bytes()).hexdigest()


def folds(y, groups, n, seed):
    if n < 2:
        raise ValueError('At least two CV folds required')
    pairs = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    for tr, va in pairs:
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise ValueError('Split has a single class; change seed/folds or supply more diverse data')
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError('Group leakage')
    return pairs


def partition(data, kind, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('test_size must be between zero and one')
    y = data.label.to_numpy()
    groups = data[kind if kind == 'scaffold' else 'smiles'].to_numpy()
    # Select closest-size valid grouped fold, solely by size, never model performance.
    n = max(2, round(1 / fraction))
    pairs = folds(y, groups, n, seed)
    return min(pairs, key=lambda pair: abs(len(pair[1]) / len(data) - fraction))


def features(data, kind, cfg, cache):
    settings = cfg.get(kind, {})
    key = digest({'smiles': data.smiles.tolist(), 'kind': kind, 'settings': settings,
                  'rdkit': rdBase.rdkitVersion, 'implementation': 'features-v1',
                  'versions': versions()})
    path = cache / (key + '.npy')
    if path.exists():
        return np.load(path, allow_pickle=False)
    if kind == 'morgan':
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=settings.get('radius', 2),
            fpSize=settings.get('n_bits', 2048), includeChirality=settings.get('chirality', True))
        x = np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in data.smiles])
    else:
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_id = settings.get('model', 'DeepChem/ChemBERTa-77M-MLM')
        revision = settings.get('revision')
        if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('ChemBERTa requires an immutable Hugging Face commit revision')
        torch.manual_seed(cfg.get('seed', 42))
        torch.use_deterministic_algorithms(True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModel.from_pretrained(model_id, revision=revision).to('cpu').eval()
        arrays = []
        batch = settings.get('batch_size', 32)
        with torch.no_grad():
            for start in range(0, len(data), batch):
                tokens = tokenizer(data.smiles.iloc[start:start+batch].tolist(), padding=True,
                    truncation=True, max_length=settings.get('max_length', 256), return_tensors='pt')
                hidden = model(**tokens).last_hidden_state
                mask = tokens['attention_mask'].unsqueeze(-1)
                arrays.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        x = np.concatenate(arrays)
        
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.save(f, x, allow_pickle=False)
    os.replace(tmp, path)
    return x


def export_features(data, x, kind, cfg, output, spec, data_hash):
    """One shared, audited CSV per dataset/target/feature identity; retain .npy cache."""
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != len(data) or not np.isfinite(x).all():
        raise ValueError('Feature export requires a finite 2D matrix aligned with cleaned dataset rows')
    source_rows = data.get('source_data_row', pd.Series(np.arange(len(data))+1)).tolist()
    identity = {'dataset': spec, 'input_sha256': data_hash, 'kind': kind,
                'settings': cfg.get(kind, {}), 'versions': versions(), 'export_format': 1,
                'smiles': data.smiles.tolist(), 'labels': data.label.tolist(),
                'source_data_rows': source_rows, 'shape': list(x.shape), 'dtype': str(x.dtype)}
    key = digest(identity)[:24]
    name = re.sub(r'[^A-Za-z0-9_-]+', '-', str(spec['name'])).strip('-_')[:40] or 'dataset'
    folder = output / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{name}__{kind}__{key}.csv'
    metadata = path.with_suffix('.json')
    if path.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved.get('csv_sha256') == hashlib.sha256(path.read_bytes()).hexdigest():
            return {'csv': '../features/' + path.name, 'metadata': '../features/' + metadata.name,
                    'sha256': saved['csv_sha256'], 'n_rows': len(data), 'n_features': x.shape[1]}
    tmp = path.with_suffix('.csv.tmp')
    prefix = 'morgan_bit_' if kind == 'morgan' else 'chemberta_embedding_'
    names = [prefix + str(i) for i in range(x.shape[1])]
    identifiers = pd.DataFrame({'cleaned_row': np.arange(len(data)), 'source_data_row': source_rows,
                                'smiles': data.smiles.to_numpy(), 'label': data.label.to_numpy()})
    with tmp.open('w', encoding='utf-8', newline='') as stream:
        for start in range(0, len(data), 500):
            rows = pd.concat([identifiers.iloc[start:start+500].reset_index(drop=True),
                              pd.DataFrame(x[start:start+500], columns=names)], axis=1)
            rows.to_csv(stream, index=False, header=start == 0, float_format='%.17g')
    os.replace(tmp, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(metadata, {'dataset': spec, 'input_sha256': data_hash, 'feature': kind,
        'settings': cfg.get(kind, {}), 'versions': versions(), 'export_id': key,
        'csv_sha256': checksum, 'n_rows': len(data), 'n_features': x.shape[1],
        'row_order': 'cleaned, deduplicated dataset order before splitting',
        'label_convention': 'original dataset 0/1 labels; not minority-remapped',
        'source_data_row': '1-based original data-record position, excluding header',
        'data_quality': data.attrs.get('data_quality', {})})
    return {'csv': '../features/' + path.name, 'metadata': '../features/' + metadata.name,
            'sha256': checksum, 'n_rows': len(data), 'n_features': x.shape[1]}


class MTuneAdapter(ClassifierMixin, BaseEstimator):
    """Supply current sklearn tags and validate inputs; delegate algorithm to mtune 0.0.8."""
    def __init__(self, base_model, method='direct', n_ensemble=11, random_state=42):
        self.base_model = base_model
        self.method = method
        self.n_ensemble = n_ensemble
        self.random_state = random_state

    def fit(self, X, y):
        from mtune import Mtune
        if metadata.version('mtune') != '0.0.8':
            raise RuntimeError('Validated integration requires mtune==0.0.8')
        counts = np.unique(y, return_counts=True)[1]
        if len(counts) != 2 or counts[0] == counts[1]:
            raise ValueError('M-Tune 0.0.8 requires unequal binary class counts (including every CV fold)')
        if self.method not in ('direct', 'ensemble'):
            raise ValueError('Unknown M-Tune method')
        if self.method == 'ensemble' and not 1 <= self.n_ensemble <= max(counts):
            raise ValueError('n_ensemble exceeds available majority rows')
        self.estimator_ = Mtune(self.base_model, self.method, self.n_ensemble, self.random_state)
        self.estimator_.fit(pd.DataFrame(np.asarray(X)), pd.Series(np.asarray(y), name='target'))
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return self.estimator_.predict(pd.DataFrame(np.asarray(X)))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(pd.DataFrame(np.asarray(X)))


def estimator(combo, cfg):
    seed = combo['seed']
    params = cfg.get('model_params', {}).get(combo['model'], {})
    if combo['model'] == 'random_forest':
        base = RandomForestClassifier(**({'random_state': seed, 'n_jobs': 1} | params))
    else:
        from catboost import CatBoostClassifier
        base = CatBoostClassifier(**({'random_seed': seed, 'thread_count': 1, 'verbose': False,
                                      'allow_writing_files': False} | params))
    if combo['strategy'] == 'mtune':
        return MTuneAdapter(base, random_state=seed, **cfg.get('mtune', {}))
    return base


def positive_label(y_train):
    """Choose the minority using outer training labels only; ties use original label 1."""
    labels, counts = np.unique(y_train, return_counts=True)
    if set(labels) != {0, 1}:
        raise ValueError('Training labels must contain both 0 and 1')
    return int(labels[np.argmin(counts)]) if counts[0] != counts[1] else 1


def minority_metrics(y, pred, prob, positive):
    """Map the fixed training minority to binary positive, including score orientation."""
    y, pred = np.asarray(y), np.asarray(pred)
    values = metrics((y == positive).astype(int), (pred == positive).astype(int), prob)
    for label in (0, 1):
        binary = metrics((y == label).astype(int), (pred == label).astype(int),
                         prob if label == positive else 1-np.asarray(prob))
        values.update({f'{name}_{label}': binary[name] for name in CLASS_METRICS})
        values[f'support_{label}'] = int(np.sum(y == label))
    return values


def evaluate_partition(model, X, y, positive):
    started = time.perf_counter()
    probabilities = model.predict_proba(X)
    class1 = probabilities[:, list(model.classes_).index(1)]
    pred = model.predict(X) if isinstance(model, MTuneAdapter) else (class1 >= .5).astype(int)
    minority_prob = probabilities[:, list(model.classes_).index(positive)]
    scores = minority_metrics(y, pred, minority_prob, positive)
    return scores, pred, class1, minority_prob, time.perf_counter()-started


def cv_metrics(model, X, y, positive):
    values, _, _, _, _ = evaluate_partition(model, X, y, positive)
    return {name: values[name] for name in CV_METRICS}


CLASS_METRICS = ('recall', 'specificity', 'precision', 'f1', 'f2', 'average_precision', 'roc_auc')
CV_METRICS = ('balanced_accuracy', 'recall', 'specificity', 'precision',
              'f1', 'f2', 'average_precision', 'roc_auc') + tuple(
                  f'{name}_{label}' for label in (0, 1) for name in CLASS_METRICS)


def refit_objective(cfg):
    # Preserve the old single-scoring config as a deprecated refit alias.
    objective = cfg.get('refit_metric', cfg.get('scoring', 'average_precision'))
    if objective not in ('average_precision', 'balanced_accuracy', 'f2'):
        raise ValueError('refit_metric must be average_precision, balanced_accuracy or f2')
    return objective



def parameter_grid(combo, cfg, y_train, cv):
    """Joint base/ensemble search; validate sizes against every training fold."""
    base_grid = cfg.get('grids', {}).get(combo['model'])
    if not base_grid:
        raise ValueError('Grid optimization requires a nonempty model grid')
    if combo['strategy'] != 'mtune':
        return base_grid
    grid = {'base_model__' + k: v for k, v in base_grid.items()}
    ensemble_grid = cfg.get('mtune_grid', {})
    if cfg.get('mtune', {}).get('method', 'direct') != 'ensemble':
        if ensemble_grid:
            raise ValueError('mtune_grid requires mtune.method=ensemble; direct mode has no ensemble size')
        return grid
    if not ensemble_grid:
        # Backward compatibility: old configs retain their fixed ensemble size.
        return grid
    if set(ensemble_grid) != {'n_ensemble'}:
        raise ValueError('mtune_grid supports only n_ensemble')
    values = ensemble_grid['n_ensemble']
    if not isinstance(values, list) or not values or any(type(v) is not int or v < 1 for v in values):
        raise ValueError('mtune_grid.n_ensemble must be a nonempty list of positive integers')
    min_majority = min(int(np.unique(y_train[a], return_counts=True)[1].max()) for a, _ in cv)
    if max(values) > min_majority:
        raise ValueError(f'n_ensemble grid exceeds smallest inner-training majority count ({min_majority})')
    grid['n_ensemble'] = values
    return grid


def metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
        f1=f1_score(y, pred, zero_division=0), f2=fbeta_score(y, pred, beta=2, zero_division=0), precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0), specificity=float(tn / (tn + fp)),
        mcc=matthews_corrcoef(y, pred), roc_auc=roc_auc_score(y, prob),
        average_precision=average_precision_score(y, prob), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def versions():
    result = {'python': platform.python_version()}
    for p in ('numpy', 'pandas', 'scikit-learn', 'rdkit', 'catboost', 'mtune', 'torch', 'transformers', 'joblib'):
        try:
            result[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            result[p] = None
    return result


def combinations(cfg):
    datasets = {d['name']: d for d in cfg['datasets']}
    if len(datasets) != len(cfg['datasets']):
        raise ValueError('Dataset names must be unique')
    axes = {'dataset': list(datasets), **AXES, 'seed': cfg.get('seeds', [42])}
    selection = cfg.get('matrix', {})
    for k, values in selection.items():
        if k not in axes or not isinstance(values, list) or not values:
            raise ValueError(f'Invalid matrix axis {k}')
        allowed = ['default', 'grid'] if k == 'optimization' else axes[k]
        if k != 'seed' and not set(values) <= set(allowed):
            raise ValueError(f'Unknown selection for {k}: {values}')
        axes[k] = values
    combos = cfg.get('experiments')
    if combos is None:
        combos = [dict(zip(axes, v)) for v in itertools.product(*axes.values())]
    for c in combos:
        if set(c) != set(axes):
            raise ValueError('Each explicit experiment needs dataset, feature, split, model, strategy, optimization, seed')
        for k, v in c.items():
            allowed = ['default', 'grid'] if k == 'optimization' else ({**axes, **AXES, 'dataset': list(datasets)})[k]
            if k != 'seed' and v not in allowed:
                raise ValueError(f'Invalid {k}: {v}')
    return combos


def preflight(specs, selected, root, run_dir):
    prepared, checks, conflict_tables = {}, [], []
    conflict_path = run_dir / 'label_conflicts.csv'
    columns = ['dataset', 'source_path', 'target_column', 'source_excel_row', 'source_data_row',
               'original_smiles', 'canonical_smiles', 'label']
    for name in selected:
        spec = specs[name]
        details = {'dataset': name, 'source_path': spec['path'],
                   'target_column': spec.get('target_column', 'label')}
        LOG.info('Validating dataset=%s target=%s path=%s', name, details['target_column'], spec['path'])
        try:
            data, checksum = load_data(spec, root)
            prepared[name] = data, checksum
            checks.append(dict(details, status='passed', **data.attrs.get('data_quality', {})))
            LOG.info('Dataset %s passed: %s retained rows; quality=%s', name, len(data), data.attrs.get('data_quality', {}))
        except (ValueError, OSError) as exc:
            check = dict(details, status='failed', error=str(exc))
            if isinstance(exc, LabelConflictError):
                rows = exc.rows.rename(columns={'smiles': 'canonical_smiles'}).assign(**details)
                rows['source_excel_row'] = rows['source_data_row'] + 1
                conflict_tables.append(rows[columns])
                check.update(conflict_groups=int(rows.canonical_smiles.nunique()),
                             conflicting_rows=len(rows), conflict_report=conflict_path.name)
            checks.append(check)
            LOG.error('Dataset %s failed: %s%s', name, exc,
                      f'; conflict report: {conflict_path}' if isinstance(exc, LabelConflictError) else '')
    if conflict_tables:
        pd.concat(conflict_tables, ignore_index=True).to_csv(conflict_path, index=False)
    atomic_json(run_dir / 'validation.json', {'datasets': checks,
        'conflict_row_numbering': {'source_excel_row': 'Original source table row: header=1, first data record=2',
                                  'source_data_row': 'Original source data record: first data record=1, excludes header'}})
    failed = [check['dataset'] for check in checks if check['status'] == 'failed']
    if failed:
        raise ValueError(f"Dataset validation failed for: {', '.join(failed)}. See {run_dir / 'validation.json'}"
                         + (f' and {conflict_path}' if conflict_tables else ''))
    return prepared


def run(config_path, plan=False, export_only=False, validate_only=False):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if plan:
        return _run(config_path, cfg, plan=True)
    logs = config_path.parent / cfg.get('output_dir', 'results') / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=time.strftime('run-%Y%m%d-%H%M%S-'), dir=logs))
    handler = logging.FileHandler(run_dir / 'run.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    previous_level = LOG.level
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    print(f'Run log: {handler.baseFilename}', flush=True)
    try:
        LOG.info('Starting config=%s validate_only=%s export_only=%s', config_path, validate_only, export_only)
        result = _run(config_path, cfg, export_only=export_only, validate_only=validate_only, run_dir=run_dir)
        LOG.info('Run completed')
        return result
    except Exception:
        LOG.exception('Run failed')
        raise
    finally:
        LOG.removeHandler(handler)
        LOG.setLevel(previous_level)
        handler.close()


def _run(config_path, cfg, plan=False, export_only=False, validate_only=False, run_dir=None):
    root = config_path.parent
    for spec in (cfg.get('datasets', []) if plan else []):
        for field in ('smiles_column', 'target_column'):
            if field in spec and (not isinstance(spec[field], str) or not spec[field].strip()):
                raise ValueError(f"Dataset {spec.get('name')!r}: fill {field} in the config before running")
    objective = refit_objective(cfg)
    combos = combinations(cfg)
    if plan:
        print(json.dumps(combos, indent=2)); return combos
    output = root / cfg.get('output_dir', 'results')
    output.mkdir(parents=True, exist_ok=True)
    cache = root / cfg.get('cache_dir', '.cache/features')
    specs = {d['name']: d for d in cfg['datasets']}
    prepared = preflight(specs, dict.fromkeys(c['dataset'] for c in combos), root, run_dir)
    if validate_only:
        print(f'Validated {len(prepared)} dataset(s). Report: {run_dir / "validation.json"}')
        return
    if export_only:
        for dataset, kind in dict.fromkeys((c['dataset'], c['feature']) for c in combos):
            spec = specs[dataset]
            data, data_hash = prepared[dataset]
            x = features(data, kind, cfg, cache)
            exported = export_features(data, x, kind, cfg, output, spec, data_hash)
            print(f"Exported: {output / exported['csv'].removeprefix('../')}")
        return
    environment = versions()
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(reporting.__file__).read_bytes()).hexdigest()
    reference = reporting.shared_run(output, cfg, environment, code_hash, atomic_json, digest)
    failures = []
    for combo in combos:
        started = time.perf_counter()
        data, data_hash = prepared[combo['dataset']]
        identity = dict(experiment=combo, config=cfg, data_hash=data_hash,
                        versions=environment, code_hash=code_hash)
        run_id = digest(identity)[:24]
        dest = output / reporting.experiment_folder(combo, run_id, cfg)
        focused = reporting.experiment_metadata(combo, cfg, data_hash, environment, code_hash, reference)
        LOG.info('Experiment %s: %s', run_id, combo)
        done = dest / 'result.json'
        if done.exists():
            saved = json.loads(done.read_text())
            if all((dest / n).exists() and hashlib.sha256((dest / n).read_bytes()).hexdigest() == h
                   for n, h in saved.get('artifacts', {}).items()) and saved.get('artifacts'):
                exported = saved.get('feature_export')
                if exported and (not (dest/exported['metadata']).exists() or not (dest/exported['csv']).exists() or hashlib.sha256((dest/exported['csv']).read_bytes()).hexdigest() != exported['sha256']):
                    x = features(data, combo['feature'], cfg, cache)
                    export_features(data, x, combo['feature'], cfg, output, specs[combo['dataset']], data_hash)
                LOG.info('Resumed verified experiment %s', dest.name)
                print(f'Resume: {dest.name}'); reporting.update_summaries(output); continue
        dest.mkdir(parents=True, exist_ok=True)
        done.unlink(missing_ok=True)
        atomic_json(dest / 'manifest.json', focused)
        stage = 'prepare artifacts'
        try:
            atomic_json(dest / 'data_quality.json', data.attrs.get('data_quality', {}))
            pd.DataFrame(data.attrs.get('excluded_rows', []),
                         columns=['source_data_row','smiles','label','reason']).to_csv(dest / 'excluded_rows.csv', index=False)
            stage = 'feature generation and export'
            LOG.info('Experiment %s: %s', run_id, stage)
            x = features(data, combo['feature'], cfg, cache)
            exported = export_features(data, x, combo['feature'], cfg, output, specs[combo['dataset']], data_hash)
            focused['feature_export'] = exported
            atomic_json(dest / 'feature_export.json', exported)
            atomic_json(dest / 'manifest.json', focused)
            stage = 'splitting'
            tr, te = partition(data, combo['split'], cfg.get('test_size', .2), combo['seed'])
            groups = data.scaffold.to_numpy() if combo['split'] == 'scaffold' else data.smiles.to_numpy()
            atomic_json(dest / 'split.json', {'train': tr.tolist(), 'test': te.tolist(),
                'canonical_smiles': data.smiles.tolist(), 'scaffolds': data.scaffold.tolist(),
                'source_data_rows': data.get('source_data_row', pd.Series(np.arange(len(data))+1)).tolist()})
            y = data.label.to_numpy()
            positive = positive_label(y[tr])
            model = estimator(combo, cfg)
            best_params, best_score = {}, None
            grid_search_seconds = None
            partition_timings = {}
            stage = 'fitting and tuning'
            LOG.info('Experiment %s: %s', run_id, stage)
            if combo['optimization'] == 'grid':
                cv = folds(y[tr], groups[tr], cfg.get('cv_folds', 5), combo['seed'])
                atomic_json(dest / 'cv.json', [{'train': tr[a].tolist(), 'validation': tr[b].tolist()} for a, b in cv])
                grid = parameter_grid(combo, cfg, y[tr], cv)
                search = GridSearchCV(model, grid, scoring=partial(cv_metrics, positive=positive), cv=cv,
                                      n_jobs=cfg.get('search_jobs', 1), error_score='raise', refit=objective)
                search_started = time.perf_counter()
                search.fit(x[tr], y[tr])
                grid_search_seconds = time.perf_counter()-search_started
                final_fit_seconds = float(search.refit_time_)
                winner = search.best_index_
                cv_fit = float(search.cv_results_['mean_fit_time'][winner])
                cv_score = float(search.cv_results_['mean_score_time'][winner])
                partition_timings['cv_validation_mean'] = {
                    'fit_seconds': cv_fit, 'score_seconds': cv_score,
                    'partition_elapsed_seconds': cv_fit+cv_score,
                    'timing_scope': 'mean per inner fold for selected candidate; excludes other candidates and final refit'}
                model = search.best_estimator_
                best_params, best_score = search.best_params_, float(search.best_score_)
                pd.DataFrame(search.cv_results_).to_csv(dest / 'cv_results.csv', index=False)
            else:
                fit_started = time.perf_counter()
                model.fit(x[tr], y[tr])
                final_fit_seconds = time.perf_counter()-fit_started
            stage = 'evaluation and reporting'
            test_metrics, pred, prob, minority_prob, test_score_seconds = evaluate_partition(model, x[te], y[te], positive)
            partition_timings['test'] = {'fit_seconds': 0., 'score_seconds': test_score_seconds,
                'partition_elapsed_seconds': test_score_seconds, 'timing_scope': 'held-out prediction and metric calculation only'}
            pd.DataFrame({'row': te, 'source_data_row': data.get('source_data_row', pd.Series(np.arange(len(data))+1)).iloc[te].to_numpy(), 'smiles': data.smiles.iloc[te].to_numpy(), 'y_true': y[te],
                          'y_pred': pred, 'p_class_1': prob, 'y_true_minority_positive': (y[te] == positive).astype(int),
                          'y_pred_minority_positive': (pred == positive).astype(int), 'p_minority': minority_prob}).to_csv(dest / 'predictions.csv', index=False)
            joblib.dump(model, dest / 'model.joblib')
            thresholds = None
            if isinstance(model, MTuneAdapter):
                actual = model.estimator_
                thresholds = {'minority_class': int(actual.minority_class), 'method': actual.method,
                              'values': [float(actual.threshold)] if actual.method == 'direct' else list(map(float, actual.thresholds_))}
            train_metrics, _, _, _, train_score_seconds = evaluate_partition(model, x[tr], y[tr], positive)
            partition_timings['train'] = {'fit_seconds': final_fit_seconds, 'score_seconds': train_score_seconds,
                'partition_elapsed_seconds': final_fit_seconds+train_score_seconds,
                'timing_scope': 'final full-training fit/refit plus training prediction and metric calculation'}
            cv_metrics_selected = reporting.cv_winner_metrics(dest, objective) if combo['optimization'] == 'grid' else {}
            reporting.write_metric_table(dest, train_metrics, test_metrics, cv_metrics_selected, len(tr), len(te),
                                         timings=partition_timings, positive=positive)
            atomic_json(dest / 'model_details.json', {
                'model': combo['model'], 'strategy': combo['strategy'], 'best_params': best_params,
                'resolved_model_params': (model.base_model if isinstance(model, MTuneAdapter) else model).get_params(),
                'mtune_method': model.method if isinstance(model, MTuneAdapter) else None,
                'selected_n_ensemble': model.n_ensemble if isinstance(model, MTuneAdapter) and model.method == 'ensemble' else None})
            artifacts = reporting.artifact_hashes(dest)
            atomic_json(done, dict(run_id=run_id, **focused, metrics=test_metrics,
                test_metrics=test_metrics, train_metrics=train_metrics, cv_validation_metrics=cv_metrics_selected,
                label_mapping={'positive_original_label': positive, 'negative_original_label': 1-positive,
                               'source': 'outer_training_counts', 'tie_policy': 'original_label_1'},
                refit_metric=objective,
                best_params=best_params, best_cv_score=best_score, thresholds=thresholds,
                partition_timings=partition_timings, grid_search_seconds=grid_search_seconds,
                n_train=len(tr), n_test=len(te), elapsed_seconds=time.perf_counter()-started, artifacts=artifacts))
            (dest / 'error.json').unlink(missing_ok=True)
            reporting.update_summaries(output)
            LOG.info('Completed %s', dest.name)
            print(f'Completed: {dest.name}')
        except Exception as exc:
            atomic_json(dest / 'error.json', {'type': type(exc).__name__, 'message': str(exc), 'stage': stage})
            LOG.exception('Experiment %s failed during %s: %s', run_id, stage, combo)
            failures.append(f'{run_id}: {exc}')
            print(f'Failed: {failures[-1]}', file=sys.stderr)
    reporting.update_summaries(output)
    if failures:
        raise RuntimeError(f'{len(failures)} experiment(s) failed; inspect error.json files')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?')
    parser.add_argument('--refresh-summaries', metavar='DIRECTORY', help='Refresh CSV summaries from saved results without training')
    parser.add_argument('--organize-results', metavar='DIRECTORY', help='Organize completed legacy results without retraining')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--export-features', action='store_true', help='Export shared features and labels without splitting or training')
    parser.add_argument('--validate-only', action='store_true', help='Validate selected datasets without generating features or training')
    args = parser.parse_args()
    if args.refresh_summaries:
        if args.config or args.plan or args.export_features or args.validate_only or args.organize_results:
            parser.error('--refresh-summaries must be used alone')
        directory = Path(args.refresh_summaries)
        if not directory.is_dir():
            parser.error('results directory does not exist')
        reporting.update_summaries(directory)
        print(f'Refreshed summaries in {directory.resolve()}')
        return
    if args.organize_results:
        if args.config or args.plan or args.export_features or args.validate_only:
            parser.error('--organize-results cannot be combined with config or --plan')
        organized, skipped = reporting.organize_existing(args.organize_results, atomic_json, digest)
        for message in organized + skipped:
            print(message)
        return
    if not args.config:
        parser.error('a config file is required')
    if sum((args.plan, args.export_features, args.validate_only)) > 1:
        parser.error('--plan, --export-features and --validate-only are mutually exclusive')
    run(args.config, args.plan, args.export_features, args.validate_only)


if __name__ == '__main__':
    main()
