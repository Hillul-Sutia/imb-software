"""Coordinate validation, feature generation, training, and saved artifacts."""

import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

from . import reporting
from ._artifacts import atomic_json, digest, versions
from .cache import cache_environment
from .config import combinations, refit_objective
from .data import load_data, preflight
from .features import export_features, features
from .splits import folds, partition, split_groups
from .training import MTuneAdapter, estimator, evaluate_partition, fit_experiment, parameter_grid, positive_label

LOG = logging.getLogger(__name__)


def run(config_path, plan=False, export_only=False, validate_only=False):
    path = Path(config_path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    with cache_environment(path.parent, cfg):
        return run_with_log(path, plan, export_only, validate_only)


def run_with_log(config_path, plan=False, export_only=False, validate_only=False):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8-sig'))
    if plan:
        return _run(config_path, cfg, plan=True)
    logs = config_path.parent / cfg.get('output_dir', 'results') / 'validation'
    logs.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    number = 1
    while True:
        run_dir = logs / (timestamp if number == 1 else f'{timestamp}_{number:02d}')
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            number += 1
    handler = logging.FileHandler(run_dir / 'run.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    run_logger = logging.getLogger(__package__)
    previous_level = run_logger.level
    run_logger.addHandler(handler)
    run_logger.setLevel(logging.INFO)
    print(f'Run log: {handler.baseFilename}', flush=True)
    try:
        atomic_json(run_dir/'cache_locations.json', {key: os.environ.get(key) for key in (
            'HF_HOME','HF_HUB_CACHE','HF_XET_CACHE','TORCH_HOME','JOBLIB_TEMP_FOLDER',
            'TEMP','PYTHONPYCACHEPREFIX','UV_CACHE_DIR','MOLPIPELINE_FEATURE_CACHE')})
        LOG.info('Starting config=%s validate_only=%s export_only=%s', config_path, validate_only, export_only)
        result = _run(config_path, cfg, export_only=export_only, validate_only=validate_only, run_dir=run_dir)
        LOG.info('Run completed')
        return result
    except Exception:
        LOG.exception('Run failed')
        raise
    finally:
        run_logger.removeHandler(handler)
        run_logger.setLevel(previous_level)
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
    cache = Path(os.environ.get('MOLPIPELINE_FEATURE_CACHE') or root / cfg.get('cache_dir', '.cache/features'))
    specs = {d['name']: d for d in cfg['datasets']}
    prepared = preflight(specs, dict.fromkeys(c['dataset'] for c in combos), root, run_dir, loader=load_data)
    if validate_only:
        print(f'Validated {len(prepared)} dataset(s). Report: {run_dir / "validation_summary.json"}')
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
    source_files = sorted(Path(__file__).parent.glob('*.py'))
    code_hash = digest([(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in source_files])
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
            groups = split_groups(data, combo['split'])
            atomic_json(dest / 'split.json', {'train': tr.tolist(), 'test': te.tolist(),
                'canonical_smiles': data.smiles.tolist(), 'scaffolds': data.scaffold.tolist(), 'split_groups': groups.tolist(),
                'source_data_rows': data.get('source_data_row', pd.Series(np.arange(len(data))+1)).tolist()})
            y = data.label.to_numpy()
            positive = positive_label(y[tr])
            model = estimator(combo, cfg)
            partition_timings = {}
            stage = 'fitting and tuning'
            LOG.info('Experiment %s: %s', run_id, stage)
            cv = grid = None
            if combo['optimization'] == 'grid':
                cv = folds(y[tr], groups[tr], cfg.get('cv_folds', 5), combo['seed'])
                atomic_json(dest / 'cv.json', [{'train': tr[a].tolist(), 'validation': tr[b].tolist()} for a, b in cv])
                grid = parameter_grid(combo, cfg, y[tr], cv)
            fitted = fit_experiment(model, x[tr], y[tr], positive, objective,
                                    cfg.get('search_jobs', 1), cv=cv, grid=grid)
            model = fitted.model
            best_params, best_score = fitted.best_params, fitted.best_score
            final_fit_seconds = fitted.fit_seconds
            grid_search_seconds = fitted.grid_search_seconds
            if fitted.cv_results is not None:
                partition_timings['cv_validation_mean'] = fitted.cv_timing
                pd.DataFrame(fitted.cv_results).to_csv(dest / 'cv_results.csv', index=False)
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
