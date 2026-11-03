"""Readable, experiment-specific result metadata and tables."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import pandas as pd


def experiment_folder(combo, run_id, cfg):
    strategy = combo['strategy']
    if strategy == 'mtune':
        strategy += '-' + cfg.get('mtune', {}).get('method', 'direct')
    values = [combo['dataset'], combo['feature'], combo['split'], combo['model'], strategy,
              combo['optimization'], 'seed-' + str(combo['seed'])]
    parts = [re.sub(r'[^A-Za-z0-9_-]+', '-', str(v)).strip('-_')[:24] or 'unnamed' for v in values]
    return '__'.join(parts) + '__' + run_id[:8]


def shared_run(output, cfg, versions, code_hash, atomic_json, digest):
    batch = digest({'config': cfg, 'versions': versions, 'code_hash': code_hash})[:24]
    relative = Path('_runs') / (batch + '.json')
    path = output / relative
    if not path.exists():
        atomic_json(path, {'config': cfg, 'versions': versions, 'code_hash': code_hash})
    return relative.as_posix()


def experiment_metadata(combo, cfg, data_hash, versions, code_hash, reference):
    model = combo['model']
    settings = {'dataset': next(d for d in cfg['datasets'] if d['name'] == combo['dataset']),
                'features': cfg.get(combo['feature'], {}), 'test_size': cfg.get('test_size', .2),
                'cv_folds': cfg.get('cv_folds', 5), 'refit_metric': cfg.get('refit_metric', cfg.get('scoring', 'average_precision')),
                'model_params': cfg.get('model_params', {}).get(model, {})}
    if combo['optimization'] == 'grid':
        settings['model_grid'] = cfg.get('grids', {}).get(model, {})
    if combo['strategy'] == 'mtune':
        settings['mtune'] = cfg.get('mtune', {})
        if combo['optimization'] == 'grid':
            settings['mtune_grid'] = cfg.get('mtune_grid', {})
    return {'experiment': combo, 'settings': settings, 'data_hash': data_hash,
            'versions': versions, 'code_hash': code_hash, 'run_config': '../' + reference}


def cv_winner_metrics(dest, objective):
    path = dest / 'cv_results.csv'
    if not path.exists():
        return {}
    scores = pd.read_csv(path)
    column = 'mean_test_' + objective
    if column not in scores:
        return {}
    row = scores.loc[scores[column].idxmax()]
    return {c[len('mean_test_'):]: float(row[c]) for c in scores.columns if c.startswith('mean_test_')}


def readable_metrics(values, positive=None):
    """Suffixes mean original input labels; legacy bare metrics mean minority-positive."""
    values = {k:v for k,v in dict(values).items() if not (k.endswith(('_0','_1')) and pd.isna(v))}
    per_class = ('recall','specificity','precision','f1','f2','average_precision','roc_auc')
    for name in per_class + ('tp','fp','fn','tn'):
        if name not in values and 'minority_'+name in values:
            values[name] = values['minority_'+name]
    if positive in (0,1):
        other = 1-positive
        for name in per_class:
            if name in values:
                values.setdefault(f'{name}_{positive}', values[name])
        if 'specificity' in values:
            values.setdefault(f'recall_{other}', values['specificity'])
        if 'recall' in values:
            values.setdefault(f'specificity_{other}', values['recall'])
        if 'roc_auc' in values:
            values.setdefault(f'roc_auc_{other}', values['roc_auc'])
        if all(k in values and pd.notna(values[k]) for k in ('tn','fp','fn','tp')):
            tn,fp,fn,tp = (float(values[k]) for k in ('tn','fp','fn','tp'))
            def ratio(a,b): return a/b if b else 0.
            for label, (t,f,n,z) in ((positive,(tp,fp,fn,tn)), (other,(tn,fn,fp,tp))):
                for name, value in {'precision':ratio(t,t+f),'recall':ratio(t,t+n),
                    'specificity':ratio(z,z+f), 'f1':ratio(2*t,2*t+f+n),
                    'f2':ratio(5*t,5*t+f+4*n),'support':t+n,
                    'tp':t,'fp':f,'fn':n,'tn':z}.items():
                    values.setdefault(f'{name}_{label}',value)
    for label in (0,1):
        for name in per_class:
            values.setdefault(f'{name}_{label}', None)
    for name in per_class + ('tp','fp','fn','tn'):
        if name in values:
            values['minority_'+name] = values.pop(name)
    return values


def cv_winner_timing(dest, objective):
    path = dest/'cv_results.csv'
    if not path.exists(): return {}
    table = pd.read_csv(path)
    if 'mean_test_'+objective not in table: return {}
    row = table.loc[table['mean_test_'+objective].idxmax()]
    if 'mean_fit_time' not in row or 'mean_score_time' not in row: return {}
    fit,score = float(row['mean_fit_time']),float(row['mean_score_time'])
    return {'fit_seconds':fit,'score_seconds':score,'partition_elapsed_seconds':fit+score,
            'timing_scope':'mean per inner fold for selected candidate; excludes other candidates and final refit'}


def timing_columns(timing):
    timing = dict(timing or {})
    seconds = timing.get('partition_elapsed_seconds')
    return {'fit_seconds': timing.get('fit_seconds'), 'score_seconds': timing.get('score_seconds'),
            'partition_elapsed_seconds': seconds,
            'partition_elapsed_minutes': seconds/60 if seconds is not None else None,
            'timing_scope':timing.get('timing_scope','not recorded for this historical partition'),
            'timing_status':'available' if seconds is not None else 'not_recorded'}


def write_metric_table(dest, train, test, cv, n_train, n_test, timings=None, positive=None):
    rows = []
    for name, values, size, note in [
        ('train', train, n_train, 'Final model evaluated on its fitting data; optimistic diagnostic'),
        ('cv_validation_mean', cv, None, 'Selected candidate mean across inner validation folds; not outer test'),
        ('test', test, n_test, 'Final model evaluated on held-out outer test data')]:
        rows.append({'partition': name, 'n_samples': size,
                     'status': 'available' if values else 'not_recorded', 'note': note,
                     **readable_metrics(values, positive), **timing_columns((timings or {}).get(name))})
    pd.DataFrame(rows).to_csv(dest / 'metrics.csv', index=False)


def artifact_hashes(dest):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir()
            if p.is_file() and p.name not in ('result.json', 'error.json')}


def save_summary(frame, path):
    try:
        frame.to_csv(path, index=False)
    except PermissionError:
        alternate = path.with_name(path.stem + '.updated.csv')
        frame.to_csv(alternate, index=False)
        print(f'{path.name} is locked or not writable; wrote {alternate.name}. Close the original file and refresh again.')


def update_summaries(output):
    index, performance, legacy_summary = [], [], []
    for path in sorted(output.glob('*/result.json')):
        try:
            r = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if 'run_id' not in r or 'experiment' not in r:
            continue
        context = {'run_id': r['run_id'], 'folder': path.parent.name, **r['experiment'],
                   'positive_original_label': r.get('label_mapping', {}).get('positive_original_label'),
                   'refit_metric': r.get('refit_metric'),
                   'grid_search_seconds': r.get('grid_search_seconds'),
                   'elapsed_seconds': r.get('elapsed_seconds'),
                   'elapsed_minutes': r['elapsed_seconds']/60 if r.get('elapsed_seconds') is not None else None}
        index.append({**context, 'status': 'completed', 'n_train': r.get('n_train'), 'n_test': r.get('n_test')})
        legacy_summary.append({**context, **r.get('metrics', {})})
        table = path.parent / 'metrics.csv'
        if table.exists():
            for row in pd.read_csv(table).to_dict('records'):
                durations = {} if row.get('partition') == 'test' else {'elapsed_seconds': None, 'elapsed_minutes': None, 'grid_search_seconds': None}
                name = row.get('partition')
                # Refresh can enrich old tables without rewriting experiment artifacts.
                timing = r.get('partition_timings', {}).get(name)
                if timing is None and name == 'cv_validation_mean':
                    timing = cv_winner_timing(path.parent, r.get('refit_metric', 'balanced_accuracy'))
                stored_times = {k: v for k,v in row.items() if k.startswith(('fit_seconds','score_seconds','partition_elapsed','timing_')) and pd.notna(v)}
                if pd.isna(row.get('partition_elapsed_seconds')):
                    stored_times = {}
                converted = readable_metrics(row, context['positive_original_label'])
                performance.append({**context, **converted, **timing_columns(timing), **stored_times, **durations})
        else:
            performance.append({**context, 'partition': 'test',
                **readable_metrics(r.get('metrics', {}), context['positive_original_label']),
                **timing_columns(r.get('partition_timings', {}).get('test'))})
    save_summary(pd.DataFrame(index, columns=list(index[0]) if index else ['run_id','folder','dataset','model','strategy','status','elapsed_seconds','elapsed_minutes']), output / 'experiments.csv')
    save_summary(pd.DataFrame(performance, columns=list(dict.fromkeys(k for row in performance for k in row)) if performance else ['run_id','folder','dataset','model','partition','status','balanced_accuracy','recall','elapsed_seconds','elapsed_minutes']), output / 'performance_summary.csv')
    save_summary(pd.DataFrame(legacy_summary, columns=list(dict.fromkeys(k for row in legacy_summary for k in row)) if legacy_summary else ['run_id','folder','dataset','model','balanced_accuracy','recall','elapsed_seconds','elapsed_minutes']), output / 'summary.csv')


def organize_existing(output, atomic_json, digest):
    """Only complete, checksum-verified legacy folders are renamed. Models are never refitted."""
    output = Path(output).resolve()
    organized, skipped = [], []
    for source in list(output.iterdir()):
        if not source.is_dir() or source.name.startswith('_') or source.name == 'features':
            continue
        done = source / 'result.json'
        if not done.exists():
            skipped.append(source.name + ': incomplete; left untouched')
            continue
        old = json.loads(done.read_text(encoding='utf-8'))
        if 'config' not in old:
            continue
        hashes = old.get('artifacts', {})
        if not hashes or any(not (source/n).is_file() or hashlib.sha256((source/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):
            skipped.append(source.name + ': checksum mismatch; left untouched')
            continue
        cfg = old['config']
        dest = output / experiment_folder(old['experiment'], old['run_id'], cfg)
        # Check final absolute paths before moving a Windows directory.
        if not source.resolve().is_relative_to(output) or not dest.resolve().is_relative_to(output):
            raise ValueError('Result path is outside the selected results directory')
        if dest.exists() and dest != source:
            skipped.append(source.name + ': destination exists; left untouched')
            continue
        reference = shared_run(output, cfg, old['versions'], old['code_hash'], atomic_json, digest)
        backup = output / '_runs' / 'legacy_metadata' / old['run_id']
        backup.mkdir(parents=True, exist_ok=True)
        for name in ('result.json', 'manifest.json'):
            if not (backup/name).exists():
                shutil.copy2(source/name, backup/name)
        if dest != source:
            source.rename(dest)
        focused = {k: v for k, v in old.items() if k != 'config'}
        metadata = experiment_metadata(old['experiment'], cfg, old['data_hash'], old['versions'], old['code_hash'], reference)
        focused.update(metadata)
        cv = cv_winner_metrics(dest, old.get('refit_metric', 'balanced_accuracy'))
        focused['test_metrics'] = old['metrics']
        focused['train_metrics'] = old.get('train_metrics', {})
        focused['cv_validation_metrics'] = cv
        atomic_json(dest/'manifest.json', metadata)
        write_metric_table(dest, focused['train_metrics'], old['metrics'], cv, old.get('n_train'), old.get('n_test'))
        atomic_json(dest/'model_details.json', {
            'model': old['experiment']['model'], 'strategy': old['experiment']['strategy'],
            'best_params': old.get('best_params', {}), 'thresholds': old.get('thresholds'),
            'note': 'Existing saved model preserved without loading or retraining'})
        focused['artifacts'] = artifact_hashes(dest)
        atomic_json(dest/'result.json', focused)
        organized.append(dest.name)
    update_summaries(output)
    return organized, skipped
