"""Discover local molecular CSVs and generate a pipeline configuration."""
import argparse
import json
import os
from pathlib import Path
import re

import pandas as pd
from rdkit import Chem, rdBase

# Immutable revision verified against the Hugging Face model API on 2026-11-03.
DEFAULT_CHEMBERTA_REVISION = "ed8a5374f2024ec8da53760af91a33fb8f6a15ff"

def detect_smiles(path):
    """Scan every row in chunks; a candidate must have valid SMILES in >=50% of rows.

    Missing, numeric and invalid entries count against the column. Only a unique
    highest-scoring candidate is selected; ties require manual selection.
    """
    valid, total = {}, 0
    with rdBase.BlockLogs():
        for frame in pd.read_csv(path, chunksize=2000):
            total += len(frame)
            for column in frame.columns:
                count = valid.setdefault(column, 0)
                for value in frame[column]:
                    if not isinstance(value, str) or not value.strip():
                        continue
                    mol = Chem.MolFromSmiles(value.strip())
                    if mol is not None and mol.GetNumAtoms() > 0:
                        count += 1
                valid[column] = count
    candidates = {c: n / total for c, n in valid.items() if total and n * 2 >= total}
    if not candidates:
        return '', candidates
    best = max(candidates.values())
    winners = [c for c, ratio in candidates.items() if ratio == best]
    return winners[0] if len(winners) == 1 else '', candidates


def discover(raw_dir, config_dir, targets=None):
    """Discover arbitrary CSVs by contents, leaving targets blank unless explicitly supplied."""
    raw_dir, config_dir = Path(raw_dir).resolve(), Path(config_dir).resolve()
    if not raw_dir.is_dir():
        raise ValueError(f'Dataset directory does not exist: {raw_dir}')
    targets = targets or {}
    datasets, notices, used, names = [], [], set(), set()
    files = sorted((p for p in raw_dir.iterdir() if p.is_file() and
                    p.name.lower().endswith(('.csv', '.csv.gz'))), key=lambda p: str(p).lower())
    for path in files:
        relative = path.relative_to(raw_dir).as_posix()
        override = targets.get(relative)
        if override is not None:
            used.add(relative)
        stem = re.sub(r'\.csv(?:\.gz)?$', '', path.name, flags=re.I)
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
            smiles, candidates = detect_smiles(path)
        except Exception as exc:
            raise ValueError(f'Cannot inspect {relative}: {exc}') from exc
        if override is not None and override not in columns:
            raise ValueError(f'{relative}: target column {override!r} is missing; columns: {columns}')
        if not smiles:
            reason = 'ambiguous top candidates' if candidates else 'no column has >=50% valid SMILES'
            notices.append(f'{relative}: {reason}; set smiles_column manually. Candidates: {candidates}')
        else:
            notices.append(f'{relative}: detected {smiles!r} ({candidates[smiles]:.1%} valid SMILES). Review before training.')
        name = stem + ('__' + override if override is not None else '')
        if name in names:
            raise ValueError(f'Duplicate dataset name {name}: use distinct file stems')
        names.add(name)
        datasets.append({'name': name, 'path': Path(os.path.relpath(path, config_dir)).as_posix(),
                         'smiles_column': smiles, 'target_column': override or '', 'invalid_smiles': 'skip',
                         'label_conflicts': 'prefer_minority', 'minority_label': 'auto'})
        if override is None:
            notices.append(f'{relative}: target_column is blank; choose the intended binary label column.')
    if set(targets) - used:
        raise ValueError(f'--target files not found: {sorted(set(targets)-used)}')
    if not datasets:
        raise ValueError('No usable CSV files found.')
    return datasets, notices


def generate(raw_dir='data/raw', output='config.grid.json', targets=None,
             preset='matrix', chemberta_revision=None, refit_metric='balanced_accuracy', mtune_method='ensemble', optimization='grid'):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose another --output filename')
    if optimization not in (None, 'default', 'grid', 'both'):
        raise ValueError('optimization must be default, grid or both')
    if mtune_method not in ('direct', 'ensemble'):
        raise ValueError('mtune_method must be direct or ensemble')
    if preset not in ('matrix', 'smoke'):
        raise ValueError('preset must be matrix or smoke')
    if refit_metric not in ('average_precision', 'balanced_accuracy', 'f2'):
        raise ValueError('Invalid refit objective')
    if chemberta_revision is not None and not re.fullmatch('[0-9a-f]{40}', chemberta_revision):
        raise ValueError('--chemberta-revision requires an immutable 40-character lowercase commit SHA')
    datasets, notices = discover(raw_dir, output.parent, targets)
    cfg = {'datasets': datasets, 'seeds': [42], 'test_size': .2, 'cv_folds': 5,
           'refit_metric': refit_metric, 'search_jobs': 1,
           'output_dir': 'results', 'cache_dir': '.cache/features', 'runtime_cache_dir': '.cache/runtime',
           'matrix': {'feature': ['morgan', 'chemberta'], 'split': ['random'],
                      'model': ['random_forest', 'catboost'], 'strategy': ['default', 'mtune'],
                      'optimization': ['default', 'grid']},
           'mtune': {'method': mtune_method},
           'morgan': {'radius': 2, 'n_bits': 2048, 'chirality': True},
           'grids': {'random_forest': {'n_estimators': [100, 300], 'max_depth': [None, 10],
                                     'min_samples_leaf': [1, 3]},
                     'catboost': {'iterations': [300, 600], 'depth': [4, 6], 'learning_rate': [.03, .1]}}}
    if mtune_method == 'ensemble':
        cfg['mtune']['n_ensemble'] = 11
        cfg['mtune_grid'] = {'n_ensemble': [3, 5, 7, 9, 11]}
    if preset == 'smoke':
        cfg['matrix'].update(split=['random'], model=['random_forest'],
                             strategy=['default'], optimization=['default'])
    if optimization is not None:
        cfg['matrix']['optimization'] = ['default', 'grid'] if optimization == 'both' else [optimization]
    if cfg['matrix']['optimization'] == ['grid']:
        cfg['mtune'].pop('n_ensemble', None)
    cfg['chemberta'] = {'model': 'DeepChem/ChemBERTa-77M-MLM',
                       'revision': chemberta_revision if chemberta_revision is not None else DEFAULT_CHEMBERTA_REVISION,
                       'batch_size': 32, 'max_length': 256}
    notices.append('Both Morgan and ChemBERTa selected. ChemBERTa requires Torch/Transformers and model weights on first use.')
    # Exclusive creation prevents overwriting a user's config even if another writer races us.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
        f.write('\n')
    count = len(datasets)
    for values in cfg['matrix'].values():
        count *= len(values)
    return cfg, notices, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/raw', help='Directory to scan (direct files only; subdirectories ignored)')
    parser.add_argument('--output', default='config.grid.json')
    parser.add_argument('--target', action='append', default=[], metavar='RELATIVE_FILE=COLUMN',
                        help='Explicit target for a file relative to data-dir; may repeat for different files')
    parser.add_argument('--preset', choices=['matrix', 'smoke'], default='matrix')
    parser.add_argument('--optimization', choices=['default', 'grid', 'both'], default='grid',
                        help='Optimization modes (default: grid; applies to either preset)')
    parser.add_argument('--mtune-method', choices=['direct', 'ensemble'], default='ensemble')
    parser.add_argument('--chemberta-revision', help='Override the bundled pinned ChemBERTa model commit SHA')
    parser.add_argument('--refit-metric', choices=['average_precision', 'balanced_accuracy', 'f2'],
                        default='balanced_accuracy')
    args = parser.parse_args()
    targets = {}
    for item in args.target:
        filename, separator, column = item.partition('=')
        if not separator or not filename or not column:
            parser.error('--target must be RELATIVE_FILE=COLUMN')
        filename = filename.replace('\\', '/')
        if filename in targets:
            parser.error(f'Duplicate --target for {filename}')
        targets[filename] = column
    try:
        cfg, notices, count = generate(args.data_dir, args.output, targets, args.preset,
                                      args.chemberta_revision, args.refit_metric, args.mtune_method, args.optimization)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(f'Created {Path(args.output).resolve()}')
    for d in cfg['datasets']:
        print(f"  {d['name']}: {d['path']} | SMILES={d['smiles_column']} | target={d['target_column']}")
    for notice in notices:
        print(notice)
    print(f'{count} experiment combinations in draft. Fill blank columns before running; full validation occurs at training.')
    print(f'Preview: python -m molpipeline "{args.output}" --plan')
    print(f'Run:     python -m molpipeline "{args.output}"')


if __name__ == '__main__':
    main()
