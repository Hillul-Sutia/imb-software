"""Read and clean molecular datasets while retaining source-row provenance."""

import hashlib
import logging
import re

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


from ._artifacts import atomic_json

LOG = logging.getLogger(__name__)


class LabelConflictError(ValueError):
    def __init__(self, rows, excluded_rows=None, input_rows=None, note=""):
        self.rows = rows
        self.excluded_rows = excluded_rows or []
        self.input_rows = input_rows
        super().__init__(f'Conflicting labels for the same canonical molecule: '
                         f'{rows.smiles.nunique()} groups, {len(rows)} source rows. {note}')


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

        records.append({
            'smiles': Chem.MolToSmiles(mol, isomericSmiles=True),
            'label': int(df[target].iloc[row]),
            'scaffold': MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
            'source_data_row': row + 1, 'original_smiles': str(value)
        })
        
    out = pd.DataFrame(records, columns=['smiles', 'label', 'scaffold', 'source_data_row', 'original_smiles'])
    conflict_policy = spec.get('label_conflicts', 'prefer_minority')

    if conflict_policy not in ('prefer_minority', 'prefer_label', 'exclude_all', 'error'):
        raise ValueError('label_conflicts must be prefer_minority, prefer_label, exclude_all or error')
    
    conflicts = out.groupby('smiles').label.transform('nunique') > 1

    audit = out.loc[conflicts].copy()
    counts = df[target].value_counts()

    detected_minority = int(counts.idxmin()) if counts.nunique() > 1 else None

    preferred = None
    preference_source = None
    majority_removed = 0

    if conflicts.any():
        if conflict_policy == 'error':
            raise LabelConflictError(audit, excluded, len(df))
        
        if conflict_policy in ('prefer_minority', 'prefer_label'):
            field = 'minority_label' if conflict_policy == 'prefer_minority' else 'preferred_label'

            preferred = spec.get(field)

            if conflict_policy == 'prefer_minority' and (preferred is None or preferred == 'auto'):
                preference_source = 'original_input_class_counts_before_filtering'
                if detected_minority is None:
                    raise LabelConflictError(audit, excluded, len(df),
                        f'Class counts are tied ({counts.to_dict()}); no automatic minority exists. '
                        'Choose error, exclude_all or an explicit preferred label.')
                preferred = detected_minority

            else:
                preference_source = 'explicit_config_label'

            if type(preferred) is not int or preferred not in (0,1):
                raise LabelConflictError(audit, excluded, len(df),
                    f'Set {field} to original label 0 or 1, or use minority_label=auto with prefer_minority.')
            
            remove = conflicts & (out.label != preferred)
        else:
            remove = conflicts

        majority_removed = int(remove.sum())
        for _, row in out.loc[remove].iterrows():
            excluded.append({'source_data_row': int(row.source_data_row), 'smiles': row.original_smiles,
                'label': int(row.label), 'reason': 'conflicting label: excluded by '+conflict_policy})
        out = out.loc[~remove]
    cleaned = out.drop_duplicates('smiles').reset_index(drop=True)
    if len(audit):
        retained_sources = set(cleaned.source_data_row)
        audit['action'] = ['retained' if row.source_data_row in retained_sources else
            ('excluded_duplicate_preferred_label' if preferred is not None and row.label == preferred else
             'excluded_conflicting_label') for _,row in audit.iterrows()]
        audit['policy'] = conflict_policy
        audit['preferred_label'] = preferred
        audit['preference_source'] = preference_source
        audit['input_count_0'] = int(counts.get(0,0))
        audit['input_count_1'] = int(counts.get(1,0))
    if cleaned.label.nunique() != 2:
        raise ValueError('Both binary classes must remain after invalid-SMILES filtering and deduplication')
    cleaned.attrs['label_conflicts'] = audit
    cleaned.attrs['excluded_rows'] = excluded
    cleaned.attrs['data_quality'] = {'input_rows': len(df), 'invalid_smiles_policy': policy,
        'invalid_smiles_removed': sum(row['reason'].startswith('RDKit') for row in excluded),
        'label_conflicts_policy': conflict_policy, 'conflict_preferred_label': preferred,
        'detected_input_minority_label': detected_minority, 'preference_source': preference_source,
        'cleaning_scope': 'dataset-level label policy before splitting; uses full input labels',
        'conflict_groups': int(audit.smiles.nunique()) if len(audit) else 0,
        'conflicting_rows_excluded': majority_removed, 'duplicate_rows_removed': len(out)-len(cleaned),
        'retained_rows': len(cleaned),
        'input_class_counts': {str(k): int(v) for k,v in df[target].value_counts().items()},
        'retained_class_counts': {str(k): int(v) for k,v in cleaned.label.value_counts().items()},
        'row_numbering': 'source_data_row is 1-based CSV data-record position, excluding header'}
    return cleaned, hashlib.sha256(path.read_bytes()).hexdigest()


def validation_dataset_name(name, used):
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip(' ._')[:64] or 'dataset'
    if safe.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))}:
        safe = 'dataset_' + safe
    candidate, number = safe, 2
    while candidate.casefold() in used:
        candidate = f'{safe}_{number:02d}'
        number += 1
    used.add(candidate.casefold())
    return candidate


def preflight(specs, selected, root, run_dir, *, loader=load_data):
    prepared, checks, used = {}, [], set()
    conflict_columns = ['dataset', 'source_path', 'target_column', 'source_excel_row', 'source_data_row',
                        'original_smiles', 'canonical_smiles', 'label']
    for name in selected:
        spec = specs[name]
        folder_name = validation_dataset_name(name, used)
        folder = run_dir / folder_name
        folder.mkdir()
        conflict_path = folder / 'label_conflicts.csv'
        details = {'dataset': name, 'source_path': spec['path'], 'report_folder': folder_name,
                   'target_column': spec.get('target_column', 'label')}
        LOG.info('Validating dataset=%s target=%s path=%s', name, details['target_column'], spec['path'])
        try:
            data, checksum = loader(spec, root)
            prepared[name] = data, checksum
            clean = data.copy()
            if 'source_data_row' not in clean:
                clean['source_data_row'] = np.arange(len(clean))+1
            if 'original_smiles' not in clean:
                clean['original_smiles'] = clean.smiles
            clean.insert(0, 'row_number', clean.source_data_row+1)
            clean = clean.rename(columns={'smiles': 'processed_smiles'})
            clean[['row_number','original_smiles','processed_smiles','label']].to_csv(folder/'cleaned.csv', index=False)
            pd.DataFrame(data.attrs.get('excluded_rows', []), columns=['source_data_row','smiles','label','reason']).assign(
                source_excel_row=lambda frame: frame.source_data_row+1).to_csv(folder/'excluded_rows.csv', index=False)
            check = dict(details, status='passed', cleaned_file=f'{folder_name}/cleaned.csv', input_sha256=checksum,
                         **data.attrs.get('data_quality', {}))
            audit = data.attrs.get('label_conflicts')
            if audit is not None and len(audit):
                rows = audit.rename(columns={'smiles':'canonical_smiles'}).assign(**details)
                rows['source_excel_row'] = rows.source_data_row+1
                rows[conflict_columns+['action','policy','preferred_label','preference_source','input_count_0','input_count_1']].to_csv(conflict_path,index=False)
                check['conflict_report'] = f'{folder_name}/label_conflicts.csv'
                LOG.warning('Dataset %s: resolved %s conflicts using %s, preferred label=%s; see %s',
                            name, rows.canonical_smiles.nunique(), spec.get('label_conflicts','prefer_minority'),
                            data.attrs['data_quality']['conflict_preferred_label'], conflict_path)
            atomic_json(folder/'data_quality.json', data.attrs.get('data_quality', {}))
            LOG.info('Dataset %s passed: %s retained rows; cleaned file=%s', name, len(data), folder/'cleaned.csv')
        except (ValueError, OSError) as exc:
            check = dict(details, status='failed', error=str(exc))
            if isinstance(exc, LabelConflictError):
                rows = exc.rows.rename(columns={'smiles': 'canonical_smiles'}).assign(**details)
                rows['source_excel_row'] = rows['source_data_row'] + 1
                rows[conflict_columns].to_csv(conflict_path, index=False)
                pd.DataFrame(exc.excluded_rows, columns=['source_data_row','smiles','label','reason']).assign(
                    source_excel_row=lambda frame: frame.source_data_row+1).to_csv(folder/'excluded_rows.csv', index=False)
                check.update(conflict_groups=int(rows.canonical_smiles.nunique()),
                             conflicting_rows=len(rows), conflict_report=f'{folder_name}/label_conflicts.csv',
                             input_rows=exc.input_rows, invalid_smiles_removed=len(exc.excluded_rows))
            LOG.error('Dataset %s failed: %s; dataset report: %s', name, exc, folder/'validation.json')
        atomic_json(folder/'validation.json', check)
        checks.append(check)
    summary = run_dir/'validation_summary.json'
    atomic_json(summary, {'datasets': checks,
        'conflict_row_numbering': {'source_excel_row': 'Original source table row: header=1, first data record=2',
                                  'source_data_row': 'Original source data record: first data record=1, excludes header'}})
    failed = [check['dataset'] for check in checks if check['status'] == 'failed']
    if failed:
        raise ValueError(f"Dataset validation failed for: {', '.join(failed)}. See {summary}; "
                         'conflict reports, when present, are in the respective dataset folders')
    return prepared
