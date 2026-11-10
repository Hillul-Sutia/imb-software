"""Generate, cache, and export molecular representations."""

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from ._artifacts import atomic_json, digest, versions


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

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=os.environ.get('HF_HUB_CACHE'))
        model = AutoModel.from_pretrained(model_id, revision=revision, cache_dir=os.environ.get('HF_HUB_CACHE')).to('cpu').eval()
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
