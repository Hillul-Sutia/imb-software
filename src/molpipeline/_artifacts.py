"""Common artifact I/O and provenance helpers."""

import hashlib
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def versions():
    result = {'python': platform.python_version()}
    for p in ('numpy', 'pandas', 'scikit-learn', 'rdkit', 'catboost', 'mtune', 'torch', 'transformers', 'joblib'):
        try:
            result[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            result[p] = None
    return result
