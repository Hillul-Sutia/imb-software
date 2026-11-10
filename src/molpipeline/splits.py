"""Create grouped outer holdouts and inner CV assignments."""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


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


def split_groups(data, kind):
    """Keep cyclic scaffolds together; group acyclic molecules by canonical SMILES."""
    if kind == 'scaffold':
        return data.scaffold.where(data.scaffold.ne(''), 'ACYCLIC_' + data.smiles).to_numpy()
    return data.smiles.to_numpy()


def partition(data, kind, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('test_size must be between zero and one')
    y = data.label.to_numpy()
    groups = split_groups(data, kind)
    if kind == 'scaffold':
        splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
        train, test = next(splitter.split(np.zeros(len(y)), groups=groups))
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError('Scaffold holdout has a single class; change seed/test_size or supply more diverse data')
        return train, test
    # Select closest-size valid grouped fold, solely by size, never model performance.
    n = max(2, round(1 / fraction))
    pairs = folds(y, groups, n, seed)
    return min(pairs, key=lambda pair: abs(len(pair[1]) / len(data) - fraction))
