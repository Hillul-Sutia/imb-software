import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from molpipeline.splits import partition, split_groups


def test_scaffold_outer_matches_script_and_keeps_cyclic_groups_together():
    data = pd.DataFrame({
        'smiles': [f'molecule-{i}' for i in range(24)],
        'scaffold': [''] * 12 + ['A'] * 4 + ['B'] * 4 + ['C'] * 4,
        'label': [0, 1] * 12,
    })
    groups = split_groups(data, 'scaffold')
    assert list(groups[:12]) == ['ACYCLIC_' + smiles for smiles in data.smiles[:12]]
    expected = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
                    .split(data, groups=groups))
    actual = partition(data, 'scaffold', .25, 42)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert set(groups[actual[0]]).isdisjoint(groups[actual[1]])


def test_scaffold_outer_rejects_single_class_partitions():
    data = pd.DataFrame({'smiles': ['A1', 'A2', 'B1', 'B2'],
                         'scaffold': ['A', 'A', 'B', 'B'], 'label': [0, 0, 1, 1]})
    with pytest.raises(ValueError, match='Scaffold holdout has a single class'):
        partition(data, 'scaffold', .5, 42)
