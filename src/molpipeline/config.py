"""Experiment configuration and matrix expansion."""

import itertools


AXES = {
    'feature': ['morgan', 'chemberta'], 
    'split': ['random', 'scaffold'],
    'model': ['random_forest', 'catboost'], 
    'strategy': ['default', 'mtune'],
    'optimization': ['default']
}


def refit_objective(cfg):
    # Preserve the old single-scoring config as a deprecated refit alias.
    objective = cfg.get('refit_metric', cfg.get('scoring', 'average_precision'))
    if objective not in ('average_precision', 'balanced_accuracy', 'f2'):
        raise ValueError('refit_metric must be average_precision, balanced_accuracy or f2')
    return objective


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
