"""Classifier construction, fitting, tuning, and evaluation."""

from dataclasses import dataclass
from functools import partial
import importlib.metadata as metadata
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, fbeta_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV


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



@dataclass
class FitResult:
    model: object
    best_params: dict
    best_score: float | None
    fit_seconds: float
    grid_search_seconds: float | None = None
    cv_timing: dict | None = None
    cv_results: dict | None = None


def fit_experiment(model, X, y, positive, objective, jobs, cv=None, grid=None):
    """Fit only on the supplied outer training data."""
    started = time.perf_counter()
    if cv is None:
        model.fit(X, y)
        return FitResult(model, {}, None, time.perf_counter()-started)
    search = GridSearchCV(model, grid, scoring=partial(cv_metrics, positive=positive), cv=cv,
                          n_jobs=jobs, error_score='raise', refit=objective)
    started = time.perf_counter()
    search.fit(X, y)
    duration = time.perf_counter()-started
    winner = search.best_index_
    cv_fit = float(search.cv_results_['mean_fit_time'][winner])
    cv_score = float(search.cv_results_['mean_score_time'][winner])
    timing = {'fit_seconds': cv_fit, 'score_seconds': cv_score,
              'partition_elapsed_seconds': cv_fit+cv_score,
              'timing_scope': 'mean per inner fold for selected candidate; excludes other candidates and final refit'}
    return FitResult(search.best_estimator_, search.best_params_, float(search.best_score_),
                     float(search.refit_time_), duration, timing, search.cv_results_)
