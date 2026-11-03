import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import molpipeline as m


def molecules():
    # Distinct ring sizes yield distinct Murcko groups, each with both classes.
    return pd.DataFrame([{'smiles':('C1' + 'C'*n + '1') if j == 0 else ('N1'+'C'*n+'1'),
                          'label':j % 2} for n in range(3,19) for j in range(2)])


def test_schema_duplicates_and_conflicts(tmp_path):
    p=tmp_path/'d.csv'
    pd.DataFrame({'smiles':['CCO','OCC','CCN'], 'label':[0,0,1]}).to_csv(p,index=False)
    d,_=m.load_data({'path':'d.csv'},tmp_path)
    assert len(d)==2
    pd.DataFrame({'smiles':['CCO','OCC'], 'label':[0,1]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='Conflicting'):m.load_data({'path':'d.csv'},tmp_path)


@pytest.mark.parametrize('split',['random','scaffold'])
def test_split_and_inner_cv_no_leakage(tmp_path,split):
    molecules().to_csv(tmp_path/'d.csv',index=False)
    d,_=m.load_data({'path':'d.csv'},tmp_path)
    tr,te=m.partition(d,split,.25,42)
    g=d.scaffold.to_numpy() if split=='scaffold' else d.smiles.to_numpy()
    assert not set(g[tr]) & set(g[te])
    for a,b in m.folds(d.label.to_numpy()[tr],g[tr],3,42):
        assert not set(g[tr[a]]) & set(g[tr[b]])
        assert not set(tr[a]) & set(te)
        assert not set(tr[b]) & set(te)


def test_mtune_real_package_threshold_and_grid():
    rng=np.random.RandomState(3)
    x=rng.normal(size=(90,6));y=np.array([0]*60+[1]*30)
    model=m.MTuneAdapter(RandomForestClassifier(n_estimators=5,random_state=7),method='direct').fit(x,y)
    actual=model.estimator_
    idx=list(actual.models_[0].classes_).index(actual.minority_class)
    expected=actual.models_[0].predict_proba(pd.DataFrame(x))[:,idx].mean()
    assert actual.threshold==pytest.approx(expected)
    search=GridSearchCV(m.MTuneAdapter(RandomForestClassifier(random_state=7)),
        {'base_model__n_estimators':[5,10]},scoring='balanced_accuracy',cv=3,error_score='raise')
    search.fit(x,y)
    assert len(search.predict(x))==90
    with pytest.raises(ValueError,match='unequal'):
        model.fit(x[:60],np.array([0]*30+[1]*30))


def test_mtune_ensemble_minority_zero():
    x=np.random.RandomState(1).normal(size=(60,4));y=np.array([0]*20+[1]*40)
    model=m.MTuneAdapter(RandomForestClassifier(n_estimators=5,random_state=3),method='ensemble',n_ensemble=3).fit(x,y)
    assert model.estimator_.minority_class==0
    assert len(model.estimator_.thresholds_)==3
    assert set(model.predict(x)) <= {0,1}


def test_cartesian_and_selection():
    cfg={'datasets':[{'name':str(i)} for i in range(4)]}
    assert len(m.combinations(cfg))==64
    cfg['matrix']={'optimization':['default','grid']}
    assert len(m.combinations(cfg))==128
    cfg['matrix'].update(feature=['morgan'],model=['random_forest'])
    assert len(m.combinations(cfg))==32


def test_end_to_end_resume_and_invalidation(tmp_path):
    molecules().to_csv(tmp_path/'d.csv',index=False)
    cfg={'datasets':[{'name':'demo','path':'d.csv'}], 'cv_folds':3,
         'matrix':{'feature':['morgan'],'model':['random_forest'],'strategy':['default'],
                   'split':['random','scaffold'],'optimization':['default','grid']},
         'model_params':{'random_forest':{'n_estimators':5}},
         'grids':{'random_forest':{'max_depth':[2,3]}}}
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg))
    m.run(p)
    results=list((tmp_path/'results').glob('*/result.json'))
    assert len(results)==4
    exports=[json.loads(f.read_text())['feature_export']['csv'] for f in results]
    assert len(set(exports))==1
    assert len(list((tmp_path/'results'/'features').glob('*.csv')))==1
    timestamps=[f.stat().st_mtime_ns for f in results]
    m.run(p)
    assert timestamps==[f.stat().st_mtime_ns for f in results]
    exported=results[0].parent/exports[0]
    exported.unlink()
    m.run(p)
    assert exported.exists()
    assert timestamps==[f.stat().st_mtime_ns for f in results]
    prediction=results[0].parent/'predictions.csv';prediction.write_text('corrupt')
    m.run(p)
    assert 'p_class_1' in pd.read_csv(prediction)
    cfg['seeds']=[23];p.write_text(json.dumps(cfg));m.run(p)
    assert len(list((tmp_path/'results').glob('*/result.json')))==8


def test_catboost_and_mtune_grid():
    pytest.importorskip('catboost')
    rng=np.random.RandomState(4)
    x=rng.normal(size=(60,5));y=np.array([0]*40+[1]*20)
    combo={'model':'catboost','seed':4,'strategy':'mtune'}
    model=m.estimator(combo,{'model_params':{'catboost':{'iterations':3,'depth':2}},'mtune':{'method':'ensemble','n_ensemble':3}})
    search=GridSearchCV(model,{'base_model__depth':[2,3]},cv=2,
        scoring=m.partial(m.cv_metrics,positive=1),refit='average_precision',error_score='raise')
    search.fit(x,y)
    assert search.predict_proba(x).shape==(60,2)


def test_feature_cache_and_settings(tmp_path):
    d=pd.DataFrame({'smiles':['CCO','CCN']})
    a=m.features(d,'morgan',{'morgan':{'n_bits':64}},tmp_path)
    b=m.features(d,'morgan',{'morgan':{'n_bits':64}},tmp_path)
    assert np.array_equal(a,b)
    assert len(list(tmp_path.glob('*.npy')))==1
    assert m.features(d,'morgan',{'morgan':{'n_bits':128}},tmp_path).shape==(2,128)
    assert len(list(tmp_path.glob('*.npy')))==2


@pytest.mark.parametrize('positive',[0,1])
def test_minority_mapping_continuous_scores(positive):
    y_binary=np.array([1,0,0,0,1,0])
    pred_binary=np.array([1,1,0,0,0,0])
    y=np.where(y_binary,positive,1-positive)
    pred=np.where(pred_binary,positive,1-positive)
    prob=np.array([.9,.8,.3,.2,.7,.1])
    assert m.positive_label(y)==positive
    actual=m.minority_metrics(y,pred,prob,positive)
    assert actual['recall']==pytest.approx(.5)
    assert actual['specificity']==pytest.approx(.75)
    assert actual['precision']==pytest.approx(.5)
    assert actual['f2']==pytest.approx(.5)
    assert actual['average_precision']==pytest.approx(5/6)
    assert actual['roc_auc']==pytest.approx(7/8)
    assert actual['average_precision'] != m.metrics(y_binary,pred_binary,pred_binary)['average_precision']


@pytest.mark.parametrize('objective',['average_precision','balanced_accuracy','f2'])
@pytest.mark.parametrize('strategy',['default','mtune','mtune_ensemble'])
def test_multimetric_training_only_grid(tmp_path,monkeypatch,objective,strategy):
    # Each scaffold has two majority rows and one minority row; label 0 is minority.
    data=pd.DataFrame({'smiles':[f'mol{i}' for i in range(90)],
                       'scaffold':np.repeat(np.arange(30),3),
                       'label':np.tile([1,1,0],30)})
    rng=np.random.RandomState(13)
    x=np.column_stack([np.arange(90),rng.normal(size=(90,3))])
    monkeypatch.setattr(m,'load_data',lambda spec,root:(data.copy(),'synthetic-input'))
    monkeypatch.setattr(m,'features',lambda *args:x)
    seen=[]
    original=RandomForestClassifier.fit
    def tracked_fit(self,X,y,*args,**kwargs):
        seen.append(set(np.asarray(X)[:,0].astype(int)))
        return original(self,X,y,*args,**kwargs)
    monkeypatch.setattr(RandomForestClassifier,'fit',tracked_fit)
    cfg={'datasets':[{'name':'synthetic','path':'unused.csv'}],'cv_folds':3,
         'refit_metric':objective,'matrix':{'feature':['morgan'],'split':['scaffold'],
         'model':['random_forest'],'strategy':['mtune' if strategy == 'mtune_ensemble' else strategy],'optimization':['grid']},
         'model_params':{'random_forest':{'n_estimators':4}},
         'grids':{'random_forest':{'max_depth':[1,3]}}}
    if strategy == 'mtune_ensemble':
        cfg['mtune']={'method':'ensemble','n_ensemble':11}
        cfg['mtune_grid']={'n_ensemble':[1,3]}
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg));m.run(p)
    result_path=next((tmp_path/'results').glob('*/result.json'))
    result=json.loads(result_path.read_text());dest=result_path.parent
    split=json.loads((dest/'split.json').read_text())
    cv=json.loads((dest/'cv.json').read_text())
    training=set(split['train']);test=set(split['test'])
    assert seen and all(rows <= training and not rows & test for rows in seen)
    if strategy == 'mtune_ensemble':
        selected=result['best_params']['n_ensemble']
        assert set.union(*seen[-selected:])==training
    else:
        assert seen[-1]==training
    for pair in cv:
        assert set(pair['train'])|set(pair['validation'])==training
        assert not set(data.scaffold.iloc[pair['train']]) & set(data.scaffold.iloc[pair['validation']])
    scores=pd.read_csv(dest/'cv_results.csv')
    for name in m.CV_METRICS:
        assert f'mean_test_{name}' in scores
        assert f'std_test_{name}' in scores
        assert f'split0_test_{name}' in scores
    assert result['best_cv_score']==pytest.approx(scores[f'mean_test_{objective}'].max())
    if strategy == 'mtune_ensemble':
        assert set(scores.param_n_ensemble)=={1,3}
        assert len(scores)==4
        loaded=m.joblib.load(dest/'model.joblib')
        assert loaded.n_ensemble==result['best_params']['n_ensemble']
        assert len(loaded.estimator_.models_)==loaded.n_ensemble
    assert 'config' not in result
    assert 'synthetic' in dest.name and 'scaffold' in dest.name
    assert (dest/result['run_config']).exists()
    table=pd.read_csv(dest/'metrics.csv').set_index('partition')
    assert table.loc['test','balanced_accuracy']==pytest.approx(result['metrics']['balanced_accuracy'])
    assert table.loc['train','balanced_accuracy']==pytest.approx(result['train_metrics']['balanced_accuracy'])
    assert table.loc['cv_validation_mean','balanced_accuracy']==pytest.approx(result['cv_validation_metrics']['balanced_accuracy'])
    assert (dest/'model_details.json').exists()
    for partition_name in ('train','cv_validation_mean','test'):
        assert table.loc[partition_name,'partition_elapsed_seconds'] >= 0
        assert table.loc[partition_name,'score_seconds'] >= 0
        assert table.loc[partition_name,'fit_seconds'] >= 0
    assert table.loc['test','fit_seconds']==0
    assert result['grid_search_seconds'] > 0
    assert table.loc['test','recall_0']==pytest.approx(result['metrics']['recall_0'])
    assert table.loc['test','precision_1']==pytest.approx(result['metrics']['precision_1'])
    assert 'recall' not in table.columns
    assert 'minority_recall' in table.columns
    assert (tmp_path/'results'/'performance_summary.csv').exists()
    assert result['refit_metric']==objective
    assert result['label_mapping']['positive_original_label']==0
    prediction=pd.read_csv(dest/'predictions.csv')
    assert np.allclose(prediction.p_minority,1-prediction.p_class_1)
    expected=m.minority_metrics(prediction.y_true,prediction.y_pred,prediction.p_minority,0)
    for name in m.CV_METRICS:
        assert result['metrics'][name]==pytest.approx(expected[name])
    before=result_path.stat().st_mtime_ns;m.run(p)
    assert result_path.stat().st_mtime_ns==before


def test_refit_configuration():
    assert m.refit_objective({})=='average_precision'
    assert m.refit_objective({'scoring':'balanced_accuracy'})=='balanced_accuracy'
    assert m.refit_objective({'refit_metric':'f2','scoring':'balanced_accuracy'})=='f2'
    with pytest.raises(ValueError,match='refit_metric'):m.refit_objective({'refit_metric':'accuracy'})
    assert m.positive_label(np.array([0,1]))==1



def test_ensemble_grid_validation():
    y=np.array([0]*20+[1]*10)
    cv=m.folds(y,np.arange(30),2,42)
    combo={'strategy':'mtune','model':'random_forest'}
    cfg={'grids':{'random_forest':{'max_depth':[2]}},
         'mtune':{'method':'ensemble'},'mtune_grid':{'n_ensemble':[1,3]}}
    assert m.parameter_grid(combo,cfg,y,cv)=={'base_model__max_depth':[2],'n_ensemble':[1,3]}
    for invalid in ([],[0],[True],[2.5],[100]):
        cfg['mtune_grid']['n_ensemble']=invalid
        with pytest.raises(ValueError):m.parameter_grid(combo,cfg,y,cv)
    cfg['mtune_grid']={'n_ensemble':[1,3]};cfg['mtune']['method']='direct'
    with pytest.raises(ValueError,match='direct mode'):m.parameter_grid(combo,cfg,y,cv)
    cfg.pop('mtune_grid')
    assert m.parameter_grid(combo,cfg,y,cv)=={'base_model__max_depth':[2]}
    combo['strategy']='default'
    assert m.parameter_grid(combo,cfg,y,cv)=={'max_depth':[2]}


def test_invalid_smiles_strict_and_audited_skip(tmp_path):
    bad='O=C1O[Al]23(OC1=O)(OC(=O)C(=O)O2)OC(=O)C(=O)O3'
    source=tmp_path/'d.csv'
    pd.DataFrame({'smiles':['CCO',bad,'CCN','OCC'],'label':[0,1,1,0]}).to_csv(source,index=False)
    before=source.read_bytes()
    spec={'path':'d.csv'}
    with pytest.raises(ValueError,match='CSV data row 2'):m.load_data(spec,tmp_path)
    data,_=m.load_data({**spec,'invalid_smiles':'skip'},tmp_path)
    assert source.read_bytes()==before
    assert len(data)==2
    assert data.source_data_row.tolist()==[1,3]
    assert data.attrs['excluded_rows'][0]['source_data_row']==2
    assert data.attrs['excluded_rows'][0]['smiles']==bad
    quality=data.attrs['data_quality']
    assert quality['invalid_smiles_removed']==1
    assert quality['duplicate_rows_removed']==1
    assert quality['retained_class_counts']=={'0':1,'1':1}
    with pytest.raises(ValueError,match='invalid_smiles must'):
        m.load_data({**spec,'invalid_smiles':'unknown'},tmp_path)


def test_skip_does_not_hide_class_loss_or_label_conflicts(tmp_path):
    source=tmp_path/'d.csv'
    pd.DataFrame({'smiles':['CCO','invalid'],'label':[0,1]}).to_csv(source,index=False)
    with pytest.raises(ValueError,match='Both binary classes'):
        m.load_data({'path':'d.csv','invalid_smiles':'skip'},tmp_path)
    pd.DataFrame({'smiles':['CCO','OCC','invalid'],'label':[0,1,1]}).to_csv(source,index=False)
    with pytest.raises(ValueError,match='Conflicting'):
        m.load_data({'path':'d.csv','invalid_smiles':'skip'},tmp_path)


def test_skip_audit_saved_in_experiment(tmp_path):
    df=molecules()
    df.loc[len(df)]=['not-a-smiles',1]
    df.to_csv(tmp_path/'data.csv',index=False)
    cfg={'datasets':[{'name':'audit','path':'data.csv','invalid_smiles':'skip'}],
         'matrix':{'feature':['morgan'],'split':['random'],'model':['random_forest'],
                   'strategy':['default'],'optimization':['default']},
         'model_params':{'random_forest':{'n_estimators':3}}}
    p=tmp_path/'config.json';p.write_text(json.dumps(cfg));m.run(p)
    dest=next((tmp_path/'results').glob('*/result.json')).parent
    assert len(pd.read_csv(dest/'excluded_rows.csv'))==1
    assert json.loads((dest/'data_quality.json').read_text())['invalid_smiles_removed']==1
    split=json.loads((dest/'split.json').read_text())
    assert len(split['source_data_rows'])==len(df)-1
    assert 'source_data_row' in pd.read_csv(dest/'predictions.csv')
    saved=json.loads((dest/'result.json').read_text())
    assert 'excluded_rows.csv' in saved['artifacts']



@pytest.mark.parametrize('positive',[0,1])
def test_original_class_metrics_and_evaluation_timer(monkeypatch,positive):
    from types import SimpleNamespace
    y=np.array([0,0,0,1,1,1])
    pred=np.array([0,1,0,1,1,0])
    p1=np.array([.1,.8,.3,.9,.7,.2])
    minority_prob=p1 if positive==1 else 1-p1
    actual=m.minority_metrics(y,pred,minority_prob,positive)
    for label in (0,1):
        expected=m.metrics((y==label).astype(int),(pred==label).astype(int),p1 if label==1 else 1-p1)
        for name in m.CLASS_METRICS:
            assert actual[f'{name}_{label}']==pytest.approx(expected[name])
    class FixedModel:
        classes_=np.array([1,0])  # Explicitly test probability-column alignment.
        def predict_proba(self,X):return np.column_stack([p1,1-p1])
    ticks=iter([10.,12.5])
    monkeypatch.setattr(m,'time',SimpleNamespace(perf_counter=lambda:next(ticks)))
    values,predicted,_,_,seconds=m.evaluate_partition(FixedModel(),np.zeros((6,1)),y,positive)
    assert seconds==2.5
    assert np.array_equal(predicted,pred)
    assert values['recall_0']==pytest.approx(2/3)
