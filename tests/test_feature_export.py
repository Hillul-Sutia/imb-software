import json
import numpy as np
import pandas as pd
import pytest
from molpipeline import pipeline as m


@pytest.mark.parametrize('kind',['morgan','chemberta'])
def test_csv_alignment_reuse_and_corruption_repair(tmp_path,kind):
    data=pd.DataFrame({'smiles':['CCO','CCN'],'label':[0,1],'source_data_row':[1,7]})
    x=np.array([[0,1,0],[1,0,1]],dtype=np.uint8) if kind=='morgan' else np.array([[.1234567890123456,-.5,.9],[.7,.3,-.1]],dtype=np.float32)
    spec={'name':'custom','path':'raw.csv','target_column':'label'}
    reference=m.export_features(data,x,kind,{},tmp_path,spec,'input')
    dest=tmp_path/'experiment'
    path=dest/reference['csv']
    saved=pd.read_csv(path)
    assert saved.source_data_row.tolist()==[1,7]
    assert saved.smiles.tolist()==data.smiles.tolist()
    assert saved.label.tolist()==[0,1]
    assert saved.cleaned_row.tolist()==[0,1]
    assert np.allclose(saved.iloc[:,4:].to_numpy(),x,rtol=1e-7)
    assert all(c.startswith('morgan_bit_' if kind=='morgan' else 'chemberta_embedding_') for c in saved.columns[4:])
    stamp=path.stat().st_mtime_ns
    assert m.export_features(data,x,kind,{},tmp_path,spec,'input')==reference
    assert path.stat().st_mtime_ns==stamp
    path.write_text('corrupt')
    m.export_features(data,x,kind,{},tmp_path,spec,'input')
    assert len(pd.read_csv(path))==2
    changed=data.copy();changed['label']=[1,0]
    new=m.export_features(changed,x,kind,{},tmp_path,spec,'input')
    assert new['csv']!=reference['csv']
    with pytest.raises(ValueError,match='aligned'):m.export_features(data,x[:1],kind,{},tmp_path,spec,'input')


def test_feature_only_exports_without_training_or_splitting(tmp_path,monkeypatch):
    pd.DataFrame({'smiles':['CCO','CCN'],'label':[0,1]}).to_csv(tmp_path/'raw.csv',index=False)
    cfg={'datasets':[{'name':'tiny','path':'raw.csv'}],
         'matrix':{'feature':['morgan'],'model':['random_forest'],'strategy':['default'],
                   'optimization':['default'],'split':['random','scaffold']},'morgan':{'n_bits':32}}
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    def forbidden(*args,**kwargs):raise AssertionError('No split or model fitting in export-only mode')
    monkeypatch.setattr(m,'partition',forbidden);monkeypatch.setattr(m,'estimator',forbidden)
    m.run(path,export_only=True)
    csvs=list((tmp_path/'results'/'features').glob('*.csv'))
    assert len(csvs)==1
    assert pd.read_csv(csvs[0]).shape==(2,36)
    assert len(list((tmp_path/'.cache'/'features').glob('*.npy')))==1
    assert not list((tmp_path/'results').glob('*/result.json'))
