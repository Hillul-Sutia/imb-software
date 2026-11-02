import json
import numpy as np
import pandas as pd
import pytest
import molpipeline as m
import result_reporting as r


def test_focused_metadata_and_safe_names():
    combo={'dataset':'first/unsafe','feature':'morgan','split':'scaffold','model':'random_forest',
           'strategy':'mtune','optimization':'grid','seed':42}
    cfg={'datasets':[{'name':'first/unsafe','path':'first.csv'},{'name':'other','path':'other.csv'}],
         'mtune':{'method':'ensemble'},'grids':{'random_forest':{'max_depth':[2]},'catboost':{'depth':[4]}}}
    name=r.experiment_folder(combo,'a'*24,cfg)
    assert '/' not in name and 'scaffold' in name and 'mtune-ensemble' in name
    focused=r.experiment_metadata(combo,cfg,'data',{},'code','_runs/run.json')
    assert 'config' not in focused
    assert focused['settings']['dataset']['name']=='first/unsafe'
    assert 'other.csv' not in json.dumps(focused)
    assert focused['settings']['model_grid']=={'max_depth':[2]}


def test_organize_preserves_completed_models_and_skips_incomplete(tmp_path):
    combo={'dataset':'demo','feature':'morgan','split':'random','model':'random_forest',
           'strategy':'default','optimization':'grid','seed':42}
    cfg={'datasets':[{'name':'demo','path':'demo.csv'},{'name':'other','path':'other.csv'}]}
    source=tmp_path/('a'*24);source.mkdir()
    (source/'model.joblib').write_bytes(b'preserve-model-without-loading')
    m.atomic_json(source/'manifest.json',{'config':cfg,'experiment':combo})
    pd.DataFrame({'mean_test_balanced_accuracy':[.4,.8],'mean_test_recall':[.5,.9]}).to_csv(source/'cv_results.csv',index=False)
    result={'run_id':'a'*24,'experiment':combo,'config':cfg,'data_hash':'data','versions':{},'code_hash':'code',
            'metrics':{'balanced_accuracy':.75},'refit_metric':'balanced_accuracy','n_train':80,'n_test':20,
            'artifacts':r.artifact_hashes(source)}
    m.atomic_json(source/'result.json',result)
    incomplete=tmp_path/('b'*24);incomplete.mkdir();(incomplete/'cv.json').write_text('[]')
    organized,skipped=r.organize_existing(tmp_path,m.atomic_json,m.digest)
    assert len(organized)==1 and len(skipped)==1
    dest=tmp_path/organized[0]
    assert (dest/'model.joblib').read_bytes()==b'preserve-model-without-loading'
    assert incomplete.exists() and not source.exists()
    focused=json.loads((dest/'result.json').read_text())
    assert 'config' not in focused and 'other.csv' not in json.dumps(focused)
    assert focused['train_metrics']=={}
    table=pd.read_csv(dest/'metrics.csv')
    assert table.loc[table.partition=='train','status'].iloc[0]=='not_recorded'
    assert focused['cv_validation_metrics']['balanced_accuracy']==.8
    assert all(m.hashlib.sha256((dest/n).read_bytes()).hexdigest()==h for n,h in focused['artifacts'].items())
    assert (dest/focused['run_config']).exists()
    assert len(pd.read_csv(tmp_path/'performance_summary.csv'))==3
    assert len(r.organize_existing(tmp_path,m.atomic_json,m.digest)[0])==0


def test_organize_rejects_corrupt_artifacts(tmp_path):
    source=tmp_path/'old';source.mkdir()
    m.atomic_json(source/'result.json',{'config':{},'artifacts':{'model.joblib':'invalid'}})
    organized,skipped=r.organize_existing(tmp_path,m.atomic_json,m.digest)
    assert not organized and 'checksum' in skipped[0] and source.exists()
