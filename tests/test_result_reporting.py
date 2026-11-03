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


def test_durations_summarized_once_per_experiment(tmp_path):
    for i,elapsed in enumerate([125.,None]):
        dest=tmp_path/str(i);dest.mkdir()
        result={'run_id':str(i),'experiment':{'dataset':'demo','model':'random_forest'},'metrics':{'accuracy':.8}}
        if elapsed is not None:result['elapsed_seconds']=elapsed
        m.atomic_json(dest/'result.json',result)
        r.write_metric_table(dest,{'accuracy':1.},{'accuracy':.8},{'balanced_accuracy':.7},80,20)
    r.update_summaries(tmp_path)
    performance=pd.read_csv(tmp_path/'performance_summary.csv')
    assert performance.elapsed_seconds.sum()==125.
    assert performance.loc[performance.partition!='test','elapsed_seconds'].isna().all()
    assert performance.loc[(performance.run_id==0)&(performance.partition=='test'),'elapsed_minutes'].iloc[0]==pytest.approx(125/60)
    index=pd.read_csv(tmp_path/'experiments.csv')
    assert index.elapsed_seconds.sum()==125.
    assert pd.read_csv(tmp_path/'summary.csv').elapsed_seconds.sum()==125.


def test_refresh_summaries_does_not_modify_experiment(tmp_path,monkeypatch):
    dest=tmp_path/'experiment';dest.mkdir()
    m.atomic_json(dest/'result.json',{'run_id':'saved','experiment':{'dataset':'demo'},
                                    'metrics':{'accuracy':.8},'elapsed_seconds':60.})
    before=(dest/'result.json').read_bytes()
    monkeypatch.setattr('sys.argv',['molpipeline','--refresh-summaries',str(tmp_path)])
    m.main()
    assert (dest/'result.json').read_bytes()==before
    table=pd.read_csv(tmp_path/'performance_summary.csv')
    assert table.elapsed_seconds.iloc[0]==60.



def test_legacy_mapping_and_unknown_partition_times(tmp_path):
    dest=tmp_path/'old';dest.mkdir()
    # Legacy confusion counts are minority-positive; the original minority is 0.
    old={'recall':.5,'specificity':.75,'precision':.5,'f1':.5,'f2':.5,
         'average_precision':.8,'roc_auc':.9,'tn':3,'fp':1,'fn':1,'tp':1}
    r.write_metric_table(dest,old,old,{'recall':.5,'specificity':.75,'precision':.5},4,4)
    m.atomic_json(dest/'result.json',{'run_id':'old','experiment':{'dataset':'old'},
        'label_mapping':{'positive_original_label':0},'metrics':old,'elapsed_seconds':50.})
    r.update_summaries(tmp_path)
    table=pd.read_csv(tmp_path/'performance_summary.csv').set_index('partition')
    assert table.loc['test','recall_0']==.5
    assert table.loc['test','recall_1']==.75
    assert table.loc['test','precision_1']==.75
    assert pd.isna(table.loc['test','average_precision_1']) if 'average_precision_1' in table else True
    assert pd.isna(table.loc['train','partition_elapsed_seconds'])
    assert table.loc['train','timing_status']=='not_recorded'
    assert 'recall' not in table


def test_locked_summary_uses_alternate_file(tmp_path,monkeypatch):
    target=tmp_path/'performance_summary.csv'
    target.write_text('original remains open')
    original=pd.DataFrame.to_csv
    def locked(self,path,*args,**kwargs):
        if str(path)==str(target):raise PermissionError('locked')
        return original(self,path,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,'to_csv',locked)
    r.save_summary(pd.DataFrame({'recall_0':[.8]}),target)
    assert target.read_text()=='original remains open'
    assert pd.read_csv(tmp_path/'performance_summary.updated.csv').recall_0.iloc[0]==.8
