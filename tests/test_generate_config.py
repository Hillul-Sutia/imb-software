import json
import pandas as pd
import pytest
import generate_config as g
import molpipeline as m


def csv(path,columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(columns).to_csv(path,index=False,compression='infer')


def test_known_files_headers_paths_and_nonoverwrite(tmp_path):
    raw=tmp_path/'data'/'raw'
    csv(raw/'BBBP.csv',{'smiles':['CC'],'p_np':[0]})
    csv(raw/'bace.csv',{'mol':['CC'],'Class':[1]})
    csv(raw/'HIV.csv',{'smiles':['CC'],'HIV_active':[0]})
    csv(raw/'clintox.csv.gz',{'smiles':['CC'],'CT_TOX':[1],'FDA_APPROVED':[0]})
    output=tmp_path/'configs'/'matrix.json'
    cfg,_,count=g.generate(raw,output)
    assert count==32==len(m.combinations(cfg))
    assert all(d['target_column']=='' for d in cfg['datasets'])
    assert all(d['invalid_smiles']=='skip' for d in cfg['datasets'])
    assert all((output.parent/d['path']).exists() for d in cfg['datasets'])
    assert cfg['refit_metric']=='balanced_accuracy'
    original=output.read_bytes()
    with pytest.raises(FileExistsError):g.generate(raw,output)
    assert output.read_bytes()==original


def test_multitarget_requires_selection(tmp_path):
    csv(tmp_path/'muv.csv.gz',{'smiles':['CC'],'MUV-466':[1],'MUV-548':[0]})
    csv(tmp_path/'BBBP.csv',{'smiles':['CC'],'p_np':[0]})
    cfg,notices,count=g.generate(tmp_path,tmp_path/'config.json')
    assert count==16 and len(cfg['datasets'])==2
    assert all(d['target_column']=='' for d in cfg['datasets'])
    assert all(d['invalid_smiles']=='skip' for d in cfg['datasets'])
    cfg,_,count=g.generate(tmp_path,tmp_path/'selected.json',targets={'muv.csv.gz':'MUV-548'},preset='smoke')
    assert count==4
    assert next(d for d in cfg['datasets'] if d['name'].startswith('muv'))['target_column']=='MUV-548'


def test_chemberta_revision_and_objective(tmp_path):
    csv(tmp_path/'BBBP.csv',{'smiles':['CC'],'p_np':[0]})
    with pytest.raises(ValueError,match='commit SHA'):
        g.generate(tmp_path,tmp_path/'invalid.json',chemberta_revision='main')
    assert not (tmp_path/'invalid.json').exists()
    cfg,_,count=g.generate(tmp_path,tmp_path/'full.json',chemberta_revision='a'*40,refit_metric='f2')
    assert count==8==len(m.combinations(cfg))
    assert cfg['chemberta']['revision']=='a'*40 and cfg['refit_metric']=='f2'


def test_missing_or_ambiguous_inputs(tmp_path):
    with pytest.raises(ValueError,match='does not exist'):g.generate(tmp_path/'absent',tmp_path/'out.json')
    with pytest.raises(ValueError,match='No usable'):g.generate(tmp_path,tmp_path/'out.json')
    csv(tmp_path/'custom.csv',{'smiles':['CC'],'a':[1],'b':[0]})
    cfg,_,_=g.generate(tmp_path,tmp_path/'draft.json')
    assert cfg['datasets'][0]['target_column']==''
    with pytest.raises(ValueError,match='missing'):g.generate(tmp_path,tmp_path/'out.json',targets={'custom.csv':'unknown'})
    with pytest.raises(ValueError,match='not found'):g.generate(tmp_path,tmp_path/'out.json',targets={'absent.csv':'a'})


def test_subdirectories_ignored(tmp_path):
    csv(tmp_path/'BBBP.csv',{'smiles':['CC'],'p_np':[0]})
    csv(tmp_path/'copy'/'BBBP.csv',{'smiles':['CC'],'p_np':[0]})
    csv(tmp_path/'nested'/'other.csv.gz',{'smiles':['CO'],'label':[1]})
    csv(tmp_path/'top.csv.gz',{'structure':['CN'],'label':[0]})
    cfg,_,_=g.generate(tmp_path,tmp_path/'out.json')
    assert {d['name'] for d in cfg['datasets']}=={'BBBP','top'}
    assert all('/copy/' not in d['path'] and '/nested/' not in d['path'] for d in cfg['datasets'])


def test_only_nested_files_not_discovered(tmp_path):
    csv(tmp_path/'nested'/'dataset.csv',{'smiles':['CC'],'label':[0]})
    with pytest.raises(ValueError,match='No usable CSV files'):
        g.generate(tmp_path,tmp_path/'out.json')


def test_generation_preserves_inputs(tmp_path):
    p=tmp_path/'BBBP.csv';csv(p,{'smiles':['INVALID'],'p_np':[0]})
    original=p.read_bytes()
    cfg,_,_=g.generate(tmp_path,tmp_path/'out.json',preset='smoke')
    assert cfg['datasets'][0]['smiles_column']==''
    assert p.read_bytes()==original



def test_generate_ensemble_grid(tmp_path):
    csv(tmp_path/'BBBP.csv',{'smiles':['CC'],'p_np':[0]})
    cfg,_,count=g.generate(tmp_path,tmp_path/'ensemble.json',mtune_method='ensemble')
    assert cfg['mtune']['method']=='ensemble'
    assert cfg['mtune_grid']=={'n_ensemble':[3,5,7,9,11]}
    assert count==8



def test_generic_smiles_detection_half_threshold(tmp_path):
    p=tmp_path/'arbitrary.csv'
    csv(p,{'structure':['CCO','c1ccccc1','invalid',None],
           'label':[0,1,0,1], 'id':[1,2,3,4]})
    selected,candidates=g.detect_smiles(p)
    assert selected=='structure' and candidates=={'structure':.5}
    cfg,_,_=g.generate(tmp_path,tmp_path/'draft.json')
    assert cfg['datasets'][0]['smiles_column']=='structure'
    assert cfg['datasets'][0]['target_column']==''
    with pytest.raises(ValueError,match='fill target_column'):m.run(tmp_path/'draft.json',plan=True)


def test_ambiguous_columns_not_silently_selected(tmp_path):
    p=tmp_path/'ambiguous.csv'
    csv(p,{'a':['CC','CCC'],'b':['CO','CN'],'label':[0,1]})
    selected,candidates=g.detect_smiles(p)
    assert selected=='' and set(candidates)=={'a','b'}
    cfg,notices,_=g.generate(tmp_path,tmp_path/'draft.json')
    assert cfg['datasets'][0]['smiles_column']==''
    assert any('ambiguous' in x for x in notices)


def test_highest_valid_fraction_and_below_half(tmp_path):
    p=tmp_path/'test.csv'
    csv(p,{'best':['CC','CO','CN','CCC'],'less':['CC','bad','CN','bad'],
           'low':['CC','bad','bad',None]})
    selected,candidates=g.detect_smiles(p)
    assert selected=='best'
    assert candidates=={'best':1.,'less':.5}


def test_full_scan_not_just_first_chunk(tmp_path):
    p=tmp_path/'large.csv'
    csv(p,{'structure':['CC']*2000+['invalid']*3000})
    selected,candidates=g.detect_smiles(p)
    assert selected=='' and candidates=={}



@pytest.mark.parametrize('mode,expected,count',[('grid',['grid'],8),('default',['default'],8),('both',['default','grid'],16)])
def test_optimization_selection(tmp_path,mode,expected,count):
    csv(tmp_path/'custom.csv',{'structure':['CC'],'label':[0]})
    cfg,_,actual=g.generate(tmp_path,tmp_path/'selected.json',optimization=mode,
                            mtune_method='ensemble',refit_metric='balanced_accuracy')
    assert cfg['matrix']['optimization']==expected
    assert actual==count
    assert cfg['refit_metric']=='balanced_accuracy'
    assert cfg['mtune_grid']['n_ensemble']==[3,5,7,9,11]


def test_explicit_optimization_overrides_preset(tmp_path):
    csv(tmp_path/'custom.csv',{'structure':['CC'],'label':[0]})
    cfg,_,count=g.generate(tmp_path,tmp_path/'grid.json',preset='smoke',optimization='grid')
    assert cfg['matrix']['optimization']==['grid'] and count==2
    with pytest.raises(ValueError,match='optimization'):
        g.generate(tmp_path,tmp_path/'invalid.json',optimization='unknown')



def test_default_api_and_command_match_requested_settings(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv(tmp_path/'data'/'raw'/'arbitrary.csv',{'structure':['CC'],'label':[0]})
    api,_,count=g.generate()
    assert (tmp_path/'config.grid.json').exists()
    assert count==8
    assert api['mtune']=={'method':'ensemble'}
    assert api['mtune_grid']['n_ensemble']==[3,5,7,9,11]
    assert api['refit_metric']=='balanced_accuracy'
    assert api['matrix']['optimization']==['grid']
    assert api['matrix']['split']==['random']
    assert api['matrix']['feature']==['morgan','chemberta']
    assert api['chemberta']['revision']==g.DEFAULT_CHEMBERTA_REVISION
    assert api['datasets'][0]['target_column']==''
    (tmp_path/'config.grid.json').rename(tmp_path/'api.json')
    monkeypatch.setattr('sys.argv',['generate_config'])
    g.main()
    cli=json.loads((tmp_path/'config.grid.json').read_text())
    assert cli==api
    assert cli['matrix']['split']==['random']
    assert all(d['invalid_smiles']=='skip' for d in cli['datasets'])
    # Existing output is protected even with no arguments.
    with pytest.raises(SystemExit) as error:g.main()
    assert error.value.code==2
    assert json.loads((tmp_path/'config.grid.json').read_text())==api


def test_explicit_old_modes_remain_available(tmp_path):
    csv(tmp_path/'custom.csv',{'structure':['CC'],'label':[0]})
    cfg,_,count=g.generate(tmp_path,tmp_path/'explicit.json',mtune_method='direct',
                            refit_metric='average_precision',optimization='both')
    assert cfg['mtune']=={'method':'direct'}
    assert 'mtune_grid' not in cfg
    assert cfg['refit_metric']=='average_precision'
    assert count==16
    cfg,_,_=g.generate(tmp_path,tmp_path/'fixed.json',optimization='default')
    assert cfg['mtune']['n_ensemble']==11
