import os
from pathlib import Path
import sys
import tempfile
import pytest
from molpipeline.cache import cache_environment


def test_runtime_routes_override_old_locations_and_restore(tmp_path,monkeypatch):
    monkeypatch.setenv('HF_HOME','old-home')
    monkeypatch.setenv('TRANSFORMERS_CACHE','old-transformers')
    monkeypatch.delenv('MOLPIPELINE_RUNTIME_CACHE',raising=False)
    old_temp,old_bytecode=tempfile.tempdir,sys.pycache_prefix
    with cache_environment(tmp_path,{}) as routes:
        root=(tmp_path/'.cache/runtime').resolve()
        for key,value in routes.items():
            assert Path(value).is_relative_to(root)
            assert os.environ[key]==value
        assert Path(tempfile.gettempdir())==root/'temp'
        assert sys.pycache_prefix==str(root/'python-bytecode')
        assert os.environ['TRANSFORMERS_CACHE']==str(root/'huggingface/hub')
    assert os.environ['HF_HOME']=='old-home'
    assert os.environ['TRANSFORMERS_CACHE']=='old-transformers'
    assert tempfile.tempdir==old_temp and sys.pycache_prefix==old_bytecode


def test_environment_cache_root_overrides_config(tmp_path,monkeypatch):
    chosen=tmp_path/'chosen-cache'
    monkeypatch.setenv('MOLPIPELINE_RUNTIME_CACHE',str(chosen))
    with cache_environment(tmp_path,{'runtime_cache_dir':'ignored'}):
        assert os.environ['HF_HOME']==str(chosen/'huggingface')


def test_restore_after_exception(tmp_path,monkeypatch):
    monkeypatch.delenv('MOLPIPELINE_RUNTIME_CACHE',raising=False)
    before=os.environ.get('JOBLIB_TEMP_FOLDER')
    with pytest.raises(RuntimeError):
        with cache_environment(tmp_path,{}):raise RuntimeError('test')
    assert os.environ.get('JOBLIB_TEMP_FOLDER')==before

