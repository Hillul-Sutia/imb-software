"""Route runtime caches without changing machine-wide settings."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile


@contextmanager
def cache_environment(config_dir, cfg):
    root = Path(
        os.environ.get('MOLPIPELINE_RUNTIME_CACHE') or \
        cfg.get('runtime_cache_dir') or '.cache/runtime'
    )

    if not root.is_absolute():
        root = Path(config_dir) / root

    root = root.resolve()

    locations = {
        'HF_HOME': root/'huggingface', 'HF_HUB_CACHE': root/'huggingface/hub',
        'HUGGINGFACE_HUB_CACHE': root/'huggingface/hub', 'HF_ASSETS_CACHE': root/'huggingface/assets',
        'HF_XET_CACHE': root/'huggingface/xet', 'TORCH_HOME': root/'torch',
        'TORCHINDUCTOR_CACHE_DIR': root/'torchinductor', 'TRITON_CACHE_DIR': root/'triton',
        'NUMBA_CACHE_DIR': root/'numba', 'MPLCONFIGDIR': root/'matplotlib',
        'CUDA_CACHE_PATH': root/'cuda', 'JOBLIB_TEMP_FOLDER': root/'joblib',
        'TEMP': root/'temp', 'TMP': root/'temp', 'TMPDIR': root/'temp',
        'XDG_CACHE_HOME': root/'xdg', 'PYTHONPYCACHEPREFIX': root/'python-bytecode',
    }
    # Older Transformers checks this legacy override before HF_HUB_CACHE.
    if 'TRANSFORMERS_CACHE' in os.environ:
        locations['TRANSFORMERS_CACHE'] = root/'huggingface/hub'

    previous = {key: os.environ.get(key) for key in locations}
    old_temp, old_bytecode = tempfile.tempdir, sys.pycache_prefix

    try:
        for key, path in locations.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
        tempfile.tempdir = str(root/'temp')
        sys.pycache_prefix = str(root/'python-bytecode')
        yield {key: str(path) for key, path in locations.items()}
        
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir, sys.pycache_prefix = old_temp, old_bytecode
