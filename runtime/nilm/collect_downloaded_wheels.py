"""Preserve completed public Torch/CUDA wheels from this user's pip staging."""
import os
import shutil
import zipfile
from pathlib import Path

dest = Path(__file__).resolve().parent / 'wheelhouse'
dest.mkdir(exist_ok=True)
for folder in Path('/tmp').glob('pip-unpack-*'):
    try:
        if folder.stat().st_uid != os.getuid():
            continue
        for source in folder.glob('*.whl'):
            if not source.name.startswith(('nvidia_', 'torch-', 'triton-')):
                continue
            target = dest / source.name
            if target.exists() or not zipfile.is_zipfile(source):
                continue
            shutil.copy2(source, target)
            print(source.name, flush=True)
    except FileNotFoundError:
        continue
