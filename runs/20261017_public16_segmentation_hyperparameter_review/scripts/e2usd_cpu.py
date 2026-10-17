"""Author E2USD CPU path; identical architecture/normalization to comparison.run."""
from __future__ import annotations
import hashlib
import random
import numpy as np
from models.time_segmentation.comparison import Unsupported, _e2usd_source, _import_e2usd, _robust_z


def run_cpu(x, params, seed=0):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and len(x) and np.isfinite(x).all()
    window, steps = int(params.get('window', 64)), int(params.get('steps', 20))
    if window < 2 or len(x) < 2*window:
        raise Unsupported(f'E2USD requires n >= 2*window; n={len(x)}, window={window}')
    if np.ptp(x) <= 1e-12:
        return [], dict(constant_signal=True, window=window, device='cpu')
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    source = _e2usd_source()
    E2USD, Adapter, DPGMM = _import_e2usd(source)
    network = dict(batch_size=1, channels=30, win_size=window, win_type='rect', depth=1,
                   nb_steps=steps, in_channels=1, kernel_size=3, lr=.003, out_channels=4,
                   reduced_size=80, cuda=False, gpu=0, M=20, N=4)
    model = E2USD(window, 1, Adapter(network.copy()), DPGMM(None)).fit(
        _robust_z(x).reshape(-1, 1).copy(), window, 1)
    states = np.asarray(model.state_seq)
    assert states.shape == (len(x),) and model.embeddings.shape[0] == len(x)-window+1
    assert np.isfinite(model.embeddings).all()
    cuts = (np.flatnonzero(states[1:] != states[:-1])+1).tolist()
    return cuts, dict(source=str(source), source_e2usd_py_sha256=hashlib.sha256((source/'E2USD/e2usd.py').read_bytes()).hexdigest(),
                      window=window, steps=steps, network=network, state_count=len(np.unique(states)),
                      embeddings=len(model.embeddings), device='cpu', gpu_name=None)
