"""Explicit, versioned adapters for the v2 boundary-only experiment."""
import importlib.util
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/home/scnu202438025446/pslg-nilm')
OLD = ROOT / 'runs/segmentation_extension_20260926_v1'
THIRD = ROOT / 'nilm_experiments/third_party'


class Unsupported(ValueError):
    pass


def run(x, method, params, seed):
    np.random.seed(seed)
    random.seed(seed)
    n = len(x)
    if method == 'pelt-rbf':
        import ruptures as rpt
        if n < 10:
            raise Unsupported('PELT min_size=5 needs >=10 samples for a candidate split')
        beta = float(params.get('penalty_multiplier', 3)) * np.log(n)
        model = rpt.Pelt(model='rbf', min_size=5, jump=1).fit(x)
        cuts = model.predict(pen=beta)[:-1]
        return cuts, dict(penalty=beta, gamma=float(model.cost.gamma), min_size=5, jump=1)

    if method.startswith('fluss'):
        import stumpy
        from stumpy.floss import _cac
        w = int(params.get('window', 50))
        exclusion = 5
        if n < 2*w or n-w+1 <= 2*exclusion*w:
            raise Unsupported(f'FLUSS valid interior requires n-window+1 > 2*5*window; window={w}')
        if np.ptp(x) <= 1e-12:
            return [], dict(constant_signal=True, window=w)
        mp = stumpy.stump(x, m=w)
        if not np.isfinite(mp[:, 0].astype(float)).any():
            raise Unsupported('No finite matrix-profile matches')
        cac = _cac(mp[:, 1].astype(np.int64), L=w, excl_factor=exclusion)
        profile = np.asarray(cac, dtype=float).copy()
        cuts = []
        threshold = float(params.get('threshold', .45))
        while np.min(profile) <= threshold:
            cp = int(np.argmin(profile))
            if cp <= 0 or cp >= n:
                raise ValueError('CAC returned endpoint')
            cuts.append(cp)
            profile[max(0, cp-exclusion*w):min(len(profile), cp+exclusion*w)] = np.inf
        fixed = {}
        if params.get('fixed_k', True):
            for k in [2, 3, 5, 8, 12]:
                _, found = stumpy.fluss(mp[:, 1].astype(np.int64), L=w, n_regimes=k, excl_factor=exclusion)
                found = sorted(set(int(v) for v in found if 0 < v < n))
                fixed[str(k)] = dict(cuts=found, attained_segments=len(found)+1,
                                    requested_segments=k, attained_requested_k=len(found)+1 == k)
        return sorted(cuts), dict(window=w, threshold=threshold, excl_factor=exclusion,
                                  fixed_k=fixed, coordinate='matrix-profile subsequence start; no offset tuning')

    if method == 'clap':
        if n < 50:
            raise Unsupported('ClaSP initialization minimum 50 samples')
        # Keep the project vendored kernel instead of pip-installing claspy.
        sys.path.insert(0, str(OLD / 'code/models/time_segmentation'))
        sys.path.insert(0, str(THIRD / 'segmentation_clap'))
        from claspy.segmentation import BinaryClaSPSegmentation
        from statsmodels.tsa.stattools import adfuller
        if np.ptp(x) <= 1e-12:
            return [], dict(constant_signal=True, initializer_cuts=[], initializer_seconds=0.)
        t0 = time.perf_counter()
        pvalue = float(adfuller(x)[1])
        distance = 'znormed_euclidean_distance' if pvalue < .05 else 'euclidean_distance'
        initializer = BinaryClaSPSegmentation(distance=distance, n_jobs=1, random_state=seed)
        initial = sorted(set(int(v) for v in initializer.fit_predict(x)))
        init_seconds = time.perf_counter()-t0
        info = dict(initializer_cuts=initial, initializer_seconds=init_seconds, adf_pvalue=pvalue,
                    distance=distance, initializer='project frozen ClaSP default significance validation + paper ADF rule',
                    classifier='rocket', merge_score='cgain', n_splits=5, sample_size=1000)
        if not initial:
            return [], dict(**info, merge_skipped='single initial segment')
        from src.clap import CLaP
        model = CLaP(window_size='suss', classifier='rocket', merge_score='cgain',
                     n_splits=5, n_jobs=1, sample_size=1000, random_state=seed)
        model.fit(x, np.array(initial, dtype=int))
        return model.get_change_points().tolist(), dict(**info, effective_window=int(model.window_size),
                                                       state_labels=model.get_segment_labels().tolist())

    if method == 'e2usd':
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not torch.cuda.is_available():
            raise RuntimeError('E2USD requires the allocated Slurm GPU')
        w = int(params.get('window', 64))
        steps = int(params.get('steps', 20))
        if n < 2*w or n-w+1 < 20:
            raise Unsupported(f'E2USD requires n>=2*window and >=20 embeddings; window={w}')
        if np.ptp(x) <= 1e-12:
            return [], dict(constant_signal=True, guard='zero variance normalization is undefined')
        sys.path.insert(0, str(THIRD / 'segmentation_e2usd'))
        # The upstream pkgutil.find_module loader was removed in Python 3.12.
        # Load the unchanged submodules through importlib instead.
        import types
        import importlib
        package = types.ModuleType('networks')
        package.__path__ = [str(THIRD / 'segmentation_e2usd/E2USD/networks')]
        sys.modules['networks'] = package
        for module in ['network', 'fncc_loss', 'fncc_loss_byol']:
            setattr(package, module, importlib.import_module('networks.'+module))
        from E2USD.e2usd import E2USD
        from E2USD.adapers import E2USD_Adaper
        from E2USD.clustering import DPGMM
        params = dict(batch_size=1, channels=30, win_size=w, win_type='rect', depth=1, nb_steps=steps,
                      in_channels=1, kernel_size=3, lr=.003, out_channels=4, reduced_size=80,
                      cuda=True, gpu=0, M=20, N=4)
        model = E2USD(w, 1, E2USD_Adaper(params.copy()), DPGMM(None)).fit(x.reshape(-1, 1).copy(), w, 1)
        states = np.asarray(model.state_seq)
        assert len(states) == n
        assert model.embeddings.shape[0] == n-w+1, 'Window coverage mismatch'
        assert np.isfinite(model.embeddings).all()
        cuts = (np.flatnonzero(states[1:] != states[:-1])+1).tolist()
        return cuts, dict(window=w, step=1, steps=steps, network=params, states=int(len(np.unique(states))),
                          dp_components=20, dp_alpha=1000, dp_max_iter=1000,
                          embeddings=int(model.embeddings.shape[0]), coverage='all samples receive >=1 window vote',
                          gpu_name=torch.cuda.get_device_name(0), gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)

    if method in ['prim-glr', 'clasp-origin', 'clasp', 'none']:
        sys.path.insert(0, str(OLD / 'code'))
        spec = importlib.util.spec_from_file_location('old_segmentation', OLD / 'code/scripts/segmentation_campaign.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.segment(x, method), dict(source='frozen v1 code')
    raise ValueError(method)
