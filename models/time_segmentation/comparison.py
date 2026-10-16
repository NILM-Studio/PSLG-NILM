"""Boundary-only adapters for the five methods in proposal experiment 5.3.

Each method receives the same one-dimensional activity power series and returns
internal, zero-based cut indices. Unsupported inputs raise ``Unsupported``;
they must never be recorded as a successful no-cut result.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import sys
import types
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

from models.time_segmentation.prim_glr import PrimGLRModel


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("prim-glr", "pelt-rbf", "clasp", "clap", "e2usd")


class Unsupported(ValueError):
    """The configured method cannot process this activity."""


class SafePrimGLR(PrimGLRModel):
    """The same boundary guard used by the verified nine-object PrimGLR run."""

    def _pick_boundaries(self, score, alpha, min_gap, refine_delta=2):
        proposed = super()._pick_boundaries(score, alpha, min_gap, refine_delta)
        chosen = []
        for point in sorted(proposed, key=lambda index: (-score[index], index)):
            if point < 2 or len(score) - point < 2:
                continue
            if all(abs(point - other) >= max(2, min_gap) for other in chosen):
                chosen.append(int(point))
        return np.array(sorted(chosen), dtype=int)


@lru_cache(maxsize=1)
def _e2usd_source() -> Path:
    requested = os.environ.get("PSLG_E2USD_SOURCE")
    candidates = [requested] if requested else [
        ROOT / "models/third_party/segmentation_e2usd",
        ROOT / "runs/20261013_multidomain_generation/executions/legacy/imported/code/vendor/segmentation_e2usd",
    ]
    manifest = json.loads((Path(__file__).with_name("e2usd_source_hashes.json")).read_text())
    for candidate in candidates:
        if candidate and (Path(candidate) / "E2USD/e2usd.py").is_file():
            source = Path(candidate).resolve()
            mismatches = []
            for name, expected in manifest["files"].items():
                path = source / "E2USD" / name
                actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.is_file() else None
                if actual != expected:
                    mismatches.append(name)
            if mismatches:
                raise RuntimeError(f"E2USD source differs from pinned author commit: {mismatches}")
            return source
    raise FileNotFoundError(
        "E2USD source unavailable; set PSLG_E2USD_SOURCE to the pinned AI4CTS/E2Usd checkout"
    )


def _import_e2usd(source: Path):
    # Upstream uses an unqualified 'networks' import. This shim only resolves
    # those unchanged modules on Python 3.12; it does not alter the algorithm.
    sys.path.insert(0, str(source))
    package = types.ModuleType("networks")
    package.__path__ = [str(source / "E2USD/networks")]
    sys.modules["networks"] = package
    for name in ("network", "fncc_loss", "fncc_loss_byol"):
        setattr(package, name, importlib.import_module("networks." + name))
    from E2USD.adapers import E2USD_Adaper
    from E2USD.clustering import DPGMM
    from E2USD.e2usd import E2USD
    return E2USD, E2USD_Adaper, DPGMM


def _robust_z(x: np.ndarray) -> np.ndarray:
    median = np.median(x)
    scale = 1.4826 * np.median(np.abs(x - median))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return np.clip((x - median) / scale, -20.0, 20.0)


def run(x, method: str, params: dict | None = None, seed: int = 0):
    """Return ``(internal_cuts, metadata)`` without hiding failures.

    ``params`` is frozen by the experiment caller. For PrimGLR it contains the
    resolved point-scale model parameters. For E2USD the default window is 64.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown comparison method {method!r}; expected {METHODS}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite, one-dimensional power series")
    params = dict(params or {})
    n = len(x)

    if method == "prim-glr":
        model = SafePrimGLR(config=params)
        cuts = model.train(x)
        info = {"source": "project PrimGLR plus verified nine-object boundary guard",
                "resolved": {key: getattr(model, key) for key in
                             ("W", "alpha", "min_gap", "n_clusters", "dur_sigma", "g_wf", "g_env")},
                "internal_random_state": 0}
    elif method == "pelt-rbf":
        if n < 10:
            raise Unsupported("PELT-RBF requires at least 10 samples with min_size=5")
        if importlib.metadata.version("ruptures") != "1.1.10":
            raise RuntimeError("PELT-RBF requires the pinned ruptures==1.1.10")
        import ruptures as rpt
        penalty = float(params.get("penalty_multiplier", 3.0)) * np.log(n)
        model = rpt.Pelt(model="rbf", min_size=5, jump=1).fit(x)
        cuts = model.predict(pen=penalty)[:-1]
        info = {"source": "ruptures.Pelt(model='rbf')", "penalty": penalty,
                "gamma": float(model.cost.gamma), "min_size": 5, "jump": 1}
    elif method in {"clap", "clasp"}:
        if n < 50:
            raise Unsupported("ClaSP's project protocol requires at least 50 samples")
        if importlib.metadata.version("claspy") != "0.2.8":
            raise RuntimeError("Clasp/CLaP require the pinned author package claspy==0.2.8")
        if method == "clap":
            if importlib.metadata.version("aeon") != "1.1.0":
                raise RuntimeError("CLaP requires pinned aeon==1.1.0")
            if importlib.metadata.version("daproli") != "0.22":
                raise RuntimeError("CLaP requires pinned daproli==0.22")
        initializer_params = params.pop("initializer_params", {}) if method == "clap" else params
        from claspy.segmentation import BinaryClaSPSegmentation
        kwargs = {"n_jobs": 1, "random_state": int(seed)}
        kwargs.update(initializer_params)
        model = BinaryClaSPSegmentation(**kwargs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            initializer_cuts = model.fit_predict(x).tolist()
        if any("2*min_seg_size" in str(item.message) for item in caught):
            raise Unsupported("ClaSP requires a longer input for its selected window size")
        info = {"source": "official claspy.BinaryClaSPSegmentation",
                "package": "claspy", "parameters": kwargs,
                "effective_window": int(model.window_size),
                "warnings": [str(item.message) for item in caught]}
        if method == "clasp":
            cuts = initializer_cuts
        else:
            # The official CLaP can only merge initial segments. With no initial
            # boundary its state sequence has one state, so no classifier is fit.
            clap_info = {"source": "official claspy.AgglomerativeCLaPDetection",
                         "initializer": info, "initializer_cuts": initializer_cuts}
            if not initializer_cuts:
                cuts = []
                clap_info.update(state_count=1, state_labels=[1],
                                 merge_skipped="single initial segment")
            else:
                from claspy.state_detection import AgglomerativeCLaPDetection
                clap_kwargs = {"n_jobs": 1, "random_state": int(seed)}
                clap_kwargs.update(params)
                clap = AgglomerativeCLaPDetection(**clap_kwargs)
                states = np.asarray(clap.fit_predict(x, change_points=np.asarray(initializer_cuts)))
                if states.shape != (n,):
                    raise RuntimeError("CLaP state sequence does not cover the input activity")
                cuts = (np.flatnonzero(states[1:] != states[:-1]) + 1).tolist()
                if cuts != clap.get_change_points().tolist():
                    raise RuntimeError("CLaP state transitions disagree with official change points")
                clap_info.update(parameters=clap_kwargs, state_count=int(len(np.unique(states))),
                                 state_labels=clap.get_segment_labels().tolist(),
                                 effective_window=int(clap.window_size))
            info = clap_info
    else:
        window = int(params.get("window", 64))
        steps = int(params.get("steps", 20))
        if window < 2 or n < 2 * window:
            raise Unsupported(f"E2USD requires n >= 2*window; n={n}, window={window}")
        import torch
        if not torch.cuda.is_available():
            raise Unsupported("E2USD requires CUDA in the selected Torch runtime")
        if np.ptp(x) <= 1e-12:
            cuts, info = [], {"constant_signal": True, "window": window}
        else:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.set_num_threads(1)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            source = _e2usd_source()
            E2USD, Adapter, DPGMM = _import_e2usd(source)
            network = {"batch_size": 1, "channels": 30, "win_size": window,
                       "win_type": "rect", "depth": 1, "nb_steps": steps,
                       "in_channels": 1, "kernel_size": 3, "lr": .003,
                       "out_channels": 4, "reduced_size": 80, "cuda": True,
                       "gpu": 0, "M": 20, "N": 4}
            torch.cuda.set_device(0)  # physical card selected by CUDA_VISIBLE_DEVICES
            model = E2USD(window, 1, Adapter(network.copy()), DPGMM(None)).fit(
                _robust_z(x).reshape(-1, 1).copy(), window, 1)
            states = np.asarray(model.state_seq)
            if len(states) != n or model.embeddings.shape[0] != n - window + 1:
                raise RuntimeError("E2USD output does not cover the input activity")
            if not np.isfinite(model.embeddings).all():
                raise RuntimeError("E2USD produced nonfinite embeddings")
            cuts = (np.flatnonzero(states[1:] != states[:-1]) + 1).tolist()
            source_hash = hashlib.sha256((source / "E2USD/e2usd.py").read_bytes()).hexdigest()
            info = {"source": str(source), "source_e2usd_py_sha256": source_hash,
                    "window": window, "steps": steps, "network": network,
                    "state_count": int(len(np.unique(states))),
                    "embeddings": int(model.embeddings.shape[0]),
                    "gpu_name": torch.cuda.get_device_name(0)}

    cuts = sorted({int(cut) for cut in cuts})
    if any(cut <= 0 or cut >= n for cut in cuts):
        raise RuntimeError(f"{method} returned an invalid endpoint cut")
    return cuts, info
