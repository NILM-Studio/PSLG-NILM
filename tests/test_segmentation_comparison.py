"""Protocol checks for the six-method experiment 5.3 adapter."""

import importlib.metadata

import numpy as np
import pytest

from models.time_segmentation.comparison import METHODS, Unsupported, _e2usd_source, run


def test_registered_methods_exclude_legacy_controls():
    assert METHODS == ("prim-glr", "pelt-rbf", "fluss-threshold", "clasp", "clap", "e2usd")
    with pytest.raises(ValueError):
        run(np.arange(40, dtype=float), "none")


def test_prim_glr_finds_known_step():
    x = np.r_[np.full(120, 20.0), np.full(120, 150.0)]
    prim, prim_info = run(x, "prim-glr")
    assert prim == [120]
    assert prim_info["internal_random_state"] == 0


def test_pelt_finds_known_step():
    pytest.importorskip("ruptures")
    x = np.r_[np.full(120, 20.0), np.full(120, 150.0)]
    pelt, info = run(x, "pelt-rbf")
    assert pelt == [120]
    assert info["min_size"] == 5


def test_short_activity_is_not_a_successful_zero_cut():
    with pytest.raises(Unsupported):
        run(np.ones(9), "pelt-rbf")
    with pytest.raises(Unsupported):
        run(np.ones(49), "clasp")
    with pytest.raises(Unsupported):
        run(np.ones(49), "clap")
    with pytest.raises(Unsupported):
        run(np.ones(127), "e2usd")
    with pytest.raises(Unsupported):
        run(np.ones(549), "fluss-threshold")


def test_fluss_threshold_uses_official_corrected_arc_curve():
    try:
        importlib.metadata.version("stumpy")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("STUMPY is installed in the isolated segmentation overlay")
    x = np.r_[np.sin(np.arange(300) * .15), np.cos(np.arange(300) * .08)]
    window, exclusion, threshold = 20, 5, .45
    cuts, info = run(x, "fluss-threshold", params={"window": window,
                                                   "threshold": threshold,
                                                   "excl_factor": exclusion})
    import stumpy
    mp = stumpy.stump(x, m=window)
    cac, fixed_k = stumpy.fluss(mp[:, 1].astype(int), L=window,
                               n_regimes=2, excl_factor=exclusion)
    direct = np.asarray(cac, dtype=float).copy()
    expected = []
    while direct.min() <= threshold:
        cut = int(direct.argmin())
        expected.append(cut)
        direct[max(0, cut - exclusion * window):
               min(len(direct), cut + exclusion * window)] = np.inf
    assert cuts == sorted(expected)
    assert info["cac_min"] == pytest.approx(float(np.min(cac)))
    assert info["n_regimes_for_cac"] == 2
    assert info["adaptation"].startswith("project CAC threshold")
    assert len(fixed_k) == 1


def test_e2usd_vendor_matches_pinned_author_source():
    assert (_e2usd_source() / "E2USD/e2usd.py").is_file()


def test_clasp_calls_official_package():
    pytest.importorskip("claspy.segmentation")
    from claspy.segmentation import BinaryClaSPSegmentation
    x = np.r_[np.ones(120) * 20, np.ones(120) * 150]
    cuts, info = run(x, "clasp", params={"window_size": 10})
    direct = BinaryClaSPSegmentation(window_size=10, n_jobs=1, random_state=0)
    assert cuts == direct.fit_predict(x).tolist()
    assert info["source"] == "official claspy.BinaryClaSPSegmentation"


def test_clap_matches_official_state_detector_on_fixed_initializer(monkeypatch):
    pytest.importorskip("aeon")
    from claspy import segmentation
    from claspy.state_detection import AgglomerativeCLaPDetection

    class FixedInitializer:
        window_size = 10

        def __init__(self, **kwargs):
            pass

        def fit_predict(self, x):
            return np.array([80, 160])

    monkeypatch.setattr(segmentation, "BinaryClaSPSegmentation", FixedInitializer)
    x = np.r_[np.full(80, 20.0), np.full(80, 150.0), np.full(80, 20.0)]
    cuts, info = run(x, "clap", params={"classifier": "dummy"}, seed=0)
    official = AgglomerativeCLaPDetection(classifier="dummy", n_jobs=1, random_state=0)
    states = official.fit_predict(x, change_points=np.array([80, 160]))
    assert cuts == (np.flatnonzero(states[1:] != states[:-1]) + 1).tolist()
    assert info["initializer_cuts"] == [80, 160]
    assert info["state_count"] == len(np.unique(states))
