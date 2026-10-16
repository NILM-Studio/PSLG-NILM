"""Protocol checks for the four-method experiment 5.3 adapter."""

import numpy as np
import pytest

from models.time_segmentation.comparison import METHODS, Unsupported, _e2usd_source, run


def test_registered_methods_exclude_legacy_controls():
    assert METHODS == ("prim-glr", "pelt-rbf", "clasp", "e2usd")
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
        run(np.ones(127), "e2usd")


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
