"""Shared helpers for the standalone visualize scripts.

Charts are DECOUPLED from the workflow: compute steps never write figures.
Every script here is a thin CLI that resolves its inputs through the run
manifest (``log/<run_id>/run_manifest.json``) and writes figures to a unified
location: ``output/<run_id>/figure/<chart-kind>/``.

All scripts take ``--run-id`` (required) and ``--config`` (default
``config/config.yaml`` — only the ``visualization:`` block is read).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_manifest(run_id: str):
    """Load the run manifest; exits with a clear message if the run is unknown."""
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from src.framework.run_manifest import RunManifest
    path = os.path.join(PROJECT_ROOT, "log", run_id, "run_manifest.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"manifest not found: {path}\n"
            f"Run main.py with --run-id {run_id} first (any steps).")
    return RunManifest.load_or_create(path)


def load_viz_config(config_path: str = None) -> dict:
    """The ``visualization:`` block (language, noise toggle, stack counts)."""
    path = config_path or os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    viz = dict(cfg.get("visualization", {}) or {})
    viz["col_index"] = (cfg.get("time_clustering", {}) or {}).get("col_index", 2)
    return viz


def fig_dir(run_id: str, kind: str) -> str:
    """Unified figure output: output/<run_id>/figure/<kind>/ (created)."""
    d = os.path.join(PROJECT_ROOT, "output", run_id, "figure", kind)
    os.makedirs(d, exist_ok=True)
    return d


def require(manifest, step_type: str, key: str):
    """Resolve a manifest artifact or exit with a helpful message."""
    p = manifest.artifact_path(step_type, key)
    if not (p and os.path.exists(p)):
        raise SystemExit(
            f"artifact {step_type}.{key} not available for run "
            f"'{manifest.data.get('run_id')}'. Run the corresponding step first.")
    return p


def require_cluster(manifest, tag: str, key: str):
    p = manifest.cluster_artifact_path(tag, key)
    if not (p and os.path.exists(p)):
        raise SystemExit(
            f"cluster artifact {tag}.{key} not available. "
            f"Known tags: {manifest.cluster_tags()}")
    return p


def resolve_cluster_tags(manifest, tag_arg: str = None):
    """--cluster-tag value -> concrete tag list (default: ALL tags)."""
    tags = manifest.cluster_tags()
    if tag_arg:
        if tag_arg not in tags:
            raise SystemExit(f"unknown --cluster-tag '{tag_arg}'. Known tags: {tags}")
        return [tag_arg]
    if not tags:
        raise SystemExit("no clustering results in this run — run --steps cluster first.")
    return tags


def segments_dir(manifest) -> str:
    return require(manifest, "extract_active_data", "segments_dir")


# ── i18n + fonts (adapted from legacy clustering_utils) ──────────────────────

@lru_cache(maxsize=1)
def _i18n_table() -> dict:
    with open(os.path.join(os.path.dirname(__file__), "resources",
                           "cluster_visualization_i18n.json"), encoding="utf-8") as f:
        return json.load(f)


def texts(language: str) -> dict:
    lang = "zh" if str(language).lower().startswith("zh") else "en"
    return _i18n_table()[lang]


def setup_fonts(language: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if str(language).lower().startswith("zh"):
        import matplotlib.font_manager as fm
        for name in ("SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei", "Noto Sans CJK SC"):
            if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                break
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
