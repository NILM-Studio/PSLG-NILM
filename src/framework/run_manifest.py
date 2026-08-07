"""Run manifest: the single source of truth for artifact locations.

The manifest replaces the legacy "proactive search" pattern where each step
guessed its upstream's output directory by拼接 a ``_on_<model>_on_<segment>``
suffix string. Instead, every step records its outputs here (paths relative to
``log_root``), and downstream steps / visualize scripts resolve them by
``(step_type, key)``.

It is written to ``log/<run_id>/run_manifest.json`` and updated after each step,
so a crashed run leaves a valid partial manifest. Re-running with the same
``--run-id`` loads the existing manifest, which lets a single later step
(e.g. ``--steps feature``) find earlier artifacts on disk.

Layout (JSON)::

    {
      "run_id": "...",
      "appliance": "fridge",
      "variants": {"segment_method": "clasp", "feature_model": "detsec", ...},
      "steps": {
        "extract_active_data": {"variant": "simple", "subdir": "...",
                                "artifacts": {"segments_dir": "ExtractActiveData_simple/segments"}},
        "time_segmentation":    {"variant": "clasp", "artifacts": {"X": ".../X.npy", ...}},
        "feature_extract":      {"variant": "detsec_on_clasp",
                                 "artifacts": {"features": ".../features.npy"},
                                 "extra": {"cache_hit": false, "cache_key": "..."}},
        "time_clustering": {                          # M3: multiple tagged results
          "results": {
            "kmeans_k3": {"subdir": "...", "artifacts": {"labels": ".../cluster_labels.npy", ...}},
            "kmeans_k4": {...}
          }
        }
      }
    }
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class RunManifest:
    """In-memory representation of ``run_manifest.json``."""

    def __init__(self, path: str, run_id: Optional[str] = None,
                 appliance: Optional[str] = None):
        self.path = path
        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "appliance": appliance,
            "variants": {},
            "steps": {},
        }

    @property
    def log_root(self) -> str:
        return os.path.dirname(self.path)

    # ── construction ─────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, path: str, run_id: Optional[str] = None,
                       appliance: Optional[str] = None) -> "RunManifest":
        """Load an existing manifest at ``path``, else create a fresh one.

        Loading preserves prior step entries so a later, narrower run
        (e.g. ``--steps feature --run-id <existing>``) can still resolve the
        upstream artifacts produced earlier.
        """
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                obj = cls(path, raw.get("run_id", run_id), raw.get("appliance", appliance))
                obj.data = raw
                # tolerate older/partial manifests
                obj.data.setdefault("variants", {})
                obj.data.setdefault("steps", {})
                return obj
            except (json.JSONDecodeError, OSError):
                pass
        return cls(path, run_id, appliance)

    # ── mutation ─────────────────────────────────────────────────

    def set_variant(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if v is not None:
                self.data.setdefault("variants", {})[k] = v

    def add_step(self, step_type: str, variant: str, subdir: str,
                 artifacts: Dict[str, str], extra: Optional[Dict[str, Any]] = None) -> None:
        """Record (or overwrite) a single-result step's outputs."""
        entry: Dict[str, Any] = {
            "variant": variant,
            "subdir": subdir,
            "artifacts": artifacts or {},
        }
        if extra:
            entry["extra"] = extra
        self.data.setdefault("steps", {})[step_type] = entry

    def add_step_artifact(self, step_type: str, key: str, rel_path: str) -> None:
        """Merge ONE artifact into a step entry WITHOUT clobbering existing keys.

        Unlike ``add_step`` (which replaces the whole entry), this preserves
        previously recorded artifacts and — for time_clustering — the
        ``results`` dict. Used e.g. by the kmeans-scan diagnostic.
        """
        entry = self.data.setdefault("steps", {}).setdefault(step_type, {"artifacts": {}})
        entry.setdefault("artifacts", {})[key] = rel_path

    def add_cluster_result(self, tag: str, subdir: str,
                           artifacts: Dict[str, str],
                           extra: Optional[Dict[str, Any]] = None) -> None:
        """Record one tagged clustering result under ``time_clustering`` (M3).

        Multiple tags (e.g. ``kmeans_k3``, ``kmeans_k4``) coexist, each with its
        own labels / metrics — this is the "generate all candidate k" design.
        """
        steps = self.data.setdefault("steps", {})
        cluster_entry = steps.setdefault("time_clustering", {"results": {}})
        results = cluster_entry.setdefault("results", {})
        res: Dict[str, Any] = {"subdir": subdir, "artifacts": artifacts or {}}
        if extra:
            res["extra"] = extra
        results[tag] = res

    # ── read ─────────────────────────────────────────────────────

    def get_step(self, step_type: str) -> Optional[Dict[str, Any]]:
        return self.data.get("steps", {}).get(step_type)

    def artifact_path(self, step_type: str, key: str) -> Optional[str]:
        """Resolve a single-result step artifact to an absolute path."""
        entry = self.get_step(step_type)
        if not entry:
            return None
        rel = entry.get("artifacts", {}).get(key)
        return self._abs(rel)

    def cluster_artifact_path(self, tag: str, key: str) -> Optional[str]:
        """Resolve a tagged clustering result artifact (M3)."""
        entry = self.get_step("time_clustering") or {}
        res = (entry.get("results") or {}).get(tag)
        if not res:
            return None
        return self._abs(res.get("artifacts", {}).get(key))

    def cluster_tags(self) -> list:
        entry = self.get_step("time_clustering") or {}
        return sorted((entry.get("results") or {}).keys())

    def _abs(self, rel: Optional[str]) -> Optional[str]:
        if rel is None:
            return None
        if os.path.isabs(rel):
            return os.path.normpath(rel)
        # stored rel uses forward slashes; normalize to the host OS on read
        return os.path.normpath(os.path.join(self.log_root, rel))

    # ── persistence ──────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(self.log_root, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
