"""Linear workflow executor (single engine for the whole project).

Replaces BOTH legacy engines:
- the old linear ``Workflow`` (with its ``.done`` skip-flags and the
  "force-run all downstream steps" cascade that made every parameter tweak
  re-run the entire pipeline), and
- the DAG ``DAGWorkflowExecutor`` (with its per-node RunKey cache, cartesian
  upstream expansion, snapshots and change reports — far more machinery than
  this project needs).

This executor is intentionally simple:
- Steps run once, in the order they were added.
- No step is skipped or force-cascaded. Steps are assumed cheap; reuse across
  invocations comes from (a) resolving upstream inputs via the run manifest and
  (b) the feature-extract content-addressed cache (M2).
- The run manifest is loaded (if an existing ``--run-id`` is reused) or created,
  then saved after every step so partial runs leave a valid manifest.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from src.framework.run_manifest import RunManifest
from src.framework.step import Step


class Workflow:
    def __init__(self, run_id: str, appliance: str, config: Dict[str, Any]):
        self.run_id = run_id
        self.appliance = appliance
        self.config = config

        self.log_root = os.path.join("log", run_id)
        self.output_root = os.path.join("output", run_id)
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(os.path.join(self.output_root, "figure"), exist_ok=True)

        manifest_path = os.path.join(self.log_root, "run_manifest.json")
        # load_or_create keeps prior step entries when reusing a run_id, so a
        # later narrow run (e.g. --steps feature) can resolve earlier artifacts.
        self.manifest = RunManifest.load_or_create(manifest_path, run_id, appliance)

        self.steps: List[Step] = []

    # ── assembly ─────────────────────────────────────────────────

    def add(self, step: Step) -> "Workflow":
        self.steps.append(step)
        return self

    def set_variants(self, **kwargs: Any) -> None:
        self.manifest.set_variant(**kwargs)

    # ── execution ────────────────────────────────────────────────

    def _base_context(self) -> Dict[str, Any]:
        paths = self.config.get("paths", {}) or {}
        return {
            "run_id": self.run_id,
            "appliance": self.appliance,
            "log_root": self.log_root,
            "output_root": self.output_root,
            "cache_dir": paths.get("cache_dir", ".cache"),
            "config": self.config,
            "manifest": self.manifest,
            "data": {},
        }

    def run(self) -> Dict[str, Any]:
        context = self._base_context()
        variants = self.manifest.data.get("variants", {})

        print(f"\n{'=' * 64}")
        print(f"[Workflow] run_id={self.run_id}  appliance={self.appliance}")
        print(f"[Workflow] variants={variants}")
        print(f"[Workflow] steps={[s.step_type for s in self.steps]}")
        print(f"{'=' * 64}")

        for step in self.steps:
            print(f"\n[Workflow] >>> {step.step_type}  (variant={step.variant or '-'})")
            context = step.run(context)
            self.manifest.save()

        self.manifest.save()
        print(f"\n[Workflow] done. artifacts under {self.log_root}")
        print(f"[Workflow] manifest: {self.manifest.path}")
        return context
