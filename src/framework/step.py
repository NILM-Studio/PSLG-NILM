"""Step base class for the PSLG-NILM-ADVANCED linear workflow.

Design changes vs. the legacy project (see docs/refactor notes):
- ``step_type`` is a stable lowercase machine id used as the manifest key and the
  path-resolution key (e.g. ``"feature_extract"``). It never changes with the model.
- ``log_subdir()`` is the human-readable on-disk folder (e.g.
  ``FeatureExtract_detsec_on_clasp``). Each step may override it for the
  ``_on_<model>_on_<segment>`` chains.
- Steps do NOT cache themselves (only FeatureExtract caches, via M2). They are
  treated as cheap and re-run on every execution. Reuse across invocations is
  achieved by resolving inputs from the run manifest, not by skip-flags.
- Steps do NOT generate figures. They write only data artifacts and register
  their paths in the manifest; chart generation lives in ``src/utils/visualize_*``.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class Step(ABC):
    """Base class for all workflow steps."""

    #: stable lowercase id (manifest key / resolver key); subclasses set this.
    step_type: str = "step"

    def __init__(self, variant: str = ""):
        self.variant = variant or ""

    @abstractmethod
    def run(self, context: dict) -> dict:
        """Execute the step. Returns the (mutated) context dict."""
        raise NotImplementedError

    # ── paths ────────────────────────────────────────────────────

    def log_subdir(self) -> str:
        """Human-readable folder name under ``log_root`` for this step's outputs.

        Default: ``{step_type}_{variant}``. Override for the multi-level
        dependency chains (e.g. FeatureExtract, TimeClustering).
        """
        return f"{self.step_type}_{self.variant}" if self.variant else self.step_type

    def log_dir(self, context: dict) -> str:
        """Absolute path to this step's output directory (created if missing)."""
        d = os.path.join(context["log_root"], self.log_subdir())
        os.makedirs(d, exist_ok=True)
        return d

    def rel(self, context: dict, abs_path: str) -> str:
        """Express an absolute path as a path relative to ``log_root``.

        Manifest artifact paths are stored relative to ``log_root`` so the
        manifest stays portable across machines / run roots. Separators are
        normalized to forward slashes so the JSON is machine-independent;
        ``RunManifest._abs`` re-normalizes on read.
        """
        return os.path.relpath(abs_path, context["log_root"]).replace(os.sep, "/")

    # ── manifest helpers ─────────────────────────────────────────

    def record(self, context: dict, artifacts: dict, extra: Optional[dict] = None) -> None:
        """Register this step's outputs in the run manifest.

        ``artifacts`` maps a logical key (e.g. ``"features"``) to a path relative
        to ``log_root``. Downstream steps and visualize scripts resolve these via
        ``Step.resolve(context, step_type, key)``.
        """
        context["manifest"].add_step(
            step_type=self.step_type,
            variant=self.variant,
            subdir=self.log_subdir(),
            artifacts=artifacts,
            extra=extra,
        )

    @staticmethod
    def resolve(context: dict, step_type: str, key: str) -> Optional[str]:
        """Return the absolute on-disk path of an artifact, or ``None``.

        Looks up the run manifest. This is how a step run in isolation finds the
        outputs of an earlier step (e.g. ``--steps feature`` finds segmentation's
        ``X.npy``) without string-convention path guessing.
        """
        manifest = context.get("manifest")
        if manifest is None:
            return None
        return manifest.artifact_path(step_type, key)
