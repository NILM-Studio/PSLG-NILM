"""Synthesize appliance cycles from a merged primitive-state result."""
from __future__ import annotations

import json
import os
from glob import glob
from typing import List

import numpy as np
import pandas as pd

from src.framework.step import Step
from src.generation import PrimitiveLibrary, RealPrimitiveSampler, StateTransitionModel
from src.generation.primitive_library import Primitive


class PrimitiveSynthesisStep(Step):
    """Real-primitive resampling baseline with a replaceable sampler contract."""

    step_type = "primitive_synthesis"

    def __init__(self, cluster_tag: str, sampler: str = "real_resample",
                 sequence_method: str = "empirical", n_cycles: int = 100,
                 random_seed: int = 42, min_blocks: int = 3,
                 max_blocks: int = 20, fs: float = 0.1666667):
        if not cluster_tag:
            raise ValueError("primitive synthesis requires --cluster-tag")
        super().__init__(variant=f"{sampler}_{sequence_method}_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.sampler_name = str(sampler).lower()
        self.sequence_method = str(sequence_method).lower()
        self.n_cycles = int(n_cycles)
        self.random_seed = int(random_seed)
        self.min_blocks = int(min_blocks)
        self.max_blocks = int(max_blocks)
        self.fs = float(fs)
        if self.fs <= 0:
            raise ValueError("primitive_synthesis.fs must be positive")

    def _cluster_artifact(self, context: dict, key: str) -> str:
        path = context["manifest"].cluster_artifact_path(self.cluster_tag, key)
        if not (path and os.path.exists(path)):
            raise FileNotFoundError(
                f"[primitive_synthesis] missing {self.cluster_tag}.{key}; "
                "run state_merge first and use a merged cluster tag")
        return path

    def _segments_dir(self, context: dict) -> str:
        path = self.resolve(context, "extract_active_data", "segments_dir")
        if not (path and os.path.isdir(path)):
            raise FileNotFoundError("[primitive_synthesis] activity CSV directory not found")
        return path

    def _load_library(self, context: dict) -> PrimitiveLibrary:
        labels = np.load(self._cluster_artifact(context, "labels")).reshape(-1)
        indices = np.load(self._cluster_artifact(context, "indices"))
        lengths = np.load(self._cluster_artifact(context, "seq_len")).reshape(-1)
        if not (len(labels) == len(indices) == len(lengths)):
            raise ValueError("[primitive_synthesis] cluster artifacts are not row-aligned")

        segments_dir = self._segments_dir(context)
        files = sorted(f for f in os.listdir(segments_dir)
                       if f.lower().endswith(".csv"))
        cache, primitives = {}, []
        for primitive_id, (label, index, length) in enumerate(zip(labels, indices, lengths)):
            activity_index, start = int(index[0]), int(index[1])
            if not (0 <= activity_index < len(files)) or int(length) <= 0:
                continue
            if activity_index not in cache:
                frame = pd.read_csv(os.path.join(segments_dir, files[activity_index]))
                column = "power" if "power" in frame.columns else frame.columns[-1]
                cache[activity_index] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy()
            source = cache[activity_index]
            end = min(len(source), start + int(length))
            if start < 0 or start >= end:
                continue
            primitives.append(Primitive(
                primitive_id=primitive_id, state_label=int(label),
                activity_index=activity_index, start=start,
                power=np.asarray(source[start:end], dtype=np.float32)))
        return PrimitiveLibrary(primitives)

    def _load_transition_model(self, context: dict) -> StateTransitionModel:
        path = self._cluster_artifact(context, "state_sequences")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return StateTransitionModel(payload.values())

    def _sampler(self, library: PrimitiveLibrary):
        if self.sampler_name == "real_resample":
            return RealPrimitiveSampler(library)
        raise ValueError(
            f"unknown primitive sampler '{self.sampler_name}'; supported: real_resample")

    def run(self, context: dict) -> dict:
        if self.n_cycles <= 0:
            raise ValueError("primitive_synthesis.n_cycles must be positive")
        library = self._load_library(context)
        transition_model = self._load_transition_model(context)
        sampler = self._sampler(library)
        rng = np.random.default_rng(self.random_seed)

        log_dir = self.log_dir(context)
        cycles_dir = os.path.join(log_dir, "cycles")
        os.makedirs(cycles_dir, exist_ok=True)
        for old_cycle in glob(os.path.join(cycles_dir, "synthetic_cycle_*.csv")):
            os.unlink(old_cycle)
        records: List[dict] = []

        for cycle_id in range(self.n_cycles):
            state_blocks = transition_model.sample(
                self.sequence_method, rng, self.min_blocks, self.max_blocks)
            powers, states, block_ids, block_records = [], [], [], []
            cursor = 0
            for block_id, (state, target_length) in enumerate(state_blocks):
                power, provenance = sampler.sample_block(state, target_length, rng)
                powers.append(power)
                states.append(np.full(len(power), state, dtype=np.int32))
                block_ids.append(np.full(len(power), block_id, dtype=np.int32))
                block_records.append({
                    "block_id": int(block_id), "state_label": int(state),
                    "start": int(cursor), "end": int(cursor + len(power)),
                    "length_samples": int(len(power)), "sources": provenance,
                })
                cursor += len(power)

            power = np.concatenate(powers).astype(np.float32)
            state_labels = np.concatenate(states)
            blocks = np.concatenate(block_ids)
            frame = pd.DataFrame({
                "sample_index": np.arange(len(power), dtype=np.int64),
                "time_seconds": np.arange(len(power), dtype=np.float64) / self.fs,
                "power": power,
                "state_label": state_labels,
                "block_id": blocks,
            })
            filename = f"synthetic_cycle_{cycle_id:05d}.csv"
            frame.to_csv(os.path.join(cycles_dir, filename), index=False)
            records.append({
                "cycle_id": int(cycle_id), "file": filename,
                "sequence_method": self.sequence_method,
                "state_sequence": [int(s) for s, _ in state_blocks],
                "length_samples": int(len(power)),
                "duration_seconds": float(len(power) / self.fs),
                "mean_power": float(np.mean(power)),
                "max_power": float(np.max(power)),
                "energy_wh": float(np.sum(power) / self.fs / 3600.0),
                "blocks": block_records,
            })

        def _dump(name: str, payload: dict | list) -> str:
            path = os.path.join(log_dir, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return path

        model_path = _dump("transition_model.json", transition_model.to_dict())
        library_path = _dump("primitive_library_summary.json", library.summary())
        manifest_path = _dump("synthesis_manifest.json", records)
        self.record(context, artifacts={
            "cycles_dir": self.rel(context, cycles_dir),
            "transition_model": self.rel(context, model_path),
            "library_summary": self.rel(context, library_path),
            "synthesis_manifest": self.rel(context, manifest_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "sampler": sampler.method,
            "sequence_method": self.sequence_method,
            "n_cycles": self.n_cycles,
            "states": library.states,
        })
        print(f"[primitive_synthesis] {self.n_cycles} cycles -> {cycles_dir}")
        return context
