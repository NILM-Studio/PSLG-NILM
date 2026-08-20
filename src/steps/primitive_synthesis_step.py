"""Synthesize appliance cycles from a merged primitive-state result."""
from __future__ import annotations

import json
import os
from glob import glob
from typing import List

import numpy as np
import pandas as pd

from src.framework.step import Step
from src.generation import (CyclePatternCatalog, PrimitiveLibrary,
                            RealPrimitiveSampler, StateTransitionModel)
from src.generation.primitive_library import Primitive


class PrimitiveSynthesisStep(Step):
    """Real-primitive resampling baseline with a replaceable sampler contract."""

    step_type = "primitive_synthesis"

    def __init__(self, cluster_tag: str, sampler: str = "real_resample",
                 sequence_method: str = "empirical", n_cycles: int = 100,
                 random_seed: int = 42, min_blocks: int = 3,
                 max_blocks: int = 20, fs: float = 0.1666667,
                 cycle_class: str = "all", class_sampling: str = "balanced",
                 mode_sampling: str = "empirical",
                 candidate_pool: int = 32,
                 within_state_smooth_samples: int = 3,
                 boundary_smooth_samples: int = 3,
                 require_cycle_validation: bool = True):
        if not cluster_tag:
            raise ValueError("primitive synthesis requires --cluster-tag")
        validation_tag = "validated_modes" if require_cycle_validation else "unvalidated"
        super().__init__(
            variant=(f"{sampler}_{sequence_method}_{cycle_class}_{validation_tag}"
                     f"_on_{cluster_tag}"))
        self.cluster_tag = cluster_tag
        self.sampler_name = str(sampler).lower()
        self.sequence_method = str(sequence_method).lower()
        self.n_cycles = int(n_cycles)
        self.random_seed = int(random_seed)
        self.min_blocks = int(min_blocks)
        self.max_blocks = int(max_blocks)
        self.fs = float(fs)
        self.cycle_class = str(cycle_class).lower()
        self.class_sampling = str(class_sampling).lower()
        self.mode_sampling = str(mode_sampling).lower()
        self.candidate_pool = int(candidate_pool)
        self.within_state_smooth_samples = int(within_state_smooth_samples)
        self.boundary_smooth_samples = int(boundary_smooth_samples)
        self.require_cycle_validation = bool(require_cycle_validation)
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

    def _load_primitives(self, context: dict) -> List[Primitive]:
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
        if not primitives:
            raise ValueError("[primitive_synthesis] primitive library is empty")
        return primitives

    def _load_catalog(self, context: dict) -> CyclePatternCatalog:
        validation = context["manifest"].get_step("cycle_validation") or {}
        validated_tag = (validation.get("extra") or {}).get("cluster_tag")
        if validated_tag and validated_tag != self.cluster_tag:
            raise ValueError(
                f"[primitive_synthesis] validated cycles use {validated_tag}, but "
                f"synthesis requested {self.cluster_tag}; rerun cycle_validate")
        validated_path = self.resolve(
            context, "cycle_validation", "validated_cycle_classes")
        if validated_path and os.path.exists(validated_path):
            with open(validated_path, encoding="utf-8") as f:
                payload = json.load(f)
            metadata = payload.get("validation", {})
            if (int(payload.get("version", 0)) < 3
                    or not metadata.get("canonical_signatures_only")
                    or not metadata.get("physical_modes_required")):
                raise ValueError(
                    "[primitive_synthesis] validated cycle catalog is from an "
                    "older workflow and may contain signature variants or "
                    "unvalidated modes; rerun --steps cycle_validate before "
                    "--steps synthesize")
            if not payload.get("classes"):
                raise ValueError(
                    "[primitive_synthesis] cycle validation accepted no full classes; "
                    "review class_validity_summary.csv or configure class_overrides")
            return CyclePatternCatalog(payload)
        if self.require_cycle_validation:
            raise FileNotFoundError(
                "[primitive_synthesis] validated cycle catalog not found; run "
                "--steps cycle_validate first for the same --run-id and --cluster-tag")

        entry = context["manifest"].get_step("cycle_classification") or {}
        classified_tag = (entry.get("extra") or {}).get("cluster_tag")
        if classified_tag and classified_tag != self.cluster_tag:
            raise ValueError(
                f"[primitive_synthesis] cycle classes use {classified_tag}, but "
                f"synthesis requested {self.cluster_tag}; rerun cycle_classify")
        path = self.resolve(context, "cycle_classification", "cycle_classes")
        if not (path and os.path.exists(path)):
            raise FileNotFoundError(
                "[primitive_synthesis] cycle classes not found; run "
                "--steps cycle_classify first for the same --run-id and --cluster-tag")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return CyclePatternCatalog(payload)

    @staticmethod
    def _canonical_blocks(blocks: List[dict]) -> List[tuple[int, int]]:
        """Collapse adjacent equal labels using the classifier's signature rule."""
        collapsed: List[tuple[int, int]] = []
        for block in blocks:
            state = int(block["state_label"])
            length = int(block.get("length_samples", 0))
            if length <= 0:
                continue
            if collapsed and collapsed[-1][0] == state:
                previous_state, previous_length = collapsed[-1]
                collapsed[-1] = (previous_state, previous_length + length)
            else:
                collapsed.append((state, length))
        return collapsed

    @staticmethod
    def _audit_catalog(catalog: CyclePatternCatalog,
                       class_ids: List[int]) -> dict:
        """Fail closed unless every synthesis member is canonical and mode-labelled."""
        audit = {"valid_class_ids": list(class_ids), "classes": {}}
        for class_id in class_ids:
            entry = catalog.classes[class_id]
            signature = [int(value) for value in entry.get(
                "representative_signature", [])]
            if not signature:
                raise ValueError(
                    f"[primitive_synthesis] class {class_id} has no representative signature")
            members = catalog.member_ids(class_id)
            if not members:
                raise ValueError(
                    f"[primitive_synthesis] class {class_id} has no validated members")
            modes = {}
            for activity_id in members:
                activity = catalog.activities[activity_id]
                observed = [state for state, _ in
                            PrimitiveSynthesisStep._canonical_blocks(
                                activity.get("blocks", []))]
                if observed != signature:
                    raise ValueError(
                        f"[primitive_synthesis] class {class_id} activity "
                        f"{activity_id} is not representative: {observed} != {signature}")
                mode_id = int(activity.get("validation_mode_id", -1))
                if mode_id < 0:
                    raise ValueError(
                        f"[primitive_synthesis] class {class_id} activity "
                        f"{activity_id} has no validated physical mode")
                modes.setdefault(str(mode_id), []).append(activity_id)
            audit["classes"][str(class_id)] = {
                "representative_signature": signature,
                "validated_members": len(members),
                "modes": {mode: len(ids) for mode, ids in sorted(modes.items())},
            }
        return audit

    def _sampler(self, library: PrimitiveLibrary):
        if self.sampler_name == "real_resample":
            return RealPrimitiveSampler(
                library,
                candidate_pool=self.candidate_pool,
                within_state_smooth_samples=self.within_state_smooth_samples,
                boundary_smooth_samples=self.boundary_smooth_samples,
            )
        raise ValueError(
            f"unknown primitive sampler '{self.sampler_name}'; supported: real_resample")

    def run(self, context: dict) -> dict:
        if self.n_cycles <= 0:
            raise ValueError("primitive_synthesis.n_cycles must be positive")
        primitives = self._load_primitives(context)
        catalog = self._load_catalog(context)
        class_ids = catalog.resolve_classes(self.cycle_class)
        catalog_audit = self._audit_catalog(catalog, class_ids)
        if self.mode_sampling not in ("balanced", "empirical"):
            raise ValueError("primitive_synthesis.mode_sampling must be balanced or empirical")
        mode_ids = {class_id: catalog.mode_ids_for_class(class_id)
                    for class_id in class_ids}
        transition_models, libraries, samplers = {}, {}, {}
        for class_id in class_ids:
            for mode_id in mode_ids[class_id]:
                key = (class_id, mode_id)
                transition_models[key] = StateTransitionModel(
                    catalog.sequences_for_mode(class_id, mode_id))
                member_ids = {int(activity_id)
                              for activity_id in catalog.member_ids(class_id, mode_id)}
                libraries[key] = PrimitiveLibrary(
                    p for p in primitives if p.activity_index in member_ids)
                samplers[key] = self._sampler(libraries[key])
        rng = np.random.default_rng(self.random_seed)

        if self.class_sampling not in ("balanced", "empirical"):
            raise ValueError("primitive_synthesis.class_sampling must be balanced or empirical")
        class_weights = np.asarray(
            [catalog.classes[class_id]["support"] for class_id in class_ids],
            dtype=np.float64)
        class_weights /= class_weights.sum()

        log_dir = self.log_dir(context)
        cycles_dir = os.path.join(log_dir, "cycles")
        os.makedirs(cycles_dir, exist_ok=True)
        for old_cycle in glob(os.path.join(cycles_dir, "synthetic_cycle_*.csv")):
            os.unlink(old_cycle)
        records: List[dict] = []
        mode_counters = {class_id: 0 for class_id in class_ids}

        for cycle_id in range(self.n_cycles):
            if len(class_ids) == 1:
                class_id = class_ids[0]
            elif self.class_sampling == "balanced":
                class_id = class_ids[cycle_id % len(class_ids)]
            else:
                class_id = class_ids[int(rng.choice(len(class_ids), p=class_weights))]
            available_modes = mode_ids[class_id]
            if len(available_modes) == 1:
                mode_id = available_modes[0]
            elif self.mode_sampling == "balanced":
                mode_id = available_modes[
                    mode_counters[class_id] % len(available_modes)]
                mode_counters[class_id] += 1
            else:
                supports = np.asarray([
                    len(catalog.member_ids(class_id, value))
                    for value in available_modes
                ], dtype=np.float64)
                supports /= supports.sum()
                mode_id = available_modes[int(rng.choice(len(available_modes), p=supports))]
            key = (class_id, mode_id)
            sampler = samplers[key]

            source_activity_id = None
            if self.sequence_method == "empirical":
                source_activity_id, blocks = catalog.sample_activity(
                    class_id, rng, mode_id=mode_id)
                state_blocks = self._canonical_blocks(blocks)
            else:
                state_blocks = transition_models[key].sample(
                    self.sequence_method, rng, self.min_blocks, self.max_blocks)
            powers, states, block_ids, block_records = [], [], [], []
            cursor = 0
            for block_id, (state, target_length) in enumerate(state_blocks):
                previous_end = float(powers[-1][-1]) if powers else None
                power, provenance = sampler.sample_block(
                    state, target_length, rng, initial_power=previous_end)
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
                "cycle_class": np.full(len(power), class_id, dtype=np.int32),
                "cycle_mode": np.full(len(power), mode_id, dtype=np.int32),
                "source_activity_id": np.full(
                    len(power), source_activity_id if source_activity_id is not None else ""),
            })
            filename = f"synthetic_cycle_{cycle_id:05d}.csv"
            frame.to_csv(os.path.join(cycles_dir, filename), index=False)
            records.append({
                "cycle_id": int(cycle_id), "file": filename,
                "cycle_class": int(class_id),
                "cycle_mode": int(mode_id),
                "source_activity_id": source_activity_id,
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

        continuity = {}
        for join_type in ("within_state", "state_boundary"):
            before, after = [], []
            for record in records:
                for block in record["blocks"]:
                    for source in block["sources"]:
                        if source.get("join_type") != join_type:
                            continue
                        before.append(float(source["join_jump_before"]))
                        after.append(float(source["join_jump_after"]))
            continuity[join_type] = {
                "count": int(len(before)),
                "before": self._jump_summary(before),
                "after": self._jump_summary(after),
            }

        model_path = _dump("transition_models.json", {
            f"class_{class_id}_mode_{mode_id}": model.to_dict()
            for (class_id, mode_id), model in transition_models.items()
        })
        library_path = _dump("primitive_library_summary.json", {
            f"class_{class_id}_mode_{mode_id}": library.summary()
            for (class_id, mode_id), library in libraries.items()
        })
        continuity_path = _dump("continuity_metrics.json", continuity)
        manifest_path = _dump("synthesis_manifest.json", records)
        audit_path = _dump("synthesis_input_audit.json", catalog_audit)
        self.record(context, artifacts={
            "cycles_dir": self.rel(context, cycles_dir),
            "transition_model": self.rel(context, model_path),
            "library_summary": self.rel(context, library_path),
            "continuity_metrics": self.rel(context, continuity_path),
            "synthesis_manifest": self.rel(context, manifest_path),
            "input_audit": self.rel(context, audit_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "sampler": self.sampler_name,
            "sequence_method": self.sequence_method,
            "cycle_class_selector": self.cycle_class,
            "generated_cycle_classes": class_ids,
            "class_sampling": self.class_sampling,
            "mode_sampling": self.mode_sampling,
            "generated_cycle_modes": {
                str(class_id): mode_ids[class_id] for class_id in class_ids
            },
            "n_cycles": self.n_cycles,
            "cycle_validation_required": self.require_cycle_validation,
            "states": sorted({state for library in libraries.values()
                              for state in library.states}),
        })
        print(f"[primitive_synthesis] {self.n_cycles} cycles -> {cycles_dir}")
        return context

    @staticmethod
    def _jump_summary(values: List[float]) -> dict:
        if not values:
            return {"mean": None, "p95": None, "max": None}
        data = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(data)),
            "p95": float(np.percentile(data, 95)),
            "max": float(np.max(data)),
        }
