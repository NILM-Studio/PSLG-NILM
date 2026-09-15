"""Build leakage-controlled real and synthetic NILM cycle datasets."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.framework.step import Step
from src.generation.cycle_conditioning import CycleNeighborIndex, cycle_profile
from src.generation.primitive_library import (Primitive, PrimitiveLibrary,
                                              RealPrimitiveSampler)


class NilmDatasetStep(Step):
    step_type = "nilm_dataset"

    def __init__(self, cluster_tag: str, aligned_series_path: str,
                 real_ratios=(0.05, 0.10, 0.20), sample_period_seconds: int = 6,
                 max_gap_seconds: int = 150, random_seed: int = 42,
                 expected_conditioning_neighbors: int = 10,
                 traditional_scale_range=(0.9, 1.1),
                 traditional_noise_ratio: float = 0.01,
                 active_threshold_watts: float = 10.0,
                 synthesis_scope: str = "legacy_global",
                 candidate_pool: int = 32,
                 within_state_smooth_samples: int = 3,
                 boundary_smooth_samples: int = 3,
                 require_train_only_structure: bool = False,
                 budget_conditioning_method: str = "independent",
                 budget_conditioning_neighbors: int = 10,
                 require_additive_measurement: bool = False):
        if not cluster_tag:
            raise ValueError("nilm_dataset requires --cluster-tag")
        if not aligned_series_path:
            raise ValueError("nilm_dataset.aligned_series is required")
        ratios = [float(value) for value in real_ratios]
        if not ratios or any(value <= 0 or value > 1 for value in ratios):
            raise ValueError("nilm_dataset.real_ratios must be in (0, 1]")
        if synthesis_scope not in ("legacy_global", "budget_local"):
            raise ValueError(
                "nilm_dataset.synthesis_scope must be legacy_global or budget_local")
        budget_conditioning_method = str(budget_conditioning_method).lower()
        if budget_conditioning_method not in ("independent", "cycle_neighbors"):
            raise ValueError(
                "nilm_dataset.budget_conditioning_method must be independent "
                "or cycle_neighbors")
        if int(budget_conditioning_neighbors) < 1:
            raise ValueError("nilm_dataset.budget_conditioning_neighbors must be positive")
        scope = "strict_" if require_train_only_structure else ""
        budget = (f"budget_local_{budget_conditioning_method}_"
                  f"k{int(budget_conditioning_neighbors)}_seed{int(random_seed)}_"
                  if synthesis_scope == "budget_local" else "")
        super().__init__(
            variant=f"{scope}{budget}cycle_augmentation_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.aligned_series_path = aligned_series_path
        self.real_ratios = sorted(set(ratios))
        self.sample_period_seconds = int(sample_period_seconds)
        self.max_gap_seconds = int(max_gap_seconds)
        self.random_seed = int(random_seed)
        self.expected_conditioning_neighbors = int(expected_conditioning_neighbors)
        self.traditional_scale_range = tuple(float(value)
                                             for value in traditional_scale_range)
        if (len(self.traditional_scale_range) != 2
                or self.traditional_scale_range[0] <= 0
                or self.traditional_scale_range[1] < self.traditional_scale_range[0]):
            raise ValueError("traditional_scale_range must contain positive [low, high]")
        self.traditional_noise_ratio = max(0.0, float(traditional_noise_ratio))
        self.active_threshold_watts = max(0.0, float(active_threshold_watts))
        self.synthesis_scope = synthesis_scope
        self.candidate_pool = max(1, int(candidate_pool))
        self.within_state_smooth_samples = max(
            0, int(within_state_smooth_samples))
        self.boundary_smooth_samples = max(0, int(boundary_smooth_samples))
        self.require_train_only_structure = bool(require_train_only_structure)
        self.budget_conditioning_method = budget_conditioning_method
        self.budget_conditioning_neighbors = int(budget_conditioning_neighbors)
        self.require_additive_measurement = bool(require_additive_measurement)

    def _measurement_audit(self, aligned_path: Path) -> dict:
        """Read the preparation audit before forming mains - target + generated."""
        audit_path = Path(str(aligned_path) + ".audit.json")
        if not audit_path.is_file():
            if self.require_additive_measurement:
                raise FileNotFoundError(
                    f"[nilm_dataset] additive synthesis requires measurement audit: {audit_path}")
            return {"audit_path": None, "mains_power_type": "unknown",
                    "appliance_power_type": "unknown",
                    "measurement_compatible_for_additive_synthesis": None}
        with audit_path.open(encoding="utf-8") as f:
            audit = json.load(f)
        compatible = audit.get("measurement_compatible_for_additive_synthesis")
        output_path = audit.get("output_path")
        path_matches = (bool(output_path)
                        and Path(output_path).resolve() == aligned_path.resolve())
        if self.require_additive_measurement:
            if (compatible is not True or audit.get("mains_power_type") != "active"
                    or audit.get("appliance_power_type") != "active"):
                raise ValueError(
                    "[nilm_dataset] additive synthesis requires compatible active/active "
                    "mains and appliance measurements; preparation audit is incompatible")
            if not path_matches:
                raise ValueError(
                    "[nilm_dataset] measurement audit output_path does not match aligned series")
        return {
            "audit_path": str(audit_path.resolve()),
            "output_path": output_path,
            "output_path_matches": path_matches,
            "mains_power_type": audit.get("mains_power_type", "unknown"),
            "appliance_power_type": audit.get("appliance_power_type", "unknown"),
            "measurement_compatible_for_additive_synthesis": compatible,
        }

    @staticmethod
    def _load_assignments(path: str) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _stratified_select(records: list[dict], count: int,
                           rng: np.random.Generator) -> list[dict]:
        if count >= len(records):
            return list(records)
        groups = {}
        for record in records:
            key = (int(record["class_id"]), int(record["mode_id"]))
            groups.setdefault(key, []).append(record)
        for values in groups.values():
            rng.shuffle(values)
        selected = []
        keys = sorted(groups)
        while len(selected) < count:
            progressed = False
            for key in keys:
                if groups[key] and len(selected) < count:
                    selected.append(groups[key].pop())
                    progressed = True
            if not progressed:
                break
        return selected

    @staticmethod
    def _stratified_order(records: list[dict],
                          rng: np.random.Generator) -> list[dict]:
        """Return one deterministic round-robin order for nested budgets."""
        groups = {}
        for record in records:
            key = (int(record["class_id"]), int(record["mode_id"]))
            groups.setdefault(key, []).append(record)
        for values in groups.values():
            rng.shuffle(values)
        ordered, keys = [], sorted(groups)
        while True:
            progressed = False
            for key in keys:
                if groups[key]:
                    ordered.append(groups[key].pop())
                    progressed = True
            if not progressed:
                return ordered

    @staticmethod
    def _resample_interval(timestamp: np.ndarray, mains: np.ndarray,
                           appliance: np.ndarray, start: int, end: int,
                           period: int, max_gap: int):
        left = int(np.searchsorted(timestamp, start, side="left"))
        right = int(np.searchsorted(timestamp, end, side="right"))
        t = timestamp[left:right]
        if len(t) < 2:
            return None, "insufficient_aligned_points"
        largest_gap = int(np.max(np.diff(t)))
        if largest_gap > max_gap:
            return None, f"gap_exceeds_{max_gap}s"
        grid_start = int(np.ceil(start / period) * period)
        grid_end = int(np.floor(end / period) * period)
        if grid_end <= grid_start:
            return None, "interval_too_short"
        grid = np.arange(grid_start, grid_end + 1, period, dtype=np.int64)
        return {
            "timestamp": grid,
            "mains": np.interp(grid, t, mains[left:right]).astype(np.float32),
            "appliance": np.interp(
                grid, t, appliance[left:right]).astype(np.float32),
            "largest_source_gap_seconds": largest_gap,
        }, None

    @staticmethod
    def _write_npz(directory: str, name: str, payload: dict) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.npz")
        np.savez_compressed(path, **payload)
        return path

    @staticmethod
    def _traditional_augment(mains: np.ndarray, appliance: np.ndarray,
                             rng: np.random.Generator, scale_range,
                             noise_ratio: float, active_threshold: float):
        """Apply magnitude scaling and active-state jitter to an NILM pair."""
        low, high = (float(scale_range[0]), float(scale_range[1]))
        scale = float(rng.uniform(low, high))
        active = appliance > active_threshold
        augmented = appliance.astype(np.float32, copy=True) * scale
        active_values = appliance[active]
        sigma = (float(np.std(active_values)) * float(noise_ratio)
                 if len(active_values) else 0.0)
        if sigma > 0:
            augmented[active] += rng.normal(0.0, sigma, int(np.sum(active))).astype(
                np.float32)
        augmented = np.maximum(augmented, 0.0).astype(np.float32)
        background = np.maximum(mains - appliance, 0.0).astype(np.float32)
        return (background + augmented).astype(np.float32), augmented, {
            "scale": scale, "active_noise_sigma": sigma,
        }

    @staticmethod
    def _relative(path: str, root: str) -> str:
        return os.path.relpath(path, root).replace(os.sep, "/")

    def _validate_synthesis(self, context: dict) -> tuple[str, str]:
        entry = context["manifest"].get_step("primitive_synthesis") or {}
        extra = entry.get("extra") or {}
        if extra.get("conditioning_method") != "cycle_neighbors":
            raise ValueError("[nilm_dataset] synthesis must use cycle_neighbors")
        if int(extra.get("conditioning_neighbors", -1)) != self.expected_conditioning_neighbors:
            raise ValueError(
                "[nilm_dataset] synthesis neighbor count does not match selected method")
        if extra.get("source_split") != "train":
            raise ValueError("[nilm_dataset] synthesis must use train split only")
        cycles = self.resolve(context, "primitive_synthesis", "cycles_dir")
        manifest = self.resolve(context, "primitive_synthesis", "synthesis_manifest")
        if not (cycles and os.path.isdir(cycles) and manifest and os.path.exists(manifest)):
            raise FileNotFoundError("[nilm_dataset] selected synthesis artifacts not found")
        return cycles, manifest

    @staticmethod
    def _exact_integer(value, name: str) -> int:
        """Reject corrupt coordinates instead of silently truncating them."""
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or (isinstance(value, (float, np.floating))
                    and (not np.isfinite(value) or value != np.floor(value)))):
            raise ValueError(f"[nilm_dataset] {name} must be a finite integer, got {value!r}")
        return int(value)

    def _budget_resources(self, context: dict, train_ids: set[int]):
        catalog_path = self.resolve(context, "cycle_split", "train_catalog")
        if not (catalog_path and os.path.exists(catalog_path)):
            raise FileNotFoundError(
                "[nilm_dataset] train catalog is required for budget-local synthesis")
        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)

        def cluster_array(key):
            path = context["manifest"].cluster_artifact_path(self.cluster_tag, key)
            if not (path and os.path.exists(path)):
                raise FileNotFoundError(
                    f"[nilm_dataset] missing {self.cluster_tag}.{key}")
            return np.load(path)

        labels = cluster_array("labels").reshape(-1)
        indices = cluster_array("indices")
        lengths = cluster_array("seq_len").reshape(-1)
        if indices.ndim != 2 or indices.shape[1] not in (2, 3):
            raise ValueError(
                "[nilm_dataset] primitive indices must have shape (n, 2) or (n, 3)")
        if not (len(labels) == len(indices) == len(lengths)):
            raise ValueError(
                "[nilm_dataset] primitive cluster artifacts are not row-aligned")

        segments_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if not (segments_dir and os.path.isdir(segments_dir)):
            raise FileNotFoundError("[nilm_dataset] extracted activity directory not found")
        files = sorted(name for name in os.listdir(segments_dir)
                       if name.lower().endswith(".csv"))
        power_cache, primitives = {}, []
        for primitive_id, (label, index, length) in enumerate(
                zip(labels, indices, lengths)):
            activity_id = self._exact_integer(index[0], f"primitive {primitive_id} activity ID")
            if activity_id not in train_ids:
                continue
            if not 0 <= activity_id < len(files):
                raise ValueError(
                    f"[nilm_dataset] training activity {activity_id} has no extracted file "
                    f"(available file IDs: 0..{len(files) - 1})")
            start = self._exact_integer(index[1], f"primitive {primitive_id} start")
            length = self._exact_integer(length, f"primitive {primitive_id} length")
            label = self._exact_integer(label, f"primitive {primitive_id} label")
            if indices.shape[1] == 3:
                index_label = self._exact_integer(index[2], f"primitive {primitive_id} index label")
                if index_label != label:
                    raise ValueError(
                        f"[nilm_dataset] primitive {primitive_id} index label differs from labels")
            if activity_id not in power_cache:
                frame = pd.read_csv(os.path.join(segments_dir, files[activity_id]))
                column = "power" if "power" in frame.columns else frame.columns[-1]
                source_power = pd.to_numeric(
                    frame[column], errors="coerce").to_numpy(dtype=np.float64)
                if (not np.isfinite(source_power).all()
                        or np.any(np.abs(source_power) > np.finfo(np.float32).max)):
                    raise ValueError(
                        f"[nilm_dataset] training activity {activity_id} contains "
                        "non-finite or non-numeric source power")
                power_cache[activity_id] = source_power.astype(np.float32)
            source = power_cache[activity_id]
            end = start + length
            if start < 0 or length <= 0 or end > len(source):
                raise ValueError(
                    f"[nilm_dataset] primitive {primitive_id} training activity {activity_id} "
                    f"interval [{start}, {end}) is outside source length {len(source)} "
                    "or has non-positive length")
            primitives.append(Primitive(
                primitive_id=int(primitive_id), state_label=label,
                activity_index=activity_id, start=start,
                power=np.asarray(source[start:end], dtype=np.float32)))
        if not primitives:
            raise ValueError("[nilm_dataset] budget-local primitive pool is empty")
        return catalog, primitives, power_cache

    @staticmethod
    def _budget_waveforms(selected_ids: set[int], activities: dict,
                          primitives: list[Primitive], source_waveforms: dict) -> dict:
        """Verify primitive slices cover each selected cycle's state template.

        Profiles must describe the same physical intervals used by the sampler.
        In particular, a truncated primitive or a dropped feature row must not
        silently shift every subsequent state's physical profile.
        """
        by_activity = {activity_id: [] for activity_id in selected_ids}
        for primitive in primitives:
            if primitive.activity_index in by_activity:
                by_activity[primitive.activity_index].append(primitive)
        verified = {}
        for activity_id in sorted(selected_ids):
            blocks = activities[str(activity_id)].get("blocks", [])
            if not blocks:
                raise ValueError(
                    f"[nilm_dataset] activity {activity_id} has invalid catalog blocks")
            state_chunks, template_cursor = [], 0
            for block_index, block in enumerate(blocks):
                location = f"activity {activity_id} catalog block {block_index}"
                length = NilmDatasetStep._exact_integer(
                    block.get("length_samples", 0), f"{location} length")
                state = NilmDatasetStep._exact_integer(block["state_label"], f"{location} state")
                if length <= 0:
                    raise ValueError(f"[nilm_dataset] {location} must have positive length")
                for coordinate, expected in (("start", template_cursor),
                                             ("end", template_cursor + length)):
                    if coordinate in block:
                        observed = NilmDatasetStep._exact_integer(
                            block[coordinate], f"{location} {coordinate}")
                        if observed != expected:
                            raise ValueError(
                                f"[nilm_dataset] {location} {coordinate}={observed} "
                                f"does not match cumulative template coordinate {expected}")
                state_chunks.append(np.full(length, state, dtype=np.int64))
                template_cursor += length
            states = np.concatenate(state_chunks)
            waveform = np.asarray(source_waveforms[activity_id], dtype=np.float32)
            if waveform.ndim != 1 or not np.isfinite(waveform).all():
                raise ValueError(
                    f"[nilm_dataset] activity {activity_id} has invalid source power")
            if len(waveform) != len(states):
                raise ValueError(
                    f"[nilm_dataset] activity {activity_id} source length "
                    f"{len(waveform)} differs from catalog duration {len(states)}")
            cursor = 0
            for primitive in sorted(by_activity[activity_id],
                                    key=lambda value: (value.start, value.primitive_id)):
                end = int(primitive.start) + len(primitive.power)
                if int(primitive.start) != cursor or end > len(states):
                    raise ValueError(
                        f"[nilm_dataset] activity {activity_id} primitive coverage "
                        f"gap/overlap at sample {cursor}, next interval "
                        f"[{primitive.start}, {end})")
                if (not np.all(states[cursor:end] == int(primitive.state_label))
                        or not np.array_equal(waveform[cursor:end], primitive.power)):
                    raise ValueError(
                        f"[nilm_dataset] activity {activity_id} primitive state/power "
                        f"does not match source interval [{cursor}, {end})")
                cursor = end
            if cursor != len(states):
                raise ValueError(
                    f"[nilm_dataset] activity {activity_id} primitive coverage ends "
                    f"at {cursor}, expected {len(states)}")
            verified[activity_id] = waveform
        return verified

    def _generate_budget_cycles(
            self, log_dir: str, tag: str, real_subset: list[dict],
            real_by_activity: dict, catalog: dict,
            all_primitives: list[Primitive],
            source_waveforms: dict[int, np.ndarray] | None = None) -> list[dict]:
        selected_ids = {int(row["activity_id"]) for row in real_subset}
        activities = catalog.get("activities", {})
        missing = selected_ids - {int(key) for key in activities}
        if missing:
            raise ValueError(
                f"[nilm_dataset] budget activities missing from catalog: {sorted(missing)}")
        if source_waveforms is None:
            source_waveforms = {
                activity_id: real_by_activity[str(activity_id)]["payload"]["appliance"]
                for activity_id in selected_ids
            }
        waveforms = self._budget_waveforms(
            selected_ids, activities, all_primitives, source_waveforms)

        group_ids = {}
        for row in real_subset:
            key = (int(row["class_id"]), int(row["mode_id"]))
            activity_id = int(row["activity_id"])
            activity = activities[str(activity_id)]
            if (int(activity.get("class_id", key[0])) != key[0]
                    or int(activity.get("validation_mode_id", key[1])) != key[1]):
                raise ValueError(
                    f"[nilm_dataset] activity {activity_id} class/mode differs from catalog")
            group_ids.setdefault(key, set()).add(activity_id)
        samplers, conditioners = {}, {}
        for key, ids in group_ids.items():
            group_primitives = [primitive for primitive in all_primitives
                                if primitive.activity_index in ids]
            samplers[key] = RealPrimitiveSampler(
                PrimitiveLibrary(group_primitives),
                candidate_pool=self.candidate_pool,
                within_state_smooth_samples=self.within_state_smooth_samples,
                boundary_smooth_samples=self.boundary_smooth_samples)
            if self.budget_conditioning_method == "cycle_neighbors":
                states = sorted({
                    int(block["state_label"]) for activity_id in ids
                    for block in activities[str(activity_id)]["blocks"]
                })
                profiles = {
                    activity_id: cycle_profile(
                        waveforms[activity_id], activities[str(activity_id)]["blocks"],
                        states)
                    for activity_id in sorted(ids)
                }
                conditioners[key] = CycleNeighborIndex(
                    profiles, neighbor_count=self.budget_conditioning_neighbors,
                    exclude_anchor=True)

        generated = []
        output_dir = os.path.join(
            log_dir, "cycles", "synthetic_budget", tag)
        for cycle_index, source_record in enumerate(real_subset):
            source_id = int(source_record["activity_id"])
            key = (int(source_record["class_id"]), int(source_record["mode_id"]))
            # Both methods exclude the template anchor for a matched ablation.
            # A one-member class/mode cannot provide cross-cycle donors.
            singleton = len(group_ids[key]) == 1
            neighbors = []
            if singleton:
                allowed_ids = {source_id}
                actual_method = "singleton_self_resample"
            elif self.budget_conditioning_method == "cycle_neighbors":
                neighbors = conditioners[key].neighbors(source_id)
                allowed_ids = {int(row["activity_id"]) for row in neighbors}
                actual_method = "cycle_neighbors"
            else:
                allowed_ids = group_ids[key] - {source_id}
                actual_method = "independent"
            blocks = activities[str(source_id)].get("blocks", [])
            rng = np.random.default_rng(np.random.SeedSequence([
                self.random_seed, source_id, 86028121,
            ]))
            powers, provenance, previous_end, cursor = [], [], None, 0
            for block_index, block in enumerate(blocks):
                state = int(block["state_label"])
                length = int(block.get("length_samples", 0))
                if length <= 0:
                    continue
                try:
                    power, sources = samplers[key].sample_block(
                        state, length, rng, initial_power=previous_end,
                        allowed_activity_ids=allowed_ids)
                except KeyError as exc:
                    raise ValueError(
                        f"[nilm_dataset] budget {tag} group {key} lacks state {state}"
                    ) from exc
                powers.append(power)
                previous_end = float(power[-1])
                provenance.append({
                    "block_index": int(block_index),
                    "state_label": state,
                    "start": int(cursor),
                    "end": int(cursor + len(power)),
                    "length_samples": int(len(power)),
                    "sources": sources,
                })
                cursor += len(power)
            if not powers:
                raise ValueError(
                    f"[nilm_dataset] activity {source_id} has no usable budget blocks")
            target = np.concatenate(powers).astype(np.float32)
            used_ids = sorted({
                int(source["activity_index"])
                for block in provenance for source in block["sources"]
            })
            if not set(used_ids).issubset(selected_ids):
                raise ValueError(
                    "[nilm_dataset] budget-local synthesis used an out-of-budget primitive")
            if not set(used_ids).issubset(allowed_ids):
                raise ValueError(
                    "[nilm_dataset] synthesis used a primitive outside its donor pool")
            source_samples = Counter()
            source_primitive_ids = set()
            for block in provenance:
                for source in block["sources"]:
                    source_samples[int(source["activity_index"])] += int(source["used_length"])
                    source_primitive_ids.add(int(source["primitive_id"]))
            self_ratio = float(source_samples[source_id] / len(target))

            source_payload = real_by_activity[str(source_id)]["payload"]
            background = np.maximum(
                source_payload["mains"] - source_payload["appliance"], 0.0)
            background_axis = np.linspace(0.0, 1.0, len(background))
            target_axis = np.linspace(0.0, 1.0, len(target))
            synthetic_background = np.interp(
                target_axis, background_axis, background).astype(np.float32)
            payload = {
                "timestamp": np.arange(len(target), dtype=np.int64)
                * self.sample_period_seconds,
                "mains": (synthetic_background + target).astype(np.float32),
                "appliance": target,
            }
            path = self._write_npz(
                output_dir, f"synthetic_{cycle_index:05d}_source_{source_id:05d}",
                payload)
            generated.append({
                "kind": "synthetic_budget_local",
                "cycle_id": int(cycle_index),
                "source_activity_id": str(source_id),
                "anchor_activity_id": str(source_id),
                "class_id": key[0], "mode_id": key[1], "split": "train",
                "length_samples": int(len(target)),
                "duration_seconds": float(len(target) * self.sample_period_seconds),
                "mean_power": float(np.mean(target)),
                "max_power": float(np.max(target)),
                "energy_wh": float(np.sum(target, dtype=np.float64)
                                   * self.sample_period_seconds / 3600.0),
                "file": self._relative(path, log_dir),
                "budget_tag": tag,
                "budget_activity_ids": sorted(selected_ids),
                "conditioning_method": self.budget_conditioning_method,
                "actual_conditioning_method": actual_method,
                "conditioning_neighbors_requested": self.budget_conditioning_neighbors,
                "conditioning_neighbors": neighbors,
                "conditioning_neighbor_count": len(neighbors),
                "conditioning_fit_activity_ids": sorted(group_ids[key]),
                "conditioning_anchor_excluded": source_id not in allowed_ids,
                "conditioning_fallback": singleton,
                "conditioning_fallback_reason": (
                    "singleton_class_mode_budget" if singleton else None),
                "donor_activity_ids": sorted(allowed_ids),
                "donor_activity_count": len(allowed_ids),
                "primitive_source_activity_ids": used_ids,
                "primitive_source_activity_count": len(used_ids),
                "primitive_source_count": len(source_primitive_ids),
                "primitive_source_samples_by_activity": {
                    str(activity_id): count for activity_id, count in sorted(
                        source_samples.items()) if count > 0
                },
                "self_source_sample_ratio": self_ratio,
                "cross_cycle_source_sample_ratio": 1.0 - self_ratio,
                "cross_cycle_source_activity_count": len(set(used_ids) - {source_id}),
                "cross_cycle_generation": bool(set(used_ids) - {source_id}),
                "blocks": provenance,
            })
        return generated

    def run(self, context: dict) -> dict:
        aligned_path = Path(self.aligned_series_path)
        if not aligned_path.exists():
            raise FileNotFoundError(f"[nilm_dataset] aligned series not found: {aligned_path}")
        measurement_audit = self._measurement_audit(aligned_path)
        assignments_path = self.resolve(context, "cycle_split", "assignments")
        segments_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if not (assignments_path and os.path.exists(assignments_path)):
            raise FileNotFoundError("[nilm_dataset] cycle split assignments not found")
        split_entry = context["manifest"].get_step("cycle_split") or {}
        if (self.require_train_only_structure
                and (split_entry.get("extra") or {}).get(
                    "structure_fit_scope") != "train_only"):
            raise ValueError("[nilm_dataset] train-only structure fit is required")
        if not (segments_dir and os.path.isdir(segments_dir)):
            raise FileNotFoundError("[nilm_dataset] extracted activity directory not found")
        synthetic_dir = synthetic_manifest_path = None
        if self.synthesis_scope == "legacy_global":
            synthetic_dir, synthetic_manifest_path = self._validate_synthesis(context)

        pair = pd.read_csv(
            aligned_path, usecols=["timestamp", "mains", "appliance"],
            dtype={"timestamp": np.int64, "mains": np.float32,
                   "appliance": np.float32})
        timestamp = pair["timestamp"].to_numpy(copy=False)
        mains = pair["mains"].to_numpy(copy=False)
        appliance = pair["appliance"].to_numpy(copy=False)
        if np.any(np.diff(timestamp) <= 0):
            raise ValueError("[nilm_dataset] aligned timestamps must be strictly increasing")

        log_dir = self.log_dir(context)
        cycle_root = os.path.join(log_dir, "cycles")
        real_records, rejected = [], []
        real_by_activity = {}
        assignments = self._load_assignments(assignments_path)
        for row in assignments:
            segment_path = os.path.join(segments_dir, row["file"])
            if not os.path.exists(segment_path):
                rejected.append({**row, "reason": "segment_file_missing"})
                continue
            segment = pd.read_csv(segment_path, usecols=["timestamp"])
            start, end = int(segment["timestamp"].min()), int(segment["timestamp"].max())
            values, reason = self._resample_interval(
                timestamp, mains, appliance, start, end,
                self.sample_period_seconds, self.max_gap_seconds)
            if reason:
                rejected.append({**row, "reason": reason})
                continue
            activity_id = str(row["activity_id"])
            path = self._write_npz(
                os.path.join(cycle_root, "real", row["split"]),
                f"activity_{int(activity_id):05d}", values)
            record = {
                "kind": "real", "activity_id": activity_id,
                "class_id": int(row["class_id"]), "mode_id": int(row["mode_id"]),
                "split": row["split"], "length_samples": int(len(values["mains"])),
                "file": self._relative(path, log_dir),
            }
            real_records.append(record)
            real_by_activity[activity_id] = {**record, "payload": values}

        synthetic_manifest = []
        if synthetic_manifest_path:
            with open(synthetic_manifest_path, encoding="utf-8") as f:
                synthetic_manifest = json.load(f)
        synthetic_records, rejected_synthetic = [], []
        for row in synthetic_manifest:
            source_id = str(row.get("source_activity_id", ""))
            source = real_by_activity.get(source_id)
            assignment = next(
                (value for value in assignments
                 if str(value["activity_id"]) == source_id), None)
            if assignment and assignment["split"] != "train":
                raise ValueError(
                    f"[nilm_dataset] synthetic cycle references non-training activity {source_id}")
            if not source:
                rejected_synthetic.append({
                    "cycle_id": int(row["cycle_id"]),
                    "source_activity_id": source_id,
                    "reason": "source_training_cycle_unavailable",
                })
                continue
            frame = pd.read_csv(
                os.path.join(synthetic_dir, row["file"]), usecols=["power"])
            target = frame["power"].to_numpy(dtype=np.float32, copy=False)
            source_payload = source["payload"]
            background = np.maximum(
                source_payload["mains"] - source_payload["appliance"], 0.0)
            source_axis = np.linspace(0.0, 1.0, len(background))
            target_axis = np.linspace(0.0, 1.0, len(target))
            synthetic_background = np.interp(
                target_axis, source_axis, background).astype(np.float32)
            synthetic_mains = synthetic_background + target
            payload = {
                "timestamp": np.arange(len(target), dtype=np.int64)
                * self.sample_period_seconds,
                "mains": synthetic_mains.astype(np.float32),
                "appliance": target.astype(np.float32),
            }
            path = self._write_npz(
                os.path.join(cycle_root, "synthetic"),
                f"synthetic_{int(row['cycle_id']):05d}", payload)
            synthetic_records.append({
                "kind": "synthetic", "cycle_id": int(row["cycle_id"]),
                "source_activity_id": source_id,
                "class_id": int(row["cycle_class"]),
                "mode_id": int(row["cycle_mode"]),
                "split": "train", "length_samples": int(len(target)),
                "file": self._relative(path, log_dir),
            })

        train = [row for row in real_records if row["split"] == "train"]
        validation = [row for row in real_records if row["split"] == "validation"]
        test = [row for row in real_records if row["split"] == "test"]
        traditional_records, traditional_by_activity = [], {}
        for record in train:
            activity_id = str(record["activity_id"])
            source_payload = real_by_activity[activity_id]["payload"]
            augment_rng = np.random.default_rng(np.random.SeedSequence([
                self.random_seed, int(activity_id), 32452843,
            ]))
            augmented_mains, augmented_appliance, parameters = (
                self._traditional_augment(
                    source_payload["mains"], source_payload["appliance"],
                    augment_rng, self.traditional_scale_range,
                    self.traditional_noise_ratio, self.active_threshold_watts))
            payload = {
                "timestamp": source_payload["timestamp"],
                "mains": augmented_mains,
                "appliance": augmented_appliance,
            }
            path = self._write_npz(
                os.path.join(cycle_root, "traditional"),
                f"traditional_{int(activity_id):05d}", payload)
            augmented_record = {
                "kind": "traditional", "source_activity_id": activity_id,
                "class_id": record["class_id"], "mode_id": record["mode_id"],
                "split": "train", "length_samples": record["length_samples"],
                "file": self._relative(path, log_dir), **parameters,
            }
            traditional_records.append(augmented_record)
            traditional_by_activity[activity_id] = augmented_record
        budget_catalog = None
        budget_primitives = []
        budget_source_waveforms = {}
        if self.synthesis_scope == "budget_local":
            budget_catalog, budget_primitives, budget_source_waveforms = (
                self._budget_resources(
                    context, {int(row["activity_id"]) for row in train}))
        budget_order = self._stratified_order(
            train, np.random.default_rng(np.random.SeedSequence([
                self.random_seed, 15485863,
            ])))
        experiments = {}
        budget_synthesis_records = {}
        for ratio in self.real_ratios:
            count = max(1, int(round(len(train) * ratio)))
            real_subset = list(budget_order[:count])
            tag = f"{int(round(ratio * 100)):02d}pct"
            if self.synthesis_scope == "budget_local":
                generated_subset = self._generate_budget_cycles(
                    log_dir, tag, real_subset, real_by_activity,
                    budget_catalog, budget_primitives, budget_source_waveforms)
                budget_synthesis_records[tag] = generated_subset
            else:
                ratio_rng = np.random.default_rng(np.random.SeedSequence([
                    self.random_seed, int(round(ratio * 10_000)),
                ]))
                generated_subset = self._stratified_select(
                    synthetic_records,
                    min(len(synthetic_records), len(real_subset)), ratio_rng)
            traditional_subset = [
                traditional_by_activity[str(row["activity_id"])]
                for row in real_subset
            ]
            experiments[tag] = {
                "real_ratio": ratio,
                "A_real_only": [row["file"] for row in real_subset],
                "B_real_plus_traditional": (
                    [row["file"] for row in real_subset]
                    + [row["file"] for row in traditional_subset]),
                "C_real_plus_generated": (
                    [row["file"] for row in real_subset]
                    + [row["file"] for row in generated_subset]),
                "selected_real_count": len(real_subset),
                "selected_traditional_count": len(traditional_subset),
                "selected_generated_count": len(generated_subset),
                "selected_real_activity_ids": [
                    str(row["activity_id"]) for row in real_subset],
                "synthesis_fit_activity_ids": [
                    str(row["activity_id"]) for row in real_subset]
                    if self.synthesis_scope == "budget_local" else None,
                "synthesis_fit_count": (len(real_subset)
                    if self.synthesis_scope == "budget_local" else None),
                "synthesis_scope": self.synthesis_scope,
                "budget_conditioning_method": (self.budget_conditioning_method
                    if self.synthesis_scope == "budget_local" else None),
                "budget_conditioning_neighbors": (self.budget_conditioning_neighbors
                    if self.synthesis_scope == "budget_local" else None),
            }
        experiments["full"] = {
            "D_full_real": [row["file"] for row in train],
            "validation": [row["file"] for row in validation],
            "test": [row["file"] for row in test],
        }

        expected_groups = Counter(
            (row["split"], int(row["class_id"]), int(row["mode_id"]))
            for row in assignments)
        accepted_groups = Counter(
            (row["split"], int(row["class_id"]), int(row["mode_id"]))
            for row in real_records)
        group_retention = []
        for (split, class_id, mode_id), total in sorted(expected_groups.items()):
            accepted = accepted_groups[(split, class_id, mode_id)]
            group_retention.append({
                "split": split, "class_id": class_id, "mode_id": mode_id,
                "total": total, "accepted": accepted,
                "rejected": total - accepted,
                "acceptance_ratio": accepted / total if total else 0.0,
            })

        manifest_path = os.path.join(log_dir, "nilm_dataset_manifest.json")
        total_real = len(real_records) + len(rejected)
        budget_synthetic_count = sum(
            len(rows) for rows in budget_synthesis_records.values())
        budget_source_ids = {
            tag: sorted({
                int(activity_id)
                for row in rows
                for activity_id in row["primitive_source_activity_ids"]
            })
            for tag, rows in budget_synthesis_records.items()
        }
        budget_fit_ids = {
            tag: {int(value) for value in experiments[tag]["synthesis_fit_activity_ids"]}
            for tag in budget_synthesis_records
        }
        budget_leakage_violations = {
            tag: sorted(set(budget_source_ids[tag]) - budget_fit_ids[tag])
            for tag in budget_synthesis_records
            if set(budget_source_ids[tag]) - budget_fit_ids[tag]
        }
        if budget_leakage_violations:
            raise ValueError(
                "[nilm_dataset] out-of-budget primitive provenance detected: "
                f"{budget_leakage_violations}")
        nested_budget_ids = [
            {int(value) for value in experiments[
                f"{int(round(ratio * 100)):02d}pct"]["selected_real_activity_ids"]}
            for ratio in self.real_ratios
        ]
        nested_subsets_verified = all(
            left.issubset(right)
            for left, right in zip(nested_budget_ids, nested_budget_ids[1:]))
        if not nested_subsets_verified:
            raise ValueError("[nilm_dataset] real-data budgets are not nested")
        effective_synthetic_count = (
            budget_synthetic_count if self.synthesis_scope == "budget_local"
            else len(synthetic_records))
        audit = {
            "aligned_series": str(aligned_path),
            "measurement_audit": measurement_audit,
            "mains_power_type": measurement_audit["mains_power_type"],
            "appliance_power_type": measurement_audit["appliance_power_type"],
            "measurement_compatible_for_additive_synthesis": measurement_audit[
                "measurement_compatible_for_additive_synthesis"],
            "require_additive_measurement": self.require_additive_measurement,
            "sample_period_seconds": self.sample_period_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "real_counts": {"train": len(train), "validation": len(validation),
                            "test": len(test), "rejected": len(rejected)},
            "real_acceptance_ratio": (
                len(real_records) / total_real if total_real else 0.0),
            "real_rejection_reasons": dict(Counter(
                row["reason"] for row in rejected)),
            "class_mode_retention": group_retention,
            "synthetic_count": effective_synthetic_count,
            "synthetic_rejected": len(rejected_synthetic),
            "synthetic_acceptance_ratio": (
                1.0 if self.synthesis_scope == "budget_local"
                else (len(synthetic_records) / len(synthetic_manifest)
                      if synthetic_manifest else 0.0)),
            "synthetic_rejection_reasons": dict(Counter(
                row["reason"] for row in rejected_synthetic)),
            "synthetic_background": "max(source_train_mains-source_train_appliance,0)",
            "synthesis_scope": self.synthesis_scope,
            "budget_conditioning": {
                "method": self.budget_conditioning_method,
                "neighbors_requested": self.budget_conditioning_neighbors,
                "profile_fit_scope": "selected_budget_class_mode_only",
                "exclude_anchor": True,
                "singleton_fallback": "singleton_self_resample",
                "random_seed": self.random_seed,
                "budgets": {
                    tag: {
                        "actual_method_counts": dict(Counter(
                            row["actual_conditioning_method"] for row in rows)),
                        "cross_cycle_generated_count": sum(
                            row["cross_cycle_generation"] for row in rows),
                        "singleton_fallback_count": sum(
                            row["conditioning_fallback"] for row in rows),
                        "mean_self_source_sample_ratio": (
                            float(np.mean([row["self_source_sample_ratio"]
                                           for row in rows])) if rows else None),
                    } for tag, rows in budget_synthesis_records.items()
                },
            } if self.synthesis_scope == "budget_local" else None,
            "nested_real_subsets": nested_subsets_verified,
            "budget_leakage_check": {
                "passed": (not budget_leakage_violations
                           if self.synthesis_scope == "budget_local" else None),
                "primitive_source_scope": ("ratio_budget_only"
                    if self.synthesis_scope == "budget_local" else "global_train_pool"),
                "empirical_cycle_structure_scope": ("ratio_budget_only"
                    if self.synthesis_scope == "budget_local" else "global_train_pool"),
                "upstream_state_representation_scope": "not_verified",
                "cycle_classification_validation_scope": (
                    (split_entry.get("extra") or {}).get("structure_fit_scope", "unknown")),
                "primitive_source_activity_ids": budget_source_ids,
                "violations": budget_leakage_violations,
            },
            "budget_synthesis_counts": {
                tag: len(rows) for tag, rows in budget_synthesis_records.items()
            },
            "traditional_augmentation": {
                "method": "magnitude_scaling_plus_active_jitter",
                "count": len(traditional_records),
                "scale_range": list(self.traditional_scale_range),
                "noise_ratio": self.traditional_noise_ratio,
                "active_threshold_watts": self.active_threshold_watts,
                "paired_with_selected_real_cycles": True,
            },
            "test_waveform_used_by_synthesis": False,
            "experiments": experiments,
            "rejected_cycles": rejected,
            "rejected_synthetic_cycles": rejected_synthetic,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        budget_manifest_path = os.path.join(
            log_dir, "budget_synthesis_manifest.json")
        with open(budget_manifest_path, "w", encoding="utf-8") as f:
            json.dump(budget_synthesis_records, f, indent=2, ensure_ascii=False)
        traditional_manifest_path = os.path.join(
            log_dir, "traditional_augmentation_manifest.json")
        with open(traditional_manifest_path, "w", encoding="utf-8") as f:
            json.dump(traditional_records, f, indent=2, ensure_ascii=False)
        self.record(context, artifacts={
            "dataset_manifest": self.rel(context, manifest_path),
            "cycles_dir": self.rel(context, cycle_root),
            "traditional_manifest": self.rel(context, traditional_manifest_path),
            "budget_synthesis_manifest": self.rel(
                context, budget_manifest_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "real_counts": audit["real_counts"],
            "synthetic_count": effective_synthetic_count,
            "synthesis_scope": self.synthesis_scope,
            "budget_conditioning_method": (self.budget_conditioning_method
                if self.synthesis_scope == "budget_local" else None),
            "budget_conditioning_neighbors": (self.budget_conditioning_neighbors
                if self.synthesis_scope == "budget_local" else None),
            "random_seed": self.random_seed,
            "waveform_holdout": True,
        })
        print(f"[nilm_dataset] real={audit['real_counts']} synthetic="
              f"{effective_synthetic_count} scope={self.synthesis_scope} -> {log_dir}")
        return context
