"""Quantitative evaluation of synthetic cycles against held-out real cycles."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.framework.step import Step


METRICS = ("duration_seconds", "energy_wh", "mean_power", "max_power")


class SynthesisEvaluationStep(Step):
    """Compare generated cycles with held-out real cycles by class and mode."""

    step_type = "synthesis_evaluation"

    def __init__(self, cluster_tag: str, fs: float = 0.1666667,
                 waveform_points: int = 256):
        if not cluster_tag:
            raise ValueError("synthesis evaluation requires --cluster-tag")
        if fs <= 0 or waveform_points < 16:
            raise ValueError("invalid synthesis evaluation sampling parameters")
        super().__init__(variant=f"heldout_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.fs = float(fs)
        self.waveform_points = int(waveform_points)

    @staticmethod
    def _load_json(path: str) -> dict | list:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _resample_shape(power: Iterable[float], points: int) -> np.ndarray:
        values = np.asarray(list(power), dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if len(values) == 0:
            return np.zeros(points, dtype=np.float64)
        if len(values) == 1:
            sampled = np.full(points, values[0], dtype=np.float64)
        else:
            sampled = np.interp(
                np.linspace(0.0, 1.0, points),
                np.linspace(0.0, 1.0, len(values)), values)
        scale = float(np.std(sampled))
        return ((sampled - float(np.mean(sampled))) / scale
                if scale > 1e-9 else np.zeros(points, dtype=np.float64))

    @staticmethod
    def _distribution_row(class_id: int, mode_id: int, metric: str,
                          real: List[float], generated: List[float]) -> dict:
        real_values = np.asarray(real, dtype=np.float64)
        generated_values = np.asarray(generated, dtype=np.float64)
        row = {
            "class_id": class_id, "mode_id": mode_id, "metric": metric,
            "real_count": len(real_values), "generated_count": len(generated_values),
        }
        if not len(real_values) or not len(generated_values):
            row.update({key: None for key in (
                "real_mean", "real_median", "generated_mean", "generated_median",
                "wasserstein", "normalized_wasserstein", "ks_statistic", "ks_pvalue")})
            return row
        distance = float(wasserstein_distance(real_values, generated_values))
        q25, q75 = np.percentile(real_values, [25, 75])
        scale = max(float(q75 - q25), abs(float(np.median(real_values))), 1e-9)
        ks = ks_2samp(real_values, generated_values)
        row.update({
            "real_mean": float(np.mean(real_values)),
            "real_median": float(np.median(real_values)),
            "generated_mean": float(np.mean(generated_values)),
            "generated_median": float(np.median(generated_values)),
            "wasserstein": distance,
            "normalized_wasserstein": float(distance / scale),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        })
        return row

    def _real_records(self, catalog: dict, files: List[str], segments_dir: str,
                      include_waveform: bool) -> List[dict]:
        class_lookup = {}
        for entry in catalog.get("classes", []):
            for activity_id in entry.get("member_ids", []):
                class_lookup[str(activity_id)] = int(entry["class_id"])
        records = []
        for activity_id, activity in catalog.get("activities", {}).items():
            index = int(activity_id)
            if not 0 <= index < len(files):
                continue
            frame = pd.read_csv(os.path.join(segments_dir, files[index]))
            column = "power" if "power" in frame.columns else frame.columns[-1]
            power = np.clip(pd.to_numeric(
                frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
                0.0, None)
            state_durations: Dict[int, float] = {}
            for block in activity.get("blocks", []):
                state = int(block["state_label"])
                state_durations[state] = state_durations.get(state, 0.0) + (
                    int(block.get("length_samples", 0)) / self.fs)
            record = {
                "activity_id": str(activity_id), "file": files[index],
                "class_id": class_lookup[str(activity_id)],
                "mode_id": int(activity["validation_mode_id"]),
                "duration_seconds": float(len(power) / self.fs),
                "energy_wh": float(np.sum(power) / self.fs / 3600.0),
                "mean_power": float(np.mean(power)),
                "max_power": float(np.max(power)),
                "state_durations": state_durations,
            }
            if include_waveform:
                record["shape"] = self._resample_shape(power, self.waveform_points)
            records.append(record)
        return records

    def run(self, context: dict) -> dict:
        split = context["manifest"].get_step("cycle_split") or {}
        split_tag = (split.get("extra") or {}).get("cluster_tag")
        if split_tag and split_tag != self.cluster_tag:
            raise ValueError(
                f"[synthesis_evaluation] split uses {split_tag}, requested {self.cluster_tag}")
        train_path = self.resolve(context, "cycle_split", "train_catalog")
        test_path = self.resolve(context, "cycle_split", "test_catalog")
        synthesis_path = self.resolve(
            context, "primitive_synthesis", "synthesis_manifest")
        cycles_dir = self.resolve(context, "primitive_synthesis", "cycles_dir")
        segments_dir = self.resolve(context, "extract_active_data", "segments_dir")
        required = [train_path, test_path, synthesis_path, cycles_dir, segments_dir]
        if not all(path and os.path.exists(path) for path in required):
            raise FileNotFoundError(
                "[synthesis_evaluation] run cycle_split and train-only synthesize first")

        train_catalog = self._load_json(train_path)
        test_catalog = self._load_json(test_path)
        synthetic_manifest = self._load_json(synthesis_path)
        files = sorted(name for name in os.listdir(segments_dir)
                       if name.lower().endswith(".csv"))
        train_records = self._real_records(
            train_catalog, files, segments_dir, include_waveform=True)
        test_records = self._real_records(
            test_catalog, files, segments_dir, include_waveform=True)

        generated_records = []
        for source in synthetic_manifest:
            frame = pd.read_csv(os.path.join(cycles_dir, source["file"]))
            state_durations = {
                int(state): float(len(rows) / self.fs)
                for state, rows in frame.groupby("state_label")
            }
            generated_records.append({
                **{key: source[key] for key in METRICS},
                "cycle_id": int(source["cycle_id"]), "file": source["file"],
                "class_id": int(source["cycle_class"]),
                "mode_id": int(source["cycle_mode"]),
                "state_durations": state_durations,
                "shape": self._resample_shape(frame["power"], self.waveform_points),
            })

        def grouped(records: List[dict]) -> Dict[tuple[int, int], List[dict]]:
            result: Dict[tuple[int, int], List[dict]] = {}
            for record in records:
                result.setdefault(
                    (record["class_id"], record["mode_id"]), []).append(record)
            return result

        train_groups, test_groups = grouped(train_records), grouped(test_records)
        generated_groups = grouped(generated_records)
        expected_groups = sorted(test_groups)
        distribution_rows, state_rows, novelty_rows, baseline_rows = [], [], [], []
        for class_id, mode_id in expected_groups:
            real = test_groups[(class_id, mode_id)]
            generated = generated_groups.get((class_id, mode_id), [])
            for metric in METRICS:
                distribution_rows.append(self._distribution_row(
                    class_id, mode_id, metric,
                    [record[metric] for record in real],
                    [record[metric] for record in generated]))
            states = sorted({state for record in real + generated
                             for state in record["state_durations"]})
            for state in states:
                state_rows.append(self._distribution_row(
                    class_id, mode_id, f"state_{state}_duration_seconds",
                    [record["state_durations"].get(state, 0.0) for record in real],
                    [record["state_durations"].get(state, 0.0)
                     for record in generated]))

            train_shapes = train_groups.get((class_id, mode_id), [])
            for record in generated:
                if train_shapes:
                    distances = np.asarray([
                        float(np.sqrt(np.mean((record["shape"] - source["shape"]) ** 2)))
                        for source in train_shapes
                    ])
                    nearest = int(np.argmin(distances))
                    nearest_id = train_shapes[nearest]["activity_id"]
                    nearest_distance = float(distances[nearest])
                else:
                    nearest_id, nearest_distance = "", None
                if real:
                    test_distances = np.asarray([
                        float(np.sqrt(np.mean(
                            (record["shape"] - target["shape"]) ** 2)))
                        for target in real
                    ])
                    nearest_test = int(np.argmin(test_distances))
                    nearest_test_id = real[nearest_test]["activity_id"]
                    nearest_test_distance = float(test_distances[nearest_test])
                else:
                    nearest_test_id, nearest_test_distance = "", None
                peers = [other for other in generated if other is not record]
                diversity = (min(float(np.sqrt(np.mean(
                    (record["shape"] - other["shape"]) ** 2))) for other in peers)
                             if peers else None)
                novelty_rows.append({
                    "cycle_id": record["cycle_id"], "file": record["file"],
                    "class_id": class_id, "mode_id": mode_id,
                    "nearest_train_activity_id": nearest_id,
                    "nearest_train_shape_rmse": nearest_distance,
                    "nearest_test_activity_id": nearest_test_id,
                    "nearest_test_shape_rmse": nearest_test_distance,
                    "nearest_generated_shape_rmse": diversity,
                })

            for record in real:
                if train_shapes:
                    train_distances = np.asarray([
                        float(np.sqrt(np.mean(
                            (record["shape"] - source["shape"]) ** 2)))
                        for source in train_shapes
                    ])
                    nearest_train = int(np.argmin(train_distances))
                    nearest_train_id = train_shapes[nearest_train]["activity_id"]
                    nearest_train_distance = float(train_distances[nearest_train])
                else:
                    nearest_train_id, nearest_train_distance = "", None
                if generated:
                    generated_distances = np.asarray([
                        float(np.sqrt(np.mean(
                            (record["shape"] - source["shape"]) ** 2)))
                        for source in generated
                    ])
                    nearest_generated = int(np.argmin(generated_distances))
                    nearest_generated_id = generated[nearest_generated]["cycle_id"]
                    nearest_generated_distance = float(
                        generated_distances[nearest_generated])
                else:
                    nearest_generated_id, nearest_generated_distance = "", None
                baseline_rows.append({
                    "activity_id": record["activity_id"], "file": record["file"],
                    "class_id": class_id, "mode_id": mode_id,
                    "nearest_train_activity_id": nearest_train_id,
                    "nearest_train_shape_rmse": nearest_train_distance,
                    "nearest_generated_cycle_id": nearest_generated_id,
                    "nearest_generated_shape_rmse": nearest_generated_distance,
                })

        missing_groups = [
            {"class_id": class_id, "mode_id": mode_id,
             "test_count": len(test_groups[(class_id, mode_id)])}
            for class_id, mode_id in expected_groups
            if not generated_groups.get((class_id, mode_id))
        ]
        normalized = [row["normalized_wasserstein"] for row in distribution_rows
                      if row["normalized_wasserstein"] is not None]
        novelty = [row["nearest_train_shape_rmse"] for row in novelty_rows
                   if row["nearest_train_shape_rmse"] is not None]
        diversity = [row["nearest_generated_shape_rmse"] for row in novelty_rows
                     if row["nearest_generated_shape_rmse"] is not None]
        generated_to_test = [row["nearest_test_shape_rmse"] for row in novelty_rows
                             if row["nearest_test_shape_rmse"] is not None]
        test_to_train = [row["nearest_train_shape_rmse"] for row in baseline_rows
                         if row["nearest_train_shape_rmse"] is not None]
        test_to_generated = [row["nearest_generated_shape_rmse"]
                             for row in baseline_rows
                             if row["nearest_generated_shape_rmse"] is not None]
        mean_generated_to_train = float(np.mean(novelty)) if novelty else None
        mean_test_to_train = float(np.mean(test_to_train)) if test_to_train else None
        mean_test_to_generated = (
            float(np.mean(test_to_generated)) if test_to_generated else None)
        novelty_ratio = (
            mean_generated_to_train / mean_test_to_train
            if mean_generated_to_train is not None
            and mean_test_to_train is not None and mean_test_to_train > 1e-9
            else None)
        coverage_ratio = (
            mean_test_to_generated / mean_test_to_train
            if mean_test_to_generated is not None
            and mean_test_to_train is not None and mean_test_to_train > 1e-9
            else None)
        summary = {
            "evaluation_scope": "heldout_test_waveforms",
            "structure_fit_scope": "all_validated_cycles",
            "train_real_cycles": len(train_records),
            "test_real_cycles": len(test_records),
            "generated_cycles": len(generated_records),
            "expected_class_modes": len(expected_groups),
            "evaluated_class_modes": len(expected_groups) - len(missing_groups),
            "missing_generated_groups": missing_groups,
            "mean_normalized_wasserstein": (
                float(np.mean(normalized)) if normalized else None),
            "mean_nearest_train_shape_rmse": (
                mean_generated_to_train),
            "mean_generated_to_test_shape_rmse": (
                float(np.mean(generated_to_test)) if generated_to_test else None),
            "mean_nearest_generated_shape_rmse": (
                float(np.mean(diversity)) if diversity else None),
            "mean_test_to_train_shape_rmse": mean_test_to_train,
            "mean_test_to_generated_shape_rmse": mean_test_to_generated,
            "generated_novelty_to_real_baseline_ratio": novelty_ratio,
            "generated_coverage_to_real_baseline_ratio": coverage_ratio,
        }

        log_dir = self.log_dir(context)
        distribution_path = os.path.join(log_dir, "distribution_metrics.csv")
        state_path = os.path.join(log_dir, "state_duration_metrics.csv")
        novelty_path = os.path.join(log_dir, "novelty_metrics.csv")
        baseline_path = os.path.join(log_dir, "real_holdout_shape_baseline.csv")
        summary_path = os.path.join(log_dir, "quality_summary.json")
        metric_fields = [
            "class_id", "mode_id", "metric", "real_count", "generated_count",
            "real_mean", "real_median", "generated_mean", "generated_median",
            "wasserstein", "normalized_wasserstein", "ks_statistic", "ks_pvalue",
        ]
        self._write_csv(distribution_path, distribution_rows, metric_fields)
        self._write_csv(state_path, state_rows, metric_fields)
        self._write_csv(novelty_path, novelty_rows, [
            "cycle_id", "file", "class_id", "mode_id",
            "nearest_train_activity_id", "nearest_train_shape_rmse",
            "nearest_test_activity_id", "nearest_test_shape_rmse",
            "nearest_generated_shape_rmse",
        ])
        self._write_csv(baseline_path, baseline_rows, [
            "activity_id", "file", "class_id", "mode_id",
            "nearest_train_activity_id", "nearest_train_shape_rmse",
            "nearest_generated_cycle_id", "nearest_generated_shape_rmse",
        ])
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.record(context, artifacts={
            "distribution_metrics": self.rel(context, distribution_path),
            "state_duration_metrics": self.rel(context, state_path),
            "novelty_metrics": self.rel(context, novelty_path),
            "real_holdout_shape_baseline": self.rel(context, baseline_path),
            "quality_summary": self.rel(context, summary_path),
        }, extra={"cluster_tag": self.cluster_tag, **summary})
        print(f"[synthesis_evaluation] {len(generated_records)} generated vs "
              f"{len(test_records)} held-out cycles -> {log_dir}")
        if missing_groups:
            print(f"[synthesis_evaluation] missing generated groups: {missing_groups}")
        return context
