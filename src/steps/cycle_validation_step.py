"""Select physically complete, statistically consistent appliance cycles."""
from __future__ import annotations

import copy
import csv
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.framework.step import Step
from src.generation.cycle_validation import (discover_metric_modes,
                                             infer_cycle_grammar,
                                             robust_z_scores,
                                             signature_purity)


class CycleValidationStep(Step):
    """Validate cycle classes and members before primitive synthesis.

    The step deliberately sits after cycle classification. It infers common
    state grammar from supported classes, checks physical cycle boundaries,
    removes robust metric outliers within each class, and emits a filtered
    catalog consumed by synthesis.
    """

    step_type = "cycle_validation"

    def __init__(self, cluster_tag: str, fs: float = 0.1666667,
                 min_class_support: int = 30,
                 min_signature_purity: float = 0.5,
                 min_valid_member_ratio: float = 0.7,
                 core_state_min_prevalence: float = 0.8,
                 terminal_state_min_prevalence: float = 0.7,
                 min_duration_seconds: float = 300.0,
                 boundary_window_seconds: float = 60.0,
                 boundary_absolute_watts: float = 50.0,
                 boundary_peak_ratio: float = 0.15,
                 max_missing_ratio: float = 0.01,
                 robust_z_threshold: float = 3.5,
                 max_metric_modes: int = 3,
                 min_mode_support: int = 15,
                 mode_bic_min_gain: float = 10.0,
                 mode_random_state: int = 42,
                 class_overrides: Dict[str, str] | None = None):
        if not cluster_tag:
            raise ValueError("cycle validation requires --cluster-tag")
        super().__init__(variant=f"multimodal_robust_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.fs = float(fs)
        self.min_class_support = max(1, int(min_class_support))
        self.min_signature_purity = float(min_signature_purity)
        self.min_valid_member_ratio = float(min_valid_member_ratio)
        self.core_state_min_prevalence = float(core_state_min_prevalence)
        self.terminal_state_min_prevalence = float(terminal_state_min_prevalence)
        self.min_duration_seconds = float(min_duration_seconds)
        self.boundary_window_seconds = float(boundary_window_seconds)
        self.boundary_absolute_watts = float(boundary_absolute_watts)
        self.boundary_peak_ratio = float(boundary_peak_ratio)
        self.max_missing_ratio = float(max_missing_ratio)
        self.robust_z_threshold = float(robust_z_threshold)
        self.max_metric_modes = max(1, int(max_metric_modes))
        self.min_mode_support = max(2, int(min_mode_support))
        self.mode_bic_min_gain = float(mode_bic_min_gain)
        self.mode_random_state = int(mode_random_state)
        self.class_overrides = {str(k): str(v).lower()
                                for k, v in (class_overrides or {}).items()}
        allowed = {"valid_full", "valid_short", "uncertain", "invalid"}
        unknown = set(self.class_overrides.values()) - allowed
        if unknown:
            raise ValueError(f"unknown cycle validation override status: {sorted(unknown)}")
        if self.fs <= 0:
            raise ValueError("cycle_validation.fs must be positive")

    def _load_inputs(self, context: dict) -> tuple[dict, str, List[str]]:
        entry = context["manifest"].get_step("cycle_classification") or {}
        classified_tag = (entry.get("extra") or {}).get("cluster_tag")
        if classified_tag and classified_tag != self.cluster_tag:
            raise ValueError(
                f"[cycle_validation] classes use {classified_tag}, requested {self.cluster_tag}")
        catalog_path = self.resolve(context, "cycle_classification", "cycle_classes")
        if not (catalog_path and os.path.exists(catalog_path)):
            raise FileNotFoundError(
                "[cycle_validation] cycle classes not found; run cycle_classify first")
        segments_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if not (segments_dir and os.path.isdir(segments_dir)):
            raise FileNotFoundError("[cycle_validation] activity CSV directory not found")
        with open(catalog_path, encoding="utf-8") as f:
            payload = json.load(f)
        files = sorted(name for name in os.listdir(segments_dir)
                       if name.lower().endswith(".csv"))
        return payload, segments_dir, files

    def _activity_metrics(self, activity_id: str, record: dict,
                          segments_dir: str, files: List[str]) -> dict:
        index = int(activity_id)
        if not 0 <= index < len(files):
            return {"load_error": True, "file": "", "missing_ratio": 1.0}
        filename = files[index]
        frame = pd.read_csv(os.path.join(segments_dir, filename))
        column = "power" if "power" in frame.columns else frame.columns[-1]
        raw = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        missing_ratio = float(np.mean(~np.isfinite(raw))) if len(raw) else 1.0
        power = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        positive = np.clip(power, 0.0, None)
        window = max(1, int(round(self.boundary_window_seconds * self.fs)))
        peak = float(np.max(positive)) if len(positive) else 0.0
        boundary_limit = max(self.boundary_absolute_watts,
                             self.boundary_peak_ratio * peak)
        return {
            "load_error": False,
            "file": filename,
            "length_samples": int(len(power)),
            "duration_seconds": float(len(power) / self.fs),
            "mean_power": float(np.mean(positive)) if len(positive) else 0.0,
            "max_power": peak,
            "energy_wh": float(np.sum(positive) / self.fs / 3600.0),
            "missing_ratio": missing_ratio,
            "start_power_median": float(np.median(positive[:window])) if len(positive) else 0.0,
            "end_power_median": float(np.median(positive[-window:])) if len(positive) else 0.0,
            "boundary_limit": float(boundary_limit),
            "signature": [int(v) for v in record.get("signature", [])],
            "distance_to_representative": record.get("distance_to_representative"),
        }

    @staticmethod
    def _write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def run(self, context: dict) -> dict:
        payload, segments_dir, files = self._load_inputs(context)
        classes = payload.get("classes", [])
        grammar = infer_cycle_grammar(
            classes, self.min_class_support, self.core_state_min_prevalence,
            self.terminal_state_min_prevalence)
        required_states = set(grammar["required_core_states"])
        terminal_states = set(grammar["allowed_terminal_states"])

        activity_rows: Dict[str, dict] = {}
        for activity_id, record in payload.get("activities", {}).items():
            if int(record.get("class_id", -1)) < 0:
                continue
            metrics = self._activity_metrics(
                activity_id, record, segments_dir, files)
            metrics.update({"activity_id": str(activity_id),
                            "class_id": int(record["class_id"])})
            activity_rows[str(activity_id)] = metrics

        class_lookup = {int(entry["class_id"]): entry for entry in classes}
        metric_names = ["duration_seconds", "energy_wh", "mean_power", "max_power"]

        # Hard checks describe observable data quality only. State grammar is a
        # semantic warning: a cold/short wash may omit a common heating state
        # without being a corrupt or incomplete recording.
        for row in activity_rows.values():
            signature = row.get("signature", [])
            representative = class_lookup[row["class_id"]].get(
                "representative_signature", [])
            row["is_representative_signature"] = (
                tuple(signature) == tuple(int(value) for value in representative))
            reasons = []
            if row.get("load_error"):
                reasons.append("load_error")
            if row.get("missing_ratio", 1.0) > self.max_missing_ratio:
                reasons.append("missing_data")
            if row.get("duration_seconds", 0.0) < self.min_duration_seconds:
                reasons.append("too_short")
            if row.get("start_power_median", np.inf) > row.get("boundary_limit", 0.0):
                reasons.append("active_start_boundary")
            if row.get("end_power_median", np.inf) > row.get("boundary_limit", 0.0):
                reasons.append("active_end_boundary")
            warnings = []
            if not row["is_representative_signature"]:
                warnings.append("signature_variant")
            if required_states and not required_states.issubset(set(signature)):
                warnings.append("missing_common_state")
            if terminal_states and (not signature or signature[-1] not in terminal_states):
                warnings.append("uncommon_terminal_state")
            row["passes_hard_checks"] = not reasons
            row["_hard_reasons"] = reasons
            row["structural_warnings"] = ";".join(warnings)

        # Discover supported physical modes inside each state-pattern class.
        # This prevents a legitimate long/energy-intensive program from being
        # rejected merely because the class's dominant program is shorter.
        mode_rows, mode_diagnostics, representative_rows = [], {}, []
        for class_id, entry in sorted(class_lookup.items()):
            eligible = [str(v) for v in entry.get("member_ids", [])
                        if str(v) in activity_rows
                        and activity_rows[str(v)]["passes_hard_checks"]]
            eligible = [mid for mid in eligible
                        if activity_rows[mid]["is_representative_signature"]]
            if not eligible:
                mode_diagnostics[str(class_id)] = {"selected_modes": 0}
                continue
            matrix = np.asarray([
                [activity_rows[mid][name] for name in metric_names]
                for mid in eligible
            ], dtype=np.float64)
            labels, diagnostics = discover_metric_modes(
                matrix, max_modes=self.max_metric_modes,
                min_mode_support=self.min_mode_support,
                bic_min_gain=self.mode_bic_min_gain,
                random_state=self.mode_random_state + class_id)
            mode_diagnostics[str(class_id)] = diagnostics
            for mid, label in zip(eligible, labels):
                activity_rows[mid]["mode_id"] = int(label)

            for mode_id in sorted(set(labels.tolist())):
                mode_members = [mid for mid in eligible
                                if activity_rows[mid]["mode_id"] == mode_id]
                for metric in metric_names:
                    scores = robust_z_scores(
                        activity_rows[mid][metric] for mid in mode_members)
                    for mid, score in zip(mode_members, scores):
                        activity_rows[mid][f"{metric}_robust_z"] = float(score)
                valid_mode_members = [
                    mid for mid in mode_members
                    if not any(activity_rows[mid][f"{name}_robust_z"]
                               > self.robust_z_threshold for name in metric_names)
                ]
                mode_row = {
                    "class_id": class_id,
                    "mode_id": int(mode_id),
                    "support": len(mode_members),
                    "valid_members": len(valid_mode_members),
                    "outlier_members": len(mode_members) - len(valid_mode_members),
                }
                for name in metric_names:
                    values = [activity_rows[mid][name] for mid in mode_members]
                    mode_row[f"median_{name}"] = float(np.median(values))
                mode_rows.append(mode_row)

                if valid_mode_members:
                    z_matrix = np.asarray([
                        [activity_rows[mid][f"{name}_robust_z"] for name in metric_names]
                        for mid in valid_mode_members
                    ], dtype=np.float64)
                    distances = np.linalg.norm(np.nan_to_num(
                        z_matrix, nan=0.0, posinf=1e6, neginf=1e6), axis=1)
                    ranked = np.argsort(distances)
                    choices = [("medoid", int(ranked[0]))]
                    if len(ranked) > 1:
                        choices.append(("near", int(ranked[1])))
                        choices.append(("far", int(ranked[-1])))
                    seen = set()
                    for role, position in choices:
                        mid = valid_mode_members[position]
                        if mid in seen:
                            continue
                        seen.add(mid)
                        representative_rows.append({
                            "class_id": class_id, "mode_id": int(mode_id),
                            "role": role, "activity_id": mid,
                            "file": activity_rows[mid]["file"],
                            "distance": float(distances[position]),
                        })

        for row in activity_rows.values():
            reasons = list(row.pop("_hard_reasons"))
            if row["passes_hard_checks"] and not row["is_representative_signature"]:
                reasons.append("signature_variant")
                row["mode_id"] = -2
            elif row["passes_hard_checks"]:
                if any(row.get(f"{name}_robust_z", np.inf) > self.robust_z_threshold
                       for name in metric_names):
                    reasons.append("mode_metric_outlier")
            else:
                row["mode_id"] = -1
            row["is_valid_member"] = not reasons
            row["rejection_reasons"] = ";".join(reasons)
            row["signature"] = "->".join(map(str, row.get("signature", [])))

        class_rows = []
        valid_class_ids, valid_activity_ids = [], []
        for class_id, entry in sorted(class_lookup.items()):
            members = [str(v) for v in entry.get("member_ids", [])
                       if str(v) in activity_rows]
            valid_members = [mid for mid in members
                             if activity_rows[mid]["is_valid_member"]]
            representative_members = [
                mid for mid in members
                if activity_rows[mid]["is_representative_signature"]]
            ratio = float(len(valid_members) / max(len(members), 1))
            purity = signature_purity(entry)
            signature = [int(v) for v in entry.get("representative_signature", [])]
            reasons = []
            if int(entry.get("support", 0)) < self.min_class_support:
                reasons.append("low_support")
            if required_states and not required_states.issubset(set(signature)):
                reasons.append("missing_common_state")
            if terminal_states and (not signature or signature[-1] not in terminal_states):
                reasons.append("uncommon_terminal_state")
            median_duration = float((entry.get("duration_samples") or {}).get("median", 0.0)) / self.fs
            if not valid_members:
                status = "invalid"
                reasons.append("no_valid_members")
            elif median_duration < self.min_duration_seconds:
                status = "valid_short" if not reasons else "uncertain"
                reasons.append("short_program")
            elif reasons:
                status = "uncertain"
            elif purity < self.min_signature_purity or ratio < self.min_valid_member_ratio:
                status = "uncertain"
                if purity < self.min_signature_purity:
                    reasons.append("low_signature_purity")
                if ratio < self.min_valid_member_ratio:
                    reasons.append("low_valid_member_ratio")
            else:
                status = "valid_full"
            override = self.class_overrides.get(str(class_id))
            if override:
                status = override
                reasons.append("manual_override")
            if status == "valid_full" and valid_members:
                valid_class_ids.append(class_id)
                valid_activity_ids.extend(valid_members)
            class_rows.append({
                "class_id": class_id,
                "status": status,
                "support": int(entry.get("support", 0)),
                "representative_members": len(representative_members),
                "valid_members": len(valid_members),
                "valid_member_ratio": ratio,
                "signature_purity": purity,
                "representative_signature": "->".join(map(str, signature)),
                "median_duration_seconds": median_duration,
                "reasons": ";".join(reasons),
            })

        valid_id_set = set(valid_activity_ids)
        filtered = copy.deepcopy(payload)
        filtered["version"] = max(2, int(filtered.get("version", 1)))
        filtered["validation"] = {
            "method": "grammar_boundary_gmm_mode_robust_mad",
            "valid_class_ids": valid_class_ids,
            "valid_activity_ids": sorted(valid_id_set, key=int),
            "grammar": grammar,
            "mode_diagnostics": mode_diagnostics,
        }
        filtered["activities"] = {
            key: value for key, value in filtered.get("activities", {}).items()
            if key in valid_id_set
        }
        for activity_id, record in filtered["activities"].items():
            record["validation_mode_id"] = int(activity_rows[activity_id]["mode_id"])
        filtered_classes = []
        for entry in filtered.get("classes", []):
            if int(entry["class_id"]) not in valid_class_ids:
                continue
            entry["member_ids"] = [mid for mid in entry.get("member_ids", [])
                                   if str(mid) in valid_id_set]
            entry["support"] = len(entry["member_ids"])
            if entry["member_ids"]:
                filtered_classes.append(entry)
        filtered["classes"] = filtered_classes
        filtered["n_classes"] = len(filtered_classes)
        filtered["n_activities"] = len(filtered["activities"])

        log_dir = self.log_dir(context)
        report_path = os.path.join(log_dir, "cycle_validity_report.csv")
        class_path = os.path.join(log_dir, "class_validity_summary.csv")
        whitelist_path = os.path.join(log_dir, "class_whitelist.json")
        catalog_path = os.path.join(log_dir, "validated_cycle_classes.json")
        grammar_path = os.path.join(log_dir, "inferred_cycle_grammar.json")
        mode_path = os.path.join(log_dir, "cycle_mode_summary.csv")
        representatives_path = os.path.join(log_dir, "mode_representatives.csv")
        diagnostics_path = os.path.join(log_dir, "mode_diagnostics.json")
        self._write_csv(report_path, sorted(activity_rows.values(),
                                            key=lambda row: int(row["activity_id"])), [
            "activity_id", "file", "class_id", "mode_id", "signature",
            "is_representative_signature",
            "passes_hard_checks", "is_valid_member", "rejection_reasons",
            "structural_warnings", "duration_seconds", "energy_wh", "mean_power",
            "max_power", "missing_ratio", "start_power_median",
            "end_power_median", "boundary_limit", "distance_to_representative",
            *[f"{name}_robust_z" for name in metric_names],
        ])
        self._write_csv(class_path, class_rows, [
            "class_id", "status", "support", "representative_members",
            "valid_members", "valid_member_ratio",
            "signature_purity", "representative_signature", "median_duration_seconds",
            "reasons",
        ])
        self._write_csv(mode_path, mode_rows, [
            "class_id", "mode_id", "support", "valid_members", "outlier_members",
            *[f"median_{name}" for name in metric_names],
        ])
        self._write_csv(representatives_path, representative_rows, [
            "class_id", "mode_id", "role", "activity_id", "file", "distance",
        ])
        with open(whitelist_path, "w", encoding="utf-8") as f:
            json.dump({"valid_class_ids": valid_class_ids,
                       "valid_activity_ids": sorted(valid_id_set, key=int)},
                      f, indent=2, ensure_ascii=False)
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        with open(grammar_path, "w", encoding="utf-8") as f:
            json.dump(grammar, f, indent=2, ensure_ascii=False)
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(mode_diagnostics, f, indent=2, ensure_ascii=False)

        self.record(context, artifacts={
            "cycle_report": self.rel(context, report_path),
            "class_summary": self.rel(context, class_path),
            "whitelist": self.rel(context, whitelist_path),
            "validated_cycle_classes": self.rel(context, catalog_path),
            "grammar": self.rel(context, grammar_path),
            "mode_summary": self.rel(context, mode_path),
            "mode_representatives": self.rel(context, representatives_path),
            "mode_diagnostics": self.rel(context, diagnostics_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "valid_class_ids": valid_class_ids,
            "n_valid_activities": len(valid_id_set),
        })
        print(f"[cycle_validation] inferred core states={sorted(required_states)}, "
              f"terminal states={sorted(terminal_states)}")
        for row in class_rows:
            print(f"  class_{row['class_id']}: {row['status']} "
                  f"({row['valid_members']}/{row['support']} valid) "
                  f"reasons={row['reasons'] or '-'}")
        for row in mode_rows:
            print(f"    class_{row['class_id']}/mode_{row['mode_id']}: "
                  f"{row['valid_members']}/{row['support']} valid, "
                  f"median_duration={row['median_duration_seconds']:.0f}s, "
                  f"median_energy={row['median_energy_wh']:.1f}Wh")
        print(f"[cycle_validation] {len(valid_id_set)} cycles in "
              f"{len(valid_class_ids)} full classes -> {log_dir}")
        return context
