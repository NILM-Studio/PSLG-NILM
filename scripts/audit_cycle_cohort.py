"""Trace source cycles through chronological splitting and downstream filtering.

Reads saved metadata only, never fits rules or changes membership. Test rows
are inspected for protocol coverage, not generator performance or tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.prepare_downstream_run import validate_run_id


SPLITS = ("train", "validation", "test")


def audit_cohort(run_root):
    root = Path(run_root).resolve()
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = {"run_manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    config_snapshot = root / "downstream_config.yaml"
    if config_snapshot.is_file():
        inputs["downstream_config"] = hashlib.sha256(config_snapshot.read_bytes()).hexdigest()

    def read(step, key):
        path = Path(manifest["steps"][step]["artifacts"][key])
        path = path if path.is_absolute() else root / path
        inputs[f"{step}.{key}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream)) if path.suffix == ".csv" else json.load(stream)

    def keyed(rows):
        result = {str(int(row["activity_id"])): row for row in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate activity IDs in cohort metadata")
        return result

    def truth(value):
        if value not in ("True", "False"):
            raise ValueError(f"invalid boolean in cycle report: {value!r}")
        return value == "True"

    temporal = keyed(read("temporal_holdout", "assignments"))
    holdout_summary = read("temporal_holdout", "summary")
    classification = read("cycle_classification", "cycle_classes")
    activities = classification["activities"]
    validity = keyed(read("cycle_validation", "cycle_report"))
    class_rows = read("cycle_validation", "class_summary")
    classes = {int(row["class_id"]): row for row in class_rows}
    whitelist = read("cycle_validation", "whitelist")
    catalog = read("cycle_validation", "validated_cycle_classes")
    retained = set(catalog["activities"])
    final = keyed(read("cycle_split", "assignments"))
    split_summary = read("cycle_split", "summary")
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(len(classes) == len(class_rows), "duplicate class IDs")
    check(classification.get("fit_scope") == "train_only"
          and catalog.get("validation", {}).get("fit_scope") == "train_only"
          and split_summary.get("structure_fit_scope") == "train_only", "structure fit is not train-only")
    check(all(row["split"] in (*SPLITS, "purged", "outside_cohort") for row in temporal.values()),
          "unknown temporal split")
    eligible = {key for key, row in temporal.items() if row["split"] in SPLITS}
    check(set(activities) == eligible, "classification membership differs from retained temporal cohort")
    classified = {key for key, row in activities.items() if int(row["class_id"]) >= 0}
    train_ids = {key for key, row in temporal.items() if row["split"] == "train"}
    for row in classification["classes"]:
        fit_ids = set(map(str, row["fit_member_ids"]))
        members = set(map(str, row["member_ids"]))
        check(fit_ids <= train_ids and fit_ids <= members <= classified,
              f"class {row['class_id']}: fitted or mapped outside training/cohort membership")
    check(set(validity) == classified, "validity report does not cover classified activities exactly")
    check(retained == set(final) == set(map(str, whitelist["valid_activity_ids"])),
          "validated catalog, whitelist and final assignments differ")
    check(retained <= classified, "retained cycle was not classified")
    check(set(map(int, whitelist["valid_class_ids"]))
          == {key for key, row in classes.items() if row["status"] == "valid_full"},
          "class whitelist differs from validity decisions")
    for key, row in activities.items():
        check(row.get("source_split") == temporal.get(key, {}).get("split"),
              f"{key}: classification split changed")
    cohort = holdout_summary.get("cohort", {})
    declared_cohort = manifest["steps"]["temporal_holdout"].get("extra", {}).get("cohort", {})
    check(cohort == declared_cohort, "cohort summary and manifest differ")
    if "source_activities" in holdout_summary:
        check(holdout_summary["source_activities"] == len(temporal), "source activity count differs")
        check(holdout_summary["cohort_excluded_count"]
              == sum(row["split"] == "outside_cohort" for row in temporal.values()), "cohort exclusion count differs")
    for key, row in temporal.items():
        start, end = float(row["start_timestamp"]), float(row["end_timestamp"])
        check(math.isfinite(start) and math.isfinite(end) and start <= end,
              f"{key}: invalid source interval")
        inside = ((cohort.get("start_timestamp") is None or start >= cohort["start_timestamp"])
                  and (cohort.get("end_timestamp") is None or end < cohort["end_timestamp"]))
        check(inside == (row["split"] != "outside_cohort"), f"{key}: temporal cohort window violated")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ends = [float(row["end_timestamp"]) for row in temporal.values() if row["split"] == left]
        starts = [float(row["start_timestamp"]) for row in temporal.values() if row["split"] == right]
        check(not ends or not starts or max(ends) < min(starts), f"temporal overlap: {left}/{right}")
    for key, row in final.items():
        source = activities.get(key, {})
        checks = validity.get(key, {})
        class_id = int(row["class_id"])
        check(row["split"] in SPLITS and row["split"] == temporal.get(key, {}).get("split"),
              f"{key}: final split differs")
        check(row["file"] == temporal.get(key, {}).get("file"), f"{key}: source filename differs")
        check(int(source.get("class_id", -1)) == class_id
              and int(checks.get("class_id", -1)) == class_id
              and classes.get(class_id, {}).get("status") == "valid_full"
              and checks.get("is_valid_member") == "True", f"{key}: retained ineligible cycle")
        check(int(row["mode_id"]) >= 0
              and int(row["mode_id"]) == int(checks.get("mode_id", -1))
              == int(catalog["activities"].get(key, {}).get("validation_mode_id", -1)),
              f"{key}: validated mode differs")
    stages, group_counts = {}, {}
    for split in SPLITS:
        ids = {key for key, row in temporal.items() if row["split"] == split}
        mapped = ids & classified
        reported = mapped & set(validity)
        valid = {key for key in reported if truth(validity[key]["is_valid_member"])}
        kept = ids & retained
        stages[split] = {
            "temporal_assigned": len(ids), "classified": len(mapped),
            "hard_checks_passed": sum(truth(validity[key]["passes_hard_checks"]) for key in reported),
            "hard_and_canonical": sum(truth(validity[key]["passes_hard_checks"])
                                      and truth(validity[key]["is_representative_signature"]) for key in reported),
            "individually_valid": len(valid), "retained": len(kept),
            "valid_but_class_excluded": len(valid - retained),
            "class_counts": dict(Counter(str(activities[key]["class_id"]) for key in ids & set(activities))),
            "excluded_valid_by_class": dict(Counter(str(activities[key]["class_id"]) for key in valid - retained)),
            "member_rejections": dict(Counter(validity[key]["rejection_reasons"] for key in reported if key not in valid)),
            "classification_rejections": dict(Counter(activities[key].get("outlier_reason", "unknown")
                                                        for key in ids & set(activities) - classified)),
        }
        check(holdout_summary["counts"][split] == len(ids), f"{split}: holdout count differs")
        check(split_summary["counts"][split] == len(kept), f"{split}: final count differs")
        group_counts[split] = Counter(f"{row['class_id']}/{row['mode_id']}"
                                      for row in final.values() if row["split"] == split)
    group_rows = [{"class_mode": group, **{split: group_counts[split][group] for split in SPLITS}}
                  for group in sorted(set().union(*(set(values) for values in group_counts.values())))]
    shared_validation = sorted(set(group_counts["train"]) & set(group_counts["validation"]))
    shared_test = sorted(set(group_counts["train"]) & set(group_counts["test"]))
    blockers = [f"empty_{split}_after_filtering" for split in SPLITS if not stages[split]["retained"]]
    if not shared_validation:
        blockers.append("no_train_validation_class_mode_overlap")
    if not shared_test:
        blockers.append("no_train_test_class_mode_overlap")
    return {
        "run_id": manifest.get("run_id"), "metadata_integrity_passed": not errors,
        "evaluation_ready": not errors and not blockers,
        "status": "metadata_integrity_failed" if errors else "evaluation_blocked" if blockers else "ready_for_composition",
        "errors": errors, "blockers": blockers, "cohort": cohort,
        "source_cycles": len(temporal),
        "outside_cohort": sum(row["split"] == "outside_cohort" for row in temporal.values()),
        "boundary_purged": sum(row["split"] == "purged" for row in temporal.values()),
        "stages": stages, "class_decisions": class_rows, "groups": group_rows,
        "train_groups_without_validation": sorted(set(group_counts["train"]) - set(group_counts["validation"])),
        "train_groups_without_test": sorted(set(group_counts["train"]) - set(group_counts["test"])),
        "input_sha256": inputs,
        "scope": "Saved metadata reconciliation and coverage only; no raw waveform check or test performance evaluation.",
        "limitations": ["A calendar cohort does not independently certify device identity or stationarity.",
                        "Inherited representation fit scope remains unverified.",
                        "Nonempty matching groups do not establish statistical sufficiency or cover every generated case."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-evaluation-ready", action="store_true")
    args = parser.parse_args()
    try:
        result = audit_cohort(Path("log") / validate_run_id(args.run_id))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"metadata_integrity_passed": False, "evaluation_ready": False,
                  "status": "metadata_integrity_failed", "errors": [str(exc)]}
    # Existing reports are evidence, including when a previous attempt failed.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(1 if not result["metadata_integrity_passed"] else
                     2 if args.require_evaluation_ready and not result["evaluation_ready"] else 0)


if __name__ == "__main__":
    main()
