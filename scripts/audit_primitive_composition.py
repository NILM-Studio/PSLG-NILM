"""Read back a composition study and verify pairing and waveform provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.validate_generation_run import sha256_file
from src.generation.primitive_composition import METHODS, waveform_metrics


def audit_composition(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "composition_manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "input_identity.json").read_text(encoding="utf-8"))
    fingerprint = identity.pop("fingerprint")
    errors, files_checked = [], 0

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
          == fingerprint == manifest["input_fingerprint"], "input identity mismatch")
    check(manifest["configuration"] == identity["configuration"], "configuration differs from input identity")
    check(manifest["methods"] == list(METHODS), "method matrix differs")
    check(manifest.get("seam_smoothing") is False, "unexpected seam smoothing")
    check(manifest.get("first_candidate_shared") is True, "unpaired first-state policy")
    budgets = {(row["seed"], row["budget_tag"]): set(row["selected_activity_ids"])
               for row in manifest["budgets"]}
    check(len(budgets) == len(manifest["budgets"]), "duplicate budgets")
    for seed in manifest["configuration"]["seeds"]:
        previous = set()
        for row in sorted((row for row in manifest["budgets"] if row["seed"] == seed),
                          key=lambda row: row["real_ratio"]):
            selected = set(row["selected_activity_ids"])
            check(previous <= selected, f"seed{seed}: nonnested budgets")
            previous = selected
    cases = manifest["cases"]
    check(len({case["case_id"] for case in cases}) == len(cases), "duplicate paired cases")
    skipped = manifest["skipped"]
    check(len({row["case_id"] for row in skipped}) == len(skipped), "duplicate skipped cases")
    check(not ({case["case_id"] for case in cases} & {row["case_id"] for row in skipped}),
          "case both completed and skipped")
    for row in manifest["budgets"]:
        observed = sum(case["seed"] == row["seed"] and case["budget_tag"] == row["budget_tag"]
                       for case in cases + skipped)
        check(observed == row["requested_anchors"], "unaccounted requested anchors")
    for case in cases:
        name = case["case_id"]
        anchor = case["anchor_activity_id"]
        budget = budgets[(case["seed"], case["budget_tag"])]
        donors = set(case["donor_activity_ids"])
        fit = set(case["reference_model"]["fit_activity_ids"])
        check(set(case["budget_activity_ids"]) == budget and anchor in budget, f"{name}: budget differs")
        check(donors <= budget and anchor not in donors and fit == donors, f"{name}: donor/fit budget leak")
        minimum = manifest["configuration"]["min_fit_cycles"]
        model = case["reference_model"]
        check(model["minimum_source_cycles"] == minimum, f"{name}: reference support threshold differs")
        for transition, values in model["transitions"].items():
            source_ids = set(values["source_activity_ids"])
            check(source_ids <= fit and len(source_ids) == len(values["source_activity_ids"]),
                  f"{name}/{transition}: transition reference source leak")
            check(values["supported"] == (len(source_ids) >= minimum),
                  f"{name}/{transition}: transition support differs")
        expected_fallback = [index for index, (left, right) in enumerate(zip(case["template"], case["template"][1:]))
                             if not model["transitions"].get(
                                 f"{left['state_label']}->{right['state_label']}", {}).get("supported", False)]
        candidate_hash = hashlib.sha256(json.dumps(case["candidates"], sort_keys=True, allow_nan=False).encode()).hexdigest()
        check(candidate_hash == case["candidate_lattice_sha256"], f"{name}: candidate lattice changed")
        check(len(case["candidates"]) == len(case["template"]), f"{name}: template/candidate block count differs")
        for template, options in zip(case["template"], case["candidates"]):
            check(bool(options), f"{name}: empty candidate pool")
            for option in options:
                check(option["activity_id"] in donors and option["activity_id"] != anchor,
                      f"{name}: candidate source leak")
                check(option["state_label"] == template["state_label"]
                      and option["target_length"] == template["length_samples"], f"{name}: candidate template changed")
        lengths = [item["length_samples"] for item in case["template"]]
        expected_states = np.concatenate([np.full(item["length_samples"], item["state_label"], dtype=np.int64)
                                          for item in case["template"]])
        check(set(case["results"]) == set(METHODS), f"{name}: missing comparison arm")
        first_indices = set()
        for method, row in case["results"].items():
            check(row["transition_fallback_edges"] == expected_fallback
                  and row["transition_supported_edges"] == len(lengths) - 1 - len(expected_fallback),
                  f"{name}/{method}: transition fallback reporting differs")
            path = (root / row["file"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                errors.append(f"{name}/{method}: missing or external waveform")
                continue
            check(sha256_file(path) == row["file_sha256"], f"{name}/{method}: waveform file changed")
            with np.load(path, allow_pickle=False) as payload:
                power = payload["appliance"]
                states = payload["state_label"]
                timestamp = payload["timestamp"]
            files_checked += 1
            check(power.ndim == 1 and len(power) == sum(lengths), f"{name}/{method}: duration changed")
            check(np.array_equal(states, expected_states), f"{name}/{method}: state order changed")
            check(np.array_equal(timestamp, np.arange(sum(lengths)) * manifest["configuration"]["sample_period"]),
                  f"{name}/{method}: time axis changed")
            check(np.isfinite(power).all() and np.all(power >= 0), f"{name}/{method}: invalid power")
            check(len(row["candidate_indices"]) == len(lengths) == len(row["sources"]),
                  f"{name}/{method}: incomplete source trace")
            first_indices.add(row["candidate_indices"][0])
            cursor, used_ids = 0, set()
            for length, index, source, options in zip(lengths, row["candidate_indices"], row["sources"], case["candidates"]):
                if not 0 <= index < len(options):
                    errors.append(f"{name}/{method}: invalid candidate index")
                    continue
                check(source == options[index], f"{name}/{method}: selected source not in shared pool")
                used_ids.add(source["activity_id"])
                actual_hash = hashlib.sha256(np.asarray(power[cursor:cursor + length], dtype="<f8").tobytes()).hexdigest()
                check(actual_hash == source["power_sha256"], f"{name}/{method}: block differs from candidate")
                cursor += length
            check(used_ids == set(row["source_activity_ids"]) and used_ids <= donors,
                  f"{name}/{method}: source summary differs")
            check(row["single_donor_cycle"] == (len(used_ids) == 1), f"{name}/{method}: replay flag differs")
            if power.ndim == 1 and len(power) == sum(lengths) and np.isfinite(power).all() and np.all(power >= 0):
                expected_metrics = waveform_metrics(power, lengths, manifest["configuration"]["sample_period"])
                check(expected_metrics == row["metrics"], f"{name}/{method}: waveform metrics differ")
        check(len(first_indices) == 1, f"{name}: initial block is not paired")
    return {"passed": not errors, "paired_cases_checked": len(cases), "waveform_files_checked": files_checked,
            "errors": errors, "scope": "Pairing, declared source budgets, file hashes and numerical integrity only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit_composition(args.study_dir)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result = {"passed": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
