"""Read back a composition study and verify pairing and waveform provenance."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.validate_generation_run import sha256_file
from src.generation.primitive_composition import LEGACY_METHODS, METHODS, waveform_metrics
from src.generation.composition_evaluation import transition_observations


def audit_composition(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "composition_manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((root / "input_identity.json").read_text(encoding="utf-8"))
    fingerprint = identity.pop("fingerprint")
    errors, files_checked, references_checked = [], 0, 0
    schema = manifest.get("schema_version", 1)
    if schema not in (1, 2):
        raise ValueError(f"unsupported composition schema: {schema}")
    methods = LEGACY_METHODS if schema == 1 else METHODS

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
          == fingerprint == manifest["input_fingerprint"], "input identity mismatch")
    check(manifest["configuration"] == identity["configuration"], "configuration differs from input identity")
    check(manifest["methods"] == list(methods), "method matrix differs")
    check(manifest.get("seam_smoothing") is False, "unexpected seam smoothing")
    check(manifest.get("first_candidate_shared") is True, "unpaired first-state policy")
    if schema == 2:
        weight = manifest["configuration"]["target_weight"]
        check(np.isfinite(weight) and weight >= 0, "invalid target weight")
        check(manifest["unit_selection"]["target_weight"] == weight, "unit selection target weight differs")
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
                if schema == 2:
                    source_length = option["source_length"]
                    target_length = option["target_length"]
                    valid_lengths = (isinstance(source_length, int) and source_length > 0
                                     and isinstance(target_length, int) and target_length > 0)
                    check(valid_lengths, f"{name}: invalid candidate lengths")
                    if valid_lengths:
                        ratio = target_length / source_length
                        check(option["duration_ratio"] == ratio, f"{name}: candidate duration ratio differs")
                        warp = manifest["configuration"]["max_warp"]
                        check(1 / warp <= ratio <= warp, f"{name}: candidate exceeds duration bounds")
                check(option["activity_id"] in donors and option["activity_id"] != anchor,
                      f"{name}: candidate source leak")
                check(option["state_label"] == template["state_label"]
                      and option["target_length"] == template["length_samples"], f"{name}: candidate template changed")
        lengths = [item["length_samples"] for item in case["template"]]
        expected_states = np.concatenate([np.full(item["length_samples"], item["state_label"], dtype=np.int64)
                                          for item in case["template"]])
        check(set(case["results"]) == set(methods), f"{name}: missing comparison arm")
        if schema == 2:
            reference = case["anchor_reference"]
            reference_path = (root / reference["file"]).resolve()
            check(reference["activity_id"] == anchor, f"{name}: reference anchor differs")
            if not reference_path.is_relative_to(root) or not reference_path.is_file():
                errors.append(f"{name}: missing or external anchor reference")
            else:
                check(sha256_file(reference_path) == reference["file_sha256"], f"{name}: anchor reference file changed")
                with np.load(reference_path, allow_pickle=False) as payload:
                    reference_power = payload["appliance"]
                    check(reference_power.ndim == 1 and len(reference_power) == sum(lengths)
                          and np.isfinite(reference_power).all() and np.all(reference_power >= 0),
                          f"{name}: invalid anchor reference power")
                    check(np.array_equal(payload["state_label"], expected_states), f"{name}: anchor reference states differ")
                    check(np.array_equal(payload["timestamp"], np.arange(sum(lengths)) * manifest["configuration"]["sample_period"]),
                          f"{name}: anchor reference time differs")
                references_checked += 1
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
                if schema == 2:
                    observations = transition_observations(
                        power, lengths, [item["state_label"] for item in case["template"]],
                        sample_period=manifest["configuration"]["sample_period"])
                    check(observations == row["transition_observations"],
                          f"{name}/{method}: transition observations differ")
                    donor_samples = Counter()
                    for source in row["sources"]:
                        donor_samples[source["activity_id"]] += source["target_length"]
                    check(row["largest_donor_sample_fraction"] == max(donor_samples.values()) / sum(lengths),
                          f"{name}/{method}: donor fraction differs")
                    check(row["nearest_donor_activity_id"] in donors,
                          f"{name}/{method}: nearest donor outside budget")
                    check(np.isfinite(row["nearest_donor_resampled_nrmse"])
                          and row["nearest_donor_resampled_nrmse"] >= 0,
                          f"{name}/{method}: invalid donor NRMSE")
                    target = float(sum(abs(np.log(source["target_length"] / source["source_length"]))
                                       for source in row["sources"]))
                    check(np.isclose(target, row["duration_target_cost"], rtol=1e-12, atol=1e-12),
                          f"{name}/{method}: duration target cost differs")
                    boundary = float(sum(np.log1p(abs(value))
                                         for value in expected_metrics["signed_boundary_jumps_watts"]))
                    check(np.isclose(boundary, row["boundary_objective"], rtol=1e-12, atol=1e-12),
                          f"{name}/{method}: boundary objective differs")
                    check(np.isfinite(row["transition_objective"]) and row["transition_objective"] >= 0,
                          f"{name}/{method}: invalid transition objective")
                    objective = row["transition_objective"] + weight * target
                    check(np.isclose(objective, row["unit_selection_objective"], rtol=1e-12, atol=1e-12),
                          f"{name}/{method}: unit selection objective differs")
        if schema == 2 and "unit_selection" in case["results"]:
            selected_cost = case["results"]["unit_selection"]["unit_selection_objective"]
            check(all(selected_cost <= row["unit_selection_objective"] + 1e-9
                      for row in case["results"].values()), f"{name}: unit selection dominated by a compared path")
        check(len(first_indices) == 1, f"{name}: initial block is not paired")
    return {"passed": not errors, "paired_cases_checked": len(cases), "waveform_files_checked": files_checked,
            "reference_files_checked": references_checked,
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
