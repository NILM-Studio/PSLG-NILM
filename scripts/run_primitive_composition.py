"""CPU-only paired real-state composition study; no mains or neural training.

Inputs are inherited, frozen upstream artifacts and a train-only cycle split.
Validation cycles are used for descriptive diagnostics only, never selection.
Run with python -m scripts.run_primitive_composition --help.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance

from scripts.prepare_downstream_run import validate_run_id
from scripts.validate_generation_run import sha256_file, upstream_report
from src.framework.run_manifest import RunManifest
from src.generation.primitive_composition import (
    METHODS, StateBlock, StateCycle, TransitionReference, candidate_lattice,
    compose, lattice_digest, waveform_metrics,
)
from src.steps.nilm_dataset_step import NilmDatasetStep


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def load_cycles(context, split, expected_rows, cluster_tag):
    helper = NilmDatasetStep(cluster_tag, "unused_composition_requires_no_mains.csv")
    ids = {int(row["activity_id"]) for row in expected_rows}
    catalog, primitives, waveforms = helper._budget_resources(context, ids, f"{split}_catalog")
    activities = catalog["activities"]
    if set(map(int, activities)) != ids:
        raise ValueError(f"{split} catalog IDs differ from cycle split assignments")
    if (catalog.get("source_split", {}).get("name") != split
            or catalog.get("source_split", {}).get("structure_fit_scope") != "train_only"):
        raise ValueError(f"catalog is not declared {split}")
    verified = helper._budget_waveforms(ids, activities, primitives, waveforms)
    by_activity = defaultdict(list)
    for primitive in primitives:
        by_activity[primitive.activity_index].append(primitive)
    result = []
    for row in sorted(expected_rows, key=lambda value: int(value["activity_id"])):
        activity_id = int(row["activity_id"])
        activity = activities[str(activity_id)]
        group = int(row["class_id"]), int(row["mode_id"])
        if (int(activity["class_id"]), int(activity["validation_mode_id"])) != group:
            raise ValueError(f"class/mode mismatch for {activity_id}")
        if activity.get("source_split") != split:
            raise ValueError(f"activity {activity_id} is not declared {split}")
        blocks, cursor = [], 0
        for index, template in enumerate(activity["blocks"]):
            end = cursor + int(template["length_samples"])
            original_ids = tuple(p.primitive_id for p in by_activity[activity_id]
                                 if cursor <= p.start < end)
            blocks.append(StateBlock(activity_id, index, int(template["state_label"]), cursor,
                                     verified[activity_id][cursor:end], original_ids))
            cursor = end
        result.append(StateCycle(activity_id, *group, tuple(blocks)))
    return result


def load_inputs(run_root, cluster_tag, device_change_date=None):
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = RunManifest.load_or_create(str(manifest_path))
    entry = manifest.get_step("cycle_split") or {}
    if (entry.get("extra", {}).get("structure_fit_scope") != "train_only"
            or entry.get("extra", {}).get("cluster_tag") != cluster_tag):
        raise ValueError("rebuild temporal_holdout,cycle_classify,cycle_validate,cycle_split "
                         "in a new downstream run; train-only matching cycle structure required")
    # Integrity inspection is not model fitting; it may inspect all source eras.
    upstream = upstream_report(run_root, cluster_tag, device_change_date)
    if not upstream["upstream_interface_passed"]:
        raise ValueError("inherited interface failed; run validate_generation_run --stage upstream "
                         "and inspect its examples before composition")
    assignments_path = manifest.artifact_path("cycle_split", "assignments")
    holdout_path = manifest.artifact_path("temporal_holdout", "assignments")
    if not assignments_path or not holdout_path:
        raise ValueError("cycle and temporal assignments are required")
    with open(assignments_path, newline="", encoding="utf-8") as source:
        assignments = list(csv.DictReader(source))
    with open(holdout_path, newline="", encoding="utf-8") as source:
        temporal_rows = list(csv.DictReader(source))
    temporal = {int(row["activity_id"]): row for row in temporal_rows}
    if len(temporal) != len(temporal_rows) or len({row["activity_id"] for row in assignments}) != len(assignments):
        raise ValueError("duplicate activity IDs in assignments")
    for row in assignments:
        activity_id = int(row["activity_id"])
        if row["split"] not in ("train", "validation", "test") or temporal.get(activity_id, {}).get("split") != row["split"]:
            raise ValueError(f"cycle/temporal assignments disagree for activity {activity_id}")
    for earlier, later in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ends = [float(row["end_timestamp"]) for row in temporal_rows if row["split"] == earlier]
        starts = [float(row["start_timestamp"]) for row in temporal_rows if row["split"] == later]
        if ends and starts and max(ends) >= min(starts):
            raise ValueError(f"temporal overlap {earlier}/{later}; rebuild purged holdout")
    source_files = sorted(Path(manifest.artifact_path("extract_active_data", "segments_dir")).glob("*.csv"))
    actual_ranges = defaultdict(list)
    for row in assignments:
        activity_id = int(row["activity_id"])
        source_path = source_files[activity_id]
        if source_path.name != row["file"]:
            raise ValueError(f"source filename no longer matches activity {activity_id}")
        timestamps = pd.read_csv(source_path, usecols=["timestamp"])["timestamp"].to_numpy()
        start, end = float(timestamps[0]), float(timestamps[-1])
        declared = temporal[activity_id]
        # Legacy holdout stores integer seconds; allow truncation but not stale
        # source membership. Actual timestamps independently check isolation.
        if abs(start - float(declared["start_timestamp"])) >= 1 or abs(end - float(declared["end_timestamp"])) >= 1:
            raise ValueError(f"source timestamps no longer match holdout activity {activity_id}")
        actual_ranges[row["split"]].append((start, end))
    for earlier, later in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if (actual_ranges[earlier] and actual_ranges[later]
                and max(end for _, end in actual_ranges[earlier]) >= min(start for start, _ in actual_ranges[later])):
            raise ValueError(f"actual source timestamp overlap {earlier}/{later}")
    context = {"manifest": manifest, "log_root": str(run_root)}
    training_rows = [row for row in assignments if row["split"] == "train"]
    validation_rows = [row for row in assignments if row["split"] == "validation"]
    if not training_rows:
        raise ValueError("no eligible training cycles")
    training = load_cycles(context, "train", training_rows, cluster_tag)
    validation = (load_cycles(context, "validation", validation_rows, cluster_tag)
                  if validation_rows else [])
    hashes = {"run_manifest": sha256_file(manifest_path),
              "cycle_assignments": sha256_file(Path(assignments_path)),
              "temporal_assignments": sha256_file(Path(holdout_path))}
    for split in ("train", "validation"):
        path = manifest.artifact_path("cycle_split", f"{split}_catalog")
        if path:
            hashes[f"{split}_catalog"] = sha256_file(Path(path))
    return training, validation, upstream, hashes


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "median": float(np.median(values)) if len(values) else None,
            "p10": float(np.percentile(values, 10)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None}


def validation_diagnostics(cases, validation, period):
    """Independent descriptive comparisons, not a model-selection score."""
    references = defaultdict(list)
    for cycle in validation:
        references[cycle.group].append(waveform_metrics(
            cycle.power, [len(block.power) for block in cycle.blocks], period))
    generated = defaultdict(list)
    for case in cases:
        for method, row in case["results"].items():
            generated[(case["seed"], case["budget_tag"], tuple(case["class_mode"]), method)].append(row["metrics"])
    rows = []
    for (seed, budget, group, method), metrics in sorted(generated.items()):
        reference = references.get(group, [])
        entry = {"seed": seed, "budget_tag": budget, "class_mode": list(group), "method": method,
                 "generated_cycles": len(metrics), "validation_cycles": len(reference),
                 "status": "descriptive_only" if reference else "no_same_group_validation"}
        for name in ("energy_wh", "mean_watts", "peak_watts"):
            a = [item[name] for item in metrics]
            b = [item[name] for item in reference]
            entry[name] = {"generated": _distribution(a), "validation": _distribution(b),
                           "wasserstein": float(wasserstein_distance(a, b)) if b else None}
        a = [jump for item in metrics for jump in item["signed_boundary_jumps_watts"]]
        b = [jump for item in reference for jump in item["signed_boundary_jumps_watts"]]
        entry["signed_boundary_jump_watts"] = {
            "generated": _distribution(a), "validation": _distribution(b),
            "wasserstein": float(wasserstein_distance(a, b)) if a and b else None,
            "note": "Pooled different transition types; descriptive, not a physical validity certificate."}
        rows.append(entry)
    return rows


def run_study(run_root, output_dir=None, *, cluster_tag="kmeans_k4_merged",
              ratios=(0.05, 0.1, 0.2, 1.0), seeds=(42,), candidates=8,
              max_warp=2.0, min_fit_cycles=3, neighbors=3, window=5,
              max_anchors=30, sample_period=6.0, device_change_date=None):
    ratios, seeds = sorted(set(map(float, ratios))), sorted(set(map(int, seeds)))
    if (not ratios or any(not np.isfinite(ratio) or not 0 < ratio <= 1 for ratio in ratios)
            or not seeds or min(seeds) < 0 or candidates < 1 or max_anchors < 0
            or not np.isfinite(sample_period) or sample_period <= 0
            or not np.isfinite(max_warp) or max_warp < 1 or min_fit_cycles < 2
            or neighbors < 1 or window < 1):
        raise ValueError("invalid composition parameters")
    if len({f"{100 * ratio:g}pct" for ratio in ratios}) != len(ratios):
        raise ValueError("budget ratios produce ambiguous output tags")
    run_root = Path(run_root).resolve()
    training, validation, upstream, hashes = load_inputs(run_root, cluster_tag, device_change_date)
    configuration = dict(cluster_tag=cluster_tag, ratios=ratios, seeds=seeds, candidates=candidates,
                         max_warp=max_warp, min_fit_cycles=min_fit_cycles, neighbors=neighbors,
                         window=window, max_anchors=max_anchors, sample_period=sample_period,
                         device_change_date=device_change_date)
    project = Path(__file__).resolve().parents[1]
    identity = {"configuration": configuration, "source_run": str(run_root), "inputs": hashes,
                "source_signal_digest": upstream["source_digest"],
                "upstream_artifact_sha256": upstream["artifact_sha256"],
                "numpy_version": np.__version__, "scipy_version": scipy.__version__,
                "code": {name: sha256_file(project / name) for name in (
                    "scripts/run_primitive_composition.py", "src/generation/primitive_composition.py",
                    "scripts/audit_primitive_composition.py", "src/steps/nilm_dataset_step.py",
                    "scripts/validate_generation_run.py")}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    output = (Path(output_dir) if output_dir else run_root / f"primitive_composition_{fingerprint[:16]}").resolve()
    # Never overwrite a previous study, even if only partially written.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input_identity.json", {"fingerprint": fingerprint, **identity})
    write_json(output / "upstream_validation.json", upstream)
    cycles = {cycle.activity_id: cycle for cycle in training}
    records = [{"activity_id": str(cycle.activity_id), "class_id": cycle.class_id,
                "mode_id": cycle.mode_id} for cycle in training]
    cases, skipped, budgets = [], [], []
    for seed in seeds:
        order = NilmDatasetStep._stratified_order(records, np.random.default_rng(seed))
        for ratio in ratios:
            tag = f"{100 * ratio:g}pct"
            selected = [cycles[int(row["activity_id"])] for row in order[:max(1, int(np.ceil(ratio * len(order))))]]
            budget_ids = sorted(cycle.activity_id for cycle in selected)
            anchors = selected[:max_anchors] if max_anchors else selected
            budget_info = {"seed": seed, "budget_tag": tag, "real_ratio": ratio,
                           "selected_activity_ids": budget_ids,
                           "selected_cycles": len(selected), "requested_anchors": len(anchors)}
            budgets.append(budget_info)
            for anchor in anchors:
                donors = [cycle for cycle in selected if cycle.group == anchor.group
                          and cycle.activity_id != anchor.activity_id]
                case_id = f"seed{seed}_{tag}_anchor{anchor.activity_id}"
                lattice, reason = candidate_lattice(anchor, donors, seed, candidates, max_warp)
                if len(anchor.blocks) < 2:
                    reason = "fewer_than_two_state_blocks"
                if reason:
                    skipped.append({"case_id": case_id, "seed": seed, "budget_tag": tag,
                                    "anchor_activity_id": anchor.activity_id, "reason": reason})
                    continue
                reference = TransitionReference(donors, min_fit_cycles, neighbors, window)
                results = compose(lattice, reference, seed, anchor.activity_id)
                case_dir = output / case_id
                case_dir.mkdir()
                lengths = [len(block.power) for block in anchor.blocks]
                states = np.concatenate([np.full(len(block.power), block.state, dtype=np.int64)
                                         for block in anchor.blocks])
                timestamp = np.arange(len(states), dtype=np.float64) * sample_period
                case = {"case_id": case_id, "seed": seed, "budget_tag": tag,
                        "anchor_activity_id": anchor.activity_id, "class_mode": list(anchor.group),
                        "budget_activity_ids": budget_ids,
                        "donor_activity_ids": sorted(cycle.activity_id for cycle in donors),
                        "template": [{"state_label": block.state, "length_samples": len(block.power)}
                                     for block in anchor.blocks],
                        "candidate_lattice_sha256": lattice_digest(lattice),
                        "candidates": [[item.provenance() for item in options] for options in lattice],
                        "reference_model": reference.summary(), "results": {}}
                for method, value in results.items():
                    relative = f"{case_id}/{method}.npz"
                    np.savez_compressed(output / relative, timestamp=timestamp,
                                        appliance=value["power"], state_label=states)
                    provenance = [item.provenance() for item in value["selected"]]
                    sources = sorted({item["activity_id"] for item in provenance})
                    donor_errors = []
                    for donor in donors:
                        reference_power = np.interp(np.linspace(0, 1, len(value["power"])),
                                                    np.linspace(0, 1, len(donor.power)), donor.power)
                        donor_errors.append(float(np.sqrt(np.mean((value["power"] - reference_power) ** 2))
                                                  / max(float(np.max(reference_power)), 1.0)))
                    case["results"][method] = {
                        "file": relative, "file_sha256": sha256_file(output / relative),
                        "candidate_indices": value["candidate_indices"], "sources": provenance,
                        "source_activity_ids": sources, "single_donor_cycle": len(sources) == 1,
                        "nearest_donor_resampled_nrmse": min(donor_errors),
                        "boundary_objective": value["boundary_objective"],
                        "transition_objective": value["transition_objective"],
                        "transition_supported_edges": value["transition_supported_edges"],
                        "transition_fallback_edges": value["transition_fallback_edges"],
                        "metrics": waveform_metrics(value["power"], lengths, sample_period),
                    }
                cases.append(case)
            print(f"[composition] seed={seed} budget={tag} sources={len(selected)} "
                  f"anchors={len(anchors)} completed_cases={len(cases)}", flush=True)
    write_json(output / "composition_manifest.json", {
        "schema_version": 1, "input_fingerprint": fingerprint, "methods": list(METHODS),
        "configuration": configuration, "budgets": budgets, "cases": cases, "skipped": skipped,
        "waveform_source": "real_contiguous_inherited_state_blocks",
        "duration_adaptation": "shared_linear_interpolation", "seam_smoothing": False,
        "first_candidate_shared": True, "upstream_representation_holdout": "not_verified",
    })
    from scripts.audit_primitive_composition import audit_composition
    audit = audit_composition(output)
    learned_cases = sum(case["results"]["transition_dp"]["transition_supported_edges"] > 0 for case in cases)
    summary = {
        "output_dir": str(output), "input_fingerprint": fingerprint,
        "status": ("integrity_failed" if not audit["passed"] else "no_paired_cases" if not cases else "no_learned_transition_support" if not learned_cases
                   else "ready_for_descriptive_review"),
        "paired_cases": len(cases), "cases_with_supported_transitions": learned_cases,
        "skipped_cases": len(skipped), "skip_reasons": dict(Counter(row["reason"] for row in skipped)),
        "audit": audit, "configuration": configuration, "budgets": budgets,
        "validation_diagnostics": validation_diagnostics(cases, validation, sample_period),
        "paired_comparisons": [{"case_id": case["case_id"], "method": method,
            "boundary_objective_delta_vs_random": row["boundary_objective"] - case["results"]["random"]["boundary_objective"],
            "transition_objective_delta_vs_boundary_dp": row["transition_objective"] - case["results"]["boundary_dp"]["transition_objective"],
            "single_donor_cycle": row["single_donor_cycle"],
            "nearest_donor_resampled_nrmse": row["nearest_donor_resampled_nrmse"],
            "supported_edges": row["transition_supported_edges"],
            "fallback_edges": len(row["transition_fallback_edges"]),
        } for case in cases for method, row in case["results"].items()],
        "limitations": [
            "Optimization cost improvement is not independent evidence of quality or NILM improvement.",
            "No neural generation or NILM training in this study; NPZ contains no mains.",
            "Budget denominator: validated training catalog cycles, before mains alignment.",
            "Shared upstream representation and class/mode resources are outside the waveform budget.",
            "Validation is descriptive only; no test-based method selection or reported test performance.",
            "Singletons/missing donors are skipped for all arms; report coverage, not just successful cases.",
            "Same-donor replay and poor transition support can make apparently good scores uninformative.",
        ],
    }
    write_json(output / "composition_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster-tag", default="kmeans_k4_merged")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ratios", default="0.05,0.1,0.2,1.0")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-warp", type=float, default=2)
    parser.add_argument("--min-fit-cycles", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--max-anchors", type=int, default=30,
                        help="Per budget/seed diagnostic cap; 0 means all selected cycles")
    parser.add_argument("--sample-period", type=float, default=6)
    parser.add_argument("--device-change-date")
    args = parser.parse_args()
    try:
        run_id = validate_run_id(args.run_id)
        summary = run_study(Path("log") / run_id, args.output_dir, cluster_tag=args.cluster_tag,
                            ratios=args.ratios.split(","), seeds=args.seeds.split(","),
                            candidates=args.candidates, max_warp=args.max_warp,
                            min_fit_cycles=args.min_fit_cycles, neighbors=args.neighbors,
                            window=args.window, max_anchors=args.max_anchors,
                            sample_period=args.sample_period, device_change_date=args.device_change_date)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"[composition] {exc}\n")
    print(json.dumps({key: summary[key] for key in (
        "output_dir", "status", "paired_cases", "cases_with_supported_transitions", "skipped_cases", "audit")},
        indent=2, ensure_ascii=False))
    raise SystemExit(0 if summary["audit"]["passed"] and summary["paired_cases"] else 2)


if __name__ == "__main__":
    main()
