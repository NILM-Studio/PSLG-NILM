"""Verify budget provenance and continuous dataset references without a GPU."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


GROUPS = ("A_real_only", "B_real_plus_traditional", "C_real_plus_generated")


def audit(dataset: Path, continuous: Path) -> dict:
    def read(root, name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    cycle = read(dataset, "nilm_dataset_manifest.json")
    generated = read(dataset, "budget_synthesis_manifest.json")
    timeline = read(continuous, "nilm_dataset_manifest.json")
    errors, budgets = [], {}

    def check(condition, message):
        if not condition:
            errors.append(message)

    def paths(root, files):
        result = [(root / name).resolve() for name in files]
        for path in set(result):
            check(path.is_file(), f"missing file: {path}")
        return Counter(result)

    check(cycle.get("synthesis_scope") == "budget_local", "cycle scope is not budget_local")
    check(timeline.get("synthesis_scope") == "budget_local", "continuous scope is not budget_local")
    entries = sorted(
        ((tag, row) for tag, row in cycle["experiments"].items() if tag != "full"),
        key=lambda item: item[1]["real_ratio"])
    check(bool(entries), "no budget experiments")
    check(set(generated) == {tag for tag, _ in entries}, "generated budget tags differ")
    previous = set()
    for tag, row in entries:
        selected = {str(value) for value in row["selected_real_activity_ids"]}
        fit = {str(value) for value in (row.get("synthesis_fit_activity_ids") or [])}
        n = row["selected_real_count"]
        check(previous <= selected, f"{tag}: real subsets are not nested")
        check(len(selected) == n and fit == selected, f"{tag}: fit IDs differ from real budget")
        check(all(row.get(key) == n for key in (
            "synthesis_fit_count", "selected_traditional_count", "selected_generated_count")),
            f"{tag}: A/B/C counts differ")
        previous = selected
        rows = generated.get(tag, [])
        check(len(rows) == n, f"{tag}: generated manifest count differs")
        check(Counter(str(r["source_activity_id"]) for r in rows) == Counter(selected),
              f"{tag}: anchors do not match selected cycles one-to-one")
        used = set()
        for record in rows:
            sources = {str(source["activity_index"])
                       for block in record["blocks"] for source in block["sources"]}
            used |= sources
            check(bool(sources) and sources <= selected, f"{tag}: empty or out-of-budget sources")
            check(sources == {str(v) for v in record["primitive_source_activity_ids"]},
                  f"{tag}: primitive source summary differs from blocks")
            check({str(v) for v in record["budget_activity_ids"]} == selected,
                  f"{tag}: recorded budget differs")
        check(Counter(row[GROUPS[2]]) == Counter(row[GROUPS[0]]) +
              Counter(record["file"] for record in rows), f"{tag}: C files differ from provenance")
        continuous_row = timeline["experiments"].get(tag, {})
        check({str(v) for v in (continuous_row.get("synthesis_fit_activity_ids") or [])} == fit,
              f"{tag}: continuous budget is stale")
        backgrounds = []
        for key in GROUPS:
            active = paths(dataset, row[key])
            actual = paths(continuous, continuous_row.get(key, []))
            check(not (active - actual), f"{tag}/{key}: continuous data misses active files")
            backgrounds.append(actual - active)
        check(bool(backgrounds[0]) and all(set(bg) == set(backgrounds[0]) for bg in backgrounds),
              f"{tag}: unique OFF backgrounds differ or are empty")
        check(backgrounds[1] == backgrounds[2], f"{tag}: B/C OFF repetitions differ")
        budgets[tag] = {"real": n, "generated": len(rows), "source_cycles": len(used)}
    return {"budget_provenance_passed": not errors, "budgets": budgets, "errors": errors,
            "upstream_representation_holdout": "not_verified",
            "note": "Budget provenance does not certify DETSEC/clustering fit isolation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster-tag", default="kmeans_k4_merged")
    args = parser.parse_args()
    root = Path("log") / args.run_id
    try:
        result = audit(
            root / f"nilm_dataset_strict_budget_local_cycle_augmentation_on_{args.cluster_tag}",
            root / f"nilm_continuous_dataset_strict_temporal_on_{args.cluster_tag}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"budget_provenance_passed": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["budget_provenance_passed"] else 1)


if __name__ == "__main__":
    main()
