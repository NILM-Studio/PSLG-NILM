"""Freeze four-object activity identity and chronological activity groups."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "runs/20261016_public_activity_segmentation_recalibration"
OUT = Path(os.environ["PSLG_EXECUTION_DIR"])
OBJECTS = ("refit_h7_dishwasher", "refit_h20_tumble_dryer",
           "ukdale_h2_washing_machine", "ukdale_h1_washing_machine_hotpoint")
EXPECTED = (725, 136, 56, 890)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def blocks(records):
    ordered = sorted(records, key=lambda row: (row["context_start"], row["activity_id"]))
    groups = []
    for row in ordered:
        if not groups or row["context_start"] > groups[-1]["end"]:
            groups.append({"end": row["context_end"], "ids": []})
        groups[-1]["end"] = max(groups[-1]["end"], row["context_end"])
        groups[-1]["ids"].append(row["activity_id"])
    return groups


def split(groups, total):
    sums = []
    count = 0
    for group in groups:
        count += len(group["ids"])
        sums.append(count)
    if len(groups) < 3:
        raise ValueError("Fewer than three disjoint chronological activity groups")
    first = min(range(1, len(groups)-1), key=lambda i: abs(sums[i-1] - .6 * total))
    second = min(range(first+1, len(groups)), key=lambda i: abs(sums[i-1] - .8 * total))
    result = {}
    for name, chunk in zip(("development", "validation", "evaluation"),
                           (groups[:first], groups[first:second], groups[second:])):
        result[name] = [aid for group in chunk for aid in group["ids"]]
    assert sum(map(len, result.values())) == total
    return result


def main():
    prepared_map = load(PREVIOUS / "data/prepared_map.json")
    source_map = load(PREVIOUS / "data/baseline_evidence.json")
    result = {"source": str(PREVIOUS), "objects": {}}
    hashes = []
    for obj, expected in zip(OBJECTS, EXPECTED):
        base = Path(prepared_map[obj]) / "analysis" / obj
        acts_path = base / "activities.json"
        acts = load(acts_path)
        assert len(acts) == expected, (obj, len(acts), expected)
        source = next(item for item in source_map["targets"] if item["target"] == obj)
        old = Path(source["source_execution"]) / "analysis" / obj
        assert acts == load(old / "activities.json"), obj
        rows = []
        for a in acts:
            aid = a["activity_id"]
            path = base / "activities" / f"{aid:05d}.npz"
            original = old / "activities" / f"{aid:05d}.npz"
            assert sha(path) == sha(original), (obj, aid)
            with np.load(path) as z:
                n = len(z["power"])
                assert n == a["n_points"] and len(z["timestamp"]) == n
            rows.append({"activity_id": aid, "path": str(path), "sha256": sha(path),
                         "n_points": n, "core_start": a["core_start"],
                         "context_start": a["context_start"], "context_end": a["context_end"],
                         "core_a": a["core_a"], "core_b": a["core_b"],
                         "left_censored": a["left_censored"],
                         "right_censored": a["right_censored"]})
        groups = blocks(rows)
        ids = split(groups, len(rows))
        by_id = {row["activity_id"]: row for row in rows}
        assert len(by_id) == len(rows)
        result["objects"][obj] = {"activity_manifest_sha256": sha(acts_path),
                                  "source_execution": str(base), "group_count": len(groups),
                                  "split": ids, "records": rows,
                                  "counts": {name: len(values) for name, values in ids.items()}}
        hashes.append({"object": obj, "activities_sha256": sha(acts_path),
                       "npz_merkle_sha256": hashlib.sha256("".join(row["sha256"] for row in rows).encode()).hexdigest()})
        print(obj, result["objects"][obj]["counts"], "groups", len(groups), flush=True)
    save(EXP / "data/frozen_inputs_and_split.json", result)
    save(OUT / "analysis/frozen_inputs_and_split.json", result)
    save(OUT / "inputs.json", {"source": str(PREVIOUS), "hashes": hashes,
                               "selection": "chronological overlap groups; 60/20/20 target"})
    save(OUT / "analysis/validation.json", {"status": "passed", "activity_count": sum(EXPECTED),
                                            "all_npz_equal_prior_source": True,
                                            "objects": list(OBJECTS)})


if __name__ == "__main__":
    main()
