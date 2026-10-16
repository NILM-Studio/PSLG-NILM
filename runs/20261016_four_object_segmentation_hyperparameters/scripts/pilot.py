"""Registered source-package runtime and feasibility pilot on development activities."""
from __future__ import annotations

import json
import hashlib
import importlib.metadata
import math
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.time_segmentation.comparison import METHODS, Unsupported, run

OUT = Path(os.environ["PSLG_EXECUTION_DIR"])
NOMINAL = {"refit_h7_dishwasher": 8, "refit_h20_tumble_dryer": 8,
           "ukdale_h2_washing_machine": 6, "ukdale_h1_washing_machine_hotpoint": 6}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def params(method, nominal):
    if method == "prim-glr":
        return {"W": math.ceil(60/nominal), "min_gap": max(2, math.ceil(12/nominal)),
                "alpha": .8, "dur_sigma": .35, "g_wf": 16, "g_env": 8,
                "n_clusters": None}
    if method == "pelt-rbf":
        return {"penalty_multiplier": 3.0}
    if method == "fluss-threshold":
        return {"window": math.ceil(120/nominal), "threshold": .45, "excl_factor": 5}
    if method in {"clasp", "clap"}:
        return {}
    return {"window": 64, "steps": 20}


def main():
    data = load(EXP / "data/frozen_inputs_and_split.json")
    package_names = ("numpy", "scipy", "scikit-learn", "ruptures", "stumpy",
                     "claspy", "aeon", "daproli", "torch")
    packages = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    save(OUT / "environment.json", {"python": sys.version, "executable": sys.executable,
                                    "host": platform.node(), "packages": packages,
                                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                    "PYTHONPATH": os.environ.get("PYTHONPATH")})
    code = (Path(__file__), ROOT / "models/time_segmentation/comparison.py",
            ROOT / "models/time_segmentation/prim_glr.py",
            EXP / "configs/search_space.json")
    provenance = []
    for path in code:
        destination = OUT / "analysis/source_code" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "snapshot": str(destination)})
    save(OUT / "analysis/code_provenance.json", provenance)
    save(OUT / "inputs.json", {"frozen_split": str(EXP / "data/frozen_inputs_and_split.json"),
                               "frozen_split_sha256": hashlib.sha256((EXP / "data/frozen_inputs_and_split.json").read_bytes()).hexdigest()})
    rows = []
    for obj, item in data["objects"].items():
        train = set(item["split"]["development"])
        eligible = sorted((row for row in item["records"] if row["activity_id"] in train),
                          key=lambda row: (row["n_points"], row["activity_id"]))
        selected = [eligible[round((len(eligible)-1)*q)] for q in (.1, .5, .9)]
        print("PILOT_OBJECT", obj, [(r["activity_id"],r["n_points"]) for r in selected], flush=True)
        for method in METHODS:
            for activity in selected:
                started = time.monotonic()
                row = {"object": obj, "activity_id": activity["activity_id"],
                       "n_points": activity["n_points"], "method": method,
                       "params": params(method, NOMINAL[obj]), "seed": 0}
                try:
                    with np.load(activity["path"]) as z:
                        x = z["power"].copy()
                    cuts, info = run(x, method, row["params"], seed=0)
                    row.update(status="success", cuts=len(cuts), metadata=info)
                except Unsupported as exc:
                    row.update(status="unsupported", reason=str(exc))
                except Exception as exc:
                    row.update(status="failed", reason=repr(exc), traceback=traceback.format_exc())
                row["wall_seconds"] = time.monotonic() - started
                rows.append(row)
                save(OUT / "analysis/pilot_results.json", rows)
                print("PILOT", obj, method, activity["activity_id"], row["status"],
                      round(row["wall_seconds"], 2), row.get("reason", ""), flush=True)
    save(OUT / "analysis/validation.json", {"status": "complete", "attempts": len(rows),
                                            "success": sum(r["status"] == "success" for r in rows),
                                            "unsupported": sum(r["status"] == "unsupported" for r in rows),
                                            "failed": sum(r["status"] == "failed" for r in rows)})


if __name__ == "__main__":
    main()
