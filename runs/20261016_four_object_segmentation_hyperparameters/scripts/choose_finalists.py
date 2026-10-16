"""Predeclared deterministic finalist selection from registered development screens."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = json.loads((ROOT / "configs/search_space.json").read_text())
OBJECTS = [item["id"] for item in SPACE["objects"]]


def score(rows):
    good = [r["metrics"] for r in rows if r["status"] == "success"]
    if not good:
        return 1e9, {}
    parts = {
        "nrmse": statistics.mean(m["core_nrmse"] for m in good),
        "cut_density": statistics.mean(m["core_cut_density_per10min"] for m in good),
        "short30_fraction": statistics.mean(m["core_short30_fraction"] for m in good),
        "proxy_miss_fraction": 1 - sum(m["proxy_matched"] for m in good) /
            max(1, sum(m["proxy_total"] for m in good)),
        "zero_cut_fraction": statistics.mean(m["zero_internal_cut"] for m in good),
    }
    total = parts["nrmse"] + .01*parts["cut_density"] + .02*sum(
        parts[name] for name in ("short30_fraction", "proxy_miss_fraction", "zero_cut_fraction"))
    return total, parts


def screen_rows(obj, method):
    if method == "fluss-threshold" and (ROOT / "amendment1_fluss.md").exists():
        condition = "screen_amendment1_" + obj + "_fluss_threshold"
    else:
        condition = "screen_" + obj + "_" + method.replace("-", "_")
    paths = sorted((ROOT / "executions" / condition).glob("*/analysis/records.json"))
    for path in reversed(paths):
        run = json.loads((path.parents[1] / "run.json").read_text())
        if run.get("exit_code") == 0:
            return path, json.loads(path.read_text())
    raise RuntimeError(f"No complete screen for {obj} {method}")


def baseline(method, params):
    if method == "prim-glr":
        return params["W_seconds_target"] == 60 and params["min_gap_seconds_target"] == 12 and params["alpha"] == .8
    if method == "pelt-rbf":
        return params["penalty_multiplier"] == 3 and params["min_size"] == 5
    if method == "fluss-threshold":
        return params["window_seconds_target"] == 120 and params["threshold"] == .45
    if method == "clasp":
        return params["window_size"] == "suss" and params["threshold"] == "default"
    if method == "clap":
        return params["window_size"] == "suss" and params["n_splits"] == 5 and params["sample_size"] == 1000
    if method == "e2usd":
        return params["window"] == 64 and params["steps"] == 20
    raise ValueError(method)


def select(obj, method):
    path, rows = screen_rows(obj, method)
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config_id"]].append(row)
    ranked = []
    for config_id, group in by_config.items():
        value, parts = score(group)
        ranked.append({"config_id": config_id, "params": group[0]["params"],
                       "success": sum(row["status"] == "success" for row in group),
                       "unsupported": sum(row["status"] == "unsupported" for row in group),
                       "failed": sum(row["status"] == "failed" for row in group),
                       "score": value, "score_components": parts,
                       "wall_seconds": sum(row["wall_seconds"] for row in group),
                       "is_baseline": baseline(method, group[0]["params"])})
    if any(row["failed"] for row in ranked):
        raise RuntimeError(f"Failed candidate in {path}")
    ranked.sort(key=lambda row: (-row["success"], row["score"], row["wall_seconds"],
                                 json.dumps(row["params"], sort_keys=True)))
    finalists = ranked[:3]
    default = next(row for row in ranked if row["is_baseline"])
    if default["config_id"] not in {row["config_id"] for row in finalists}:
        finalists.append(default)
    if method == "prim-glr":
        import copy
        expanded = []
        for row in finalists:
            for sigma in (.2,.35,.6):
                param = copy.deepcopy(row["params"])
                param["dur_sigma"] = sigma
                expanded.append({"params": param, "parent_config_id": row["config_id"],
                                 "development_success": row["success"],
                                 "development_score": row["score"]})
    else:
        expanded = [{"params": row["params"], "parent_config_id": row["config_id"],
                     "development_success": row["success"],
                     "development_score": row["score"]} for row in finalists]
    unique = {json.dumps(row["params"],sort_keys=True):row for row in expanded}
    return {"screen_execution": str(path.parents[1]), "ranking": ranked,
            "finalists": list(unique.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    result = {obj: select(obj,args.method) for obj in OBJECTS}
    destination = ROOT / "data" / f"finalists_{args.method.replace('-', '_')}.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(destination)
    for obj, block in result.items():
        print(obj, "candidates",len(block["ranking"]),"finalists",len(block["finalists"]),
              "best",block["ranking"][0]["params"])


if __name__ == "__main__":
    main()
