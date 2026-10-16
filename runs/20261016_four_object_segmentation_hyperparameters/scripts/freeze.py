"""Freeze one validation-selected parameter set per object and method."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from choose_finalists import score, OBJECTS

ROOT = Path(__file__).resolve().parents[1]


def freeze(obj, method):
    condition="validate_"+obj+"_"+method.replace("-","_")
    paths=sorted((ROOT/"executions"/condition).glob("*/analysis/records.json"))
    found=None
    for path in reversed(paths):
        run=json.loads((path.parents[1]/"run.json").read_text())
        if run.get("exit_code")==0:
            found=path
            break
    if found is None:raise RuntimeError(f"No complete validation for {obj} {method}")
    rows=json.loads(found.read_text())
    by=defaultdict(list)
    for row in rows:by[row["config_id"]].append(row)
    ranking=[]
    for config_id,group in by.items():
        value,parts=score(group)
        ranking.append({"config_id":config_id,"params":group[0]["params"],
           "success":sum(row["status"]=="success" for row in group),
           "unsupported":sum(row["status"]=="unsupported" for row in group),
           "failed":sum(row["status"]=="failed" for row in group),
           "score":value,"score_components":parts,
           "wall_seconds":sum(row["wall_seconds"] for row in group)})
    if any(row["failed"] for row in ranking):raise RuntimeError(f"Validation has failures: {found}")
    ranking.sort(key=lambda row:(-row["success"],row["score"],row["wall_seconds"],
                                 json.dumps(row["params"],sort_keys=True)))
    winner=ranking[0]
    params={k:v for k,v in winner["params"].items() if not k.endswith("_seconds_target")}
    return {"validation_execution":str(found.parents[1]),"selected":winner,
            "run_params":params,"ranking":ranking}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--method",required=True)
    args=parser.parse_args()
    result={obj:freeze(obj,args.method) for obj in OBJECTS}
    path=ROOT/"data"/f"selected_{args.method.replace('-', '_')}.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    if args.method=="clasp":
        secondary={obj:{"params":entry["run_params"],
                        "validation_execution":entry["validation_execution"]}
                   for obj,entry in result.items()}
        (ROOT/"data/selected_clasp.json").write_text(
            json.dumps(secondary,ensure_ascii=False,indent=2)+"\n")
    print(path)
    for obj,entry in result.items():
        print(obj,"selected",entry["run_params"],"success",entry["selected"]["success"],
              "score",round(entry["selected"]["score"],5))


if __name__=="__main__":main()
