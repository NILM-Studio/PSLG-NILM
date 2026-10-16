"""Freeze one validation-selected parameter set per object and method."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from choose_finalists import score, OBJECTS

ROOT = Path(__file__).resolve().parents[1]


def freeze(obj, method, decision):
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
    choice=decision["decisions"][obj][method]
    if choice.get("approved") is not True or not choice.get("note"):
        raise RuntimeError(f"Human review approval and note required: {obj} {method}")
    winner=next((row for row in ranking if row["config_id"]==choice["config_id"]),None)
    if winner is None:
        raise RuntimeError(f"Reviewed config not among validated finalists: {obj} {method}")
    if method=="clap":
        approved_path=ROOT/"data/selected_clasp.json"
        if not approved_path.exists():
            raise RuntimeError("CLaP requires human-approved Clasp initialization")
        approved=json.loads(approved_path.read_text())[obj]["params"]
        if winner["params"]["initializer_params"]!=approved:
            raise RuntimeError(f"Exploratory CLaP uses a different Clasp initializer for {obj}; rerun required")
    params={k:v for k,v in winner["params"].items() if not k.endswith("_seconds_target")}
    return {"validation_execution":str(found.parents[1]),"selected":winner,
            "human_review":choice,
            "run_params":params,"ranking":ranking}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--method",required=True)
    parser.add_argument("--review-decisions",type=Path,required=True)
    args=parser.parse_args()
    decision=json.loads(args.review_decisions.read_text())
    if not decision.get("reviewer") or not decision.get("reviewed_at"):
        raise RuntimeError("Named reviewer and review timestamp required")
    safe=args.method.replace("-","_")
    for split in ("development","validation"):
        manifest=ROOT/"review/manifests"/f"{split}_{safe}.json"
        if not manifest.exists():
            raise RuntimeError(f"Missing ten-panel {split} gallery for {args.method}")
        expected=hashlib.sha256(manifest.read_bytes()).hexdigest()
        if decision.get("gallery_sha256",{}).get(split)!=expected:
            raise RuntimeError(f"Review decisions do not match {split} ten-panel gallery")
    result={obj:freeze(obj,args.method,decision) for obj in OBJECTS}
    path=ROOT/"data"/f"selected_{args.method.replace('-', '_')}.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    if args.method=="clasp":
        secondary={obj:{"params":entry["run_params"],
                        "validation_execution":entry["validation_execution"],
                        "human_review":entry["human_review"]}
                   for obj,entry in result.items()}
        (ROOT/"data/selected_clasp.json").write_text(
            json.dumps(secondary,ensure_ascii=False,indent=2)+"\n")
    print(path)
    for obj,entry in result.items():
        print(obj,"selected",entry["run_params"],"success",entry["selected"]["success"],
              "score",round(entry["selected"]["score"],5))


if __name__=="__main__":main()
