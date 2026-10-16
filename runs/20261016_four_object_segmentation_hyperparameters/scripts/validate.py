"""Run preregistered finalists on untouched validation activities."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import screen
from models.time_segmentation.comparison import Unsupported, run

OUT = Path(os.environ["PSLG_EXECUTION_DIR"])
EXP = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=screen.METHODS, required=True)
    parser.add_argument("--object", choices=tuple(screen.FROZEN["objects"]), required=True)
    args = parser.parse_args()
    obj, method = args.object, args.method
    finalists_path = EXP / "data" / f"finalists_{method.replace('-', '_')}.json"
    finalists = json.loads(finalists_path.read_text())[obj]["finalists"]
    selection = json.loads((EXP / "data/screening_selection.json").read_text())
    ids = set(selection[obj]["validation"])
    activities = [row for row in screen.FROZEN["objects"][obj]["records"]
                  if row["activity_id"] in ids]
    proxy = json.loads((Path(screen.FROZEN["objects"][obj]["source_execution"])
                        / "proxy_events.json").read_text())
    screen.save(OUT / "inputs.json", {
        "object":obj, "method":method, "activity_ids":sorted(ids),
        "activity_hashes":{str(row["activity_id"]):row["sha256"] for row in activities},
        "frozen_split_sha256":screen.sha(EXP/"data/frozen_inputs_and_split.json"),
        "finalists_sha256":screen.sha(finalists_path)})
    source = [Path(__file__),Path(screen.__file__),screen.ROOT/"models/time_segmentation/comparison.py",
              screen.ROOT/"models/time_segmentation/prim_glr.py"]
    snapshots=[]
    for file in source:
        dest=OUT/"analysis/source_code"/file.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest)
        snapshots.append({"path":str(file),"sha256":screen.sha(file),"snapshot":str(dest)})
    screen.save(OUT/"analysis/code_provenance.json",snapshots)
    packages={}
    for name in ("numpy","ruptures","stumpy","claspy","aeon","daproli","torch"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    screen.save(OUT/"environment.json",{"host":platform.node(),"python":sys.version,
       "executable":sys.executable,"packages":packages,
       "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES")})
    records=[]
    started=time.monotonic()
    for activity in activities:
        with np.load(activity["path"]) as z:
            t,x,ends=z["timestamp"].copy(),z["power"].copy(),z["end_timestamp"].copy()
        cache={}
        for finalist in finalists:
            params=finalist["params"]
            for seed in ([0,1,2] if method=="e2usd" else [0]):
                row={"object":obj,"method":method,"activity_id":activity["activity_id"],
                     "n_points":len(x),"config_id":screen.key(params),
                     "parent_config_id":finalist["parent_config_id"],
                     "params":params,"seed":seed}
                st=time.monotonic()
                try:
                    if method=="fluss-threshold":
                        window=params["window"]
                        if window not in cache:cache[window]=screen.fluss_profiles(x,window)
                        cuts=screen.fluss_cuts(cache[window],params["threshold"],window)
                    else:
                        runtime={k:v for k,v in params.items() if not k.endswith("_seconds_target")}
                        cuts,_=run(x,method,runtime,seed=seed)
                    row.update(status="success",cuts=cuts,
                               metrics=screen.metrics(activity,t,x,ends,cuts,
                                                      proxy[str(activity["activity_id"])]))
                except Unsupported as exc:
                    row.update(status="unsupported",reason=str(exc))
                except Exception as exc:
                    row.update(status="failed",reason=repr(exc),traceback=traceback.format_exc())
                row["wall_seconds"]=time.monotonic()-st
                records.append(row)
        screen.save(OUT/"analysis/records.json",records)
        screen.save(OUT/"analysis/progress.json",{"object":obj,"method":method,
                    "completed_activities":len({r["activity_id"] for r in records}),
                    "total_activities":len(activities),"finalists":len(finalists),
                    "elapsed_seconds":time.monotonic()-started})
        print("VALIDATE",obj,method,"activity",activity["activity_id"],
              "done",len({r["activity_id"] for r in records}),"/",len(activities),flush=True)
    screen.save(OUT/"analysis/validation.json",{"status":"complete",
       "activities":len(activities),"finalists":len(finalists),"attempts":len(records),
       "success":sum(r["status"]=="success" for r in records),
       "unsupported":sum(r["status"]=="unsupported" for r in records),
       "failed":sum(r["status"]=="failed" for r in records),
       "wall_seconds":time.monotonic()-started})
    if any(r["status"]=="failed" for r in records):
        raise SystemExit("Finalist failures recorded; inspect before freezing")


if __name__=="__main__":
    main()
