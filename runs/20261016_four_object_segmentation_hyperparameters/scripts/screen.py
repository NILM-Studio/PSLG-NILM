"""Registered per-object, per-method candidate screening on frozen development IDs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
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
from models.time_segmentation.comparison import Unsupported, run

OUT = Path(os.environ["PSLG_EXECUTION_DIR"])
SEARCH_PATH = Path(os.environ.get("PSLG_SEARCH_SPACE", EXP / "configs/search_space.json"))
SPACE = json.loads(SEARCH_PATH.read_text(encoding="utf-8"))
FROZEN = json.loads((EXP / "data/frozen_inputs_and_split.json").read_text(encoding="utf-8"))
NOMINAL = {item["id"]: item["nominal_seconds"] for item in SPACE["objects"]}
METHODS = ("prim-glr", "pelt-rbf", "fluss-threshold", "clasp", "clap", "e2usd")


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_ids(obj, split_name, count=12):
    item = FROZEN["objects"][obj]
    ids = set(item["split"][split_name])
    rows = [row for row in item["records"] if row["activity_id"] in ids]
    ordered = sorted(rows, key=lambda row: (row["core_start"], row["activity_id"]))
    length_order = sorted(rows, key=lambda row: (row["n_points"], row["activity_id"]))
    length_rank = {row["activity_id"]: index for index, row in enumerate(length_order)}
    buckets = {}
    for i, row in enumerate(ordered):
        key = (min(3, i*4//len(rows)), min(2, length_rank[row["activity_id"]]*3//len(rows)))
        buckets.setdefault(key, []).append(row)
    selected = []
    for key in sorted(buckets):
        bucket = buckets[key]
        selected.append(bucket[len(bucket)//2])
    selected_ids = {row["activity_id"] for row in selected}
    for row in ordered:
        if len(selected) >= min(count, len(rows)):
            break
        if row["activity_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["activity_id"])
    return sorted(selected[:min(count,len(rows))], key=lambda row: (row["core_start"],row["activity_id"]))


def create_selection():
    path = EXP / "data/screening_selection.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    selection = {obj: {name: [row["activity_id"] for row in choose_ids(obj,name)]
                       for name in ("development", "validation")}
                 for obj in FROZEN["objects"]}
    save(path, selection)
    return selection


def candidates(method, obj):
    dt = NOMINAL[obj]
    if method == "prim-glr":
        spec = SPACE["prim_glr"]
        windows = spec["W_seconds"] + (spec["h20_extra_W_seconds"] if obj == "refit_h20_tumble_dryer" else [])
        gaps = spec["min_gap_seconds"] + (spec["h20_extra_min_gap_seconds"] if obj == "refit_h20_tumble_dryer" else [])
        for w, gap, alpha in itertools.product(windows, gaps, spec["alpha"]):
            yield {"W": max(3,math.ceil(w/dt)), "min_gap": max(2,math.ceil(gap/dt)),
                   "alpha": alpha, "dur_sigma": spec["dur_sigma_main"],
                   "g_wf": 16, "g_env": 8, "n_clusters": None,
                   "W_seconds_target": w, "min_gap_seconds_target": gap}
    elif method == "pelt-rbf":
        for penalty, min_size in itertools.product(SPACE["pelt_rbf"]["penalty_multiplier"],
                                                    [5,math.ceil(60/dt)]):
            yield {"penalty_multiplier": penalty, "min_size": min_size,
                   "jump": SPACE["pelt_rbf"]["fixed"]["jump"]}
    elif method == "fluss-threshold":
        for w, threshold in itertools.product(SPACE["fluss_threshold"]["window_seconds"],
                                               SPACE["fluss_threshold"]["threshold"]):
            yield {"window": math.ceil(w/dt), "window_seconds_target": w,
                   "threshold": threshold, "excl_factor": 5}
    elif method == "clasp":
        for w, threshold in itertools.product(SPACE["clasp"]["window_size"],
                                               SPACE["clasp"]["significance_threshold"]):
            points = w if w == "suss" else math.ceil(int(w.split("_")[0])/dt)
            yield {"window_size": points, "validation": "significance_test",
                   "threshold": threshold}
    elif method == "clap":
        initializer_path = Path(os.environ.get("PSLG_CLASP_INITIALIZER_FILE",
                                               EXP / "data/selected_clasp.json"))
        selected = json.loads(initializer_path.read_text(encoding="utf-8"))
        initializer = selected[obj]["params"]
        for w, splits, size in itertools.product(SPACE["clap"]["window_size"],
                                                  SPACE["clap"]["n_splits"],
                                                  SPACE["clap"]["sample_size"]):
            points = w if w == "suss" else math.ceil(int(w.split("_")[0])/dt)
            yield {"initializer_params": initializer, "window_size": points,
                   "n_splits": splits, "sample_size": size}
    else:
        for window, steps in itertools.product(SPACE["e2usd"]["window_points"],
                                               SPACE["e2usd"]["steps"]):
            yield {"window": window, "steps": steps}


def key(params):
    return hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:12]


def proxy_match(events, cut_times, tolerance):
    i = j = matched = 0
    p = sorted(events)
    c = sorted(cut_times)
    while i < len(p) and j < len(c):
        if c[j] < p[i] - tolerance:
            j += 1
        elif c[j] > p[i] + tolerance:
            i += 1
        else:
            i += 1
            j += 1
            matched += 1
    return matched


def metrics(activity, t, x, ends, cuts, proxy):
    n = len(x)
    assert cuts == sorted(set(cuts)) and all(0 < cut < n for cut in cuts)
    a, b = activity["core_a"], activity["core_b"]
    bounds = [0] + cuts + [n]
    weight = ends - t
    assert np.isfinite(weight).all() and np.all(weight > 0)
    core = x[a:b]
    scale = max(float(np.quantile(core,.95) - np.quantile(core,.05)), 1.)
    durations = []
    sse = support = 0.
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        lo, hi = max(lo,a), min(hi,b)
        if lo >= hi:
            continue
        mean = float(np.average(x[lo:hi], weights=weight[lo:hi]))
        duration = float(weight[lo:hi].sum())
        durations.append(duration)
        sse += float((weight[lo:hi] * ((x[lo:hi]-mean)/scale)**2).sum())
        support += duration
    internal = [cut for cut in cuts if a < cut < b]
    events = proxy["timestamps"]
    matched = proxy_match(events, [float(t[c]) for c in internal], proxy["tolerance_seconds"])
    return {"core_cuts": len(internal), "core_segments": len(durations),
            "core_support_seconds": support, "core_normalized_sse": sse,
            "core_nrmse": math.sqrt(sse/support) if support else None,
            "core_short30": sum(d<30 for d in durations),
            "core_short30_fraction": sum(d<30 for d in durations)/len(durations) if durations else None,
            "core_cut_density_per10min": len(internal)*600/support if support else None,
            "proxy_total": len(events), "proxy_matched": matched,
            "all_cuts": len(cuts), "zero_internal_cut": len(internal)==0}


def fluss_profiles(x, window):
    if len(x) < 2*window or len(x)-window+1 <= 10*window:
        raise Unsupported(f"FLUSS insufficient length n={len(x)} window={window}")
    if np.ptp(x) <= 1e-12:
        return None
    os.environ["NUMBA_DISABLE_CUDA"] = "1"
    import numba
    numba.config.DISABLE_CUDA = 1
    import stumpy
    mp = stumpy.stump(x,m=window)
    if not np.isfinite(mp[:,0].astype(float)).any():
        raise Unsupported("FLUSS matrix profile has no finite neighbor distance")
    cac,_ = stumpy.fluss(mp[:,1].astype(int),L=window,n_regimes=2,excl_factor=5)
    cac = np.asarray(cac,float)
    if not np.isfinite(cac).all():
        raise RuntimeError("Nonfinite FLUSS CAC")
    return cac


def fluss_cuts(cac, threshold, window):
    if cac is None:
        return []
    profile = cac.copy()
    cuts = []
    while profile.min() <= threshold:
        cut = int(profile.argmin())
        cuts.append(cut)
        profile[max(0,cut-5*window):min(len(profile),cut+5*window)] = np.inf
    return sorted(set(cuts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",choices=METHODS,required=True)
    parser.add_argument("--object",choices=tuple(FROZEN["objects"]),required=True)
    args = parser.parse_args()
    method, obj = args.method, args.object
    selection = create_selection()
    activity_ids = set(selection[obj]["development"])
    activities = [row for row in FROZEN["objects"][obj]["records"] if row["activity_id"] in activity_ids]
    proxy = json.loads((Path(FROZEN["objects"][obj]["source_execution"]) / "proxy_events.json").read_text(encoding="utf-8"))
    configs = list({key(p):p for p in candidates(method,obj)}.values())
    inputs={"object":obj,"method":method,"activity_ids":sorted(activity_ids),
                               "activity_hashes": {str(a["activity_id"]):a["sha256"] for a in activities},
                               "frozen_split_sha256":sha(EXP/"data/frozen_inputs_and_split.json"),
                               "search_space_path":str(SEARCH_PATH),
                               "search_space_sha256":sha(SEARCH_PATH)}
    if method=="clap":
        initializer_path=Path(os.environ.get("PSLG_CLASP_INITIALIZER_FILE",
                                                EXP/"data/selected_clasp.json"))
        inputs["clasp_initializer_path"]=str(initializer_path)
        inputs["clasp_initializer_sha256"]=sha(initializer_path)
    save(OUT / "inputs.json", inputs)
    source = [Path(__file__), ROOT/"models/time_segmentation/comparison.py",
              ROOT/"models/time_segmentation/prim_glr.py"]
    snapshots = []
    for file in source:
        destination = OUT/"analysis/source_code"/file.name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,destination)
        snapshots.append({"path":str(file),"sha256":sha(file),"snapshot":str(destination)})
    save(OUT/"analysis/code_provenance.json",snapshots)
    packages = {}
    for name in ("numpy","ruptures","stumpy","claspy","aeon","daproli","torch"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    save(OUT/"environment.json", {"host":platform.node(),"python":sys.version,
                                   "executable":sys.executable,"packages":packages,
                                   "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   "PYTHONPATH":os.environ.get("PYTHONPATH")})
    records=[]
    started=time.monotonic()
    for activity in activities:
        with np.load(activity["path"]) as z:
            t,x,ends = z["timestamp"].copy(),z["power"].copy(),z["end_timestamp"].copy()
        profile_cache = {}
        for params in configs:
            config_id=key(params)
            row={"object":obj,"method":method,"activity_id":activity["activity_id"],
                 "n_points":len(x),"config_id":config_id,"params":params,"seed":0}
            st=time.monotonic()
            try:
                if method=="fluss-threshold":
                    window=params["window"]
                    if window not in profile_cache:
                        profile_cache[window]=fluss_profiles(x,window)
                    cuts=fluss_cuts(profile_cache[window],params["threshold"],window)
                else:
                    runtime_params={k:v for k,v in params.items() if not k.endswith("_seconds_target")}
                    cuts,_=run(x,method,runtime_params,seed=0)
                row.update(status="success",cuts=cuts,
                           metrics=metrics(activity,t,x,ends,cuts,proxy[str(activity["activity_id"])]))
            except Unsupported as exc:
                row.update(status="unsupported",reason=str(exc))
            except Exception as exc:
                row.update(status="failed",reason=repr(exc),traceback=traceback.format_exc())
            row["wall_seconds"]=time.monotonic()-st
            records.append(row)
        save(OUT/"analysis/progress.json", {"object":obj,"method":method,
                                            "completed_activities":len({r["activity_id"] for r in records}),
                                            "total_activities":len(activities),"configs":len(configs),
                                            "last_activity_id":activity["activity_id"],
                                            "elapsed_seconds":time.monotonic()-started})
        save(OUT/"analysis/records.json",records)
        print("SCREEN",obj,method,"activity",activity["activity_id"],
              "done",len({r["activity_id"] for r in records}),"/",len(activities),flush=True)
    save(OUT/"analysis/validation.json", {"status":"complete","activities":len(activities),
                                           "configs":len(configs),"attempts":len(records),
                                           "success":sum(r["status"]=="success" for r in records),
                                           "unsupported":sum(r["status"]=="unsupported" for r in records),
                                           "failed":sum(r["status"]=="failed" for r in records),
                                           "wall_seconds":time.monotonic()-started})
    if any(r["status"]=="failed" for r in records):
        raise SystemExit("Candidate failures recorded; inspect before selection")


if __name__=="__main__":
    main()
