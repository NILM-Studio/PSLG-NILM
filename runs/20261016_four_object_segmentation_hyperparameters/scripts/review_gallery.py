"""Ten-panel human-review gallery: one activity, up to ten parameter settings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from choose_finalists import OBJECTS, baseline, screen_rows, score, select

EXP = Path(__file__).resolve().parents[1]
FROZEN = json.loads((EXP/"data/frozen_inputs_and_split.json").read_text())
SELECTION = json.loads((EXP/"data/screening_selection.json").read_text())


def read_rows(obj,method,split):
    if split=="development":
        path,rows=screen_rows(obj,method)
        ranked=select(obj,method)["ranking"]
    else:
        safe=method.replace("-","_")
        condition=f"validate_{obj}_{safe}"
        paths=sorted((EXP/"executions"/condition).glob("*/analysis/records.json"))
        path=None
        for candidate in reversed(paths):
            state=json.loads((candidate.parents[1]/"run.json").read_text())
            if state.get("exit_code")==0:
                path=candidate
                break
        if path is None:raise RuntimeError(f"Validation incomplete: {obj} {method}")
        rows=json.loads(path.read_text())
        grouped=defaultdict(list)
        for row in rows:grouped[row["config_id"]].append(row)
        ranked=[]
        for config_id,group in grouped.items():
            value,_=score(group)
            ranked.append({"config_id":config_id,"params":group[0]["params"],
                "success":sum(r["status"]=="success" for r in group),
                "score":value,"is_baseline":baseline(method,group[0]["params"])})
        ranked.sort(key=lambda x:(-x["success"],x["score"],
                         json.dumps(x["params"],sort_keys=True)))
        if method=="e2usd":
            ranked=[dict(candidate,seed=seed) for candidate in ranked for seed in (0,1,2)]
    return path,rows,ranked


def panel_order(ranked):
    chosen=ranked[:10]
    defaults=[row for row in ranked if row["is_baseline"]]
    if defaults and defaults[0]["config_id"] not in {row["config_id"] for row in chosen}:
        chosen=chosen[:9]+[defaults[0]]
    return chosen


def label(method,params):
    if method=="prim-glr":
        return (f"W={params['W_seconds_target']}s/{params['W']}pt "
                f"gap={params['min_gap_seconds_target']}s/{params['min_gap']}pt "
                f"α={params['alpha']} σ={params['dur_sigma']}")
    if method=="pelt-rbf":
        return f"penalty×{params['penalty_multiplier']} min_size={params['min_size']}pt"
    if method=="fluss-threshold":
        return f"window={params['window_seconds_target']}s/{params['window']}pt CAC≤{params['threshold']}"
    if method=="clasp":
        return f"window={params['window_size']} threshold={params['threshold']}"
    if method=="clap":
        return f"window={params['window_size']} folds={params['n_splits']} sample={params['sample_size']}"
    return f"window={params['window']}pt steps={params['steps']}"


def draw(obj,method,split,activity,ordered,lookup):
    aid=activity["activity_id"]
    with np.load(activity["path"]) as data:
        timestamps=data["timestamp"].copy()
        power=data["power"].copy()
    seconds=(timestamps-timestamps[0])
    core_a,core_b=activity["core_a"],activity["core_b"]
    ymin,ymax=np.percentile(power,[.5,99.5])
    pad=max((ymax-ymin)*.08,10)
    ymin=min(float(power.min()),float(ymin))-pad
    ymax=max(float(power.max()),float(ymax))+pad
    fig,axes=plt.subplots(5,2,figsize=(17,16),layout="constrained")
    entries=[]
    for position,ax in enumerate(axes.flat):
        ax.set_xlim(0,max(float(seconds[-1]),1))
        ax.set_ylim(ymin,ymax)
        if position>=len(ordered):
            ax.text(.5,.5,"No additional candidate",ha="center",va="center",
                    transform=ax.transAxes,color="#607d8b",fontsize=11)
            ax.set_axis_off()
            entries.append({"panel":position+1,"status":"unused"})
            continue
        candidate=ordered[position]
        seed=candidate.get("seed",0)
        row=lookup.get((aid,candidate["config_id"],seed))
        ax.plot(seconds,power,color="#263238",linewidth=.8)
        ax.axvspan(0,seconds[core_a],color="#b0bec5",alpha=.4)
        ax.axvspan(seconds[core_b-1],seconds[-1],color="#b0bec5",alpha=.4)
        ax.axvline(seconds[core_a],color="#546e7a",linewidth=.8)
        ax.axvline(seconds[core_b-1],color="#546e7a",linewidth=.8)
        cut_lines=[]
        if row and row["status"]=="success":
            for cut in row["cuts"]:
                relative=float(seconds[cut])
                ax.axvline(relative,color="#c62828",linestyle="--",linewidth=.9,alpha=.8)
                cut_lines.append(relative)
            core_cuts=row["metrics"]["core_cuts"]
            status=f"{core_cuts} core cuts / {len(row['cuts'])+1} all segments"
        else:
            status=row["status"] if row else "not evaluated"
            if row and row.get("reason"):
                ax.text(.02,.94,row["reason"][:70],transform=ax.transAxes,
                        color="#b71c1c",fontsize=7,va="top")
        name=label(method,candidate["params"])
        if method=="e2usd" and split=="validation":
            name+=f" seed={seed}"
        ax.set_title(f"{position+1}. {name} | {status}",fontsize=8)
        ax.set_xlabel("Elapsed seconds",fontsize=8)
        ax.set_ylabel("Power (W)",fontsize=8)
        ax.tick_params(labelsize=7)
        entries.append({"panel":position+1,"config_id":candidate["config_id"],
            "params":candidate["params"],"is_baseline":candidate["is_baseline"],
            "seed":seed,
            "status":row["status"] if row else "missing",
            "cuts":row.get("cuts",[]) if row else [],
            "cut_timestamps_utc":[float(timestamps[c]) for c in row.get("cuts",[])] if row else [],
            "cut_elapsed_seconds":cut_lines,"drawn_cut_lines":len(cut_lines),
            "source_execution":row.get("source_execution") if row else None})
    flags=" ".join(["CENSORED L" if activity["left_censored"] else "",
                    "CENSORED R" if activity["right_censored"] else ""]).strip() or "uncensored"
    dependency=""
    if method=="clap" and ordered:
        init=ordered[0]["params"]["initializer_params"]
        dependency=(f" | provisional Clasp init: window={init['window_size']} "
                    f"p={init['threshold']}")
    fig.suptitle(f"{obj} | {method} | {split} activity {aid} | {flags}{dependency}\n"
                 f"Same waveform and axes in all ten panels; red dashed lines = original cuts",
                 fontsize=12)
    stem=EXP/"review/figures"/split/obj/method.replace("-","_")/f"activity_{aid}"
    stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix(".png"),dpi=140)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    return {"object":obj,"method":method,"split":split,"activity_id":aid,
            "input_path":activity["path"],"input_sha256":activity["sha256"],
            "n_points":len(power),"core_a":core_a,"core_b":core_b,
            "censored_left":activity["left_censored"],
            "censored_right":activity["right_censored"],
            "png":str(stem.with_suffix(".png").relative_to(EXP)),
            "svg":str(stem.with_suffix(".svg").relative_to(EXP)),
            "panels":entries}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--method",required=True)
    parser.add_argument("--split",choices=("development","validation"),required=True)
    args=parser.parse_args()
    figures=[]
    sources={}
    for obj in OBJECTS:
        path,rows,ranked=read_rows(obj,args.method,args.split)
        sources[obj]=str(path)
        ordered=panel_order(ranked)
        lookup={(r["activity_id"],r["config_id"],r["seed"]):r for r in rows}
        activities={row["activity_id"]:row for row in FROZEN["objects"][obj]["records"]}
        for aid in SELECTION[obj][args.split][:10]:
            figures.append(draw(obj,args.method,args.split,activities[aid],ordered,lookup))
    safe=args.method.replace("-","_")
    dest=EXP/"review/manifests"/f"{args.split}_{safe}.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    payload={"experiment":"20261016_four_object_segmentation_hyperparameters",
             "purpose":"human_parameter_review","method":args.method,
             "split":args.split,"sources":sources,"figures":figures}
    dest.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    if "PSLG_EXECUTION_DIR" in os.environ:
        out=Path(os.environ["PSLG_EXECUTION_DIR"])/"analysis/gallery_manifest.json"
        out.write_text(json.dumps({"manifest":str(dest),
            "sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
            "figures":len(figures),"panels":len(figures)*10},indent=2)+"\n")
    print("GALLERY",args.method,args.split,len(figures),"figures",len(figures)*10,"panels")


if __name__=="__main__":main()
