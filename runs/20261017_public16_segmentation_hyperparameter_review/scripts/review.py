"""Registered complete-grid review on the provided public16 v2 activities."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
METHODS = ("pelt-rbf", "fluss-threshold", "clasp", "clap", "e2usd")
DATASET = ROOT / "datasets/public16_activity_states_20261017_v2"
PLAN = EXP / "configs/plan.json"
REPORT = EXP / "reports/full_grid_review"
OUT = Path(os.environ["PSLG_EXECUTION_DIR"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.item()) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def key(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def candidates(method, dt):
    if method == "pelt-rbf":
        for size, penalty in itertools.product(dict.fromkeys([5, math.ceil(60 / dt)]), (1.5, 3., 6., 12.)):
            yield {"penalty_multiplier": penalty, "min_size": size, "jump": 5}
    elif method == "fluss-threshold":
        for window, threshold in itertools.product((60, 120, 180), (.35, .45, .55, .65, .75, .85, .95)):
            yield {"window": math.ceil(window / dt), "window_seconds_target": window,
                   "threshold": threshold, "excl_factor": 5}
    elif method == "clasp":
        for window, threshold in itertools.product(("suss", math.ceil(60 / dt), math.ceil(120 / dt)),
                                                   ("default", 1e-8, 1e-4, .01)):
            yield {"window_size": window, "validation": "significance_test", "threshold": threshold}
    elif method == "clap":
        for window, splits, size in itertools.product(("suss", math.ceil(60 / dt)), (3, 5), (500, 1000)):
            yield {"initializer_params": {"window_size": "suss", "validation": "significance_test",
                                           "threshold": "default"},
                   "window_size": window, "n_splits": splits, "sample_size": size}
    elif method == "e2usd":
        for window, steps in itertools.product((32, 64), (20, 40)):
            yield {"window": window, "steps": steps}
    else:
        raise ValueError(method)


def baseline(method, params, dt):
    if method == "pelt-rbf":
        return params["penalty_multiplier"] == 3 and params["min_size"] == 5
    if method == "fluss-threshold":
        return params["window_seconds_target"] == 120 and params["threshold"] == .45
    if method == "clasp":
        return params["window_size"] == "suss" and params["threshold"] == "default"
    if method == "clap":
        return params["window_size"] == "suss" and params["n_splits"] == 5 and params["sample_size"] == 1000
    return params["window"] == 64 and params["steps"] == 20


def provenance():
    files = [Path(__file__), ROOT / "models/time_segmentation/comparison.py"]
    if (Path(__file__).parent / "e2usd_cpu.py").exists():
        files.append(Path(__file__).parent / "e2usd_cpu.py")
    snap = OUT / "analysis/source_code"
    snap.mkdir(exist_ok=True)
    for path in files:
        shutil.copy2(path, snap / path.name)
    save(OUT / "analysis/source_code.json", [{"path": str(p), "sha256": sha(p)} for p in files])
    versions = {}
    for name in ("numpy", "matplotlib", "ruptures", "stumpy", "claspy", "aeon", "daproli", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    save(OUT / "environment.runtime.json", {"python": sys.version, "host": platform.node(),
         "executable": sys.executable, "packages": versions,
         "environment": {k: os.getenv(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH",
           "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "PSLG_E2USD_DEVICE")}})


def prepare():
    assert not PLAN.exists(), "Plan exists; preserve the registered selection"
    data = load(DATASET / "manifest.json")
    assert data["object_count"] == 16 and data["activity_count"] == 4202
    prepared_map = ROOT / "runs/20261017_public16_primglr_feature_pca16/data/prepared_map.json"
    prepared = load(prepared_map)
    objects = []
    for entry in data["objects"]:
        obj = entry["object"]
        path = Path(entry["activity_manifest"])
        records = load(path)["records"]
        assert len(records) == entry["selected_interval"]["selected_activity_count"]
        refpath = Path(prepared[obj]["path"]) / "activities.json"
        refs = {r["activity_id"]: r for r in load(refpath)}
        selected = []
        # Selection depends on duration and ID only, before any comparison result.
        for split, quantile in (("development", .25), ("validation", .75)):
            rows = sorted([r for r in records if r["split"] == split],
                          key=lambda r: (r["duration_seconds"], r["activity_id"]))
            assert rows, (obj, split)
            row = copy.deepcopy(rows[round((len(rows) - 1) * quantile)])
            assert Path(row["npz"]).is_relative_to(DATASET / obj / "activities")
            assert sha(row["npz"]) == row["npz_sha256"]
            ref = refs[row["activity_id"]]
            assert ref["input_sha256"] == row["npz_sha256"] and ref["split"] == split
            row["primglr_reference"] = {k: ref[k] for k in ("full_context_cuts", "params", "effective_W_points")}
            row["selection_rule"] = f"duration_quantile_{quantile}_within_existing_{split}"
            selected.append(row)
        dt = float(entry["selected_interval"]["nominal_seconds"])
        grids = {m: [{"config_id": key(p), "params": p, "is_baseline": baseline(m, p, dt)}
                      for p in candidates(m, dt)] for m in METHODS}
        assert [len(grids[m]) for m in METHODS] == [8, 21, 12, 8, 4]
        objects.append({"object": obj, "nominal_seconds": dt, "activity_manifest": str(path),
                        "activity_manifest_sha256": sha(path), "activity_count": len(records),
                        "timezone": entry["selected_interval"]["source_timezone"],
                        "primglr_records": str(refpath), "primglr_records_sha256": sha(refpath),
                        "activities": selected, "grids": grids})
    plan = {"experiment": EXP.name, "dataset_root": str(DATASET),
            "dataset_manifest_sha256": sha(DATASET / "manifest.json"),
            "object_count": 16, "cohort_activity_count": 4202,
            "figures_per_object_method": 2, "expected_figures": 160,
            "methods": list(METHODS), "e2usd_seeds": [0, 1, 2], "other_seed": 0,
            "clap_initializer": "official_default_Clasp_provisional_pending_human_choice",
            "human_selection_required": True, "objects": objects}
    save(PLAN, plan)
    save(OUT / "inputs.json", {"dataset_manifest": str(DATASET / "manifest.json"),
         "dataset_manifest_sha256": plan["dataset_manifest_sha256"], "prepared_map": str(prepared_map),
         "prepared_map_sha256": sha(prepared_map)})
    save(OUT / "analysis/validation.json", {"status": "passed", "selected_activities": 32,
         "expected_figures": 160, "plan_sha256": sha(PLAN),
         "selected_evaluation_activities": 0, "Primglr_recomputed": False})
    print("PLAN", PLAN, "32 existing activities; 160 complete-grid figures", flush=True)


def label(method, p, dt):
    if method == "pelt-rbf":
        return f"penalty x{p['penalty_multiplier']:g}, min={p['min_size']}pt (~{p['min_size']*dt:g}s), jump=5"
    if method == "fluss-threshold":
        return f"window={p['window_seconds_target']}s/{p['window']}pt, CAC <= {p['threshold']:.2f}"
    if method == "clasp":
        window = p["window_size"]
        return f"window={window}" + (" (auto)" if isinstance(window, str) else f"pt (~{window*dt:g}s)") + f", p={p['threshold']}"
    if method == "clap":
        return f"window={p['window_size']}, folds={p['n_splits']}, sample={p['sample_size']}"
    return f"window={p['window']}pt (~{p['window']*dt:g}s), steps={p['steps']}"


def cached_fluss(x, params, cache):
    """Same official CAC/extraction as comparison.run; reuse CAC across thresholds."""
    import numpy as np
    from models.time_segmentation.comparison import Unsupported
    window, excl = params["window"], params["excl_factor"]
    n = len(x)
    if n < 2*window or n-window+1 <= 2*excl*window:
        raise Unsupported(f"FLUSS insufficient length n={n} window={window} excl_factor={excl}")
    if importlib.metadata.version("stumpy") != "1.14.1":
        raise RuntimeError("FLUSS requires pinned stumpy==1.14.1")
    info = {"source": "official stumpy.stump + stumpy.fluss corrected arc curve",
            "adaptation": "project CAC threshold and exclusion-zone local minima",
            "window": window, "threshold": params["threshold"], "excl_factor": excl,
            "n_regimes_for_cac": 2, "cut_coordinate": "subsequence start index"}
    if np.ptp(x) <= 1e-12:
        return [], {**info, "constant_signal": True}
    if window not in cache:
        os.environ["NUMBA_DISABLE_CUDA"] = "1"
        import numba
        numba.config.DISABLE_CUDA = 1
        import stumpy
        mp = stumpy.stump(x, m=window)
        if not np.isfinite(mp[:, 0].astype(float)).any():
            raise Unsupported("FLUSS matrix profile has no finite nearest-neighbor distances")
        cac, _ = stumpy.fluss(mp[:, 1].astype(np.int64), L=window, n_regimes=2, excl_factor=excl)
        cac = np.asarray(cac, dtype=float)
        if not np.isfinite(cac).all():
            raise RuntimeError("FLUSS returned a nonfinite corrected arc curve")
        cache[window] = cac
    profile = cache[window].copy()
    info.update(cac_min=float(profile.min()), matrix_profile_subsequences=len(profile))
    cuts = []
    while profile.min() <= params["threshold"]:
        cut = int(profile.argmin())
        if not 0 < cut < n:
            raise RuntimeError("FLUSS selected an endpoint cut")
        cuts.append(cut)
        profile[max(0, cut-excl*window):min(len(profile), cut+excl*window)] = np.inf
    return sorted(set(cuts)), info


def draw(obj, method, activity, configs, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    with np.load(activity["npz"]) as z:
        t, x = z["timestamp"].copy(), z["power"].copy()
    elapsed = t - t[0]
    a, b = activity["core_a"], activity["core_b"]
    seeds = [0, 1, 2] if method == "e2usd" else [0]
    panels = [(c, seed) for c in configs for seed in seeds]
    cols = {"pelt-rbf": 4, "fluss-threshold": 7, "clasp": 4, "clap": 4, "e2usd": 3}[method]
    height = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(height, cols, figsize=(4.6*cols, 2.8*height + .8), squeeze=False,
                             layout="constrained")
    span = max(float(np.ptp(x)), 10.)
    ylim = (float(x.min()) - span*.06, float(x.max()) + span*.08)
    lookup = {(r["config_id"], r["seed"]): r for r in rows}
    entries = []
    for i, (config, seed) in enumerate(panels):
        ax = axes.flat[i]
        r = lookup[config["config_id"], seed]
        ax.plot(elapsed, x, color="#263238", lw=.75)
        ax.axvspan(0, elapsed[a], color="#b0bec5", alpha=.4)
        ax.axvspan(elapsed[b-1], elapsed[-1], color="#b0bec5", alpha=.4)
        ax.axvline(elapsed[a], color="#607d8b", lw=.6)
        ax.axvline(elapsed[b-1], color="#607d8b", lw=.6)
        lines = []
        for cut in r.get("cuts", []):
            line = ax.axvline(elapsed[cut], color="#c62828", ls="--", lw=.85, alpha=.85)
            line.set_gid(f"cut_{i+1}_{cut}")
            lines.append(float(elapsed[cut]))
        ax.set_xlim(0, max(float(elapsed[-1]), 1.))
        ax.set_ylim(*ylim)
        status = (f"{r['core_cuts']} core cuts / {len(lines)} total cuts"
                  if r["status"] == "success" else r["status"].upper())
        name = f"{i+1}. {config['config_id']}" + (" [BASE]" if config["is_baseline"] else "")
        if method == "e2usd":
            name += f" seed={seed}"
        ax.set_title(name + "\n" + label(method, config["params"], obj["nominal_seconds"]) + "\n" + status, fontsize=7.5)
        if r.get("reason"):
            ax.text(.02, .95, r["reason"][:120], transform=ax.transAxes, fontsize=6.5, color="#b71c1c", va="top", wrap=True)
        ax.set_xlabel("Elapsed seconds", fontsize=7)
        ax.set_ylabel("Power (W)", fontsize=7)
        ax.tick_params(labelsize=6.5)
        entries.append({"panel": i+1, "config_id": config["config_id"], "params": config["params"],
                        "seed": seed, "is_baseline": config["is_baseline"], "status": r["status"],
                        "reason": r.get("reason"), "cuts": r.get("cuts", []),
                        "cut_timestamps": [float(t[c]) for c in r.get("cuts", [])],
                        "cut_elapsed_seconds": lines, "drawn_cut_lines": len(lines),
                        "source_execution": str(OUT), "source_record_index": r["record_index"],
                        "metadata": r.get("metadata", {})})
    assert len(entries) == len(panels)
    dependency = " | provisional DEFAULT Clasp initializer" if method == "clap" else ""
    fig.suptitle(f"{obj['object']} | {method} | {activity['split']} activity {activity['activity_id']}{dependency}\n"
                 f"ALL {len(configs)} configurations | same waveform/axes | red = method cuts | grey = context | BASE = registered baseline",
                 fontsize=10)
    stem = REPORT / "figures" / obj["object"] / method.replace("-", "_") / f"{activity['split']}_{activity['activity_id']:05d}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for suffix in ("png", "svg", "pdf"):
        target = stem.with_suffix("." + suffix)
        fig.savefig(target, dpi=140)
        artifacts[suffix] = str(target.relative_to(REPORT))
        artifacts[suffix + "_sha256"] = sha(target)
    plt.close(fig)
    return {"object": obj["object"], "method": method, "split": activity["split"],
            "activity_id": activity["activity_id"], "input_path": activity["npz"],
            "input_sha256": activity["npz_sha256"], "n_points": len(x), "core_a": a, "core_b": b,
            "nominal_seconds": obj["nominal_seconds"], "timezone": obj["timezone"],
            "left_censored": activity["left_censored"], "right_censored": activity["right_censored"],
            "primglr_reference": activity["primglr_reference"], "panels": entries, **artifacts}


def worker(ident, method):
    import numpy as np
    from models.time_segmentation.comparison import Unsupported, run
    plan = load(PLAN)
    obj = next(o for o in plan["objects"] if o["object"] == ident)
    assert sha(DATASET / "manifest.json") == plan["dataset_manifest_sha256"]
    assert sha(obj["activity_manifest"]) == obj["activity_manifest_sha256"]
    configs = obj["grids"][method]
    seeds = plan["e2usd_seeds"] if method == "e2usd" else [plan["other_seed"]]
    save(OUT / "inputs.json", {"plan": str(PLAN), "plan_sha256": sha(PLAN), "object": ident,
         "method": method, "activity_hashes": {str(a["activity_id"]): a["npz_sha256"] for a in obj["activities"]},
         "config_ids": [c["config_id"] for c in configs], "seeds": seeds})
    records, figures = [], []
    for activity in obj["activities"]:
        assert activity["split"] in ("development", "validation")
        assert sha(activity["npz"]) == activity["npz_sha256"]
        with np.load(activity["npz"]) as z:
            x, t = z["power"].copy(), z["timestamp"].copy()
        assert len(x) == activity["n_points"] and np.isfinite(x).all() and np.all(np.diff(t) > 0)
        current, fluss_cache = [], {}
        for config in configs:
            for seed in seeds:
                p = {k: v for k, v in config["params"].items() if not k.endswith("_seconds_target")}
                row = {"object": ident, "method": method, "activity_id": activity["activity_id"],
                       "split": activity["split"], "input_sha256": activity["npz_sha256"],
                       "config_id": config["config_id"], "params": config["params"], "seed": seed,
                       "record_index": len(records)}
                start = time.monotonic()
                try:
                    if method == "e2usd" and os.getenv("PSLG_E2USD_DEVICE") == "cpu":
                        from e2usd_cpu import run_cpu
                        cuts, metadata = run_cpu(x, p, seed=seed)
                    else:
                        cuts, metadata = (cached_fluss(x, p, fluss_cache) if method == "fluss-threshold"
                                          else run(x, method, p, seed=seed))
                    assert cuts == sorted(set(cuts)) and all(0 < c < len(x) for c in cuts)
                    row.update(status="success", cuts=cuts, metadata=metadata,
                               core_cuts=sum(activity["core_a"] < c < activity["core_b"] for c in cuts))
                except Unsupported as error:
                    row.update(status="unsupported", reason=str(error))
                except Exception as error:
                    row.update(status="failed", reason=repr(error), traceback=traceback.format_exc())
                row["wall_seconds"] = time.monotonic() - start
                records.append(row)
                current.append(row)
                save(OUT / "analysis/records.json", records)
                save(OUT / "analysis/progress.json", {"object": ident, "method": method,
                     "completed": len(records), "total": len(configs)*len(seeds)*len(obj["activities"]),
                     "failed": sum(r["status"] == "failed" for r in records)})
                print("RESULT", ident, method, activity["split"], activity["activity_id"], config["config_id"], seed,
                      row["status"], round(row["wall_seconds"], 2), flush=True)
        figures.append(draw(obj, method, activity, configs, current))
    summary = {"object": ident, "method": method, "config_count": len(configs), "seeds": seeds,
               "plan_sha256": sha(PLAN), "figures": figures, "source_execution": str(OUT),
               "records_sha256": sha(OUT / "analysis/records.json"),
               "status_counts": {s: sum(r["status"] == s for r in records) for s in ("success", "unsupported", "failed")}}
    save(OUT / "analysis/gallery_manifest.json", summary)
    # Publish only structurally complete artifacts, including clearly labelled technical failures.
    save(REPORT / "manifests" / f"{ident}_{method.replace('-', '_')}.json", summary)
    save(OUT / "analysis/validation.json", {"status": "passed" if not summary["status_counts"]["failed"] else "failed",
                                           **summary["status_counts"], "figures": len(figures)})
    if summary["status_counts"]["failed"]:
        raise SystemExit("Technical failures recorded; inspect before delivery")


def index(require_complete=False):
    import xml.etree.ElementTree as ET
    from PIL import Image
    plan = load(PLAN)
    figures, manifests, missing, choices = [], [], [], []
    for obj in plan["objects"]:
        for method in METHODS:
            path = REPORT / "manifests" / f"{obj['object']}_{method.replace('-', '_')}.json"
            if not path.exists():
                missing.append(f"{obj['object']}/{method}")
                continue
            manifest = load(path)
            assert manifest["plan_sha256"] == sha(PLAN)
            assert not manifest["status_counts"]["failed"], path
            assert len(manifest["figures"]) == 2
            expected = {(c["config_id"], s) for c in obj["grids"][method] for s in ([0, 1, 2] if method == "e2usd" else [0])}
            for fig in manifest["figures"]:
                assert {(p["config_id"], p["seed"]) for p in fig["panels"]} == expected
                assert len(fig["panels"]) == len(expected)
                for ext in ("png", "svg", "pdf"):
                    assert sha(REPORT / fig[ext]) == fig[ext + "_sha256"]
                assert (REPORT / fig["pdf"]).read_bytes().startswith(b"%PDF-")
                with Image.open(REPORT / fig["png"]) as image:
                    image.verify()
                tree = ET.parse(REPORT / fig["svg"])
                cut_elements = [e for e in tree.iter() if e.attrib.get("id", "").startswith("cut_")]
                assert len(cut_elements) == sum(len(p["cuts"]) for p in fig["panels"])
                for p in fig["panels"]:
                    assert p["status"] in ("success", "unsupported")
                    assert p["drawn_cut_lines"] == len(p["cuts"]) == len(p["cut_timestamps"])
                figures.append(fig)
            manifests.append({"path": str(path.relative_to(REPORT)), "sha256": sha(path)})
            for c in obj["grids"][method]:
                choices.append({"object": obj["object"], "method": method, "config_id": c["config_id"],
                     "params": json.dumps(c["params"], ensure_ascii=False, sort_keys=True),
                     "is_baseline": c["is_baseline"], "human_decision": "", "human_note": "",
                     "full_validation_required_before_freeze": True})
    if require_complete:
        assert not missing and len(figures) == 160, missing
    REPORT.mkdir(parents=True, exist_ok=True)
    save(REPORT / "figure_manifest.json", {"experiment": EXP.name, "plan_sha256": sha(PLAN),
         "status": "awaiting_human_review" if not missing else "building", "figure_count": len(figures),
         "expected_figures": 160, "missing": missing, "manifests": manifests, "figures": figures})
    if choices:
        with (REPORT / "all_parameter_choices.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(choices[0]))
            writer.writeheader()
            writer.writerows(choices)
    lines = ["# public16 v2：五种切分模型完整参数图审", "",
       f"状态：{'待人工审核' if not missing else '生成中'}；已完成 {len(figures)}/160 张总图，每张提供 PNG / SVG / PDF。", "",
       "每个对象、每种方法合计2张：开发集1条、验证集1条。全部方法共用同一对活动。每张图展示完整参数网格，按参数值排列，无Top-N截断。",
       "数据直接读取 datasets/public16_activity_states_20261017_v2 的既有活动，保留原时间/功率/核心/上下文。PrimGLR结果仅复用为清单中的参照，未重新切分。",
       "黑线为原始功率，红虚线为本方法切点，灰区为上下文。BASE是本实验登记基线，不代表所有参数均为作者默认。SVG/PDF可放大检查。",
       "CLaP使用官方默认Clasp初始化，图标题明确标为provisional；用户另选Clasp后，必须重绘对应CLaP网格。E2USD每组参数并列seed=0/1/2。",
       "不适用组合显示UNSUPPORTED及原因，不以零切点代替。源未标注时区的SNM/DARCK时间不改称UTC。",
       "两条活动用于参数目视审核，不提供新队列的完整验证排序；选择后须单独登记补齐验证再冻结，evaluation活动未用于本图审。", ""]
    for obj in plan["objects"]:
        lines += [f"## {obj['object']}", "", f"原采样标称间隔：{obj['nominal_seconds']}s；原活动数：{obj['activity_count']}。", ""]
        for method in METHODS:
            fs = [f for f in figures if f["object"] == obj["object"] and f["method"] == method]
            links = []
            for f in fs:
                links.append(f"{f['split']} #{f['activity_id']}：" + " / ".join(f"[{e.upper()}]({f[e]})" for e in ("png", "svg", "pdf")))
            lines.append(f"- **{method}**（{len(obj['grids'][method])}组）：" + ("；".join(links) if links else "待生成"))
        lines.append("")
    lines += ["完整候选记录见 [all_parameter_choices.csv](all_parameter_choices.csv)，逐格切点、来源与文件哈希见 [figure_manifest.json](figure_manifest.json)。", ""]
    (REPORT / "index.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save(OUT / "analysis/gallery_qa.json", {"status": "passed", "figures": len(figures), "missing": missing,
         "plan_sha256": sha(PLAN), "figure_manifest_sha256": sha(REPORT / "figure_manifest.json"),
         "complete_parameter_coverage": not missing, "scientific_parameter_approval": "pending_user_review"})
    print("INDEX", len(figures), "/160", "missing", len(missing), flush=True)


def coordinate(gpu, gpu_workers):
    from create_execution import register
    plan = load(PLAN)
    jobs = [(o["object"], m) for o in plan["objects"] for m in METHODS]
    index()
    def launch(job):
        ident, method = job
        path = register(EXP.name, "review_" + ident + "_" + method.replace("-", "_"),
                        {"plan": str(PLAN), "plan_sha256": sha(PLAN), "object": ident, "method": method})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if method == "e2usd" else "",
                   PYTHONPATH=str(ROOT / "runtime/segmentation/overlay_site"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
        code = subprocess.call([sys.executable, str(ROOT / "scripts/run_execution.py"), str(path), "--",
                                sys.executable, str(Path(__file__).resolve()), "--object", ident, "--method", method],
                               cwd=ROOT, env=env)
        result = {"object": ident, "method": method, "execution": str(path), "exit_code": code}
        print("JOB_DONE", json.dumps(result), flush=True)
        return result
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as cpu_pool, concurrent.futures.ThreadPoolExecutor(max_workers=gpu_workers) as gpu_pool:
        futures = [(gpu_pool if m == "e2usd" else cpu_pool).submit(launch, (o, m)) for o, m in jobs]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            save(EXP / "data/coordinator_progress.json", {"completed": len(results), "total": len(jobs), "jobs": results})
    if any(r["exit_code"] for r in results):
        save(OUT / "analysis/failed_jobs.json", results)
        raise SystemExit("Inspect failed jobs; originals preserved")
    # The final QA is another registered execution.
    qa = register(EXP.name, "complete_grid_delivery_qa", {"plan": str(PLAN), "plan_sha256": sha(PLAN)})
    code = subprocess.call([sys.executable, str(ROOT / "scripts/run_execution.py"), str(qa), "--",
                            sys.executable, str(Path(__file__).resolve()), "--index", "--require-complete"], cwd=ROOT)
    assert code == 0
    save(OUT / "analysis/validation.json", {"status": "passed", "jobs": results, "qa_execution": str(qa), "figures": 160})
    print("COMPLETE: 160 full-grid figures; awaiting human review", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--coordinate", action="store_true")
    p.add_argument("--index", action="store_true")
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--object")
    p.add_argument("--method", choices=METHODS)
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--gpu-workers", type=int, default=2)
    args = p.parse_args()
    provenance()
    if args.prepare:
        prepare()
    elif args.coordinate:
        coordinate(args.gpu, args.gpu_workers)
    elif args.index:
        index(args.require_complete)
    else:
        assert args.object and args.method
        worker(args.object, args.method)


if __name__ == "__main__":
    main()
