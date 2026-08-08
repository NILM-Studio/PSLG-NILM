"""Benchmark per-interval clasp-origin (full ClaSP) cost on REAL washing-machine
cycles. clasp-origin is single-threaded (n_jobs=1) and auto-detects segment
count — so we measure both timing AND how many segments it produces.

Run: NUMBA_DISABLE_CUDA=1 python determined/probe_clasp_origin_timing.py
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# repo root on path so `from models.time_segmentation...` resolves like main.py
ROOT = "/labdata2/lexingruan/pslg-nilm"
sys.path.insert(0, os.path.join(ROOT, "models", "time_segmentation"))
sys.path.insert(0, ROOT)

import numpy as np
from models.time_segmentation.clasp_origin import ClaspOriginModel

H5 = os.path.join(ROOT, "datasets/ukdale/ukdale.h5")
THRESHOLD, T_DROP, T_MIN_WORK, CONTEXT_SEC, FS = 5.0, 150, 180, 90, 0.1666667

print("loading series...", flush=True)
from nilmtk import DataSet
s = DataSet(H5).buildings[1].elec["washing machine"].power_series_all_data()
power = np.nan_to_num(s.to_numpy(dtype=np.float64), nan=0.0)
times = (s.index.view(np.int64)//10**9).astype(np.float64)

idx = np.where(power >= THRESHOLD)[0]
groups = []
cur = [idx[0]]
def flush(g):
    if times[g[-1]] - times[g[0]] >= T_MIN_WORK:
        groups.append((g[0], g[-1]))
for i in range(1, len(idx)):
    if times[idx[i]] - times[idx[i-1]] <= T_DROP:
        cur.append(idx[i])
    else:
        flush(cur); cur = [idx[i]]
flush(cur)
print(f"cycles={len(groups)}", flush=True)

ctx = int(CONTEXT_SEC * FS)
lengths = np.array([min(len(power), e+ctx+1) - max(0, s0-ctx) for s0,e in groups])
print(f"interval pts: min={lengths.min()} median={int(np.median(lengths))} "
      f"p75={int(np.percentile(lengths,75))} max={lengths.max()}")

# pick representative sizes: smallest, ~median, ~p75, ~p90, max
targets = {"min": int(np.argmin(lengths))}
for q in (50,75,90):
    v = np.percentile(lengths,q)
    targets[q] = int(np.argmin(np.abs(lengths-v)))
targets["max"] = int(np.argmax(lengths))

print(f"\n=== timing clasp-origin on {len(targets)} representative cycles ===", flush=True)
results = []
for label, gi in targets.items():
    s0,e = groups[gi]
    seg = power[max(0,s0-ctx):min(len(power),e+ctx+1)].astype(np.float64)
    m = ClaspOriginModel(config={"distance": "euclidean_distance"})
    t = time.time()
    try:
        cps = m.train(seg)
        dt = time.time()-t
        ncps = len(cps) if cps is not None else 0
        print(f"  [{label:>4}] cycle#{gi:4d} pts={len(seg):5d} -> {ncps} change_points "
              f"({ncps+1} segs)  {dt:.2f}s", flush=True)
        results.append((len(seg), ncps, dt))
    except Exception as ex:
        dt = time.time()-t
        print(f"  [{label:>4}] cycle#{gi:4d} pts={len(seg):5d} -> ERROR ({ex}) after {dt:.2f}s", flush=True)
        results.append((len(seg), 0, dt))

print("\n=== extrapolation ===")
if results:
    import numpy as np
    pts = np.array([r[0] for r in results]); segs = np.array([r[1]+1 for r in results]); ts = np.array([r[2] for r in results])
    print(f"  segs/cycle: {segs.tolist()}")
    # ClaSP ~ superlinear in n; fit log-model t = a * n^b as a rough guide
    valid = ts > 0
    if valid.sum() >= 2:
        coefs = np.polyfit(np.log(pts[valid]), np.log(ts[valid]), 1)
        b, a = coefs[0], np.exp(coefs[1])
        print(f"  fit: t ≈ {a:.4f} * n^{b:.2f} s")
        # predict per-cycle time for ALL cycles, sum
        pred = a * lengths**b
        total = pred.sum()
        print(f"  est. total segmentation over {len(groups)} cycles: {total:.0f}s = {total/60:.1f}min = {total/3600:.2f}h")
    # also a blunt mean extrapolation
    print(f"  mean sampled time {ts.mean():.2f}s × {len(groups)} = {ts.mean()*len(groups)/60:.1f}min")
