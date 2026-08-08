"""Cleaner clasp-origin steady-state benchmark: 18 random cycles, discard the
first 3 (numba JIT warmup), report the distribution. Each cycle is distinct data
so module-level caching (if any) cannot mask real cost. Also checks whether a
RE-RUN of the same cycle is instant (reveals internal caching).

Run: NUMBA_DISABLE_CUDA=1 python determined/probe_clasp_steady.py
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = "/labdata2/lexingruan/pslg-nilm"
sys.path.insert(0, os.path.join(ROOT, "models", "time_segmentation"))
sys.path.insert(0, ROOT)
import numpy as np
from models.time_segmentation.clasp_origin import ClaspOriginModel

H5 = os.path.join(ROOT, "datasets/ukdale/ukdale.h5")
THRESHOLD, T_DROP, T_MIN_WORK, CONTEXT_SEC, FS = 5.0, 150, 180, 90, 0.1666667

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
    (cur.append(idx[i]) if times[idx[i]]-times[idx[i-1]] <= T_DROP else (flush(cur), cur:=[idx[i]]))
flush(cur)
ctx = int(CONTEXT_SEC * FS)
print(f"cycles={len(groups)}", flush=True)

rng = np.random.RandomState(42)
pick = rng.choice(len(groups), 18, replace=False)
print(f"\n=== 18 cycles (first 3 = JIT warmup, discarded) ===", flush=True)
pts_list, ts_list, segs_list = [], [], []
for n, gi in enumerate(pick):
    s0,e = groups[gi]
    seg = power[max(0,s0-ctx):min(len(power),e+ctx+1)].astype(np.float64)
    m = ClaspOriginModel(config={"distance": "euclidean_distance"})
    t = time.time()
    cps = m.train(seg); dt = time.time()-t
    ncps = len(cps) if cps is not None else 0
    tag = "warmup" if n < 3 else "steady"
    print(f"  [{tag}] #{gi:4d} pts={len(seg):5d} -> {ncps}cp ({ncps+1} segs)  {dt:.2f}s", flush=True)
    if n >= 3:
        pts_list.append(len(seg)); ts_list.append(dt); segs_list.append(ncps+1)

# re-run one steady cycle to detect internal caching
gi = pick[5]; s0,e = groups[gi]
seg = power[max(0,s0-ctx):min(len(power),e+ctx+1)].astype(np.float64)
t = time.time(); ClaspOriginModel(config={"distance":"euclidean_distance"}).train(seg); dt2=time.time()-t
print(f"\n  re-run same cycle #{gi} (cache probe): {dt2:.2f}s  (first run was {ts_list[0]:.2f}s)")

print(f"\n=== steady-state ({len(ts_list)} cycles) ===")
pts=np.array(pts_list); ts=np.array(ts_list); segs=np.array(segs_list)
print(f"  time(s): min={ts.min():.3f} median={np.median(ts):.3f} mean={ts.mean():.3f} max={ts.max():.3f}")
print(f"  segs/cycle: mean={segs.mean():.1f}  total est samples ≈ {int(segs.mean()*len(groups)):,}")
# sum of per-cycle times using measured mean (NOT length-fit, given the weird scaling)
print(f"  est. total segmentation: median×{len(groups)} = {np.median(ts)*len(groups)/60:.1f}min"
      f"  | mean×{len(groups)} = {ts.mean()*len(groups)/60:.1f}min"
      f"  | max×{len(groups)} = {ts.max()*len(groups)/60:.1f}min")
