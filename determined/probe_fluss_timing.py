"""Benchmark per-interval fluss cost on REAL washing-machine cycles, then
extrapolate to all 1,424 cycles. Read-only timing probe.

Run: NUMBA_DISABLE_CUDA=1 python determined/probe_fluss_timing.py
"""
from __future__ import annotations
import os, sys, time
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np, pandas as pd
sys.path.insert(0, "/labdata2/lexingruan/pslg-nilm/models/time_segmentation")
from fluss import fluss   # vendored

H5 = "/labdata2/lexingruan/pslg-nilm/datasets/ukdale/ukdale.h5"
THRESHOLD, T_DROP, T_MIN_WORK, CONTEXT_SEC, FS = 5.0, 150, 180, 90, 0.1666667
WINDOW, N_REGIMES, EXCL = 10, 3, 2

print("loading series...", flush=True)
from nilmtk import DataSet
s = DataSet(H5).buildings[1].elec["washing machine"].power_series_all_data()
power = np.nan_to_num(s.to_numpy(dtype=np.float64), nan=0.0)
times = (s.index.view(np.int64)//10**9).astype(np.float64)

# group active points into cycles (same logic as SimpleThresholdDetector)
idx = np.where(power >= THRESHOLD)[0]
boundaries = []   # (start_idx, end_idx) into the padded interval
cur = [idx[0]]
groups = []
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
# build interval lengths (with context padding)
lengths = np.array([min(len(power), e+ctx+1) - max(0, s0-ctx) for s0,e in groups])
print(f"interval pts: min={lengths.min()} median={int(np.median(lengths))} "
      f"p75={int(np.percentile(lengths,75))} max={lengths.max()} total={int(lengths.sum()):,}")

# sample cycles across the size distribution: smallest, median, p75, p90, max + a few random
sample_idx = sorted(set(
    [int(np.argmin(lengths)), int(np.median(np.arange(len(lengths)))),
     np.searchsorted(np.sort(lengths), np.percentile(lengths,75)),
     np.searchsorted(np.sort(lengths), np.percentile(lengths,90)),
     int(np.argmax(lengths))]
    + np.random.RandomState(0).randint(0, len(groups), 10).tolist()))
sample_idx = [i % len(groups) for i in sample_idx][:14]

print(f"\n=== timing fluss on {len(sample_idx)} sampled cycles (window={WINDOW}, n_regimes={N_REGIMES}) ===", flush=True)
# warm up numba JIT on the smallest so timing reflects steady-state
t_jit0 = time.time()
s0,e = groups[sample_idx[0]]
seg = power[max(0,s0-ctx):min(len(power),e+ctx+1)]
_ = fluss(seg.astype(np.float64), window_size=WINDOW, n_regimes=N_REGIMES, excl_factor=EXCL, visualize=False)
print(f"  (JIT warmup on 1 cycle: {time.time()-t_jit0:.2f}s)", flush=True)

timings, n_cps = [], []
for gi in sample_idx[1:]:
    s0,e = groups[gi]
    seg = power[max(0,s0-ctx):min(len(power),e+ctx+1)].astype(np.float64)
    t = time.time()
    _, cps = fluss(seg, window_size=WINDOW, n_regimes=N_REGIMES, excl_factor=EXCL, visualize=False)
    dt = time.time()-t
    timings.append(dt); n_cps.append(len(cps))
    print(f"  cycle#{gi:4d}  pts={len(seg):5d}  cp={len(cps)}  {dt*1000:6.1f}ms", flush=True)

timings = np.array(timings)
print(f"\n=== per-cycle fluss timing (post-JIT) ===")
print(f"  min={timings.min()*1000:.1f}ms  median={np.median(timings)*1000:.1f}ms  "
      f"mean={timings.mean()*1000:.1f}ms  max={timings.max()*1000:.1f}ms")
# rough extrapolation: sum over all cycles, weighting by length (fluss ~ O(n) for stump on GPU? no, CPU O(n^2/m))
# Use mean-per-point rate from samples
per_pt = np.array([t/len(power[max(0,groups[gi][0]-ctx):min(len(power),groups[gi][1]+ctx+1)])
                   for t,gi in zip(timings, sample_idx[1:])])
rate = np.median(per_pt)
est_total = rate * lengths.sum()
print(f"  est. total segmentation (sum over {len(groups)} cycles, by per-point rate): "
      f"{est_total:.0f}s = {est_total/60:.1f}min")
print(f"  est. total segmentation (mean-per-cycle × {len(groups)}): "
      f"{timings.mean()*len(groups):.0f}s = {timings.mean()*len(groups)/60:.1f}min")
print(f"  est. samples produced (cycles × ~{N_REGIMES} regimes): ~{len(groups)*N_REGIMES:,}")
