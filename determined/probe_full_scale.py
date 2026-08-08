"""Probe the FULL UK-DALE washing_machine series: how big is it, how many work
cycles does SimpleThresholdDetector extract, and what would segmentation + a
bilstm_ae training run cost. Read-only — no training, no artifact writes.

Run:  python determined/probe_full_scale.py
"""
from __future__ import annotations
import numpy as np, pandas as pd, time

H5 = "/labdata2/lexingruan/pslg-nilm/datasets/ukdale/ukdale.h5"
BUILDING, APPLIANCE = 1, "washing machine"

# extract_active_data config (config_ukdale_test.yaml)
THRESHOLD = 5.0       # W
T_DROP = 150          # s  (gap tolerance within a cycle)
T_MIN_WORK = 180      # s  (min cycle length)
CONTEXT_SEC = 90      # s  (padding before/after each interval)
FS = 0.1666667        # Hz (1 sample / 6 s) — assumed sample rate
# time_segmentation config
WINDOW = 10           # segmentation window

t0 = time.time()
print("loading nilmtk dataset (heavy import)...", flush=True)
from nilmtk import DataSet
ds = DataSet(H5)
elec = ds.buildings[BUILDING].elec
meter = elec[APPLIANCE]
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
series = meter.power_series_all_data()
n = len(series)
print(f"\n=== RAW SERIES ===")
print(f"  rows        : {n:,}")
print(f"  time span   : {series.index[0]} -> {series.index[-1]}")
print(f"  load time   : {time.time()-t1:.1f}s")
dt = np.diff(series.index.view(np.int64)//10**9)
print(f"  median dt   : {np.median(dt):.1f}s   (config assumes 6.0s; fs={FS})")
dt_mode = np.median(dt)
fs_real = 1.0/dt_mode if dt_mode>0 else FS
print(f"  real fs     : {fs_real:.4f} Hz")

power = series.to_numpy(dtype=np.float64)
nan_ct = int(np.isnan(power).sum())
print(f"  NaN points  : {nan_ct:,} ({100*nan_ct/n:.1f}%)")
power = np.nan_to_num(power, nan=0.0)
above = power >= THRESHOLD
active_pts = int(above.sum())
print(f"  >= {THRESHOLD}W     : {active_pts:,} points ({100*active_pts/n:.2f}% of series)")

# --- replicate SimpleThresholdDetector.detect grouping, stats only ---
times = (series.index.view(np.int64)//10**9).astype(np.float64)
idx = np.where(above)[0]
print(f"\n=== ACTIVE-INTERVAL GROUPING (threshold={THRESHOLD}W, t_drop={T_DROP}s, t_min_work={T_MIN_WORK}s) ===")
durations, cycle_pts, ctx_pts = [], [], []
if len(idx):
    cur = [idx[0]]
    def flush(g):
        d = times[g[-1]] - times[g[0]]
        if d >= T_MIN_WORK:
            durations.append(d)
            cycle_pts.append(len(g))
            ctx_pts.append(len(g) + 2*int(CONTEXT_SEC*FS))
    for i in range(1, len(idx)):
        if times[idx[i]] - times[idx[i-1]] <= T_DROP:
            cur.append(idx[i])
        else:
            flush(cur); cur = [idx[i]]
    flush(cur)

durations = np.array(durations); cycle_pts = np.array(cycle_pts); ctx_pts = np.array(ctx_pts)
K = len(durations)
print(f"  work cycles : {K:,}")
if K:
    pct = lambda a,p: np.percentile(a,p)
    print(f"  duration(s) : min={durations.min():.0f}  p25={pct(durations,25):.0f}  "
          f"median={np.median(durations):.0f}  p75={pct(durations,75):.0f}  "
          f"max={durations.max():.0f}  mean={durations.mean():.0f}")
    h = durations/3600
    print(f"  duration(h) : min={h.min():.2f}  median={np.median(h):.2f}  "
          f"max={h.max():.2f}  mean={h.mean():.2f}")
    print(f"  cycle points: min={cycle_pts.min()}  median={int(np.median(cycle_pts))}  "
          f"p75={int(pct(cycle_pts,75))}  max={cycle_pts.max()}  total={int(cycle_pts.sum()):,}")
    print(f"  +ctx points : total(with {CONTEXT_SEC}s padding)={int(ctx_pts.sum()):,}")

# --- segmentation tensor estimate ---
print(f"\n=== SEGMENTATION ESTIMATE (window_size={WINDOW}) ===")
if K:
    # fluss cuts each interval into sub-windows of WINDOW points; samples ≈ ceil(points/WINDOW)
    per_cycle_samples = np.ceil(cycle_pts / WINDOW).astype(int)
    n_samples = int(per_cycle_samples.sum())
    # but cap max_len per window at WINDOW (fixed-window segmentation)
    max_len = WINDOW
    dim = 1   # single power feature (the smoke run showed dim=4 after some featurization in seg)
    print(f"  est. sub-windows (samples) : {n_samples:,}   (= sum ceil(cycle_pts/{WINDOW}))")
    print(f"  tensor shape (samples x len x dim) : ({n_samples:,}, {max_len}, {dim})")
    mem_gb = n_samples*max_len*dim*8/1e9
    print(f"  tensor memory : {mem_gb:.3f} GB (float64)")
    print(f"  bilstm_ae est: 50 epoch x {n_samples} samples, batch 32 -> "
          f"{int(np.ceil(n_samples/32))*50:,} steps")

print(f"\n=== total probe time: {time.time()-t0:.1f}s ===")
