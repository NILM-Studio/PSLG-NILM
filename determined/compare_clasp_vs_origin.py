#!/usr/bin/env python
"""clasp vs clasp-origin: metric comparison with a tolerance sweep.

Metrics (no true change-point labels exist; OSR/USR use the counterpart method
as reference, documented in the report):

  F1@k   : per-cycle change-point agreement, tol k = max(2, frac*len) samples
  Coverage: fraction of activity-cycle samples covered by primitives
            (= 1.0 for both by construction)
  OSR_M  : mean over cycles of max(0, n_M - n_ref) / max(1, n_ref)
  USR_M  : mean over cycles of max(0, n_ref - n_M) / max(1, n_ref)
  d_nn   : mean over cycles of median nearest-neighbour cp distance (samples)
"""
import json
import os
import numpy as np

ROOT = "/labdata2/lexingruan/pslg-nilm"
DATASETS = ["eco", "greend", "iawe", "redd", "refit", "ukdale"]
A, B = "clasp", "clasp-origin"
TOLS = [0.005, 0.01, 0.02, 0.05]
TOL_ABS_MIN = 2


def run_id_for(m, ds):
    return "20260808_clasp-clasp_" + ds if m == "clasp" else "20260808_clasp-origin_" + ds


def load_run(ds, m):
    run_id = run_id_for(m, ds)
    base = os.path.join(ROOT, "log_det_test", run_id)
    mpath = os.path.join(base, "run_manifest.json")
    if not os.path.exists(mpath):
        return None
    man = json.load(open(mpath))
    seg = man["steps"]["time_segmentation"]
    indices = np.load(os.path.join(base, seg["artifacts"]["indices"]))
    lengths = np.load(os.path.join(base, seg["artifacts"]["lengths"]))
    n_cycles = int(man["steps"]["extract_active_data"]["artifacts"]["count"])
    cps, cyc_len = {}, {}
    for row in range(indices.shape[0]):
        c, s, L = int(indices[row, 0]), int(indices[row, 1]), int(lengths[row, 0])
        cyc_len[c] = max(cyc_len.get(c, 0), s + L)
        if s > 0:
            cps.setdefault(c, []).append(s)
    for c in cps:
        cps[c] = np.array(sorted(cps[c]), dtype=int)
    return dict(ds=ds, n_cycles=n_cycles, n_samples=int(seg["extra"]["n_samples"]),
                cps=cps, cyc_len=cyc_len)


def match_counts(p, q, tol):
    used = np.zeros(len(q), dtype=bool)
    matched = 0
    for x in p:
        if not len(q):
            break
        d = np.abs(q - x)
        best = int(np.argmin(d))
        if d[best] <= tol and not used[best]:
            used[best] = True
            matched += 1
    return matched


def cycle_metrics(cA, cB, clen, frac):
    rows = []
    for c in sorted(set(cA) & set(cB)):
        pa, pb = cA[c], cB[c]
        tol = max(TOL_ABS_MIN, round(frac * clen[c]))
        m = match_counts(pa, pb, tol)
        nA, nB = len(pa), len(pb)
        pa_p = m / nA if nA else 0.0
        pa_r = m / nB if nB else 0.0
        f1A = 2 * pa_p * pa_r / (pa_p + pa_r) if (pa_p + pa_r) else 0.0
        pb_p = m / nB if nB else 0.0
        pb_r = m / nA if nA else 0.0
        f1B = 2 * pb_p * pb_r / (pb_p + pb_r) if (pb_p + pb_r) else 0.0
        # median nearest-neighbour distances
        def nn(x, y):
            if not len(x) or not len(y):
                return float("nan")
            d = np.array([np.min(np.abs(y - v)) for v in x])
            return float(np.median(d))
        rows.append(dict(nA=nA, nB=nB, f1A=f1A, f1B=f1B,
                         nnA=nn(pa, pb), nnB=nn(pb, pa)))
    return rows


def main():
    print("== raw granularity ==")
    print(f"{'dataset':<8}{'cyc':>6}{'prims_A':>8}{'prims_B':>9}{'cp_A':>7}{'cp_B':>8}")
    stats = {}
    for ds in DATASETS:
        rA, rB = load_run(ds, A), load_run(ds, B)
        if not rA or not rB:
            print(f"{ds:<8}  (missing run)"); continue
        nA = sum(len(v) for v in rA["cps"].values()); nB = sum(len(v) for v in rB["cps"].values())
        print(f"{ds:<8}{rA['n_cycles']:>6}{rA['n_samples']:>8}{rB['n_samples']:>9}{nA:>7}{nB:>8}")
        stats[ds] = (rA, rB)

    print("\n== F1@k tolerance sweep (mean over common cycles) ==")
    hdr = f"{'dataset':<8}" + "".join(f"{'F1_A@'+str(int(t*100))+'%':>11}{'F1_B@'+str(int(t*100))+'%':>11}" for t in TOLS)
    print(hdr)
    for ds, (rA, rB) in stats.items():
        line = f"{ds:<8}"
        for t in TOLS:
            rows = cycle_metrics(rA["cps"], rB["cps"], rA["cyc_len"], t)
            f1A = float(np.mean([r["f1A"] for r in rows])) if rows else float("nan")
            f1B = float(np.mean([r["f1B"] for r in rows])) if rows else float("nan")
            line += f"{f1A:>11.3f}{f1B:>11.3f}"
        print(line)

    print("\n== OSR / USR (A=clasp vs B=clasp-origin reference, tol 1%) ==")
    print(f"{'dataset':<8}{'OSR_A':>8}{'USR_A':>8}{'OSR_B':>8}{'USR_B':>8}{'CovA':>7}{'CovB':>7}{'dnn_A':>8}{'dnn_B':>8}")
    for ds, (rA, rB) in stats.items():
        rows = cycle_metrics(rA["cps"], rB["cps"], rA["cyc_len"], 0.01)
        if not rows:
            continue
        osrA = np.mean([max(0, r["nA"] - r["nB"]) / max(1, r["nB"]) for r in rows])
        usrA = np.mean([max(0, r["nB"] - r["nA"]) / max(1, r["nB"]) for r in rows])
        osrB = np.mean([max(0, r["nB"] - r["nA"]) / max(1, r["nA"]) for r in rows])
        usrB = np.mean([max(0, r["nA"] - r["nB"]) / max(1, r["nA"]) for r in rows])
        nnA = float(np.nanmean([r["nnA"] for r in rows]))
        nnB = float(np.nanmean([r["nnB"] for r in rows]))
        print(f"{ds:<8}{osrA:>8.3f}{usrA:>8.3f}{osrB:>8.3f}{usrB:>8.3f}{1.0:>7.2f}{1.0:>7.2f}{nnA:>8.0f}{nnB:>8.0f}")


if __name__ == "__main__":
    main()
