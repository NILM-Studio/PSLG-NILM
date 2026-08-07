"""Prepare a small UK-DALE test slice via nilmtk.

Loads the washing machine (building 1) power series from the nilmtk-formatted
HDF5 file, finds the first active region, and exports a 100-row window around
it as ``input/ukdale_washing_machine_100.csv`` (columns: timestamp, power).

Usage:
    python scripts/prepare_ukdale.py [--h5 datasets/ukdale/ukdale.h5]
        [--building 1] [--appliance "washing machine"] [--n-rows 100]
        [--active-threshold 20] [--out input/ukdale_washing_machine_100.csv]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Export a small UK-DALE slice via nilmtk.")
    ap.add_argument("--h5", default="datasets/ukdale/ukdale.h5")
    ap.add_argument("--building", type=int, default=1)
    ap.add_argument("--appliance", default="washing machine")
    ap.add_argument("--n-rows", type=int, default=100)
    ap.add_argument("--active-threshold", type=float, default=20.0,
                    help="W; the window is anchored slightly before the first "
                         "point above this, so the slice actually contains an event")
    ap.add_argument("--out", default=None,
                    help="default: input/ukdale_<appliance>_<n>.csv")
    args = ap.parse_args()

    from nilmtk import DataSet  # lazy: nilmtk import is heavy

    out = args.out or os.path.join(
        "input", f"ukdale_{args.appliance.replace(' ', '_')}_{args.n_rows}.csv")

    ds = DataSet(args.h5)
    elec = ds.buildings[args.building].elec
    meter = elec[args.appliance]
    print(f"[prepare] {args.appliance} (building {args.building}): loading power series...")
    series = meter.power_series_all_data()
    print(f"[prepare] full series: {len(series)} points "
          f"({series.index[0]} -> {series.index[-1]})")

    active = np.nonzero(series.to_numpy() > args.active_threshold)[0]
    if len(active) == 0:
        raise SystemExit(f"[prepare] no point above {args.active_threshold}W in the series")
    # anchor the window a few rows before the first active point so the slice
    # shows a clean off -> on transition
    start = max(0, int(active[0]) - 10)
    window = series.iloc[start:start + args.n_rows]
    print(f"[prepare] slice: rows {start}..{start + len(window) - 1}, "
          f"{window.index[0]} -> {window.index[-1]}, "
          f"max={window.max():.1f}W, active={int((window > args.active_threshold).sum())}/{len(window)}")

    df = pd.DataFrame({
        "timestamp": window.index.view(np.int64) // 10**9,  # unix seconds
        "power": window.to_numpy(dtype=np.float64),
    })
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[prepare] wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
