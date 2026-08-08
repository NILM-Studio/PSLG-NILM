"""Export a full appliance power series from any dataset HDF5 as CSV.

Reads a nilmtk HDF5 store (pandas HDFStore), locates the appliance matching
``--appliance`` inside ``--building``, and writes the full power-active series
as ``timestamp,power`` CSV (same schema as ``prepare_ukdale.py`` output).

Usage:
    python scripts/prepare_dataset_series.py --h5 datasets/datasets/eco/eco.h5 \
        --building 1 --appliance "washing machine" \
        --out input/eco_washing_machine.csv

Matching rules (in priority order, per dataset):
  1. exact original_name match
  2. original_name containing the appliance string (case-insensitive)
  3. type == appliance
  4. type containing the appliance string
  5. if ``--allow-washer-dryer``: type == 'washer dryer' (falls back to 'washing machine')

The matched meter's power-active column is used when present; otherwise the
first available power column.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _first_power_col(df: pd.DataFrame):
    if isinstance(df.columns, pd.MultiIndex):
        for c in df.columns:
            if c[0] == "power":
                return c
        raise KeyError(f"no power column in {list(df.columns)}")
    if "power" in df.columns:
        return "power"
    raise KeyError(f"no power column in {list(df.columns)}")


def find_meter(store: pd.HDFStore, building: int, appliance: str,
               allow_washer_dryer: bool = False):
    """Return (meter_key, appliance_meta) for the best-matching appliance."""
    prefix = f"/building{building}/elec/meter"
    from nilmtk import DataSet

    ds = DataSet(store._handle.filename)
    try:
        elec = ds.buildings[building].elec
        candidates = []
        for a in elec.appliances:
            md = a.metadata or {}
            on = str(md.get("original_name", "") or "")
            ty = str(md.get("type", "") or "")
            score = 0
            if appliance.lower() in on.lower():
                score = 4
            elif on and appliance.lower() == on.lower():
                score = 5
            if appliance.lower() in ty.lower():
                score = max(score, 3)
            elif ty.lower() == appliance.lower():
                score = max(score, 4)
            if allow_washer_dryer and appliance.lower() == "washing machine" \
                    and ty.lower() == "washer dryer":
                score = max(score, 2)
            if score:
                candidates.append((score, a))
        if not candidates:
            return None, None
        candidates.sort(key=lambda t: -t[0])
        app = candidates[0][1]
        md = app.metadata
        mnum = md["meters"][0]
        key = f"{prefix}{mnum}"
        return key, md
    finally:
        ds.close()


def main():
    ap = argparse.ArgumentParser(description="Export a full appliance power series as CSV.")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--building", type=int, default=1)
    ap.add_argument("--appliance", default="washing machine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-washer-dryer", action="store_true",
                    help="accept a 'washer dryer' meter as a washing machine")
    args = ap.parse_args()

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if not os.path.exists(args.h5):
        raise SystemExit(f"h5 not found: {args.h5}")

    with pd.HDFStore(args.h5, mode="r") as store:
        key, md = find_meter(store, args.building, args.appliance,
                             allow_washer_dryer=args.allow_washer_dryer)
        if key is None:
            avail = [k for k in store.keys() if k.startswith(f"/building{args.building}/elec/meter")]
            raise SystemExit(
                f"no appliance matching '{args.appliance}' in building {args.building}.\n"
                f"available meter keys (first 30): {avail[:30]}")
        print(f"[prepare] matched meter {key} meta={md}", flush=True)

        print(f"[prepare] reading {key} ...", flush=True)
        df = store.select(key)
        print(f"[prepare] {len(df):,} rows, cols={list(df.columns)}", flush=True)

        col = _first_power_col(df)
        print(f"[prepare] using power column: {col}", flush=True)
        # index: datetime64 (possibly tz-aware) -> unix seconds
        idx = df.index.view(np.int64) // 10**9
        power = df[col].to_numpy(dtype=np.float64)
        n = len(idx)
        del df
        # stream-write CSV in chunks to bound peak memory
        with open(out, "w") as fh:
            fh.write("timestamp,power\n")
            for start in range(0, n, 1_000_000):
                chunk = pd.DataFrame({
                    "timestamp": idx[start:start + 1_000_000],
                    "power": power[start:start + 1_000_000],
                })
                chunk.to_csv(fh, index=False, header=False)
                print(f"[prepare] wrote {min(start + 1_000_000, n):,}/{n:,} rows", flush=True)
        print(f"[prepare] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
