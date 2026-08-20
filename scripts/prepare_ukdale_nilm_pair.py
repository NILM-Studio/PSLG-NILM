"""Align a UK-DALE aggregate meter with one appliance channel.

The raw UK-DALE channels are sampled on offset time grids. This utility uses
the appliance timestamps as the output grid and selects the nearest aggregate
reading within a strict tolerance. Unmatched appliance rows are omitted rather
than interpolated across data gaps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_power_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, usecols=["timestamp", "power"])
        timestamp = frame["timestamp"].to_numpy(dtype=np.int64, copy=False)
        power = frame["power"].to_numpy(dtype=np.float32, copy=False)
    else:
        frame = pd.read_csv(
            path, sep=r"\s+", header=None, names=["timestamp", "power"],
            dtype={"timestamp": np.int64, "power": np.float32},
        )
        timestamp = frame["timestamp"].to_numpy(dtype=np.int64, copy=False)
        power = frame["power"].to_numpy(dtype=np.float32, copy=False)
    if len(timestamp) == 0:
        raise ValueError(f"empty power series: {path}")
    if np.any(np.diff(timestamp) <= 0):
        raise ValueError(f"timestamps must be strictly increasing: {path}")
    if not np.isfinite(power).all():
        raise ValueError(f"power contains NaN or Inf: {path}")
    return timestamp, power


def nearest_alignment(reference_timestamp: np.ndarray,
                      query_timestamp: np.ndarray,
                      tolerance_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest reference row and validity mask for each query row."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    right = np.searchsorted(reference_timestamp, query_timestamp, side="left")
    right = np.clip(right, 0, len(reference_timestamp) - 1)
    left = np.clip(right - 1, 0, len(reference_timestamp) - 1)
    right_distance = np.abs(reference_timestamp[right] - query_timestamp)
    left_distance = np.abs(reference_timestamp[left] - query_timestamp)
    choose_left = left_distance <= right_distance
    nearest = np.where(choose_left, left, right)
    distance = np.abs(reference_timestamp[nearest] - query_timestamp)
    return nearest.astype(np.int64, copy=False), distance <= tolerance_seconds


def interval_summary(timestamp: np.ndarray) -> dict:
    if len(timestamp) < 2:
        return {"median_seconds": None, "p95_seconds": None,
                "max_seconds": None}
    delta = np.diff(timestamp).astype(np.float64)
    return {
        "median_seconds": float(np.median(delta)),
        "p95_seconds": float(np.percentile(delta, 95)),
        "max_seconds": float(np.max(delta)),
    }


def prepare(mains_path: Path, appliance_path: Path, output_path: Path,
            tolerance_seconds: float, chunksize: int = 500_000) -> dict:
    mains_timestamp, mains_power = load_power_series(mains_path)
    appliance_timestamp, appliance_power = load_power_series(appliance_path)
    nearest, valid = nearest_alignment(
        mains_timestamp, appliance_timestamp, tolerance_seconds)
    matched_index = nearest[valid]
    output_timestamp = appliance_timestamp[valid]
    output_mains = mains_power[matched_index]
    output_appliance = appliance_power[valid]
    offset = mains_timestamp[matched_index] - output_timestamp

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({
        "timestamp": output_timestamp,
        "mains": output_mains,
        "appliance": output_appliance,
    })
    output.to_csv(output_path, index=False, chunksize=chunksize)

    audit = {
        "mains_path": str(mains_path),
        "appliance_path": str(appliance_path),
        "output_path": str(output_path),
        "alignment_method": "nearest_mains_to_appliance_grid",
        "tolerance_seconds": float(tolerance_seconds),
        "mains_rows": int(len(mains_timestamp)),
        "appliance_rows": int(len(appliance_timestamp)),
        "matched_rows": int(np.sum(valid)),
        "unmatched_rows": int(np.sum(~valid)),
        "matched_ratio": float(np.mean(valid)),
        "timestamp_offset_seconds": {
            "min": int(np.min(offset)) if len(offset) else None,
            "median": float(np.median(offset)) if len(offset) else None,
            "max": int(np.max(offset)) if len(offset) else None,
        },
        "output_interval": interval_summary(output_timestamp),
        "power_quality": {
            "mains_min": float(np.min(output_mains)) if len(output_mains) else None,
            "mains_max": float(np.max(output_mains)) if len(output_mains) else None,
            "appliance_min": float(np.min(output_appliance)) if len(output_appliance) else None,
            "appliance_max": float(np.max(output_appliance)) if len(output_appliance) else None,
            "appliance_above_mains_rows": int(np.sum(output_appliance > output_mains)),
            "appliance_above_mains_ratio": float(np.mean(output_appliance > output_mains))
            if len(output_mains) else None,
        },
    }
    audit_path = output_path.with_suffix(output_path.suffix + ".audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mains", required=True, type=Path)
    parser.add_argument("--appliance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tolerance-seconds", type=float, default=3.1)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    audit = prepare(
        args.mains, args.appliance, args.out,
        args.tolerance_seconds, args.chunksize)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"audit -> {args.out.with_suffix(args.out.suffix + '.audit.json')}")


if __name__ == "__main__":
    main()
