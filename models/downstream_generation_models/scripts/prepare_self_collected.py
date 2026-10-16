#!/usr/bin/env python3
"""Create a small, deterministic 6-second NILM window set for smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


APPLIANCES = ("Air_condition", "Microwave", "WashMachine")


def file_prefix_sha256(path: Path, limit: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/home/avis/pslg-nilm"))
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--sample-seconds", type=int, default=6)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--max-windows-per-appliance", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    deploy_root = Path(__file__).resolve().parents[1]
    output = args.output or deploy_root.parents[1] / "runs/20261015_downstream_generation_validation/data" / "self_collected_smoke.npz"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = args.hours * 3600
    raw_windows, status_windows, onoff_windows, appliance_ids = [], [], [], []
    sources = []
    cadence = f"{args.sample_seconds}s"

    for appliance_id, name in enumerate(APPLIANCES):
        path = args.project_root / "datasets" / "self_collected" / f"{name}.csv"
        frame = pd.read_csv(
            path,
            nrows=rows,
            usecols=["datetime", "active power", "status", "on/off status"],
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frame = frame.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
        sampled = frame.resample(cadence, origin="epoch").mean().dropna()
        sampled["status"] = (sampled["status"] >= 0.5).astype(np.int8)
        sampled["on/off status"] = (sampled["on/off status"] >= 0.5).astype(np.int8)

        usable = min(len(sampled) // args.window, args.max_windows_per_appliance)
        if usable < 2:
            raise RuntimeError(f"Not enough complete windows in {path}: {usable}")
        end = usable * args.window
        watts = sampled["active power"].to_numpy(np.float32)[:end].reshape(usable, args.window, 1)
        status = sampled["status"].to_numpy(np.int8)[:end].reshape(usable, args.window)
        onoff = sampled["on/off status"].to_numpy(np.int8)[:end].reshape(usable, args.window)

        raw_windows.append(watts)
        status_windows.append(status)
        onoff_windows.append(onoff)
        appliance_ids.append(np.full(usable, appliance_id, dtype=np.int16))
        sources.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "prefix_1m_sha256": file_prefix_sha256(path),
                "rows_read": len(frame),
                "resampled_rows": len(sampled),
                "windows": usable,
                "first_timestamp": sampled.index[0].isoformat(),
                "last_timestamp": sampled.index[-1].isoformat(),
            }
        )

    raw = np.concatenate(raw_windows).astype(np.float32)
    status = np.concatenate(status_windows)
    onoff = np.concatenate(onoff_windows)
    appliance = np.concatenate(appliance_ids)
    low, high = np.quantile(raw, [0.01, 0.99]).astype(np.float32)
    if not np.isfinite(low + high) or high <= low:
        raise RuntimeError(f"Invalid robust scale: low={low}, high={high}")
    normalized = np.clip(2.0 * (raw - low) / (high - low) - 1.0, -1.0, 1.0).astype(np.float32)

    np.savez_compressed(
        output,
        x=normalized,
        raw_watts=raw,
        status=status,
        on_off_status=onoff,
        appliance_id=appliance,
        appliance_names=np.asarray(APPLIANCES),
        scale_low=np.asarray(low),
        scale_high=np.asarray(high),
        sample_seconds=np.asarray(args.sample_seconds),
        window_length=np.asarray(args.window),
    )
    artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "artifact": str(output),
        "artifact_sha256": artifact_sha256,
        "shape": list(normalized.shape),
        "dtype": str(normalized.dtype),
        "scale": {"method": "global q01/q99 then clip", "low": float(low), "high": float(high)},
        "sample_seconds": args.sample_seconds,
        "window_length": args.window,
        "sources": sources,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
