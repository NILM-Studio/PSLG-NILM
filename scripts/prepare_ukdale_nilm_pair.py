"""Align a UK-DALE aggregate meter with one appliance channel.

The raw UK-DALE channels are sampled on offset time grids. This utility uses
the appliance timestamps as the output grid and selects the nearest aggregate
reading within a strict tolerance. Unmatched appliance rows are omitted rather
than interpolated across data gaps.

UK-DALE mains.dat has four whitespace-separated columns: fractional UNIX
timestamp, active power (W), apparent power (VA), RMS voltage (V). The active
column is selected by default for that format; generic two-column files have
unknown measurement semantics unless explicitly declared. Format reference:
https://www.nature.com/articles/sdata20157 (Data Records, 1 second data).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


POWER_TYPES = ("unknown", "active", "apparent")
RAW_FORMATS = ("auto", "channel", "ukdale-mains")


def parse_time_bound(value: str | None) -> float | None:
    """Parse ISO dates/times; an omitted timezone explicitly means UTC."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("time bounds must be valid ISO dates or timestamps")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return float(timestamp.timestamp())


def time_range(timestamp: np.ndarray) -> dict:
    if not len(timestamp):
        return {"start_unix": None, "end_unix": None,
                "start_utc": None, "end_utc": None}
    start, end = float(timestamp[0]), float(timestamp[-1])
    return {
        "start_unix": start, "end_unix": end,
        "start_utc": pd.Timestamp(start, unit="s", tz="UTC").isoformat(),
        "end_utc": pd.Timestamp(end, unit="s", tz="UTC").isoformat(),
    }


def _load_power_series(path: Path, *, raw_format: str = "auto",
                       power_type: str = "unknown", chunksize: int = 500_000,
                       start: float | None = None,
                       end: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    if raw_format not in RAW_FORMATS or power_type not in POWER_TYPES:
        raise ValueError("unsupported input format or power measurement type")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    is_csv = path.suffix.lower() == ".csv"
    if is_csv and raw_format == "ukdale-mains":
        raise ValueError("ukdale-mains requires a four-column whitespace .dat file")
    reader = (pd.read_csv(path, usecols=["timestamp", "power"], chunksize=chunksize)
              if is_csv else pd.read_csv(
                  path, sep=r"\s+", header=None, chunksize=chunksize))
    detected_format = "timestamp_power_csv" if is_csv else None
    selected_type = power_type
    timestamp_parts, power_parts = [], []
    rows_read, excluded_rows, negative_rows = 0, 0, 0
    source_start, source_end = None, None
    power_column = "power" if is_csv else 1
    with reader:
        for frame in reader:
            if not is_csv:
                ncols = len(frame.columns)
                current_format = {2: "channel", 4: "ukdale-mains"}.get(ncols)
                if current_format is None:
                    raise ValueError(f"expected 2 channel or 4 mains.dat columns: {path}")
                if ((raw_format != "auto" and current_format != raw_format)
                        or (detected_format is not None
                            and detected_format != current_format)):
                    raise ValueError(f"column count does not match {raw_format}: {path}")
                detected_format = current_format
                if current_format == "ukdale-mains":
                    selected_type = "apparent" if power_type == "apparent" else "active"
                    power_column = 2 if selected_type == "apparent" else 1
            timestamp = frame["timestamp" if is_csv else 0].to_numpy(dtype=np.float64)
            power = frame[power_column].to_numpy(dtype=np.float32)
            if not len(timestamp):
                continue
            if not np.isfinite(timestamp).all() or not np.isfinite(power).all():
                raise ValueError(f"timestamp or selected power contains NaN or Inf: {path}")
            if (np.any(np.diff(timestamp) <= 0)
                    or (source_end is not None and timestamp[0] <= source_end)):
                raise ValueError(f"timestamps must be strictly increasing: {path}")
            if source_start is None:
                source_start = float(timestamp[0])
            source_end = float(timestamp[-1])
            rows_read += len(timestamp)
            selected = np.ones(len(timestamp), dtype=bool)
            if start is not None:
                selected &= timestamp >= start
            if end is not None:
                selected &= timestamp < end
            excluded_rows += int(np.sum(~selected))
            negative_rows += int(np.sum(power[selected] < 0))
            if selected.any():
                timestamp_parts.append(timestamp[selected])
                power_parts.append(power[selected])
    if not timestamp_parts:
        raise ValueError(f"empty power series in requested time range: {path}")
    timestamp, power = np.concatenate(timestamp_parts), np.concatenate(power_parts)
    audit = {
        "format": detected_format,
        "timestamp_column": "timestamp" if is_csv else 0,
        "power_column": power_column,
        "column_index_base": None if is_csv else 0,
        "requested_power_type": power_type,
        "power_type": selected_type,
        "measurement_type_source": ("ukdale_mains_column_definition"
                                    if detected_format == "ukdale-mains"
                                    else "explicit_argument" if power_type != "unknown"
                                    else "not_declared"),
        "power_unit": {"active": "W", "apparent": "VA", "unknown": None}[selected_type],
        "rows_read": rows_read, "valid_rows": rows_read,
        "invalid_rows": 0, "invalid_row_policy": "reject_input",
        "range_excluded_rows": excluded_rows, "selected_rows": len(timestamp),
        "selected_negative_power_rows": negative_rows,
        "source_time_range": time_range(np.asarray([source_start, source_end])),
        "selected_time_range": time_range(timestamp),
    }
    return timestamp, power, audit


def load_power_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible loader; preserve fractional UNIX seconds."""
    timestamp, power, _ = _load_power_series(path)
    return timestamp, power


def nearest_alignment(reference_timestamp: np.ndarray,
                      query_timestamp: np.ndarray,
                      tolerance_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest reference row and validity mask for each query row."""
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be finite and non-negative")
    if not len(reference_timestamp):
        raise ValueError("reference_timestamp must not be empty")
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
            tolerance_seconds: float, chunksize: int = 500_000, *,
            mains_format: str = "auto", mains_power_type: str = "unknown",
            appliance_power_type: str = "unknown", start: str | None = None,
            end: str | None = None, instance_boundary: str | None = None) -> dict:
    start_unix, end_unix = parse_time_bound(start), parse_time_bound(end)
    boundary_unix = parse_time_bound(instance_boundary)
    if start_unix is not None and end_unix is not None and start_unix >= end_unix:
        raise ValueError("start must be earlier than end (end is exclusive)")
    mains_timestamp, mains_power, mains_audit = _load_power_series(
        mains_path, raw_format=mains_format, power_type=mains_power_type,
        chunksize=chunksize, start=start_unix, end=end_unix)
    appliance_timestamp, appliance_power, appliance_audit = _load_power_series(
        appliance_path, raw_format="channel", power_type=appliance_power_type,
        chunksize=chunksize, start=start_unix, end=end_unix)
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
        "schema_version": 2,
        "mains_path": str(mains_path),
        "appliance_path": str(appliance_path),
        "output_path": str(output_path),
        "alignment_method": "nearest_mains_to_appliance_grid",
        "tie_break": "earlier_mains_reading",
        "interpolation_method": "none",
        "interpolated_rows": 0,
        "mains_power_type": mains_audit["power_type"],
        "appliance_power_type": appliance_audit["power_type"],
        "measurement_compatible_for_additive_synthesis": (
            mains_audit["power_type"] == appliance_audit["power_type"] == "active"),
        "additive_synthesis_check_scope": "measurement_types_only",
        "mains_input": mains_audit,
        "appliance_input": appliance_audit,
        "requested_time_range": {
            "start": start, "end": end, "start_unix": start_unix, "end_unix": end_unix,
            "start_inclusive": True, "end_exclusive": True,
            "timezone_for_naive_dates": "UTC",
        },
        "output_time_range": time_range(output_timestamp),
        "tolerance_seconds": float(tolerance_seconds),
        "mains_rows": int(len(mains_timestamp)),
        "appliance_rows": int(len(appliance_timestamp)),
        "matched_rows": int(np.sum(valid)),
        "unmatched_rows": int(np.sum(~valid)),
        "matched_ratio": float(np.mean(valid)),
        "distinct_mains_rows_used": int(len(np.unique(matched_index))),
        "reused_mains_rows": int(len(matched_index) - len(np.unique(matched_index))),
        "timestamp_offset_seconds": {
            "min": float(np.min(offset)) if len(offset) else None,
            "median": float(np.median(offset)) if len(offset) else None,
            "max": float(np.max(offset)) if len(offset) else None,
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
    audit["instance_periods"] = {
        "boundary": instance_boundary,
        "boundary_unix": boundary_unix,
        "source": "explicit_argument" if boundary_unix is not None else "not_declared",
        "before_boundary_rows": (int(np.sum(output_timestamp < boundary_unix))
                                 if boundary_unix is not None else None),
        "at_or_after_boundary_rows": (int(np.sum(output_timestamp >= boundary_unix))
                                      if boundary_unix is not None else None),
        "note": "Counts use the supplied boundary; no machine identity is inferred.",
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
    parser.add_argument("--mains-format", choices=RAW_FORMATS, default="auto",
                        help="auto detects 2-column channel or 4-column UK-DALE mains.dat")
    parser.add_argument("--mains-power-type", choices=POWER_TYPES, default="unknown",
                        help="For mains.dat, active is default; apparent selects column 3. "
                             "For two-column input, this declares its measurement type.")
    parser.add_argument("--appliance-power-type", choices=POWER_TYPES, default="unknown")
    parser.add_argument("--start", help="Inclusive ISO date/time; naive values are UTC")
    parser.add_argument("--end", help="Exclusive ISO date/time; naive values are UTC")
    parser.add_argument("--instance-boundary",
                        help="Optional ISO machine-change boundary for before/after row counts")
    args = parser.parse_args()
    audit = prepare(
        args.mains, args.appliance, args.out,
        args.tolerance_seconds, args.chunksize,
        mains_format=args.mains_format, mains_power_type=args.mains_power_type,
        appliance_power_type=args.appliance_power_type,
        start=args.start, end=args.end, instance_boundary=args.instance_boundary)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"audit -> {args.out.with_suffix(args.out.suffix + '.audit.json')}")


if __name__ == "__main__":
    main()
