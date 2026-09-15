"""CPU-only checks of inherited primitives and the actual budget NILM inputs.

This checks an upstream interface, not the scientific validity or ownership of
the segmentation/clustering method. It never rewrites upstream artifacts.
Run as ``python -m scripts.validate_generation_run --run-id ID --stage upstream``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_budget_dataset import GROUPS, audit, dataset_directories


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage(length: int, intervals: list[tuple[int, int]]) -> dict:
    """Measure gaps and multiply covered samples in half-open source spans."""
    cursor = covered = overlap = invalid = 0
    for start, end in sorted(intervals):
        if start < 0 or end <= start or end > length:
            invalid += 1
        start, end = max(start, 0), min(end, length)
        if end <= start:
            continue
        overlap += max(0, min(cursor, end) - start)
        covered += max(0, end - max(cursor, start))
        cursor = max(cursor, end)
    return {"source_samples": length, "covered_samples": covered,
            "gap_samples": length - covered, "overlap_samples": overlap,
            "invalid_intervals": invalid}


def artifact(root: Path, manifest: dict, step: str, key: str,
             cluster_tag: str | None = None) -> Path:
    entry = manifest["steps"][step]
    if cluster_tag is not None:
        entry = entry["results"][cluster_tag]
    path = Path(entry["artifacts"][key])
    return path if path.is_absolute() else root / path


def upstream_report(root: Path, cluster_tag: str,
                    device_change_date: str | None = None) -> dict:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    paths = {key: artifact(root, manifest, "time_clustering", key, cluster_tag)
             for key in ("labels", "indices", "seq_len", "state_sequences")}
    labels = np.load(paths["labels"], allow_pickle=False).reshape(-1)
    indices = np.load(paths["indices"], allow_pickle=False)
    lengths = np.load(paths["seq_len"], allow_pickle=False).reshape(-1)
    if indices.ndim != 2 or indices.shape[1] < 2:
        raise ValueError("primitive indices must have activity/start columns")
    if not (len(labels) == len(indices) == len(lengths)) or not len(labels):
        raise ValueError("primitive arrays are empty or not row-aligned")
    for name, values in (("labels", labels), ("indices", indices[:, :2]),
                         ("lengths", lengths)):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{name} must contain finite integers")
    sequences = json.loads(paths["state_sequences"].read_text(encoding="utf-8"))
    segments = artifact(root, manifest, "extract_active_data", "segments_dir")
    files = sorted(segments.glob("*.csv"))
    spans, labeled_spans = defaultdict(list), defaultdict(list)
    for index, length, label in zip(indices, lengths, labels):
        spans[int(index[0])].append((int(index[1]), int(index[1] + length)))
        labeled_spans[int(index[0])].append(
            (int(index[1]), int(index[1] + length), int(label)))
    errors, examples, eras, counts = [], [], Counter(), Counter()
    activity_ids = {int(key) for key in sequences} | set(spans)
    unsupported = sorted(activity_ids - set(range(len(files))))
    if unsupported:
        errors.append(f"primitive/state sequence activity IDs have no CSV: {unsupported[:10]}")
    boundary = (pd.to_datetime(device_change_date, utc=True).timestamp()
                if device_change_date else None)
    source_digest = hashlib.sha256()
    timestamps = []
    for activity_id in sorted(activity_ids):
        if activity_id < 0 or activity_id >= len(files):
            continue
        path = files[activity_id]
        frame = pd.read_csv(path, usecols=["timestamp", "power"])
        t = frame["timestamp"].to_numpy(dtype=np.float64)
        p = frame["power"].to_numpy(dtype=np.float64)
        checks = coverage(len(frame), spans.get(activity_id, []))
        checks["invalid_signal"] = bool(
            len(t) == 0 or not np.isfinite(t).all() or not np.isfinite(p).all()
            or np.any(np.diff(t) <= 0) or np.any(p < 0))
        blocks = sequences.get(str(activity_id), [])
        checks["state_samples"] = sum(int(block["length_samples"]) for block in blocks)
        checks["state_length_mismatch"] = checks["state_samples"] != len(frame)
        state_labels = np.full(len(frame), np.iinfo(np.int64).min, dtype=np.int64)
        cursor, layout_errors = 0, 0
        for block in blocks:
            fields = [block[key] for key in ("start", "end", "length_samples", "state_label")]
            if any(not np.isfinite(value) or value != int(value) for value in fields):
                layout_errors += 1
                continue
            start, end, length, state = map(int, fields)
            if start != cursor or end - start != length or length <= 0 or end > len(frame):
                layout_errors += 1
            if 0 <= start < end <= len(frame):
                state_labels[start:end] = state
            cursor = end
        checks["state_layout_errors"] = layout_errors + int(cursor != len(frame))
        checks["primitive_state_mismatches"] = sum(
            not np.all(state_labels[start:end] == label)
            for start, end, label in labeled_spans[activity_id]
            if 0 <= start < end <= len(frame))
        bad = any(checks[key] for key in (
            "gap_samples", "overlap_samples", "invalid_intervals", "invalid_signal",
            "state_length_mismatch", "state_layout_errors", "primitive_state_mismatches"))
        counts["activities"] += 1
        counts["invalid_activities"] += int(bad)
        for key in ("source_samples", "gap_samples", "overlap_samples"):
            counts[key] += checks[key]
        if bad and len(examples) < 10:
            examples.append({"activity_id": activity_id, "file": path.name, **checks})
        if len(t) and np.isfinite(t).all():
            timestamps.extend([float(t[0]), float(t[-1])])
            if boundary is not None:
                eras["before" if t[-1] < boundary else
                     "after" if t[0] >= boundary else "crosses_change"] += 1
        source_digest.update(f"{activity_id}:{path.name}:{sha256_file(path)}\n".encode())
    if not counts["activities"]:
        errors.append("no usable source activities")
    if counts["invalid_activities"]:
        errors.append("source coverage/signal checks failed; inspect examples before generation")
    return {
        "upstream_interface_passed": not errors, "cluster_tag": cluster_tag,
        "counts": dict(counts), "examples": examples, "errors": errors,
        "unreferenced_source_activities": len(set(range(len(files))) - activity_ids),
        "source_timestamp_range": [min(timestamps), max(timestamps)] if timestamps else None,
        "device_change_date": device_change_date, "device_era_counts": dict(eras),
        "source_digest": source_digest.hexdigest(),
        "artifact_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "upstream_representation_holdout": "not_verified",
        "scope": "Inherited segmentation/clustering interface; no upstream method changes.",
    }


def inspect_pair(path: Path, period: float) -> dict:
    if not np.isfinite(period) or period <= 0:
        raise ValueError("sample_period_seconds must be finite and positive")
    with np.load(path, allow_pickle=False) as payload:
        values = {key: np.asarray(payload[key])
                  for key in ("timestamp", "mains", "appliance")}
    t, m, a = (values[key] for key in ("timestamp", "mains", "appliance"))
    if (any(value.ndim != 1 for value in values.values()) or not len(t)
            or not (len(t) == len(m) == len(a))):
        raise ValueError(f"invalid or empty pair shape: {path}")
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"non-finite pair: {path}")
    if np.any(m < 0) or np.any(a < 0):
        raise ValueError(f"negative power: {path}")
    if len(t) > 1 and not np.allclose(np.diff(t), period, rtol=0, atol=1e-6):
        raise ValueError(f"pair is not on its declared uniform time grid: {path}")
    return {"samples": len(t), "on_samples": int(np.sum(a >= 20)),
            "start": float(t[0]), "end": float(t[-1]),
            "appliance_energy_wh": float(np.sum(a, dtype=np.float64) * period / 3600),
            "appliance_peak_watts": float(np.max(a)),
            "mains_below_appliance_samples": int(np.sum(m < a)),
            "appliance_over_4000_samples": int(np.sum(a > 4000))}


def dataset_report(root: Path) -> dict:
    dataset, continuous = dataset_directories(root)
    result = audit(dataset, continuous)
    cycle = json.loads((dataset / "nilm_dataset_manifest.json").read_text(encoding="utf-8"))
    timeline = json.loads((continuous / "nilm_dataset_manifest.json").read_text(encoding="utf-8"))
    errors = list(result["errors"])
    measurement = {
        key: cycle.get(key) for key in ("mains_power_type", "appliance_power_type",
                                       "measurement_compatible_for_additive_synthesis")}
    if (measurement["measurement_compatible_for_additive_synthesis"] is not True
            or measurement["mains_power_type"] != "active"
            or measurement["appliance_power_type"] != "active"):
        errors.append("additive generation is not certified active/active by preparation audit")
    if (not cycle.get("aligned_series") or not timeline.get("aligned_series")
            or Path(cycle["aligned_series"]).resolve() != Path(timeline["aligned_series"]).resolve()):
        errors.append("cycle and continuous aligned_series differ or are not recorded")
    if timeline.get("source_cycle_manifest_sha256") != sha256_file(dataset / "nilm_dataset_manifest.json"):
        errors.append("continuous dataset source manifest hash is missing or stale; rebuild continuous")
    period = float(timeline["sample_period_seconds"])
    summaries = {}

    def pair(relative):
        path = (continuous / relative).resolve()
        if path not in summaries:
            summaries[path] = inspect_pair(path, period)
        return summaries[path]

    distributions = {}
    for tag, row in timeline["experiments"].items():
        for key in (GROUPS if tag != "full" else ("D_full_real", "validation", "test")):
            records = [pair(relative) for relative in row[key]]
            samples = sum(record["samples"] for record in records)
            distributions[f"{tag}/{key}"] = {
                "references": len(records), "samples": samples,
                "unique_files": len(set(row[key])),
                "on_fraction_at_20w": (sum(record["on_samples"] for record in records)
                                       / samples if samples else None),
                "appliance_energy_wh": sum(record["appliance_energy_wh"] for record in records),
                "mains_below_appliance_samples": sum(
                    record["mains_below_appliance_samples"] for record in records),
                "appliance_over_4000_samples": sum(
                    record["appliance_over_4000_samples"] for record in records),
            }
            if not samples:
                errors.append(f"empty group {tag}/{key}")
    validation_start = float(timeline["temporal_boundaries"]["validation_start"])
    test_start = float(timeline["temporal_boundaries"]["test_start"])
    if validation_start >= test_start:
        errors.append("invalid validation/test boundaries")
    full = timeline["experiments"]["full"]
    for key, lower, upper in (("D_full_real", None, validation_start),
                               ("validation", validation_start, test_start),
                               ("test", test_start, None)):
        for relative in full[key]:
            record = pair(relative)
            if ((lower is not None and record["start"] < lower)
                    or (upper is not None and record["end"] >= upper)):
                errors.append(f"real {key} crosses temporal boundary: {relative}")
    # Synthetic timestamps are relative. Test real and OFF inputs separately.
    real_paths = {(dataset / name).resolve()
                  for row in cycle["experiments"].values()
                  for name in row.get("A_real_only", [])}
    for tag, row in timeline["experiments"].items():
        if tag == "full":
            continue
        # Traditional scaling preserves absolute source timestamps; generated
        # cycles use a relative time axis and are checked via source provenance.
        for name in row["B_real_plus_traditional"]:
            record = pair(name)
            if record["end"] >= validation_start:
                errors.append(f"{tag}: real/traditional/OFF training crosses validation boundary: {name}")
    off_paths = set()
    for tag, row in timeline["experiments"].items():
        key = "D_full_real" if tag == "full" else "A_real_only"
        active = {(dataset / name).resolve() for name in cycle["experiments"][tag][key]}
        off_paths.update((continuous / name).resolve() for name in row[key]
                         if (continuous / name).resolve() not in active)
    off_threshold = float(timeline["active_threshold_watts"])
    if not np.isfinite(off_threshold) or off_threshold < 0:
        errors.append("invalid OFF threshold")
    for path in off_paths:
        if summaries[path]["appliance_peak_watts"] > off_threshold:
            errors.append(f"OFF background contains active target samples: {path}")
    generated = json.loads((dataset / "budget_synthesis_manifest.json").read_text(encoding="utf-8"))
    diversity = {}
    for tag, rows in generated.items():
        self_only = sum(set(map(str, row["primitive_source_activity_ids"]))
                        == {str(row["source_activity_id"])} for row in rows)
        diversity[tag] = {
            "generated_cycles": len(rows), "anchor_only_cycles": self_only,
            "anchor_only_fraction": self_only / len(rows) if rows else None,
            "mean_source_cycles": float(np.mean([
                len(set(row["primitive_source_activity_ids"])) for row in rows])) if rows else None,
        }
    result.update({"numeric_and_temporal_checks_passed": not errors,
                   "errors": errors, "training_distributions": distributions,
                   "budget_source_diversity": diversity,
                   "unique_npz_checked": len(summaries),
                   "unique_budget_real_files": len(real_paths),
                   "dataset_dir": str(dataset), "continuous_dir": str(continuous),
                   "measurement": measurement,
                   "synthesis_method": (cycle.get("budget_conditioning") or {}).get("method", "not_recorded"),
                   "limitations": [
                       "Upstream encoder/clustering fit scope is not certified.",
                       "Activity-cycle budget excludes shared structure/OFF resources.",
                       "Numerical checks do not establish NILM improvement."]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster-tag", default="kmeans_k4_merged")
    parser.add_argument("--stage", choices=("upstream", "dataset", "all"), default="upstream")
    parser.add_argument("--device-change-date", help="Optional ISO date for source-era counts")
    parser.add_argument("--output", type=Path, help="Optional report file; source artifacts stay read-only")
    args = parser.parse_args()
    root = Path("log") / args.run_id
    result = {"run_id": args.run_id, "stage": args.stage, "checks": {}}
    for stage in (("upstream", "dataset") if args.stage == "all" else (args.stage,)):
        try:
            result["checks"][stage] = (upstream_report(root, args.cluster_tag, args.device_change_date)
                                       if stage == "upstream" else dataset_report(root))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            result["checks"][stage] = {"errors": [str(exc)]}
    result["validation_passed"] = all(not row.get("errors") for row in result["checks"].values())
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["validation_passed"] else 1)


if __name__ == "__main__":
    main()
