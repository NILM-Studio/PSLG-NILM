"""Create a fresh downstream run that references inherited upstream artifacts.

Usage: python -m scripts.prepare_downstream_run --source-run-id OLD --run-id NEW
Only a new manifest is written. Referenced data remain at their original paths.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


# The temporal_state_merge implementation registers its step as state_merge;
# its merged cluster artifacts are also stored in time_clustering.results.
UPSTREAM_STEPS = (
    "extract_active_data", "time_segmentation", "feature_extract",
    "time_clustering", "state_merge",
)
REFERENCE_NOTE = (
    "Only manifest references were created; no upstream data were copied or modified. "
    "The source-manifest hash does not make the referenced artifacts immutable. "
    "Run validate_generation_run to record/check the actual artifact contents."
)


def validate_run_id(value: str) -> str:
    if (not isinstance(value, str) or not value or value in (".", "..")
            or value != value.strip() or any(char in value for char in "/\\:")
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("run ID must be a single non-empty path component, without traversal")
    return value


def _absolute_artifacts(value, source_root: Path):
    """Resolve artifacts wherever they occur, including nested cluster results."""
    if isinstance(value, list):
        return [_absolute_artifacts(item, source_root) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        if key != "artifacts":
            result[key] = _absolute_artifacts(item, source_root)
            continue
        if not isinstance(item, dict):
            raise ValueError("inherited artifacts must be a mapping of names to paths")
        resolved = {}
        for name, reference in item.items():
            if reference is None:
                resolved[name] = None
            elif isinstance(reference, str) and reference and "\0" not in reference:
                path = Path(reference)
                resolved[name] = str((path if path.is_absolute() else source_root / path).resolve())
            else:
                raise ValueError(f"invalid inherited artifact path for {name!r}")
        result[key] = resolved
    return result


def prepare(source_run_id: str, run_id: str, log_root: Path = Path("log")) -> dict:
    source_run_id, run_id = validate_run_id(source_run_id), validate_run_id(run_id)
    if source_run_id == run_id:
        raise ValueError("source and destination run IDs must differ")
    root = Path(log_root).resolve()
    source_root = root / source_run_id
    target_root = root / run_id
    if target_root.exists() or target_root.is_symlink():
        raise FileExistsError(f"destination run already exists: {target_root}")
    source_manifest = source_root / "run_manifest.json"
    source_bytes = source_manifest.read_bytes()
    source = json.loads(source_bytes)
    if not isinstance(source, dict) or not isinstance(source.get("steps"), dict):
        raise ValueError("source manifest must contain a steps mapping")
    inherited = {
        step: _absolute_artifacts(source["steps"][step], source_root)
        for step in UPSTREAM_STEPS if step in source["steps"]
    }
    if not inherited:
        raise ValueError("source manifest contains none of the allowed upstream steps")
    if any(not isinstance(entry, dict) for entry in inherited.values()):
        raise ValueError("inherited upstream step entries must be mappings")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    manifest = {
        "run_id": run_id,
        "appliance": source.get("appliance"),
        "variants": copy.deepcopy(source.get("variants", {})),
        "steps": inherited,
        "inherited_upstream": {
            "source_run_id": source_run_id,
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": source_sha256,
            "ownership": "inherited_senior_work",
            "fit_scope": "not_verified",
            "inherited_steps": list(inherited),
            "artifacts_copied": False,
            "artifact_immutability_guaranteed": False,
            "note": REFERENCE_NOTE,
        },
    }
    for key in ("config", "config_path", "config_sha256"):
        if key in source:
            manifest[key] = copy.deepcopy(source[key])
    # Serialize and validate everything before reserving the destination. mkdir
    # is exclusive, so a concurrent creator cannot be silently overwritten.
    serialized = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    target_root.mkdir(exist_ok=False)
    destination = target_root / "run_manifest.json"
    with destination.open("x", encoding="utf-8") as output:
        output.write(serialized)
    return {
        "run_id": run_id,
        "manifest_path": str(destination),
        "source_manifest_sha256": source_sha256,
        "inherited_steps": list(inherited),
        "dropped_steps": sorted(set(source["steps"]) - set(inherited)),
        "ownership": "inherited_senior_work",
        "fit_scope": "not_verified",
        "note": REFERENCE_NOTE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        summary = prepare(args.source_run_id, args.run_id)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"[prepare_downstream_run] {exc}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
