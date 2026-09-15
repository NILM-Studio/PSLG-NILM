"""Content-based identities for completed NILM experiments; no TensorFlow import."""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
import platform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = (
    "scripts/train_nilm_seq2point.py",
    "src/nilm/seq2point.py",
    "src/nilm/experiment_identity.py",
)
PARAMETERS = (
    "seed", "epochs", "patience", "batch_size", "window_length",
    "train_stride", "validation_stride", "test_stride", "learning_rate",
    "dropout", "mains_scale", "appliance_scale", "on_threshold",
)


def _json_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_experiment_identity(args, dataset_root: Path, manifest: dict,
                              ratio: str, group: str,
                              train_files: list[str]) -> dict:
    """Hash actual input bytes and ordered references, including OFF repeats.

    Files are streamed once per identity, even when referenced repeatedly. The
    manifest read from disk must agree with the one used to select experiments.
    No mtime/size shortcut is used: replacing an NPZ in place invalidates reuse.
    """
    root = Path(dataset_root).resolve()
    manifest_path = root / "nilm_dataset_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if json.loads(manifest_bytes) != manifest:
        raise ValueError(
            "[seq2point] dataset manifest changed after loading; reload it before training")
    full = manifest["experiments"]["full"]
    digests = {}
    inputs = {}
    for split, files in (("train", train_files),
                         ("validation", full["validation"]),
                         ("test", full["test"])):
        rows = []
        for name in files:
            path = (root / name).resolve()
            if path not in digests:
                digests[path] = _file_digest(path)
            rows.append({"file": str(name), "sha256": digests[path]})
        inputs[split] = rows
    payload = {
        "dataset_root": str(root),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": inputs,
        "experiment": {"ratio": ratio, "group": group},
        "parameters": {name: getattr(args, name) for name in PARAMETERS},
        "code_sha256": {name: _file_digest(PROJECT_ROOT / name)
                        for name in CODE_FILES},
        "runtime": {
            "python": platform.python_version(),
            **{name: _package_version(name) for name in (
                "numpy", "keras", "tensorflow", "tensorflow-cpu", "tensorflow-macos")},
        },
    }
    return {"schema_version": 1, "sha256": _json_digest(payload), "inputs": payload}


def reusable_metrics(output: Path, identity: dict, force: bool = False) -> dict | None:
    """Reuse only a completed result carrying exactly the current identity."""
    output = Path(output)
    metrics_path = output / "metrics.json"
    if force or not metrics_path.exists():
        return None
    guidance = ("Use a new --output-root to preserve the old experiment, or "
                "--force to retrain and replace its outputs.")
    identity_path = output / "experiment_identity.json"
    try:
        with identity_path.open(encoding="utf-8") as source:
            saved = json.load(source)
        with metrics_path.open(encoding="utf-8") as source:
            metrics = json.load(source)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"[seq2point] cannot verify completed result at {output}: "
            f"missing or invalid identity/metrics. {guidance}") from exc
    if (saved != identity or not isinstance(metrics, dict)
            or metrics.get("experiment_fingerprint") != identity["sha256"]):
        raise ValueError(
            f"[seq2point] refusing stale result at {output}: dataset content, "
            f"training/model parameters, code, or runtime identity differs. {guidance}")
    return metrics
