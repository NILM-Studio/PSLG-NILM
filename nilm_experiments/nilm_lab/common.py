from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def require_approval(config_path, approval_path):
    """Fail closed before opening real dataset files or fitting any model."""
    if not approval_path:
        raise PermissionError("Real-data operations are locked: user review and an approval file are required.")
    a = read_json(approval_path)
    if a.get("approved") is not True or a.get("config_sha256") != digest(config_path):
        raise PermissionError("Approval must authorize this exact configuration SHA256.")
    if not a.get("reviewer") or not a.get("reviewed_at"):
        raise PermissionError("Approval requires reviewer and reviewed_at.")
    c = load_config(config_path)
    manifest = Path(config_path).resolve().parent / c["manifest"]
    if a.get("manifest_sha256") != digest(manifest):
        raise PermissionError("Approval must also authorize the exact split/channel manifest SHA256.")


def load_config(path):
    import yaml
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    c["_config_path"] = str(path)
    c["_config_hash"] = digest(path)
    return c


def code_snapshot():
    paths = list((ROOT / "nilm_lab").glob("*.py"))
    paths += list((ROOT / "third_party/nilmformer/src/nilmformer").rglob("*.py"))
    paths += [ROOT / "third_party/nilmformer/src/baselines/nilm/fcn.py", ROOT / "third_party/bert4nilm/model.py", ROOT / "sources.lock.json"]
    paths += list((ROOT.parent / 'models/extract_active_data').glob('*.py'))
    for folder in ['src/steps','src/utils','models/feature_extract','models/time_segmentation']:
        paths += list((ROOT.parent/folder).glob('*.py'))
    paths = sorted(set(paths))
    return {str(p.relative_to(ROOT.parent)).replace("\\", "/"): digest(p) for p in sorted(paths)}
