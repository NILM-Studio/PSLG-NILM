"""Standalone NILM experiment stages.

This module is the Slurm-facing boundary for NILM numeric work. It does not
import or invoke main.py; it consumes a frozen run manifest and delegates
the isolated numeric implementation to nilm_lab.workflow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.framework.run_manifest import RunManifest
from src.framework.run_paths import run_directories
from .common import load_config
from . import workflow


COMMANDS = {
    "data": ("nilm_data", {"split_manifest": "split_manifest.json", "data_qc": "data_qc.json"}),
    "labels": ("nilm_labels", {"metadata": "metadata.json", "quality": "label_qc.json"}),
    "train": ("nilm_train", {"trials": "trials.json"}),
    "select": ("nilm_select", {"selection": "selection.json"}),
    "evaluate": ("nilm_evaluate", {
        "prediction_manifest": "prediction_manifest.json", "results": "results.json"}),
    "report": ("nilm_report", {"report": "report.md", "results": "report.json"}),
}


def _relative_artifacts(root: Path, output: Path, names: dict[str, str]) -> dict[str, str]:
    return {
        key: str((output / filename).relative_to(root)).replace("\\", "/")
        for key, filename in names.items()
    }


def run(command: str, config_path: str, run_id: str) -> Path:
    if command not in COMMANDS:
        raise ValueError(f"unknown NILM command: {command}")
    project = Path(__file__).resolve().parents[2]
    config = load_config(config_path)
    data_root, _ = run_directories(run_id, project)
    data_root = data_root.resolve()
    manifest_path = data_root / "run_manifest.json"
    manifest = RunManifest.load_or_create(str(manifest_path), run_id, config["run"]["appliance"])

    step_type, names = COMMANDS[command]
    if command == "data" and manifest.data.get("steps"):
        raise ValueError("data must be the first stage for a run")
    if command != "data" and "nilm_data" not in manifest.data.get("steps", {}):
        raise ValueError("run manifest has no frozen nilm_data; run the data stage first")
    if command in manifest.data.get("steps", {}) and command not in {"train", "report"}:
        raise FileExistsError(f"{step_type} is already registered; use a new run-id")

    output = data_root / step_type
    output.mkdir(parents=True, exist_ok=True)
    request = output / "request.json"
    request.write_text(json.dumps({
        "command": command,
        "config": config,
        "log_root": str(data_root),
        "output": str(output),
        "manifest": manifest.data,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    workflow.main_request({
        "command": command,
        "config": config,
        "log_root": str(data_root),
        "output": str(output),
        "manifest": manifest.data,
    })
    manifest.add_step(step_type, command, str(output.relative_to(data_root)).replace("\\", "/"),
                      _relative_artifacts(data_root, output, names))
    manifest.save()
    request.unlink(missing_ok=True)
    print(f"[nilm_lab] {command} complete: {output}", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(args.command, args.config, args.run_id)


if __name__ == "__main__":
    main()
