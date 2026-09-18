from pathlib import Path
import json

from nilm_experiments.nilm_lab import standalone


def test_standalone_stage_registers_manifest_without_main(tmp_path, monkeypatch):
    project = tmp_path
    monkeypatch.setattr(standalone, "run_directories",
                        lambda run_id, root: (project / "runs" / run_id,
                                              project / "runs" / run_id))

    def fake_worker(req):
        output = Path(req["output"])
        output.mkdir(parents=True, exist_ok=True)
        if req["command"] == "data":
            (output / "split_manifest.json").write_text("{}")
            (output / "data_qc.json").write_text("[]")
        else:
            (output / "metadata.json").write_text("{}")
            (output / "label_qc.json").write_text("[]")

    monkeypatch.setattr(standalone.workflow, "main_request", fake_worker)
    config = tmp_path / "config.yaml"
    config.write_text("run:\n  appliance: fridge\nruntime:\n  require_slurm: false\n")
    standalone.run("data", str(config), "fixture")
    standalone.run("labels", str(config), "fixture")

    manifest = json.loads((project / "runs/fixture/run_manifest.json").read_text())
    assert set(manifest["steps"]) == {"nilm_data", "nilm_labels"}
    assert manifest["steps"]["nilm_labels"]["artifacts"]["metadata"] == "nilm_labels/metadata.json"
    assert not (project / "runs/fixture/nilm_labels/request.json").exists()
