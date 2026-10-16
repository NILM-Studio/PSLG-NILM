#!/usr/bin/env python3
"""Run one tiny official TimeGAN iteration in its TensorFlow 1.15 environment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "runs/20261015_downstream_generation_validation/data" / "self_collected_smoke.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import tensorflow as tf
    if not tf.__version__.startswith("1.15"):
        raise RuntimeError(f"TimeGAN requires the isolated TensorFlow 1.15 environment, got {tf.__version__}")

    x = np.load(args.data, allow_pickle=False)["x"][:8, :24].astype(np.float32)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="pslg-timegan-") as temp_name:
        temp = Path(temp_name)
        shutil.copytree(ROOT / "vendor" / "TimeGAN", temp / "TimeGAN")
        source = temp / "TimeGAN" / "timegan.py"
        text = source.read_text(encoding="utf-8")
        old = "generated_data = generated_data * max_val"
        if old not in text:
            raise RuntimeError("Expected upstream TimeGAN compatibility line was not found")
        source.write_text(text.replace(old, "generated_data = np.asarray(generated_data) * max_val", 1), encoding="utf-8")
        sys.path.insert(0, str(temp / "TimeGAN"))
        from timegan import timegan
        generated = np.asarray(
            timegan(
                list(x),
                {"module": "gru", "hidden_dim": 4, "num_layer": 2, "iterations": 1, "batch_size": 4},
            )
        )

    detail = {
        "status": "pass", "runtime": {"python": sys.version.split()[0], "tensorflow": tf.__version__},
        "input_shape": list(x.shape), "generated_shape": list(generated.shape),
        "generated_finite": bool(np.isfinite(generated).all()), "seconds": round(time.time() - started, 3),
        "compatibility_patch": "temporary-copy list-to-ndarray renormalization only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(detail, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
