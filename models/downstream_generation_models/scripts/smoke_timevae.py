#!/usr/bin/env python3
"""One real-data TimeVAE optimization step and prior sample."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "runs/20261015_downstream_generation_validation/data" / "self_collected_smoke.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "vendor" / "timeVAE" / "src"))
    from vae.timevae import TimeVAE

    np.random.seed(7)
    tf.random.set_seed(7)
    x = np.load(args.data, allow_pickle=False)["x"][:8, :64].astype(np.float32)
    started = time.time()
    model = TimeVAE(
        seq_len=64, feat_dim=1, latent_dim=4, hidden_layer_sizes=[8, 16],
        trend_poly=1, custom_seas=None, use_residual_conn=True, batch_size=4,
    )
    metrics = model.train_step(tf.convert_to_tensor(x[:4]))
    generated = model.get_prior_samples(2)
    detail = {
        "status": "pass",
        "runtime": {"python": sys.version.split()[0], "tensorflow": tf.__version__},
        "input_shape": list(x.shape),
        "generated_shape": list(generated.shape),
        "generated_finite": bool(np.isfinite(generated).all()),
        "metrics": {k: float(v.numpy()) for k, v in metrics.items()},
        "seconds": round(time.time() - started, 3),
    }
    if not all(math.isfinite(value) for value in detail["metrics"].values()):
        raise RuntimeError(f"non-finite TimeVAE metrics: {detail['metrics']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(detail, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
