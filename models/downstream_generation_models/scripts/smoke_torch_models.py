#!/usr/bin/env python3
"""Small real-data build/forward/backward/sample checks for the PyTorch models."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"


def add_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite scalar: {value}")
    return value


def diffusion_ts(x: torch.Tensor, device: torch.device) -> dict:
    add_path(VENDOR / "Diffusion-TS")
    from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

    model = Diffusion_TS(
        seq_length=x.shape[1], feature_size=1, n_layer_enc=1, n_layer_dec=1,
        d_model=32, timesteps=8, sampling_timesteps=2, n_heads=2,
        mlp_hidden_times=2, kernel_size=1, padding_size=0, use_ff=False,
    ).to(device)
    batch = x[:2]
    loss = model(batch)
    loss.backward()
    model.eval()
    with torch.no_grad():
        generated = model.sample((1, x.shape[1], 1))
    return {"loss": finite(loss.item()), "generated_shape": list(generated.shape), "finite": bool(torch.isfinite(generated).all())}


def csdi(x: torch.Tensor, device: torch.device) -> dict:
    add_path(VENDOR / "CSDI")
    from main_model import CSDI_Physio

    config = {
        "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3, "itr_per_epoch": 1},
        "diffusion": {
            "layers": 1, "channels": 16, "nheads": 2, "diffusion_embedding_dim": 32,
            "beta_start": 0.0001, "beta_end": 0.1, "num_steps": 4,
            "schedule": "quad", "is_linear": False,
        },
        "model": {"is_unconditional": 0, "timeemb": 16, "featureemb": 8, "target_strategy": "random"},
    }
    model = CSDI_Physio(config, device, target_dim=1).to(device)
    data = x[:2]
    mask = torch.ones_like(data)
    gt_mask = mask.clone()
    gt_mask[:, data.shape[1] // 2 :, :] = 0
    batch = {
        "observed_data": data,
        "observed_mask": mask,
        "timepoints": torch.arange(data.shape[1], device=device).float().repeat(data.shape[0], 1),
        "gt_mask": gt_mask,
    }
    loss = model(batch, is_train=1)
    loss.backward()
    with torch.no_grad():
        samples, _, _, _, _ = model.evaluate(batch, n_samples=1)
    return {"loss": finite(loss.item()), "sample_shape": list(samples.shape), "finite": bool(torch.isfinite(samples).all())}


def imagentime(x: torch.Tensor, device: torch.device) -> dict:
    add_path(VENDOR / "ImagenTime")
    # The pinned upstream imports dataset loaders that are absent from its own
    # repository just to reach two scaling helpers. Delay embedding uses neither;
    # provide the exact two helpers as a narrow adapter without editing vendor/.
    import utils
    utils_data = types.ModuleType("utils.utils_data")
    utils_data.MinMaxScaler = lambda data, return_scalers=False: (
        (data - np.min(data, 0)) / (np.max(data, 0) - np.min(data, 0) + 1e-7),
        np.min(data, 0),
        np.max(data, 0),
    ) if return_scalers else (data - np.min(data, 0)) / (np.max(data, 0) - np.min(data, 0) + 1e-7)
    utils_data.MinMaxArgs = lambda data, low, high: (data - low) / (high - low + 1e-7)
    sys.modules["utils.utils_data"] = utils_data
    from models.model import ImagenTime

    args = SimpleNamespace(
        diffusion_steps=4, img_resolution=32, input_channels=1, ch_mult=[1, 1],
        unet_channels=16, attn_resolution=[16], use_stft=False, delay=2,
        embedding=32, seq_len=x.shape[1], ema=False, ema_warmup=1,
    )
    model = ImagenTime(args, device).to(device)
    signal = x[:2]
    image = model.ts_to_img(signal)
    recovered = model.img_to_ts(image)
    transform_mae = (recovered - signal).abs().mean()
    loss, logged = model.loss_fn(image)
    loss.backward()
    return {
        "loss": finite(loss.item()), "image_shape": list(image.shape),
        "roundtrip_mae": finite(transform_mae.item()), "upstream_log": logged,
    }


def timevqvae(x: torch.Tensor, device: torch.device) -> dict:
    add_path(VENDOR / "TimeVQVAE" / "src")
    from timevqvae.maskgit import MaskGIT, PriorModelConfig
    from timevqvae.vqvae import VQVAE

    model = VQVAE(
        in_channels=1, input_length=x.shape[1], n_fft=4, init_dim=2, hid_dim=16,
        downsampled_width_l=4, downsampled_width_h=16,
        encoder_n_resnet_blocks=1, decoder_n_resnet_blocks=1,
        codebook_size_l=16, codebook_size_h=16, kmeans_init=False, codebook_dim=4,
    ).to(device)
    batch = x[:2].permute(0, 2, 1)
    output = model(batch)
    loss = sum(output.recons_loss.values())
    for item in output.vq_losses.values():
        if torch.is_tensor(item):
            loss = loss + item.mean()
    loss.backward()
    prior_config = PriorModelConfig(
        hidden_dim=16, n_layers=1, heads=1, ff_mult=1, use_rmsnorm=True,
        p_unconditional=0.2, model_dropout=0.0, emb_dropout=0.0,
    )
    prior = MaskGIT(
        vqvae=model, lf_choice_temperature=1.0, hf_choice_temperature=1.0,
        lf_num_sampling_steps=2, hf_num_sampling_steps=2,
        lf_codebook_size=16, hf_codebook_size=16, transformer_embedding_dim=16,
        lf_prior_model_config=prior_config, hf_prior_model_config=prior_config,
        classifier_free_guidance_scale=1.0, n_classes=3,
    ).to(device)
    class_condition = torch.tensor([[0], [1]], device=device)
    prior_losses = prior(batch, class_condition)
    prior_losses.total_mask_prediction_loss.backward()
    prior.eval()
    with torch.no_grad():
        token_l, token_h = prior.iterative_decoding(
            num_samples=2, mode="cosine", class_condition=1, device=device,
        )
        generated = prior.decode_token_ind_to_timeseries(token_l, "lf") + prior.decode_token_ind_to_timeseries(token_h, "hf")
    return {
        "stage1_loss": finite(loss.item()),
        "stage2_prior_loss": finite(prior_losses.total_mask_prediction_loss.item()),
        "reconstruction_shape": list(output.x_recon.shape),
        "reconstruction_finite": bool(torch.isfinite(output.x_recon).all()),
        "generated_shape": list(generated.shape),
        "generated_finite": bool(torch.isfinite(generated).all()),
        "reconstruction_terms": {k: finite(v.item()) for k, v in output.recons_loss.items()},
    }


def fourier_flows(x: torch.Tensor, _device: torch.device) -> dict:
    add_path(VENDOR / "Fourier-flows")
    from fflows import FourierFlow

    train = x[:16].detach().cpu().numpy()
    model = FourierFlow(hidden=8, n_flows=2, FFT=True, normalize=True)
    losses = model.fit(train, epochs=2, batch_size=16, learning_rate=1e-3, display_step=1)
    generated = model.sample(2)
    return {
        "last_loss": finite(losses[-1]), "generated_shape": list(generated.shape),
        "finite": bool(np.isfinite(generated).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "runs/20261015_downstream_generation_validation/data" / "self_collected_smoke.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    np.random.seed(7)
    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)

    data = np.load(args.data, allow_pickle=False)
    x = torch.from_numpy(data["x"][:, :64]).float().to(device)
    results = {
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": str(device), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
        "input": {"path": str(args.data), "shape": list(x.shape)},
        "models": {},
    }
    checks = [
        ("Diffusion-TS", diffusion_ts), ("CSDI", csdi), ("ImagenTime", imagentime),
        ("TimeVQVAE", timevqvae), ("Fourier-flows", fourier_flows),
    ]
    for name, check in checks:
        started = time.time()
        try:
            detail = check(x, device)
            results["models"][name] = {"status": "pass", "seconds": round(time.time() - started, 3), **detail}
        except Exception as exc:  # keep independent checks observable
            results["models"][name] = {
                "status": "fail", "seconds": round(time.time() - started, 3),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            }
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    failed = [name for name, detail in results["models"].items() if detail["status"] != "pass"]
    if args.strict and failed:
        raise SystemExit(f"failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
