"""Frozen train/validation adapters for the approved public16 representation study.

The caller supplies already scaled, split N x 128 x 1 numpy arrays and an explicit
device. Each process may import exactly one upstream method. Vendor files remain
unchanged. No activity selection, scaling, PCA, or clustering happens here.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

VENDOR = Path(__file__).resolve().parent / "vendor"
_METHOD = None
_UPSTREAM = None
_DEFAULTS = {
    "ts2vec": dict(native_dim=320, hidden_dims=64, depth=10, weight_decay=.01),
    "timesurl": dict(native_dim=320, hidden_dims=64, depth=10, lmd=1.,
                     mask_ratio_per_seg=.25, segment_num=4, weight_decay=.0005),
    "timemae": dict(native_dim=64, d_model=64, layers=8, attn_heads=4, d_ff=256,
                    wave_length=8, mask_ratio=.6, vocab_size=192, reg_layers=4,
                    momentum=.99, dropout=.2, alpha=5., beta=1.,
                    label_smoothing=.2, weight_decay=.01, enable_res_parameter=1),
    "timedart": dict(native_dim=512, d_model=512, e_layers=2, d_layers=1,
                     n_heads=8, d_ff=2048, dropout=.1, head_dropout=.1,
                     patch_len=8, stride=8, time_steps=1000, scheduler="cosine",
                     mask_ratio=1., use_norm=0, weight_decay=0.),
}


def _upstream(method):
    """Isolate upstream bare imports from the host project's models package."""
    global _METHOD, _UPSTREAM
    if _METHOD is not None:
        if _METHOD != method:
            raise RuntimeError("Different upstream models require separate processes")
        return _UPSTREAM
    roots = {"models", "model", "utils", "layers", "lib", "collator"}
    saved = {k: v for k, v in list(sys.modules.items()) if k.split(".")[0] in roots}
    old_path = list(sys.path)
    for k in saved:
        del sys.modules[k]
    vendor = VENDOR / method / ("src" if method == "timesurl" else "")
    sys.path.insert(0, str(vendor))
    try:
        if method in ("ts2vec", "timesurl"):
            name = "ts2vec" if method == "ts2vec" else "timesurl"
            spec = importlib.util.spec_from_file_location("_pslg_formal_" + method, vendor / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            cls = module.TS2Vec if method == "ts2vec" else module.TimesURL
            collator = importlib.import_module("collator").CLDataCollator if method == "timesurl" else None
            result = (cls, module, collator)
        elif method == "timemae":
            result = (importlib.import_module("model.TimeMAE").TimeMAE, None, None)
        else:
            result = (importlib.import_module("models.TimeDART").Model, None, None)
    finally:
        for k in list(sys.modules):
            if k.split(".")[0] in roots:
                del sys.modules[k]
        sys.modules.update(saved)
        sys.path[:] = old_path
    _METHOD, _UPSTREAM = method, result
    return result


def _rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def _set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def _fixed_random(seed):
    state = _rng()
    try:
        _seed(seed)
        yield
    finally:
        _set_rng(state)


def _atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _array(x, allow_empty=False):
    if x is None and allow_empty:
        return np.empty((0, 128, 1), dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (128, 1) or (not len(x) and not allow_empty):
        raise ValueError("Expected nonempty N x 128 x 1 numpy array")
    if not np.isfinite(x).all():
        raise ValueError("Inputs must contain finite values")
    return x


def _config(config, n_train=None):
    cfg = dict(config)
    method = cfg.get("method", cfg.get("model"))
    if method not in _DEFAULTS:
        raise ValueError("method must be ts2vec, timesurl, timemae or timedart")
    if "device" not in cfg:
        raise ValueError("Explicit device required; scheduler must set CUDA_VISIBLE_DEVICES")
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("CUDA_VISIBLE_DEVICES must be explicitly assigned")
        if device.index not in (None, 0):
            raise ValueError("Use local cuda:0 after UUID visibility restriction")
        torch.cuda.set_device(0)
    actual = {**_DEFAULTS[method], **cfg, "method": method, "device": str(device)}
    actual.setdefault("learning_rate", actual.get("lr", .0001 if method == "timedart" else .001))
    actual.setdefault("batch_size", 32 if n_train is None else min(32, n_train))
    if n_train is not None:
        actual["batch_size"] = min(actual["batch_size"], n_train)
    actual.setdefault("validation_seed", 20261017)
    actual.setdefault("validation_sample_cap", 512)
    actual.setdefault("min_relative_improvement", .001)
    return actual


class FrozenEncoder:
    """One native representation model; encode never updates training state."""
    def __init__(self, config):
        self.config = dict(config)
        self.method = config["method"]
        self.device = torch.device(config["device"])
        cls, self.up, collator_cls = _upstream(self.method)
        if self.method in ("ts2vec", "timesurl"):
            kwargs = dict(input_dims=1, output_dims=config["native_dim"],
                          hidden_dims=config["hidden_dims"], depth=config["depth"],
                          device=self.device, lr=config["learning_rate"],
                          batch_size=config["batch_size"])
            if self.method == "timesurl":
                kwargs["args"] = SimpleNamespace(**config)
                self.collator = collator_cls(max_len=128, args=kwargs["args"])
            wrapper = cls(**kwargs)
            self.train_net, self.net = wrapper._net, wrapper.net
        else:
            args = {**config, "device": self.device, "data_shape": (128, 1), "num_class": 2,
                    "input_len": 128, "pred_len": 128, "task_name": "pretrain"}
            self.train_net = self.net = cls(SimpleNamespace(**args)).to(self.device)
            if self.method == "timemae":
                self.net.copy_weight()
        self.net.eval()

    @torch.no_grad()
    def encode(self, x, batch_size=None):
        x = _array(x, allow_empty=True)
        self.net.eval()
        size = batch_size or self.config["batch_size"]
        out = []
        for first in range(0, len(x), size):
            batch = torch.as_tensor(x[first:first+size], device=self.device)
            if self.method == "ts2vec":
                z = self.net(batch).amax(dim=1)
            elif self.method == "timesurl":
                time_grid = torch.linspace(0, 1, 128, device=self.device).view(1, 128, 1).expand(len(batch), -1, -1)
                z = self.net({"data": torch.cat([batch, time_grid], dim=-1),
                              "mask": torch.ones_like(batch)}).amax(dim=1)
            elif self.method == "timemae":
                z = self.net(batch)
            else:
                patches = self.net.patch(self.net.channel_independence(batch))
                tokens = self.net.positional_encoding(self.net.enc_embedding(patches))
                z = self.net.encoder(tokens, is_mask=False).mean(dim=1)
            out.append(z.cpu().numpy())
        z = np.concatenate(out) if out else np.empty((0, self.config["native_dim"]), np.float32)
        if z.shape != (len(x), self.config["native_dim"]) or not np.isfinite(z).all():
            raise FloatingPointError("Invalid native embeddings")
        return z.astype(np.float32, copy=False)

    def state(self):
        return dict(net=self.train_net.state_dict(),
                    swa=self.net.state_dict() if self.net is not self.train_net else None)

    def restore(self, state):
        if self.method == "timemae":
            # Upstream copy_weight aliases encoder/momentum parameter storage.
            # Break aliases before load, otherwise loading momentum overwrites
            # freshly restored encoder weights. Training initialization is kept.
            for param in self.train_net.momentum_encoder.parameters():
                param.data = param.data.clone()
        self.train_net.load_state_dict(state["net"])
        if state.get("swa") is not None:
            self.net.load_state_dict(state["swa"])
        self.net.eval()

    def loss(self, batch, validation=False):
        net = self.net if validation else self.train_net
        if self.method == "ts2vec":
            left, right, crop_l = self._crops(batch)
            mask = "binomial" if validation else None
            return self.up.hierarchical_contrastive_loss(
                net(left, mask=mask)[:, -crop_l:], net(right, mask=mask)[:, :crop_l], temporal_unit=0)
        if self.method == "timesurl":
            return self._timesurl_loss(net, batch, validation)
        if self.method == "timemae":
            (target, pred), (logits, labels) = net.pretrain_forward(batch)
            return self.config["alpha"] * F.mse_loss(pred, target) + self.config["beta"] * F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                label_smoothing=self.config["label_smoothing"])
        return F.mse_loss(net(batch), batch)

    def _crops(self, x):
        length = x.size(1)
        crop_l = np.random.randint(2, length + 1)
        left = np.random.randint(length - crop_l + 1)
        right = left + crop_l
        ext_left = np.random.randint(left + 1)
        ext_right = np.random.randint(right, length + 1)
        offset = np.random.randint(-ext_left, length - ext_right + 1, size=x.size(0))
        return (self.up.take_per_row(x, offset + ext_left, right - ext_left),
                self.up.take_per_row(x, offset + left, ext_right - left), crop_l)

    def _timesurl_loss(self, net, batch, validation):
        # The official collator runs on CPU, with time-sensitive sampling/masking.
        xb = batch.detach().cpu()
        time_grid = torch.linspace(0, 1, 128)
        collated = self.collator([(a, time_grid, torch.ones_like(a)) for a in xb])
        values, times, masks, originals = [collated[k].to(self.device) for k in
                                          ("value", "time", "mask", "mask_origin")]
        loss = batch.new_zeros(())
        for seq in range(values.size(1)):
            x = torch.cat([values[:, seq], times[:, seq, :, None]], dim=-1)
            m, old = masks[:, seq], originals[:, seq]
            length = x.size(1)
            crop_l = np.random.randint(2, length + 1)
            left = np.random.randint(length - crop_l + 1)
            right = left + crop_l
            ext_left = np.random.randint(left + 1)
            ext_right = np.random.randint(right, length + 1)
            offsets = np.random.randint(-ext_left, length - ext_right + 1, size=len(x))
            l_args = (offsets + ext_left, right - ext_left)
            r_args = (offsets + left, ext_right - left)
            take = self.up.take_per_row
            xl, xr = take(x, *l_args), take(x, *r_args)
            # Device-safe equivalent of upstream freq_mix (which hardcodes CUDA).
            freq = torch.fft.fft(xr[..., :-1], dim=1)
            mixing = torch.empty(freq.shape, device=freq.device).uniform_() < .5
            mixing &= freq.abs().sort(dim=1, descending=True)[1] > 2
            idx = np.arange(len(xr)); np.random.shuffle(idx)
            freq2 = torch.fft.fft(xr[idx, :, :-1], dim=1)
            mixed = torch.complex(freq.real.masked_fill(mixing, 0) + freq2.real.masked_fill(~mixing, 0),
                                  freq.imag.masked_fill(mixing, 0) + freq2.imag.masked_fill(~mixing, 0))
            xr = torch.cat([torch.fft.ifft(mixed, dim=1).abs(), xr[..., -1:]], dim=-1)
            ml, mr = take(m[..., :1], *l_args), take(m[..., :1], *r_args)
            il, ir = take(m[..., 1:], *l_args), take(m[..., 1:], *r_args)
            ol, ore = take(old, *l_args), take(old, *r_args)
            def forward(data, mask, inter, origin):
                if not validation:
                    return net({"data": data, "mask": mask, "mask_inter": inter, "mask_origin": origin})
                # eval-mode upstream omits the reconstruction branch; use the same
                # shared layers explicitly with dropout disabled and no updates.
                core = net.module
                def features(mask_):
                    z = core.input_fc(data[..., :-1] * mask_).transpose(1, 2)
                    return core.repr_dropout(core.feature_extractor(z)).transpose(1, 2)
                return features(origin), core.interphead(features(mask))
            zl, rl = forward(xl, ml, il, ol)
            zr, rr = forward(xr, mr, ir, ore)
            zl, rl, zr, rr = zl[:, -crop_l:], rl[:, -crop_l:], zr[:, :crop_l], rr[:, :crop_l]
            xl, xr, il, ir = xl[:, -crop_l:], xr[:, :crop_l], il[:, -crop_l:], ir[:, :crop_l]
            loss = loss + self.config["lmd"] * self.up.hierarchical_contrastive_loss(zl, zr, temporal_unit=0, temp=1.)
            for data, recon, inter in ((xl, rl, il), (xr, rr, ir)):
                if torch.sum(inter) > 0:
                    loss = loss + ((data[..., :-1] - recon) * inter).square().sum() / (inter.sum()+1e-10) / 2
        return loss


def load(path, device="cpu", checkpoint="best.pt"):
    """Load a frozen encoder; use checkpoint='last.pt' only for diagnostics."""
    path = Path(path)
    saved = torch.load(path / checkpoint if path.is_dir() else path, map_location="cpu", weights_only=False)
    cfg = _config({**saved["config"], "device": device})
    model = FrozenEncoder(cfg)
    model.restore(saved["model"])
    return model


def _validation(encoder, x):
    if not len(x):
        return None
    encoder.net.eval()
    total, count = 0., 0
    with _fixed_random(encoder.config["validation_seed"]), torch.no_grad():
        size = encoder.config["batch_size"]
        for first in range(0, len(x), size):
            batch = torch.as_tensor(x[first:first+size], device=encoder.device)
            value = float(encoder.loss(batch, validation=True).item())
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation objective")
            total += value * len(batch); count += len(batch)
    return total / count


def train(config, X_train, X_val, output_dir, seed, steps_cap, min_steps=500,
          validate_every=100, patience=10, fixed_steps=None):
    """Train with frozen-validation selection, or exact fixed full-refit updates.

    config.resume_checkpoint may point at last.pt for exact RNG/order resume.
    Short pilots below min_steps export final state but report legal_selection=false.
    """
    x, xv = _array(X_train), _array(X_val, allow_empty=True)
    cfg = _config(config, len(x))
    limit = int(steps_cap if fixed_steps is None else fixed_steps)
    if limit < 1 or min_steps < 0 or validate_every < 1 or patience < 1:
        raise ValueError("Invalid update or validation budget")
    if not len(xv) and fixed_steps is None:
        raise ValueError("Heldout training requires explicit validation input")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists() and not cfg.get("resume_checkpoint"):
        raise FileExistsError("Refusing to overwrite existing training")
    _seed(int(seed))
    encoder = FrozenEncoder(cfg)
    optimizer_cls = torch.optim.Adam if cfg["method"] == "timedart" else torch.optim.AdamW
    optimizer = optimizer_cls(encoder.train_net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    data_rng = np.random.default_rng(seed + 713)
    order, cursor = data_rng.permutation(len(x)), 0
    step, epoch, best_step, stale = 0, 0, None, 0
    best_loss, stopping_reference = None, None
    history = []
    if cfg.get("resume_checkpoint"):
        state = torch.load(cfg["resume_checkpoint"], map_location="cpu", weights_only=False)
        frozen_keys = set(_DEFAULTS[cfg["method"]]) | {"method", "learning_rate", "batch_size"}
        if any(cfg[k] != state["config"][k] for k in frozen_keys):
            raise ValueError("Resume configuration differs from checkpoint")
        encoder.restore(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch = state["step"], state["epoch"]
        best_step, best_loss, stale = state["best_step"], state["best_validation_loss"], state["stale_checks"]
        stopping_reference = state["stopping_reference"]
        history = state["history"]
        order, cursor = state["sampler"]["order"], state["sampler"]["cursor"]
        data_rng.bit_generator.state = state["sampler"]["rng"]
        _set_rng(state["rng"])
    val_ids = np.random.default_rng(cfg["validation_seed"]).permutation(len(xv))[:cfg["validation_sample_cap"]]
    val = xv[val_ids]
    last_validation_loss = next((h["validation_loss"] for h in reversed(history) if "validation_loss" in h), None)
    np.save(out / "validation_objective_indices.npy", val_ids)
    started, initial_step = time.perf_counter(), step
    _atomic_json(out / "config.json", {**cfg, "seed": seed, "steps_cap": steps_cap,
                                      "min_steps": min_steps, "validate_every": validate_every,
                                      "patience": patience, "fixed_steps": fixed_steps,
                                      "validation_mode": "eval_frozen_fixed_augment_rng_no_state_update",
                                      "timesurl_frequency_augmentation": "upstream_math_device_safe_allocation",
                                      "timesurl_eval_reconstruction": "explicit_shared_layers_dropout_disabled",
                                      "parameter_count": sum(p.numel() for p in encoder.train_net.parameters())})
    def checkpoint():
        return dict(format_version=1, config=cfg, seed=seed, step=step, epoch=epoch,
                    model=encoder.state(), optimizer=optimizer.state_dict(),
                    rng=_rng(), best_step=best_step, best_validation_loss=best_loss,
                    stale_checks=stale, stopping_reference=stopping_reference, history=history,
                    sampler=dict(order=order, cursor=cursor, rng=data_rng.bit_generator.state),
                    fixed_steps=fixed_steps, validation_indices=val_ids)
    try:
        while step < limit:
            size = cfg["batch_size"]
            if cursor >= len(x) or (cfg["method"] == "ts2vec" and len(x)-cursor < size):
                order, cursor = data_rng.permutation(len(x)), 0
                epoch += 1
            ix = order[cursor:cursor+size]; cursor += len(ix)
            batch = torch.as_tensor(x[ix], device=encoder.device)
            encoder.train_net.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            loss = encoder.loss(batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward(); optimizer.step()
            if encoder.net is not encoder.train_net:
                encoder.net.update_parameters(encoder.train_net)
            elif cfg["method"] == "timemae":
                encoder.net.momentum_update()
            step += 1
            elapsed = time.perf_counter() - started
            record = dict(step=step, train_loss=float(loss.item()), update_seconds=time.perf_counter()-tick)
            is_check = step % validate_every == 0 or step == limit
            if is_check and fixed_steps is None:
                value = _validation(encoder, val)
                last_validation_loss = value
                record["validation_loss"] = value
                if step >= min_steps:
                    if best_loss is None or value < best_loss:
                        best_loss, best_step = value, step
                        save_best = True
                    else:
                        save_best = False
                    if stopping_reference is None or value < stopping_reference - cfg["min_relative_improvement"] * max(abs(stopping_reference), 1e-12):
                        stopping_reference, stale = value, 0
                    else:
                        stale += 1
                else:
                    save_best = False
            else:
                save_best = False
            history.append(record)
            if save_best:
                _save(out / "best.pt", checkpoint())
            if is_check:
                _save(out / "last.pt", checkpoint())
            if step % 10 == 0 or is_check:
                _atomic_json(out / "progress.json", dict(status="running", step=step, target_steps=limit,
                    epoch=epoch, best_step=best_step, best_validation_loss=best_loss,
                    stale_checks=stale, train_loss=record["train_loss"], elapsed_seconds=elapsed,
                    current_validation_loss=last_validation_loss,
                    updates_per_second=(step-initial_step)/max(elapsed, 1e-9),
                    elapsed=elapsed, throughput=(step-initial_step)/max(elapsed, 1e-9),
                    last_update_unix=time.time(), device=str(encoder.device)))
            if fixed_steps is None and step >= min_steps and stale >= patience:
                break
        legal_selection = fixed_steps is not None or best_step is not None
        if fixed_steps is not None or best_step is None:
            best_step = step
            best_loss = None if fixed_steps is not None else _validation(encoder, val)
            _save(out / "best.pt", checkpoint())
        _save(out / "last.pt", checkpoint())
    except BaseException as exc:
        _save(out / "interrupted.pt", checkpoint())
        _atomic_json(out / "progress.json", dict(status="failed", step=step, error=repr(exc), last_update_unix=time.time()))
        raise
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    encoder.restore(saved["model"])
    ztrain, zval = encoder.encode(x), encoder.encode(xv)
    np.save(out / "native_train.npy", ztrain)
    np.save(out / "native_validation.npy", zval)
    summary = dict(status="completed", best_step=best_step, actual_steps=step,
                   best_validation_loss=best_loss, legal_selection=legal_selection,
                   fixed_steps=fixed_steps, n_train=len(x), n_validation=len(xv),
                   native_dim=cfg["native_dim"], elapsed_seconds=time.perf_counter()-started,
                   current_validation_loss=last_validation_loss,
                   updates_per_second=(step-initial_step)/max(time.perf_counter()-started, 1e-9),
                   checkpoint_path=str(out / "best.pt"), last_checkpoint_path=str(out / "last.pt"),
                   last_update_unix=time.time(), method=cfg["method"], seed=seed)
    _atomic_json(out / "summary.json", summary)
    _atomic_json(out / "progress.json", summary)
    return {**summary, "encoder": encoder, "train_features": ztrain,
            "validation_features": zval, "history": history}
