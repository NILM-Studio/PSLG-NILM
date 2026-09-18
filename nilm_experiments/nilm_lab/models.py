"""Adapters keep upstream regression paths; auxiliary logits are never denormalized."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from .common import ROOT


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class AuxiliaryAdapter(nn.Module):
    def __init__(self, name, length, classes, kwargs=None, activity_mode='none', hidden_size=32):
        super().__init__()
        self.name, self.classes, self.features = name, classes, None
        kwargs = kwargs or {}
        if name == "NILMFormer":
            vendor = ROOT / "third_party/nilmformer"
            if "src" in sys.modules:
                source = getattr(sys.modules["src"], "__file__", "") or ""
                if not Path(source).resolve().is_relative_to(vendor.resolve()):
                    raise RuntimeError("Conflicting 'src' package. Run nilm_lab in a fresh isolated process.")
            sys.path.insert(0, str(vendor))
            from src.nilmformer.model import NILMFormer
            from src.nilmformer.congif import NILMFormerConfig
            if kwargs.get("use_efficient_attention", False):
                raise ValueError("Reviewed adapter uses standard attention, not optional xformers")
            config = NILMFormerConfig(**kwargs)
            self.base = NILMFormer(config)
            layer, dim = self.base.DownstreamTaskHead, config.d_model
        elif name == "FCN":
            module = module_from_file("_upstream_fcn", ROOT / "third_party/nilmformer/src/baselines/nilm/fcn.py")
            self.base = module.FCN(length, c_in=1, downstreamtask="seq2seq")
            layer, dim = self.base.fc, 50
            # Materialize LazyLinear before creating auxiliary parameters. No BN/dropout updates.
            self.base.eval()
            with torch.no_grad():
                self.base(torch.zeros(1, 1, length))
            self.base.train()
        else:
            raise ValueError(name)
        layer.register_forward_pre_hook(self._capture)
        self.state_head = nn.Conv1d(dim, classes, 1) if classes else None
        from .sequence_heads import SequenceActivityHead, direct_activity_head, PointActivityHead, FeatureActivityHead
        if activity_mode not in {'none', 'direct', 'direct_matched', 'gru', 'point', 'feature_gru'}:
            raise ValueError('Unknown activity head')
        if activity_mode in {'gru', 'point'} and not classes:
            raise ValueError('GRU requires predicted state candidates')
        self.activity_mode = activity_mode
        self.activity_head = (SequenceActivityHead(classes, hidden_size) if activity_mode == 'gru'
                              else direct_activity_head(dim, hidden_size) if activity_mode == 'direct' else None)
        if activity_mode == 'direct_matched':
            budget = 3 * hidden_size * (classes + hidden_size + 2) + hidden_size + 1
            self.activity_head = direct_activity_head(dim, max(1, round((budget-1)/(dim+2))))
        elif activity_mode == 'point':
            self.activity_head = PointActivityHead(classes, hidden_size)
        elif activity_mode == 'feature_gru':
            self.activity_head = FeatureActivityHead(dim, kwargs.get('activity_projection_dim', 6), hidden_size)

    def _capture(self, module, inputs):
        self.features = inputs[0]

    def forward(self, x):
        power = self.base(x if self.name == "NILMFormer" else x[:, :1]).squeeze(1)
        logits = self.state_head(self.features) if self.state_head is not None else None
        activity = (self.activity_head(logits) if self.activity_mode in {'gru', 'point'} else
                    self.activity_head(self.features).squeeze(1) if self.activity_mode in {'direct', 'direct_matched'} else
                    self.activity_head(self.features) if self.activity_mode == 'feature_gru' else None)
        self.features = None  # Do not retain autograd graphs beyond this batch.
        return power, logits, activity


class BERTAdapter(nn.Module):
    """Official architecture with explicitly standardized regression-only training."""
    def __init__(self, length):
        super().__init__()
        if length % 2:
            raise ValueError("Official BERT4NILM requires an even window length")
        m = module_from_file("_upstream_bert4nilm", ROOT / "third_party/bert4nilm/model.py")
        self.base = m.BERT4NILM(SimpleNamespace(window_size=length, drop_out=0.1, output_size=1))

    def forward(self, x):
        return self.base(x[:, 0]).squeeze(-1), None, None


class SGNAdapter(nn.Module):
    """Paper Appendix A architecture / Eq.9, explicit seq2seq protocol adaptation.

    Not an official author implementation and not a numerical paper reproduction.
    Independent six-convolution power and ON subnetworks, learned standby scalar.
    Output width follows this harness's common window, rather than paper's 32.
    """
    def __init__(self, length):
        super().__init__()
        def network():
            layers, channels = [], 1
            for c, kernel in zip([30, 30, 40, 50, 50, 50], [10, 8, 6, 5, 5, 5]):
                layers += [nn.Conv1d(channels, c, kernel, padding="same"), nn.ReLU()]
                channels = c
            return nn.Sequential(*layers, nn.Flatten(), nn.Linear(50 * length, 1024), nn.ReLU(), nn.Linear(1024, length))
        self.power, self.on = network(), network()
        self.standby = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        logits = self.on(x[:, :1])
        gate = logits.sigmoid()
        power = self.power(x[:, :1]) * gate + (1 - gate) * self.standby
        return power, torch.stack([torch.zeros_like(logits), logits], dim=1), logits


def make_model(name, length, classes=0, kwargs=None, activity_mode='none', hidden_size=32):
    if name not in {'NILMFormer', 'FCN'} and activity_mode != 'none':
        raise ValueError('Sequence ablations are implemented for NILMFormer and FCN')
    if name in {"NILMFormer", "FCN"}:
        return AuxiliaryAdapter(name, length, classes, kwargs, activity_mode, hidden_size)
    if name == "BERT4NILM":
        if classes:
            raise ValueError("BERT4NILM is an external regression baseline, not a four-arm backbone")
        return BERTAdapter(length)
    if name == "SGN":
        if classes != 2:
            raise ValueError("SGN is external ON-gated regression; requires O")
        return SGNAdapter(length)
    raise ValueError(f"Unknown model: {name}")
