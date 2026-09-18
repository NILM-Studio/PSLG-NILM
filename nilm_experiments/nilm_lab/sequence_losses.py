"""Observation-only regression, shared state support and independent activity mask."""
import math
import torch
from torch.nn import functional as F


def stratified_mean(values, labels, mode):
    if mode == 'mean':
        return values.mean()
    if mode != 'onoff_balanced':
        raise ValueError('Unknown auxiliary reduction')
    parts = [values[labels == level].mean() for level in (0, 1) if (labels == level).any()]
    return torch.stack(parts).mean()


def sequence_objective(power, states, activity, batch, lambda_p, lambda_a, regression='mse', reduction='mean'):
    observed = batch['observed'].bool()
    y, z, a = batch['y'], batch['z'], batch['activity']
    if not torch.isfinite(power).all() or not torch.isfinite(y[observed]).all():
        raise ValueError('Nonfinite regression tensors')
    zero = power.sum() * 0
    if regression not in {'mse', 'huber'}:
        raise ValueError('Unknown regression loss')
    reg = (F.mse_loss(power[observed], y[observed]) if regression == 'mse' else
           F.huber_loss(power[observed], y[observed], delta=.1)) if observed.any() else zero
    primitive, active = zero, zero
    if lambda_p:
        if states is None:
            raise ValueError('Primitive loss requires state logits')
        mask = observed & (z != -100)
        if mask.any():
            if (a[mask] < 0).any():
                raise ValueError('State support requires valid activity reference')
            values = F.cross_entropy(states.transpose(1, 2)[mask], z[mask], reduction='none')
            primitive = stratified_mean(values, a[mask], reduction) / math.log(states.shape[1])
    if lambda_a:
        if activity is None:
            raise ValueError('Activity loss requires activity logits')
        mask = observed & (a != -100)
        if mask.any():
            values = F.binary_cross_entropy_with_logits(activity[mask], a[mask].float(), reduction='none')
            active = stratified_mean(values, a[mask], reduction) / math.log(2)
    return reg + lambda_p * primitive + lambda_a * active, reg, primitive, active
