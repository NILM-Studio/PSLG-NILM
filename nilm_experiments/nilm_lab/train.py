"""Canonical source-only NILM training; sequence labels never enter inference."""
from __future__ import annotations
from pathlib import Path
import random
import os
import time
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .common import write_json, digest, code_snapshot
from .data import Windows
from .models import make_model
from .sequence_losses import sequence_objective
from .activity_metrics import score_record


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def model_for_trial(config, meta, trial):
    arm = trial['arm']
    classes = 0 if arm == 'R' else 2 if arm in {'O', 'O_common'} else meta['k'] + 1
    return make_model(trial['model'], config['window_length'], classes,
        config.get('model_kwargs', {}).get(trial['model']), trial.get('activity_head', 'none'),
        config.get('sequence_head', {}).get('hidden_size', 32))


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    model.eval()
    sums = [np.zeros(len(r.t), np.float64) for r in dataset.records]
    activity_sums = [np.zeros(len(r.t), np.float64) for r in dataset.records]
    state_sums = None
    counts = [np.zeros(len(r.t), np.int64) for r in dataset.records]
    has_activity = False
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        power, logits, activity = model(batch['x'].to(device))
        if not torch.isfinite(power).all():
            raise ValueError('Nonfinite predictions')
        probs = logits.softmax(1).cpu().numpy() if logits is not None else None
        if probs is not None and state_sums is None:
            state_sums = [np.zeros((probs.shape[1], len(r.t)), np.float64) for r in dataset.records]
        if activity is None and logits is not None:
            activity = torch.logsumexp(logits[:, 1:], dim=1) - logits[:, 0]
        q = activity.sigmoid().cpu().numpy() if activity is not None else None
        has_activity |= q is not None
        for j, (idx, p) in enumerate(zip(batch['index'].tolist(), power.cpu().numpy())):
            ri, start = dataset.index[idx]
            sl = slice(start, start + dataset.length)
            sums[ri][sl] += p * dataset.scale
            counts[ri][sl] += 1
            if q is not None:
                activity_sums[ri][sl] += q[j]
            if probs is not None:
                state_sums[ri][:, sl] += probs[j]
    divide = lambda s, n: np.divide(s, n, out=np.full_like(s, np.nan), where=n > 0)
    return [dict(power=divide(s, n), count=n,
                 activity=divide(activity_sums[i], n) if has_activity else None,
                 state_probabilities=divide(state_sums[i], n[None]) if state_sums is not None else None)
            for i, (s, n) in enumerate(zip(sums, counts))]


def gradient_diagnostic(model, dataset, device, trial, tc):
    """Fixed deterministic windows; autograd.grad does not update parameters."""
    from torch.utils.data._utils.collate import default_collate
    on, off = [], []
    for i, (ri, start) in enumerate(dataset.index):
        active = dataset.targets[dataset.records[ri].id]['onoff'][start:start+dataset.length]
        pool = on if (active == 1).any() else off
        if len(pool) < 16:
            pool.append(i)
        if len(on) == 16 and len(off) == 16:
            break
    ids = sorted(on + off)
    batch = {k: v.to(device) for k, v in default_collate([dataset[i] for i in ids]).items() if k != 'index'}
    model.eval()
    # cuDNN GRU backward needs a training-mode reserve buffer. These heads
    # have one layer and no dropout; keep the backbone BN/dropout in eval.
    for module in model.modules():
        if isinstance(module, nn.GRU):
            module.train()
    outputs = model(batch['x'])
    parts = sequence_objective(*outputs, batch, trial['lambda_p'], trial['lambda_a'],
                               tc.get('loss', 'mse'), tc.get('auxiliary_reduction', 'mean'))
    params = [p for name, p in model.named_parameters() if not name.startswith(('state_head', 'activity_head'))]
    gradients = [torch.autograd.grad(v, params, retain_graph=True, allow_unused=True) for v in parts[1:]]
    norms = [float(sum((g.detach().double().square().sum() for g in gs if g is not None),
                       torch.zeros((), device=device)).sqrt()) for gs in gradients]
    cosines = []
    for gs, norm in zip(gradients[1:], norms[1:]):
        dot = sum((a.detach().double().mul(b.detach().double()).sum()
                   for a, b in zip(gradients[0], gs) if a is not None and b is not None), torch.zeros((), device=device))
        cosines.append(float(dot/(norms[0]*norm)) if norms[0]*norm else None)
    return dict(window_indices=ids, gradient_norms=norms, regression_auxiliary_cosines=cosines,
                weighted_auxiliary_norms=[norms[1]*trial['lambda_p'], norms[2]*trial['lambda_a']],
                activity_counts=[int((batch['activity'] == j).sum()) for j in (0, 1)])


def score_records(records, predictions, references, threshold, dt):
    return [score_record(r, p['power'], p['activity'], references[r.id], threshold, dt)
            for r, p in zip(records, predictions)]


def house_macro_mae(scores):
    groups = {}
    for s in scores:
        n, err = groups.setdefault((s['dataset'], s['house']), [0, 0.])
        groups[(s['dataset'], s['house'])] = [n+s['n'], err+s['n']*s['mae']]
    return float(np.mean([err/n for n, err in groups.values()]))


def fit_run(config, train_records, val_records, targets, state_meta, trial, run_dir, references):
    if any(r.split != 'train' for r in train_records) or any(r.split != 'val' for r in val_records):
        raise ValueError('Training API split violation')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    tc, seed = config['training'], trial['seed']
    device = torch.device(tc.get('device', 'cpu'))
    seed_all(seed)
    model = model_for_trial(config, state_meta, trial).to(device)
    base_hash = hashlib.sha256()
    for key, value in model.state_dict().items():
        if not key.startswith(('state_head', 'activity_head')):
            base_hash.update(key.encode())
            base_hash.update(value.detach().cpu().numpy().tobytes())
    seed_all(seed)
    tr = Windows(train_records, config['window_length'], config['train_stride'], state_meta['scale'], targets, trial['arm'])
    va = Windows(val_records, config['window_length'], config['eval_stride'], state_meta['scale'])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(tr, batch_size=tc['batch_size'], shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc['lr'], weight_decay=tc.get('weight_decay', 0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=tc.get('lr_patience', 3), factor=.5)
    best, stale, history = float('inf'), 0, []
    started = time.perf_counter()
    write_json(run_dir / 'run.json', dict(trial=trial, config=config, state_metadata=state_meta,
        base_initial_sha256=base_hash.hexdigest(), parameter_count=sum(p.numel() for p in model.parameters()),
        torch_version=str(torch.__version__), code_sha256=code_snapshot(),
        source_record_content_hashes=[r.content_hash for r in train_records+val_records],
        training_record_ids=[r.id for r in train_records], validation_record_ids=[r.id for r in val_records],
        train_window_count=len(tr), validation_window_count=len(va)))
    if tc.get('gradient_diagnostics'):
        write_json(run_dir / 'gradient_diagnostic.json', gradient_diagnostic(model, tr, device, trial, tc))
    for epoch in range(int(tc['epochs'])):
        model.train()
        running, n, empty_batches = np.zeros(4), 0, 0
        exposure = np.zeros(state_meta['k']+1, dtype=np.int64)
        empty_on, empty_off = 0, 0
        for batch in loader:
            tensors = {k: v.to(device) for k, v in batch.items() if k != 'index'}
            if not tensors['observed'].any():
                empty_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            power, logits, activity = model(tensors['x'])
            parts = sequence_objective(power, logits, activity, tensors, trial['lambda_p'],
                                       trial['lambda_a'], tc.get('loss', 'mse'), tc.get('auxiliary_reduction', 'mean'))
            zz = batch['z'].numpy()
            exposure += np.bincount(zz[zz >= 0], minlength=len(exposure))
            empty_on += int(not (batch['activity'] == 1).any())
            empty_off += int(not (batch['activity'] == 0).any())
            if not torch.isfinite(parts[0]):
                raise ValueError('Nonfinite loss')
            parts[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.get('gradient_clip', 1.), error_if_nonfinite=True)
            optimizer.step()
            size = len(batch['y'])
            running += np.array([v.item() for v in parts])*size
            n += size
        if not n:
            raise ValueError('No observed regression targets in training windows')
        scores = score_records(val_records, predict(model, va, device, tc['batch_size']), references,
                               config['threshold'], config['sample_seconds'])
        val_mae = house_macro_mae(scores)
        scheduler.step(val_mae)
        history.append(dict(epoch=epoch+1, loss=running[0]/n, reg=running[1]/n,
                            primitive_loss=running[2]/n, activity_loss=running[3]/n,
                            val_house_macro_mae=val_mae, empty_batches=empty_batches,
                            validation_scores=scores, class_exposure=exposure.tolist(),
                            batches_without_on=empty_on, batches_without_off=empty_off))
        write_json(run_dir / 'history.json', history)
        if val_mae < best:
            best, stale = val_mae, 0
            torch.save(dict(model=model.state_dict(), epoch=epoch+1), run_dir / 'best.pt')
            write_json(run_dir / 'validation_scores.json', scores)
        else:
            stale += 1
        if stale >= int(tc['patience']):
            break
    write_json(run_dir / 'finished.json', dict(best_val_mae=best, seconds=time.perf_counter()-started,
        epochs=len(history), checkpoint_sha256=digest(run_dir / 'best.pt'), test_evaluated=False))
    return model
