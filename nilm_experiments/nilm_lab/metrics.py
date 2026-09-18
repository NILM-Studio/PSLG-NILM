from __future__ import annotations

import numpy as np
from .data import ranges


def metrics(y, prediction, threshold, dt):
    y, p = np.asarray(y, float), np.maximum(np.asarray(prediction, float), 0)
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        raise ValueError("No evaluated samples")
    a, b = y[valid], p[valid]
    on, detected = a > threshold, b > threshold
    tp, fp, fn = int(np.sum(on & detected)), int(np.sum(~on & detected)), int(np.sum(on & ~detected))
    energy = float(a.sum() * dt / 3600)
    predicted = float(b.sum() * dt / 3600)
    denom = float(np.maximum(a, b).sum())
    # Operate on original grid so missing gaps cannot join independent events.
    true_events = ranges(valid & (y > threshold))
    pred_events = ranges(valid & (p > threshold))
    candidates = []
    first = 0
    for i, (s, e) in enumerate(true_events):
        while first < len(pred_events) and pred_events[first][1] <= s:
            first += 1
        j = first
        while j < len(pred_events) and pred_events[j][0] < e:
            u, v = pred_events[j]
            intersection = max(0, min(e, v) - max(s, u))
            iou = intersection / (e - s + v - u - intersection)
            if iou >= 0.5:
                candidates.append((iou, i, j))
            j += 1
    used_t, used_p = set(), set()
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
    event_denom = len(true_events) + len(pred_events)
    return {"n": int(valid.sum()), "coverage": float(valid.mean()),
            "mae": float(np.abs(a - b).mean()), "rmse": float(np.sqrt(np.square(a - b).mean())),
            "on_mae": float(np.abs(a[on] - b[on]).mean()) if on.any() else None,
            "sae": abs(predicted - energy) / energy if energy > 0 else None,
            "mr": float(np.minimum(a, b).sum()) / denom if denom > 0 else None,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "event_f1_iou05": 2 * len(used_t) / event_denom if event_denom else None,
            "energy_wh": energy, "predicted_energy_wh": predicted,
            "off_false_energy_wh": float(b[~on].sum() * dt / 3600)}


def paired_house_ci(rows, baseline="B", proposed="P", draws=5000, seed=42):
    """Average paired seeds within household, bootstrap independent household differences."""
    groups = {}
    for r in rows:
        key = (r["dataset"], r["appliance"], r["model"], r["house"], r["seed"])
        if r["arm"] in groups.setdefault(key, {}):
            raise ValueError("Duplicate run/house in summary")
        groups[key][r["arm"]] = r["mae"]
    house = {}
    for key, values in groups.items():
        if baseline not in values or proposed not in values:
            raise ValueError("Unpaired seed; no silently dropped runs")
        house.setdefault(key[:-1], []).append(values[baseline] - values[proposed])
    strata = {}
    for (dataset, appliance, model, h), values in house.items():
        strata.setdefault((dataset, appliance, model), []).append(float(np.mean(values)))
    result, rng = [], np.random.default_rng(seed)
    for key, values in strata.items():
        v = np.asarray(values)
        ci = np.percentile(rng.choice(v, (draws, len(v)), replace=True).mean(1), [2.5, 97.5]) if len(v) >= 3 else None
        result.append({"dataset": key[0], "appliance": key[1], "model": key[2], "households": len(v),
                       "mae_improvement_w": float(v.mean()), "ci95": ci.tolist() if ci is not None else None,
                       "note": "CI omitted with fewer than 3 households; seeds are not independent households."})
    return result
