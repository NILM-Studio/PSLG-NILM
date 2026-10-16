"""Explicit household splits, real paired signals, gap-safe windowing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .common import digest, read_json, signature


@dataclass
class Record:
    id: str
    dataset: str
    house: str
    split: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    fingerprint: str
    content_hash: str = ""
    context_valid: np.ndarray | None = None
    x_observed: np.ndarray | None = None


def ranges(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def validate_manifest(m):
    if m.get("protocol") not in {"cross_house", "cross_dataset"}:
        raise ValueError("Only strict cross_house / cross_dataset are implemented; time splits require separate review.")
    entries = m["records"]
    ids, houses, files = set(), {}, {}
    for e in entries:
        if e["split"] not in {"train", "val", "test"}:
            raise ValueError("Invalid split")
        if e["id"] in ids:
            raise ValueError("Duplicate record id")
        ids.add(e["id"])
        key = (e["dataset"], str(e["house"]))
        if key in houses and houses[key] != e["split"]:
            raise ValueError("Household leakage across splits")
        houses[key] = e["split"]
        # File identity is checked again using contents when loaded.
        key = (e.get("mains_path", e.get("path")), e.get("target_path", e.get("path")))
        if key in files and files[key] != e["split"]:
            raise ValueError("The same recording appears in multiple splits")
        files[key] = e["split"]
    if set(e["split"] for e in entries) != {"train", "val", "test"}:
        raise ValueError("Explicit train, val, test records required")
    if m["protocol"] == "cross_dataset":
        source = {e["dataset"] for e in entries if e["split"] != "test"}
        target = {e["dataset"] for e in entries if e["split"] == "test"}
        if source & target or len(source) != 1 or len(target) != 1:
            raise ValueError("Cross-dataset requires disjoint source and target datasets; validation stays source-only")


def _resample(frame, column, dt, coverage, native):
    s = pd.Series(pd.to_numeric(frame[column], errors="coerce").to_numpy(),
                  index=pd.to_datetime(frame["timestamp"], unit="s", utc=True)).sort_index()
    if s.index.has_duplicates:
        raise ValueError("Duplicate raw timestamps; resolve rather than silently averaging")
    s = s.where(np.isfinite(s) & (s >= 0))
    if native > dt or native <= 0:
        raise ValueError("Cannot upsample or use a nonpositive native interval")
    r = s.resample(f"{dt}s", origin="epoch", label="left", closed="left")
    # Sample-mean power on nominally regular data, no forward fill or interpolation.
    return r.mean().where(r.count() >= max(1, int(np.ceil(coverage * dt / native))))


def load_records(manifest_path, splits=("train", "val")):
    """CSV: timestamp(seconds UTC), mains, target; REFIT or UKDALE adapters explicit."""
    manifest_path = Path(manifest_path).resolve()
    m = read_json(manifest_path)
    validate_manifest(m)
    dt = int(m["sample_seconds"])
    if dt <= 0:
        raise ValueError("sample_seconds must be positive")
    out, seen_hash = [], {}
    for e in m["records"]:
        if e["split"] not in splits:
            continue
        def path(key):
            return (manifest_path.parent / e[key]).resolve()
        fmt = e.get("format", "paired_csv")
        if fmt == "ukdale_dat":
            xp, yp = path("mains_path"), path("target_path")
            fx = pd.read_csv(xp, sep=r"\s+", names=["timestamp", "mains"])
            fy = pd.read_csv(yp, sep=r"\s+", names=["timestamp", "target"])
            hashes = [digest(xp), digest(yp)]
        elif fmt in {"paired_csv", "refit_csv"}:
            p = path("path")
            columns = e.get("columns", {"timestamp": "timestamp", "mains": "mains", "target": "target"})
            f = pd.read_csv(p, usecols=list(columns.values())).rename(columns={v: k for k, v in columns.items()})
            fx, fy, hashes = f, f, [digest(p)]
        else:
            raise ValueError(f"Unknown data format: {fmt}")
        content = signature(hashes)
        if e.get("sha256") and hashes != [e["sha256"]]:
            raise ValueError("Source file hash mismatch")
        if content in seen_hash and seen_hash[content] != e["split"]:
            raise ValueError("Identical file contents across splits")
        seen_hash[content] = e["split"]
        # Explicit, predeclared pilot date range; identical for all supervision arms.
        for bound, op in (("start_timestamp", "ge"), ("end_timestamp", "lt")):
            if bound in e:
                cutoff = float(e[bound])
                fx = fx.loc[getattr(pd.to_numeric(fx["timestamp"]), op)(cutoff)]
                fy = fy.loc[getattr(pd.to_numeric(fy["timestamp"]), op)(cutoff)]
        native = float(e["native_seconds"])
        cov = float(m.get("min_coverage", 0.8))
        if not 0 < cov <= 1:
            raise ValueError("min_coverage must be in (0,1]")
        xs = _resample(fx, "mains", dt, cov, float(e.get("mains_native_seconds", native)))
        ys = _resample(fy, "target", dt, cov, float(e.get("target_native_seconds", native)))
        joined = pd.concat([xs.rename("x"), ys.rename("y")], axis=1).sort_index()
        # Each record remains a single grid; NaNs preserve gaps and windows never cross them.
        joined = joined.asfreq(f"{dt}s")
        t = joined.index.as_unit("s").asi8
        if len(t) == 0:
            raise ValueError(f"Empty record {e['id']}")
        minimum = int(m.get("min_train_block_samples", 0))
        policy = m.get("missing_policy", "strict")
        if policy not in {"strict", "aggregate_isolated_context_observed_targets_v2"}:
            raise ValueError("Unknown missing policy")
        context, x_observed = None, None
        if policy != "strict":
            if dt != 6 or native != 6:
                raise ValueError("This missing policy requires the UK-DALE native 6 s grid")
            from .missing import context_arrays
            xx, context, x_observed = context_arrays(joined.x.to_numpy(), np.zeros(len(joined)))
            joined["x"] = xx
        if minimum < 0:
            raise ValueError("Negative minimum training block length")
        if e["split"] == "train" and minimum:
            # Same training support for discovery, normalization, bins and NILM.
            # Preserve timestamps/gaps; never concatenate separate activities.
            keep = training_support(joined.x.to_numpy(), joined.y.to_numpy(), minimum) if context is None else training_support(context.astype(float), np.where(context, 0., np.nan), minimum)
            joined.loc[~keep, ["x", "y"]] = np.nan
            if context is not None:
                context &= keep
                x_observed &= keep
        fp = signature({"source_hashes": hashes, "entry": e, "sampling": [dt, cov],
                        "min_train_block_samples": minimum, "missing_policy": policy})
        out.append(Record(e["id"], e["dataset"], str(e["house"]), e["split"], t,
                          joined.x.to_numpy(np.float32), joined.y.to_numpy(np.float32), fp, content, context, x_observed))
    return out, m


def training_support(x, y, minimum):
    """Points belonging to full valid training windows (stride <= length)."""
    if minimum < 1:
        raise ValueError("Training window length must be positive")
    keep = np.zeros(len(x), dtype=bool)
    for start, end in ranges(np.isfinite(x) & np.isfinite(y)):
        if end - start >= minimum:
            keep[start:end] = True
    return keep


def train_signature(records):
    return signature(sorted((r.id, r.fingerprint) for r in records if r.split == "train"))


def time_features(t):
    ix = pd.to_datetime(t, unit="s", utc=True)
    values = [ix.minute.to_numpy(), ix.hour.to_numpy(), ix.dayofweek.to_numpy(), ix.month.to_numpy() - 1]
    # Match the published TimeRPE endpoint convention. Change only in a separate ablation.
    return np.stack([fun(2 * np.pi * v / p) for v, p in zip(values, [59, 23, 6, 11])
                     for fun in (np.sin, np.cos)]).astype(np.float32)


class Windows:
    def __init__(self, records, length, stride, scale, targets=None, arm="R", threshold=10):
        if length < 2 or not 0 < stride <= length or scale <= 0:
            raise ValueError("Invalid window/stride/scale")
        self.records, self.length, self.scale = records, length, scale
        self.targets, self.arm, self.threshold = targets, arm, threshold
        self.index = []
        self.features = [time_features(r.t) for r in records]
        for i, r in enumerate(records):
            support = np.isfinite(r.x) if r.context_valid is None else r.context_valid & np.isfinite(r.x)
            for start, end in ranges(support):
                if end - start < length:
                    continue
                starts = list(range(start, end - length + 1, stride))
                if starts[-1] != end - length:
                    starts.append(end - length)
                self.index.extend((i, s) for s in starts)
        if not self.index:
            raise ValueError("No complete gap-free windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ri, start = self.index[i]
        r, sl = self.records[ri], slice(start, start + self.length)
        x = np.concatenate([(r.x[sl] / self.scale)[None], self.features[ri][:, sl]], axis=0)
        y = r.y[sl] / self.scale
        observed = np.isfinite(y)
        z = np.full(self.length, -100, dtype=np.int64)
        if self.targets is not None:
            bundle = self.targets[r.id]
            key = {"R": "primitive", "O": "onoff", "O_common": "onoff", "B": "bins", "B_full": "bins_full",
                   "P": "primitive", "BP": "segment_bins", "S": "shape_bins"}[self.arm]
            z = bundle[key][sl].copy()
            z[~bundle['activity_valid' if self.arm in {'O', 'B_full'} else 'valid'][sl]] = -100
        elif self.arm != "R":
            raise ValueError("Auxiliary training requires exported activity/state labels")
        z[~observed] = -100
        activity = np.full(self.length, -100, dtype=np.int64)
        if self.targets is not None:
            activity = bundle["onoff"][sl].copy()
            activity[~bundle["activity_valid"][sl] | ~observed] = -100
        return {"activity": activity, "x": x.astype(np.float32), "y": np.where(observed, y, 0).astype(np.float32), "observed": observed, "z": z, "index": i}
