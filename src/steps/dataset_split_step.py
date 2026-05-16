import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.framework.step import Step


class DatasetSplitStep(Step):
    """
    Build train/test datasets with event-level masking on branch and mains series.

    Inputs:
    - Branch raw series: (len, 2), [timestamp, power]
    - Mains raw series: (len, 2), [timestamp, power]
    - few-shot / non-few-shot tensor npy (3D), used for base validation
    - few-shot / non-few-shot activity json with start/end timestamps
    """

    def __init__(
        self,
        name: str = "DatasetSplit",
        raw_series_path: Optional[str] = None,
        mains_series_path: Optional[str] = None,
        few_shot_tensor_path: Optional[str] = None,
        non_few_shot_tensor_path: Optional[str] = None,
        few_shot_activity_json_path: Optional[str] = None,
        non_few_shot_activity_json_path: Optional[str] = None,
        few_train_ratio: float = 0.5,
        non_few_train_ratio: float = 0.8,
        random_seed: int = 42,
        timestamp_tolerance_seconds: float = 0.0,
        clip_negative_mains_to_zero: bool = True,
        appliance_name: str = "",
    ):
        super().__init__(name)
        self.raw_series_path = raw_series_path
        self.mains_series_path = mains_series_path
        self.few_shot_tensor_path = few_shot_tensor_path
        self.non_few_shot_tensor_path = non_few_shot_tensor_path
        self.few_shot_activity_json_path = few_shot_activity_json_path
        self.non_few_shot_activity_json_path = non_few_shot_activity_json_path
        self.few_train_ratio = float(few_train_ratio)
        self.non_few_train_ratio = float(non_few_train_ratio)
        self.random_seed = int(random_seed)
        self.timestamp_tolerance_seconds = max(0.0, float(timestamp_tolerance_seconds))
        self.clip_negative_mains_to_zero = bool(clip_negative_mains_to_zero)
        self.appliance_name = appliance_name

    def _resolve_path(self, explicit_path: Optional[str], context_key: str, context: dict, error_hint: str) -> str:
        if explicit_path:
            return str(explicit_path)
        if context.get(context_key):
            return str(context[context_key])
        raise ValueError(error_hint)

    def _resolve_inputs(self, context: dict) -> Dict[str, str]:
        return {
            'raw_series_path': self._resolve_path(
                self.raw_series_path,
                'raw_series_path',
                context,
                "[DatasetSplit] raw_series_path is missing. Set steps.dataset_split.raw_series_path or context['raw_series_path'].",
            ),
            'mains_series_path': self._resolve_path(
                self.mains_series_path,
                'mains_series_path',
                context,
                "[DatasetSplit] mains_series_path is missing. Set steps.dataset_split.mains_series_path or context['mains_series_path'].",
            ),
            'few_shot_tensor_path': self._resolve_path(
                self.few_shot_tensor_path,
                'few_shot_activity_tensor_npy',
                context,
                "[DatasetSplit] few_shot_tensor_path is missing. Set steps.dataset_split.few_shot_tensor_path or context['few_shot_activity_tensor_npy'].",
            ),
            'non_few_shot_tensor_path': self._resolve_path(
                self.non_few_shot_tensor_path,
                'non_few_shot_activity_tensor_npy',
                context,
                "[DatasetSplit] non_few_shot_tensor_path is missing. Set steps.dataset_split.non_few_shot_tensor_path or context['non_few_shot_activity_tensor_npy'].",
            ),
            'few_shot_activity_json_path': self._resolve_path(
                self.few_shot_activity_json_path,
                'few_shot_activity_sequences_json',
                context,
                "[DatasetSplit] few_shot_activity_json_path is missing. Set steps.dataset_split.few_shot_activity_json_path or context['few_shot_activity_sequences_json'].",
            ),
            'non_few_shot_activity_json_path': self._resolve_path(
                self.non_few_shot_activity_json_path,
                'non_few_shot_activity_sequences_json',
                context,
                "[DatasetSplit] non_few_shot_activity_json_path is missing. Set steps.dataset_split.non_few_shot_activity_json_path or context['non_few_shot_activity_sequences_json'].",
            ),
        }

    def _load_series_2col(self, file_path: str, series_name: str) -> np.ndarray:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"[DatasetSplit] {series_name} file not found: {file_path}")

        ext = path.suffix.lower()
        if ext == '.npy':
            data = np.load(path)
        elif ext == '.dat':
            data = np.loadtxt(path)
        else:
            raise ValueError(f"[DatasetSplit] Unsupported {series_name} format: {ext}. Supported: .dat, .npy")

        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(
                f"[DatasetSplit] {series_name} must be 2D with at least 2 columns [timestamp, power], got {data.shape}."
            )
        return data[:, :2].astype(np.float64, copy=False)

    def _load_3d_tensor(self, file_path: str, tensor_name: str) -> np.ndarray:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"[DatasetSplit] {tensor_name} file not found: {file_path}")
        arr = np.asarray(np.load(path))
        if arr.ndim != 3:
            raise ValueError(
                f"[DatasetSplit] {tensor_name} must be 3D with shape (n_samples, seq_len, feature_dim), got {arr.shape}."
            )
        return arr

    def _load_activity_records(self, json_path: str, tag: str) -> List[Dict[str, Any]]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"[DatasetSplit] {tag} activity json not found: {json_path}")

        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError(f"[DatasetSplit] {tag} activity json must be a list, got: {type(payload).__name__}")

        valid: List[Dict[str, Any]] = []
        for rec in payload:
            if not isinstance(rec, dict):
                continue
            start = rec.get('start_timestamp')
            end = rec.get('end_timestamp')
            try:
                start_f = float(start)
                end_f = float(end)
            except Exception:
                print(f"[DatasetSplit][WARN] Skip {tag} record with invalid timestamps: {rec}")
                continue
            if end_f < start_f:
                start_f, end_f = end_f, start_f
            new_rec = dict(rec)
            new_rec['start_timestamp'] = start_f
            new_rec['end_timestamp'] = end_f
            valid.append(new_rec)

        return valid

    def _ratio_to_count(self, total: int, ratio: float) -> int:
        ratio = min(max(float(ratio), 0.0), 1.0)
        return int(round(total * ratio))

    def _split_records(self, records: List[Dict[str, Any]], train_ratio: float, rng: np.random.Generator) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not records:
            return [], []
        total = len(records)
        train_count = self._ratio_to_count(total, train_ratio)
        idx = np.arange(total)
        rng.shuffle(idx)
        train_idx = set(int(i) for i in idx[:train_count])
        train = [records[i] for i in range(total) if i in train_idx]
        test = [records[i] for i in range(total) if i not in train_idx]
        return train, test

    def _build_drop_mask(self, timestamps: np.ndarray, drop_records: List[Dict[str, Any]]) -> np.ndarray:
        mask = np.zeros(timestamps.shape[0], dtype=bool)
        tol = float(self.timestamp_tolerance_seconds)
        for rec in drop_records:
            start = float(rec['start_timestamp']) - tol
            end = float(rec['end_timestamp']) + tol
            if end < start:
                start, end = end, start
            hit = (timestamps >= start) & (timestamps <= end)
            mask = mask | hit
        return mask

    def _apply_knockout(
        self,
        raw_branch: np.ndarray,
        raw_mains: np.ndarray,
        drop_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        branch = raw_branch.copy()
        mains = raw_mains.copy()

        delta = raw_branch[:, 1] * drop_mask.astype(np.float64)
        branch[drop_mask, 1] = 0.0
        mains[:, 1] = mains[:, 1] - delta

        negative_before_clip = int(np.sum(mains[:, 1] < 0))
        negative_total = float(np.sum(np.abs(np.minimum(mains[:, 1], 0.0))))
        if self.clip_negative_mains_to_zero:
            mains[:, 1] = np.maximum(mains[:, 1], 0.0)

        quality = {
            'drop_points': int(np.sum(drop_mask)),
            'drop_ratio': float(np.mean(drop_mask)) if drop_mask.size > 0 else 0.0,
            'mains_negative_points_before_clip': negative_before_clip,
            'mains_negative_total_before_clip': negative_total,
            'clip_negative_mains_to_zero': bool(self.clip_negative_mains_to_zero),
        }
        return branch, mains, quality

    def _events_total_duration(self, records: List[Dict[str, Any]]) -> float:
        if not records:
            return 0.0
        return float(sum(max(0.0, float(rec['end_timestamp']) - float(rec['start_timestamp'])) for rec in records))

    def _mask_duration(self, timestamps: np.ndarray, mask: np.ndarray) -> float:
        if timestamps.size <= 1 or mask.size == 0:
            return 0.0
        ts = timestamps.astype(np.float64)
        dt = np.diff(ts)
        if dt.size == 0:
            return 0.0
        dt = np.append(dt, dt[-1])
        return float(np.sum(dt[mask]))

    def _composition_summary(
        self,
        name: str,
        keep_few: List[Dict[str, Any]],
        keep_non: List[Dict[str, Any]],
        timestamps: np.ndarray,
        drop_mask: np.ndarray,
        quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        few_count = len(keep_few)
        non_count = len(keep_non)
        total_count = few_count + non_count

        few_duration = self._events_total_duration(keep_few)
        non_duration = self._events_total_duration(keep_non)
        total_duration = few_duration + non_duration

        return {
            'dataset_name': name,
            'event_count': {
                'few_shot': few_count,
                'non_few_shot': non_count,
                'total': total_count,
            },
            'event_duration_seconds': {
                'few_shot': few_duration,
                'non_few_shot': non_duration,
                'total': total_duration,
            },
            'event_ratio': {
                'few_shot': float(few_count / total_count) if total_count > 0 else 0.0,
                'non_few_shot': float(non_count / total_count) if total_count > 0 else 0.0,
            },
            'duration_ratio': {
                'few_shot': float(few_duration / total_duration) if total_duration > 0 else 0.0,
                'non_few_shot': float(non_duration / total_duration) if total_duration > 0 else 0.0,
            },
            'mask_duration_seconds': {
                'dropped': self._mask_duration(timestamps, drop_mask),
                'kept': self._mask_duration(timestamps, ~drop_mask),
            },
            'quality': quality,
        }

    def _save_dataset_outputs(
        self,
        log_dir: str,
        prefix: str,
        branch: np.ndarray,
        mains: np.ndarray,
        keep_few: List[Dict[str, Any]],
        keep_non: List[Dict[str, Any]],
        drop_mask: np.ndarray,
        summary: Dict[str, Any],
    ) -> Dict[str, str]:
        branch_path = os.path.join(log_dir, f'{prefix}_branch.npy')
        mains_path = os.path.join(log_dir, f'{prefix}_mains.npy')
        keep_few_path = os.path.join(log_dir, f'{prefix}_keep_few_shot_events.json')
        keep_non_path = os.path.join(log_dir, f'{prefix}_keep_non_few_shot_events.json')
        drop_mask_path = os.path.join(log_dir, f'{prefix}_drop_mask.npy')
        summary_path = os.path.join(log_dir, f'{prefix}_composition_summary.json')

        np.save(branch_path, branch)
        np.save(mains_path, mains)
        np.save(drop_mask_path, drop_mask.astype(np.uint8))
        with open(keep_few_path, 'w', encoding='utf-8') as f:
            json.dump(keep_few, f, ensure_ascii=False, indent=2)
        with open(keep_non_path, 'w', encoding='utf-8') as f:
            json.dump(keep_non, f, ensure_ascii=False, indent=2)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return {
            'branch_npy': branch_path,
            'mains_npy': mains_path,
            'drop_mask_npy': drop_mask_path,
            'keep_few_json': keep_few_path,
            'keep_non_json': keep_non_path,
            'composition_summary_json': summary_path,
        }

    def run(self, context: dict) -> dict:
        log_dir = self.get_log_dir(context)
        os.makedirs(log_dir, exist_ok=True)

        if not (0.0 <= self.few_train_ratio <= 1.0):
            raise ValueError(f"[DatasetSplit] few_train_ratio must be in [0,1], got {self.few_train_ratio}")
        if not (0.0 <= self.non_few_train_ratio <= 1.0):
            raise ValueError(f"[DatasetSplit] non_few_train_ratio must be in [0,1], got {self.non_few_train_ratio}")

        paths = self._resolve_inputs(context)

        raw_branch = self._load_series_2col(paths['raw_series_path'], 'raw branch series')
        raw_mains = self._load_series_2col(paths['mains_series_path'], 'raw mains series')
        if raw_branch.shape[0] != raw_mains.shape[0]:
            raise ValueError(
                f"[DatasetSplit] raw branch/mains length mismatch: {raw_branch.shape[0]} vs {raw_mains.shape[0]}"
            )
        if not np.allclose(raw_branch[:, 0], raw_mains[:, 0], atol=max(1e-9, self.timestamp_tolerance_seconds)):
            raise ValueError("[DatasetSplit] raw branch and mains timestamps are not aligned.")

        few_tensor = self._load_3d_tensor(paths['few_shot_tensor_path'], 'few-shot tensor')
        non_few_tensor = self._load_3d_tensor(paths['non_few_shot_tensor_path'], 'non-few-shot tensor')
        if int(few_tensor.shape[2]) != int(non_few_tensor.shape[2]):
            raise ValueError(
                "[DatasetSplit] few-shot and non-few-shot tensors have different feature_dim: "
                f"{few_tensor.shape[2]} vs {non_few_tensor.shape[2]}"
            )

        few_records = self._load_activity_records(paths['few_shot_activity_json_path'], 'few-shot')
        non_few_records = self._load_activity_records(paths['non_few_shot_activity_json_path'], 'non-few-shot')

        rng = np.random.default_rng(self.random_seed)
        few_train, few_test = self._split_records(few_records, self.few_train_ratio, rng)
        non_train, non_test = self._split_records(non_few_records, self.non_few_train_ratio, rng)

        datasets = {
            'train': {
                'keep_few': few_train,
                'keep_non': non_train,
                'drop': few_test + non_test,
            },
            'test_a': {
                'keep_few': few_test,
                'keep_non': non_test,
                'drop': few_train + non_train,
            },
            'test_b': {
                'keep_few': few_test,
                'keep_non': [],
                'drop': few_train + non_train + non_test,
            },
        }

        output_paths: Dict[str, Dict[str, str]] = {}
        composition: Dict[str, Dict[str, Any]] = {}
        timestamps = raw_branch[:, 0]

        for name, cfg in datasets.items():
            drop_mask = self._build_drop_mask(timestamps, cfg['drop'])
            branch_ds, mains_ds, quality = self._apply_knockout(raw_branch, raw_mains, drop_mask)
            summary = self._composition_summary(name, cfg['keep_few'], cfg['keep_non'], timestamps, drop_mask, quality)
            paths_ds = self._save_dataset_outputs(
                log_dir,
                name,
                branch_ds,
                mains_ds,
                cfg['keep_few'],
                cfg['keep_non'],
                drop_mask,
                summary,
            )
            output_paths[name] = paths_ds
            composition[name] = summary

        global_summary = {
            'input_paths': paths,
            'raw_branch_shape': list(raw_branch.shape),
            'raw_mains_shape': list(raw_mains.shape),
            'few_shot_tensor_shape': list(few_tensor.shape),
            'non_few_shot_tensor_shape': list(non_few_tensor.shape),
            'hyper_parameters': {
                'few_train_ratio': self.few_train_ratio,
                'non_few_train_ratio': self.non_few_train_ratio,
                'random_seed': self.random_seed,
                'timestamp_tolerance_seconds': self.timestamp_tolerance_seconds,
                'clip_negative_mains_to_zero': self.clip_negative_mains_to_zero,
            },
            'split_counts': {
                'few_total': len(few_records),
                'few_train': len(few_train),
                'few_test': len(few_test),
                'non_few_total': len(non_few_records),
                'non_few_train': len(non_train),
                'non_few_test': len(non_test),
            },
            'datasets': composition,
            'output_paths': output_paths,
        }
        global_summary_path = os.path.join(log_dir, 'dataset_split_summary.json')
        with open(global_summary_path, 'w', encoding='utf-8') as f:
            json.dump(global_summary, f, ensure_ascii=False, indent=2)

        context['dataset_split_summary'] = global_summary
        context['dataset_split_summary_json'] = global_summary_path
        context['dataset_split_outputs'] = output_paths

        print(
            f"[DatasetSplit] train events(few/non)={len(few_train)}/{len(non_train)}, "
            f"test_a events(few/non)={len(few_test)}/{len(non_test)}, "
            f"test_b events(few/non)={len(few_test)}/0"
        )
        return context
