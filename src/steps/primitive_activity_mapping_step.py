import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.framework.step import Step


class PrimitiveActivityMappingStep(Step):
	"""
	Prepare inputs for primitive-to-activity mapping.

	Current scope:
	1) Traverse activity CSV folder and collect start/end timestamp from column 0.
	2) Traverse primitive npy folder and collect start/end timestamp from each primitive sequence.
	3) Match primitive ranges to activity ranges, then split activities into few-shot/non-few-shot groups.
	"""

	def __init__(
		self,
		name: str = "PrimitiveActivityMapping",
		activity_sequence_dir: Optional[str] = None,
		primitive_sequence_dir: Optional[str] = None,
		enable_tolerant_match: bool = False,
		timestamp_tolerance: float = 0.0,
		appliance_name: str = "",
	):
		super().__init__(name)
		self.activity_sequence_dir = activity_sequence_dir
		self.primitive_sequence_dir = primitive_sequence_dir
		self.enable_tolerant_match = bool(enable_tolerant_match)
		self.timestamp_tolerance = max(0.0, float(timestamp_tolerance))
		self.appliance_name = appliance_name

	def _resolve_activity_dir(self, context: Dict[str, Any]) -> str:
		if self.activity_sequence_dir:
			return self.activity_sequence_dir
		if context.get('activity_sequence_dir'):
			return str(context['activity_sequence_dir'])
		if context.get('input_root'):
			return str(context['input_root'])
		raise ValueError(
			"[PrimitiveActivityMapping] activity_sequence_dir is missing. "
			"Set it in config (steps.primitive_activity_mapping.activity_sequence_dir) "
			"or provide context['activity_sequence_dir']."
		)

	def _collect_activity_ranges(self, activity_dir: str) -> List[Dict[str, Any]]:
		activity_path = Path(activity_dir)
		if not activity_path.exists():
			raise FileNotFoundError(f"[PrimitiveActivityMapping] Activity directory not found: {activity_dir}")

		records: List[Dict[str, Any]] = []
		csv_files = sorted(activity_path.glob('*.csv'))

		for csv_file in csv_files:
			try:
				df = pd.read_csv(csv_file)
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Skip unreadable csv {csv_file.name}: {e}")
				continue

			if df.empty:
				print(f"[PrimitiveActivityMapping] Skip empty csv: {csv_file.name}")
				continue

			if df.shape[1] < 1:
				print(f"[PrimitiveActivityMapping] Skip csv with no columns: {csv_file.name}")
				continue

			start_ts = df.iloc[0, 0]
			end_ts = df.iloc[-1, 0]

			records.append(
				{
					'file_name': csv_file.name,
					'file_path': str(csv_file),
					'start_timestamp': start_ts.item() if hasattr(start_ts, 'item') else start_ts,
					'end_timestamp': end_ts.item() if hasattr(end_ts, 'item') else end_ts,
					'row_count': int(df.shape[0]),
					'col_count': int(df.shape[1]),
				}
			)

		return records

	def _resolve_primitive_dir(self, context: Dict[str, Any]) -> str:
		if self.primitive_sequence_dir:
			return self.primitive_sequence_dir
		if context.get('primitive_sequence_dir'):
			return str(context['primitive_sequence_dir'])
		raise ValueError(
			"[PrimitiveActivityMapping] primitive_sequence_dir is missing. "
			"Set it in config (steps.primitive_activity_mapping.primitive_sequence_dir) "
			"or provide context['primitive_sequence_dir']."
		)

	def _to_numeric_ts(self, value: Any) -> Optional[float]:
		try:
			num = float(value)
			if np.isfinite(num):
				return num
		except Exception:
			pass
		return None

	def _extract_ranges_from_array(self, arr: np.ndarray, file_name: str, file_path: str) -> List[Dict[str, Any]]:
		records: List[Dict[str, Any]] = []

		def build_record(ts_values: np.ndarray, primitive_idx: int) -> Optional[Dict[str, Any]]:
			if ts_values.ndim != 1 or ts_values.size == 0:
				return None
			valid = np.isfinite(ts_values)
			# Padding points are often zeros; keep positive finite timestamps.
			valid = valid & (ts_values > 0)
			if not np.any(valid):
				return None
			valid_idx = np.where(valid)[0]
			start_idx = int(valid_idx[0])
			end_idx = int(valid_idx[-1])
			start_ts = float(ts_values[start_idx])
			end_ts = float(ts_values[end_idx])
			if end_ts < start_ts:
				start_ts, end_ts = end_ts, start_ts
			return {
				'primitive_file_name': file_name,
				'primitive_file_path': file_path,
				'primitive_index': int(primitive_idx),
				'start_timestamp': start_ts,
				'end_timestamp': end_ts,
			}

		if arr.ndim == 3:
			for i in range(arr.shape[0]):
				sample = arr[i]
				if sample.ndim != 2 or sample.shape[1] < 1:
					continue
				record = build_record(sample[:, 0], i)
				if record is not None:
					records.append(record)
		elif arr.ndim == 2:
			ts_values = arr[:, 0] if arr.shape[1] >= 1 else arr.reshape(-1)
			record = build_record(ts_values, 0)
			if record is not None:
				records.append(record)
		elif arr.ndim == 1:
			record = build_record(arr, 0)
			if record is not None:
				records.append(record)

		return records

	def _collect_primitive_ranges(self, primitive_dir: str) -> List[Dict[str, Any]]:
		primitive_path = Path(primitive_dir)
		if not primitive_path.exists():
			raise FileNotFoundError(f"[PrimitiveActivityMapping] Primitive directory not found: {primitive_dir}")

		records: List[Dict[str, Any]] = []
		npy_files = sorted(primitive_path.glob('*.npy'))

		for npy_file in npy_files:
			try:
				arr = np.load(npy_file)
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Skip unreadable npy {npy_file.name}: {e}")
				continue

			file_records = self._extract_ranges_from_array(arr, npy_file.name, str(npy_file))
			if not file_records:
				print(f"[PrimitiveActivityMapping] No valid primitive ranges from {npy_file.name}")
				continue

			records.extend(file_records)

		for i, rec in enumerate(records):
			rec['primitive_global_index'] = int(i)

		return records

	def _match_primitive_to_activity(
		self,
		primitive_records: List[Dict[str, Any]],
		activity_records: List[Dict[str, Any]],
	) -> List[Dict[str, Any]]:
		matches: List[Dict[str, Any]] = []
		tol = float(self.timestamp_tolerance) if self.enable_tolerant_match else 0.0

		norm_activities: List[Dict[str, Any]] = []
		for a in activity_records:
			a_start = self._to_numeric_ts(a.get('start_timestamp'))
			a_end = self._to_numeric_ts(a.get('end_timestamp'))
			if a_start is None or a_end is None:
				continue
			if a_end < a_start:
				a_start, a_end = a_end, a_start
			norm_activities.append(
				{
					'file_name': a.get('file_name'),
					'file_path': a.get('file_path'),
					'start': float(a_start),
					'end': float(a_end),
				}
			)

		def choose_best_containing(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
			if not candidates:
				return None
			# Prefer the tightest containing interval.
			return min(candidates, key=lambda x: (x['end'] - x['start'], x['start']))

		def nearest_activity_and_direction(p_start: float, p_end: float) -> tuple[Optional[Dict[str, Any]], str, Optional[float]]:
			if not norm_activities:
				return None, 'no_activity', None

			best = None
			best_gap = np.inf
			best_direction = 'unmatched'

			for a in norm_activities:
				if p_end < a['start']:
					gap = float(a['start'] - p_end)
					direction = 'unmatched_left'
				elif p_start > a['end']:
					gap = float(p_start - a['end'])
					direction = 'unmatched_right'
				else:
					# Overlap but not containment.
					left_over = max(0.0, a['start'] - p_start)
					right_over = max(0.0, p_end - a['end'])
					gap = float(left_over + right_over)
					direction = 'overlap_not_contain'

				if gap < best_gap:
					best_gap = gap
					best = a
					best_direction = direction

			return best, best_direction, (None if not np.isfinite(best_gap) else float(best_gap))

		for p in primitive_records:
			p_start = self._to_numeric_ts(p.get('start_timestamp'))
			p_end = self._to_numeric_ts(p.get('end_timestamp'))
			if p_start is None or p_end is None:
				continue

			if p_end < p_start:
				p_start, p_end = p_end, p_start

			strict_candidates = [
				a for a in norm_activities
				if a['start'] <= p_start and p_end <= a['end']
			]
			strict_hit = choose_best_containing(strict_candidates)
			if strict_hit is not None:
				matches.append(
					{
						'primitive_global_index': p['primitive_global_index'],
						'primitive_file_name': p['primitive_file_name'],
						'primitive_index': p['primitive_index'],
						'primitive_start_timestamp': p_start,
						'primitive_end_timestamp': p_end,
						'activity_file_name': strict_hit['file_name'],
						'activity_file_path': strict_hit['file_path'],
						'activity_start_timestamp': strict_hit['start'],
						'activity_end_timestamp': strict_hit['end'],
						'match_type': 'contain',
						'tolerance_used': 0.0,
					}
				)
				continue

			if tol > 0:
				tolerant_candidates = [
					a for a in norm_activities
					if (a['start'] - tol) <= p_start and p_end <= (a['end'] + tol)
				]
				tolerant_hit = choose_best_containing(tolerant_candidates)
				if tolerant_hit is not None:
					matches.append(
						{
							'primitive_global_index': p['primitive_global_index'],
							'primitive_file_name': p['primitive_file_name'],
							'primitive_index': p['primitive_index'],
							'primitive_start_timestamp': p_start,
							'primitive_end_timestamp': p_end,
							'activity_file_name': tolerant_hit['file_name'],
							'activity_file_path': tolerant_hit['file_path'],
							'activity_start_timestamp': tolerant_hit['start'],
							'activity_end_timestamp': tolerant_hit['end'],
							'match_type': 'tolerant_contain',
							'tolerance_used': tol,
						}
					)
					continue

			nearest_a, direction, gap = nearest_activity_and_direction(p_start, p_end)
			if nearest_a is not None:
				if tol > 0:
					print(
						"[PrimitiveActivityMapping][WARN] tolerance mismatch: "
						f"primitive({p['primitive_file_name']}#{p['primitive_index']}, global={p['primitive_global_index']}, "
						f"[{p_start}, {p_end}]) vs activity({nearest_a['file_name']}, [{nearest_a['start']}, {nearest_a['end']}]), "
						f"direction={direction}, gap={gap}, tolerance={tol}"
					)
				else:
					print(
						"[PrimitiveActivityMapping][WARN] strict mismatch: "
						f"primitive({p['primitive_file_name']}#{p['primitive_index']}, global={p['primitive_global_index']}, "
						f"[{p_start}, {p_end}]) vs activity({nearest_a['file_name']}, [{nearest_a['start']}, {nearest_a['end']}]), "
						f"direction={direction}, gap={gap}"
					)

			matches.append(
				{
					'primitive_global_index': p['primitive_global_index'],
					'primitive_file_name': p['primitive_file_name'],
					'primitive_index': p['primitive_index'],
					'primitive_start_timestamp': p_start,
					'primitive_end_timestamp': p_end,
					'activity_file_name': nearest_a['file_name'] if nearest_a else None,
					'activity_file_path': nearest_a['file_path'] if nearest_a else None,
					'activity_start_timestamp': nearest_a['start'] if nearest_a else None,
					'activity_end_timestamp': nearest_a['end'] if nearest_a else None,
					'match_type': direction,
					'tolerance_used': tol,
					'mismatch_gap': gap,
				}
			)

		return matches

	def _build_activity_tensor(
		self,
		activity_records: List[Dict[str, Any]],
	) -> tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
		"""
		Read matched activity CSV files and stack them to a 3D tensor.

		Returns:
			tensor: np.ndarray with shape (n_samples, seq_len, feature_dim)
			seq_lens: np.ndarray with shape (n_samples,)
			feature_columns: list of aligned feature column names
			valid_records: records that are successfully converted into tensor samples
		"""
		if not activity_records:
			return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32), [], []

		feature_columns: Optional[List[str]] = None
		samples: List[np.ndarray] = []
		seq_lens: List[int] = []
		valid_records: List[Dict[str, Any]] = []

		for rec in activity_records:
			file_path = rec.get('file_path')
			if not file_path or not Path(file_path).exists():
				print(f"[PrimitiveActivityMapping] Skip missing activity csv: {file_path}")
				continue

			try:
				df = pd.read_csv(file_path)
			except Exception as e:
				print(f"[PrimitiveActivityMapping] Skip unreadable activity csv {file_path}: {e}")
				continue

			if df.empty:
				print(f"[PrimitiveActivityMapping] Skip empty activity csv: {file_path}")
				continue

			if feature_columns is None:
				feature_columns = [str(c) for c in df.columns]
			else:
				# Align columns to a stable schema so feature_dim is consistent.
				df = df.reindex(columns=feature_columns)

			for col in df.columns:
				df[col] = pd.to_numeric(df[col], errors='coerce')

			arr = df.to_numpy(dtype=np.float32, copy=False)
			if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
				print(f"[PrimitiveActivityMapping] Skip invalid activity matrix from {file_path}")
				continue

			samples.append(arr)
			seq_lens.append(int(arr.shape[0]))
			valid_records.append(rec)

		if not samples:
			return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32), (feature_columns or []), []

		max_seq_len = int(max(seq_lens))
		feature_dim = int(samples[0].shape[1])
		tensor = np.zeros((len(samples), max_seq_len, feature_dim), dtype=np.float32)

		for i, arr in enumerate(samples):
			cur_len = int(arr.shape[0])
			tensor[i, :cur_len, :] = arr

		return tensor, np.asarray(seq_lens, dtype=np.int32), (feature_columns or []), valid_records

	def run(self, context: dict) -> dict:
		log_dir = self.get_log_dir(context)
		os.makedirs(log_dir, exist_ok=True)

		activity_dir = self._resolve_activity_dir(context)
		activity_records = self._collect_activity_ranges(activity_dir)
		activity_df = pd.DataFrame(activity_records)

		primitive_dir = self._resolve_primitive_dir(context)
		primitive_records = self._collect_primitive_ranges(primitive_dir)
		primitive_df = pd.DataFrame(primitive_records)

		match_records = self._match_primitive_to_activity(primitive_records, activity_records)
		match_df = pd.DataFrame(match_records)

		matched_activity_files = {
			rec['activity_file_name']
			for rec in match_records
			if rec.get('activity_file_name') and rec.get('match_type') in ('contain', 'tolerant_contain')
		}
		few_shot_activity_records = [
			rec for rec in activity_records
			if rec.get('file_name') in matched_activity_files
		]
		non_few_shot_activity_records = [
			rec for rec in activity_records
			if rec.get('file_name') not in matched_activity_files
		]

		ranges_json_path = os.path.join(log_dir, 'activity_sequence_ranges.json')
		with open(ranges_json_path, 'w', encoding='utf-8') as f:
			json.dump(activity_records, f, ensure_ascii=False, indent=2)

		primitive_ranges_json = os.path.join(log_dir, 'primitive_sequence_ranges.json')
		with open(primitive_ranges_json, 'w', encoding='utf-8') as f:
			json.dump(primitive_records, f, ensure_ascii=False, indent=2)

		mapping_json = os.path.join(log_dir, 'primitive_activity_mapping.json')
		with open(mapping_json, 'w', encoding='utf-8') as f:
			json.dump(match_records, f, ensure_ascii=False, indent=2)

		few_shot_input_count = len(few_shot_activity_records)
		non_few_shot_input_count = len(non_few_shot_activity_records)

		few_shot_tensor, few_shot_seq_lens, few_shot_feature_columns, few_shot_activity_records = self._build_activity_tensor(few_shot_activity_records)
		non_few_shot_tensor, non_few_shot_seq_lens, non_few_shot_feature_columns, non_few_shot_activity_records = self._build_activity_tensor(non_few_shot_activity_records)

		few_shot_activity_df = pd.DataFrame(few_shot_activity_records)
		non_few_shot_activity_df = pd.DataFrame(non_few_shot_activity_records)

		few_shot_json = os.path.join(log_dir, 'few_shot_activity_sequences.json')
		non_few_shot_json = os.path.join(log_dir, 'non_few_shot_activity_sequences.json')
		with open(few_shot_json, 'w', encoding='utf-8') as f:
			json.dump(few_shot_activity_records, f, ensure_ascii=False, indent=2)
		with open(non_few_shot_json, 'w', encoding='utf-8') as f:
			json.dump(non_few_shot_activity_records, f, ensure_ascii=False, indent=2)

		few_shot_tensor_npy = os.path.join(log_dir, 'few_shot_activity_tensor.npy')
		non_few_shot_tensor_npy = os.path.join(log_dir, 'non_few_shot_activity_tensor.npy')
		np.save(few_shot_tensor_npy, few_shot_tensor)
		np.save(non_few_shot_tensor_npy, non_few_shot_tensor)

		context['activity_sequence_source_dir'] = activity_dir
		context['activity_sequence_ranges'] = activity_records
		context['activity_sequence_ranges_df'] = activity_df
		context['activity_sequence_ranges_json'] = ranges_json_path

		context['primitive_sequence_source_dir'] = primitive_dir
		context['primitive_sequence_ranges'] = primitive_records
		context['primitive_sequence_ranges_df'] = primitive_df
		context['primitive_sequence_ranges_json'] = primitive_ranges_json

		context['primitive_activity_mapping'] = match_records
		context['primitive_activity_mapping_df'] = match_df
		context['primitive_activity_mapping_json'] = mapping_json

		context['few_shot_activity_sequences'] = few_shot_activity_records
		context['few_shot_activity_sequences_df'] = few_shot_activity_df
		context['few_shot_activity_sequences_json'] = few_shot_json
		context['few_shot_activity_tensor'] = few_shot_tensor
		context['few_shot_activity_seq_lens'] = few_shot_seq_lens
		context['few_shot_activity_feature_columns'] = few_shot_feature_columns
		context['few_shot_activity_tensor_npy'] = few_shot_tensor_npy

		context['non_few_shot_activity_sequences'] = non_few_shot_activity_records
		context['non_few_shot_activity_sequences_df'] = non_few_shot_activity_df
		context['non_few_shot_activity_sequences_json'] = non_few_shot_json
		context['non_few_shot_activity_tensor'] = non_few_shot_tensor
		context['non_few_shot_activity_seq_lens'] = non_few_shot_seq_lens
		context['non_few_shot_activity_feature_columns'] = non_few_shot_feature_columns
		context['non_few_shot_activity_tensor_npy'] = non_few_shot_tensor_npy

		print(
			f"[PrimitiveActivityMapping] Activities={len(activity_records)}, primitives={len(primitive_records)}, "
			f"few-shot activities={len(few_shot_activity_records)}/{few_shot_input_count}, "
			f"non-few-shot activities={len(non_few_shot_activity_records)}/{non_few_shot_input_count}, "
			f"few-shot tensor shape={tuple(few_shot_tensor.shape)}, non-few-shot tensor shape={tuple(non_few_shot_tensor.shape)}"
		)
		return context
