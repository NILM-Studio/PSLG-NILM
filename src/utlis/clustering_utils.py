import datetime
import importlib
import json
import os
import warnings
from functools import lru_cache
from typing import Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tslearn.barycenters import dtw_barycenter_averaging


def _normalize_language(language: str) -> str:
	lang = str(language).lower()
	if lang.startswith('zh'):
		return 'zh'
	return 'en'


@lru_cache(maxsize=1)
def _load_i18n_table() -> dict:

	i18n_file = os.path.join(os.path.dirname(__file__), 'resources', 'cluster_visualization_i18n.json')
	if not os.path.exists(i18n_file):
		raise FileNotFoundError(f'i18n resource not found: {i18n_file}')

	with open(i18n_file, 'r', encoding='utf-8') as f:
		table = json.load(f)

	if not (isinstance(table, dict) and 'zh' in table and 'en' in table):
		raise ValueError(
			f"Invalid i18n resource format: {i18n_file}. "
			"Expected top-level keys 'zh' and 'en'."
		)

	return table


def _i18n(language: str) -> dict:
	lang = _normalize_language(language)
	table = _load_i18n_table()
	if lang in table and isinstance(table[lang], dict):
		return table[lang]
	return table.get('en', {})


def setup_chinese_font() -> bool:
	"""Configure Chinese fonts for matplotlib and return whether setup succeeded."""
	chinese_font_paths = [
		'/home/scnu2023024258/.local/share/fonts/wqy-microhei.ttc',
		'/home/scnu2023024258/.local/share/fonts/SourceHanSansSC-Regular.otf',
		'/home/scnu2023024258/.local/share/fonts/NotoSansCJKsc-Regular.otf',
	]

	for font_path in chinese_font_paths:
		if os.path.exists(font_path):
			try:
				font_prop = fm.FontProperties(fname=font_path)
				font_name = font_prop.get_name()
				plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
				plt.rcParams['axes.unicode_minus'] = False
				return True
			except Exception:
				continue

	chinese_font_names = [
		'WenQuanYi Micro Hei',
		'Source Han Sans SC',
		'Noto Sans CJK SC',
		'SimHei',
		'Microsoft YaHei',
	]
	all_fonts = [f.name for f in fm.fontManager.ttflist]
	for font in chinese_font_names:
		if font in all_fonts:
			plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
			plt.rcParams['axes.unicode_minus'] = False
			return True

	plt.rcParams['axes.unicode_minus'] = False
	warnings.filterwarnings('ignore', category=UserWarning, message='Glyph.*missing from font')
	return False


def visualize_dict_data_layered(
	data_dict,
	title='Layered Visualization',
	bar_width=0.8,
	x_axis=None,
	max_labels=5,
	language='zh',
	show=True,
):
	"""Visualize dict values as layered bar charts and return the figure."""
	if x_axis is None:
		raise ValueError('x_axis cannot be None')

	n_items = len(data_dict)
	if n_items == 0:
		return None

	if n_items <= 2:
		n_cols, n_rows = n_items, 1
	elif n_items <= 6:
		n_cols, n_rows = 2, (n_items + 1) // 2
	elif n_items <= 12:
		n_cols, n_rows = 3, (n_items + 2) // 3
	elif n_items <= 20:
		n_cols, n_rows = 4, (n_items + 3) // 4
	else:
		n_cols, n_rows = 5, (n_items + 4) // 5

	fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=150)
	if n_items == 1:
		axes = [axes]
	else:
		axes = axes.flatten() if hasattr(axes, 'flatten') else axes

	colors = plt.cm.tab10(np.linspace(0, 1, n_items)) if n_items <= 10 else plt.cm.hsv(np.linspace(0, 1, n_items))

	texts = _i18n(language)
	if title == 'Layered Visualization':
		title = texts['default_title']

	for idx, (key, value) in enumerate(data_dict.items()):
		if not isinstance(value, np.ndarray):
			continue

		ax = axes[idx]
		if len(x_axis) > len(value):
			value = np.pad(value, (0, len(x_axis) - len(value)), mode='constant', constant_values=0)
		elif len(x_axis) < len(value):
			raise ValueError(f'x_axis length ({len(x_axis)}) is less than value length ({len(value)})')

		x_pos = np.arange(len(value))
		ax.bar(x_pos, value, width=bar_width, color=colors[idx])
		ax.set_title(f"{texts['cluster_prefix']}_{key}")
		ax.set_xlabel(texts['time'])
		ax.set_ylabel(texts['power'])

		n_ticks = len(x_pos)
		if max_labels > 0 and n_ticks > max_labels:
			indices = np.linspace(0, n_ticks - 1, max_labels, dtype=int)
			ax.set_xticks(indices)
			ax.set_xticklabels([x_axis[i] for i in indices], rotation=45, ha='right')
		else:
			ax.set_xticks(range(len(x_pos)))
			ax.set_xticklabels(x_axis, rotation=45, ha='right')
		ax.grid(True, alpha=0.3)

	if hasattr(axes, '__iter__') and len(axes) > n_items:
		for idx in range(n_items, len(axes)):
			if hasattr(axes[idx], 'set_visible'):
				axes[idx].set_visible(False)

	fig.suptitle(title, fontsize=16)
	plt.tight_layout()
	if show:
		plt.show()
	return fig


def _build_time_bins(start_datetime, end_datetime, time_gap_type='days', time_gap_value=1):
	bin_start_datetimes = []
	current_datetime = start_datetime
	if time_gap_type == 'days':
		while current_datetime <= end_datetime:
			bin_start_datetimes.append(current_datetime)
			current_datetime += datetime.timedelta(days=time_gap_value)
	elif time_gap_type == 'months':
		while current_datetime <= end_datetime:
			bin_start_datetimes.append(current_datetime)
			year = current_datetime.year
			month = current_datetime.month + time_gap_value
			while month > 12:
				month -= 12
				year += 1
			current_datetime = current_datetime.replace(year=year, month=month)
	else:
		raise ValueError("time_gap_type must be 'days' or 'months'")
	return bin_start_datetimes


def visualize_cluster_by_time_gap(
	data_mapping,
	cluster_result,
	time_gap_type='days',
	time_gap_value=1,
	max_duration=3600 * 24,
	save_json_path='./',
	language='zh',
	show=True,
):
	"""Aggregate and visualize cluster durations over fixed time gaps."""
	texts = _i18n(language)
	if len(data_mapping) == 0 or len(cluster_result) == 0:
		raise ValueError('data_mapping and cluster_result must be non-empty')
	if len(data_mapping) != len(cluster_result):
		raise ValueError('data_mapping length must match cluster_result length')

	total_start_time = data_mapping[0]['start_timestamp']
	total_end_time = data_mapping[len(cluster_result) - 1]['end_timestamp']
	start_datetime = datetime.datetime.fromtimestamp(total_start_time)
	end_datetime = datetime.datetime.fromtimestamp(total_end_time)
	bin_start_datetimes = _build_time_bins(start_datetime, end_datetime, time_gap_type, time_gap_value)

	n_bins = len(bin_start_datetimes) - 1
	time_gap_start_datetimes = bin_start_datetimes[:n_bins]
	time_gap_end_datetimes = bin_start_datetimes[1:]
	time_gap_start_timestamps = np.array([dt.timestamp() for dt in time_gap_start_datetimes])
	time_gap_end_timestamps = np.array([dt.timestamp() for dt in time_gap_end_datetimes])
	time_gap_start_datetimes_str = np.array([dt.strftime('%y/%m/%d') for dt in time_gap_start_datetimes])
	time_gap_end_datetimes_str = np.array([dt.strftime('%y/%m/%d') for dt in time_gap_end_datetimes])

	unique_clusters = np.unique(cluster_result)
	cluster_time_stats = {cluster_id: np.zeros(n_bins) for cluster_id in unique_clusters}

	for i in range(len(cluster_result)):
		mapping_info = data_mapping[i]
		start_time = mapping_info['start_timestamp']
		end_time = mapping_info['end_timestamp']
		cluster_id = cluster_result[i]
		duration = end_time - start_time
		if duration > max_duration:
			continue

		start_dt = datetime.datetime.fromtimestamp(start_time)
		data_bin = -1
		for j in range(n_bins):
			if bin_start_datetimes[j] <= start_dt < bin_start_datetimes[j + 1]:
				data_bin = j
				break
		if 0 <= data_bin < n_bins:
			cluster_time_stats[cluster_id][data_bin] += duration

	if save_json_path:
		json_data = {}
		for cluster_id in unique_clusters:
			time_data = []
			for i in range(n_bins):
				time_data.append(
					{
						'start_timestamp': float(time_gap_start_timestamps[i]),
						'start_datetime': str(time_gap_start_datetimes_str[i]),
						'end_timestamp': float(time_gap_end_timestamps[i]),
						'end_datetime': str(time_gap_end_datetimes_str[i]),
						'interval_total_duration': float(cluster_time_stats[cluster_id][i]),
					}
				)
			json_data[str(cluster_id)] = time_data

		os.makedirs(save_json_path, exist_ok=True)
		with open(os.path.join(save_json_path, 'time_gap_data.json'), 'w', encoding='utf-8') as f:
			json.dump(json_data, f, ensure_ascii=False, indent=2)

	fig = visualize_dict_data_layered(
		cluster_time_stats,
		title=texts['time_gap_stats'],
		x_axis=time_gap_start_datetimes_str,
		language=language,
		show=show,
	)
	return fig


def cluster_result_pic_save(data_array, seq_length, cluster_result, save_dir, threshold=200, col_index=1, language='zh'):
	"""Save per-series figures grouped by cluster label."""
	import shutil
	texts = _i18n(language)

	cluster_groups = {}
	for i in range(len(data_array)):
		cluster_id = cluster_result[i]
		cluster_groups.setdefault(cluster_id, []).append(i)

	for cluster_id, indices in cluster_groups.items():
		if len(indices) > threshold:
			indices = indices[:threshold]

		cluster_dir = os.path.join(save_dir, f'cluster_{cluster_id}')
		if os.path.exists(cluster_dir):
			shutil.rmtree(cluster_dir)
		os.makedirs(cluster_dir, exist_ok=True)

		for idx, data_idx in enumerate(indices):
			length = int(np.asarray(seq_length[data_idx]).reshape(-1)[0]) if len(seq_length) > data_idx else len(data_array[data_idx])
			data = data_array[data_idx][:length][:, col_index]
			plt.figure(figsize=(10, 6))
			plt.plot(data)
			plt.title(f"{texts['cluster_prefix']} {cluster_id} - {texts['series']} {idx + 1}")
			plt.xlabel(texts['time'])
			plt.ylabel(texts['value'])
			plt.savefig(os.path.join(cluster_dir, f'item_{idx + 1}.png'))
			plt.close()


def preprocess_cluster_data(cluster_labels: np.ndarray, dist_matrix: np.ndarray, org_data: np.ndarray, feature_matrix: np.ndarray):
	"""Validate dimensions and filter out noise points (-1 labels)."""
	n_samples = len(cluster_labels)
	if dist_matrix.shape != (n_samples, n_samples):
		raise ValueError(f'dist_matrix shape {dist_matrix.shape} does not match n_samples {n_samples}')
	if org_data.shape[0] != n_samples:
		raise ValueError(f'org_data rows {org_data.shape[0]} does not match n_samples {n_samples}')
	if feature_matrix.shape[0] != n_samples:
		raise ValueError(f'feature_matrix rows {feature_matrix.shape[0]} does not match n_samples {n_samples}')

	valid_idx = cluster_labels != -1
	valid_dist_matrix = dist_matrix[valid_idx][:, valid_idx]
	valid_labels = cluster_labels[valid_idx]
	valid_org_data = org_data[valid_idx]
	valid_feature_matrix = feature_matrix[valid_idx]
	n_clusters = len(np.unique(valid_labels))
	return valid_idx, valid_dist_matrix, valid_labels, valid_org_data, valid_feature_matrix, n_clusters


@lru_cache(maxsize=1)
def _get_hdbscan_validity_index():
	"""Lazily import hdbscan validity_index to avoid hard dependency at import time."""
	try:
		module = importlib.import_module('hdbscan.validity')
		return getattr(module, 'validity_index', None)
	except Exception:
		return None


def calculate_dbcv_score(dist_matrix: np.ndarray, cluster_labels: np.ndarray):
	"""Compute DBCV score using hdbscan.validity.validity_index on precomputed distances."""
	labels = np.asarray(cluster_labels)
	if labels.size == 0:
		return None

	valid_labels = labels[labels != -1]
	if len(np.unique(valid_labels)) < 2:
		return None

	if dist_matrix.shape != (labels.size, labels.size):
		raise ValueError(
			f'dist_matrix shape {dist_matrix.shape} does not match labels length {labels.size}'
		)

	validity_index = _get_hdbscan_validity_index()
	if validity_index is None:
		print('[TimeClustering][WARN] hdbscan is not installed; DBCV is skipped. Install hdbscan to enable DBCV.')
		return None

	try:
		score = validity_index(dist_matrix, labels, metric='precomputed')
		score = float(score)
		if not np.isfinite(score):
			return None
		return score
	except Exception as e:
		print(f'[TimeClustering][WARN] Failed to compute DBCV: {e}')
		return None


def calculate_cluster_metrics(valid_dist_matrix: np.ndarray, valid_labels: np.ndarray, valid_feature_matrix: np.ndarray, cluster_labels: np.ndarray):
	"""Compute silhouette, DB, and CH metrics."""
	n_clusters = len(np.unique(valid_labels))
	if n_clusters < 2:
		return None, None, None

	sil_score = silhouette_score(valid_dist_matrix, valid_labels, metric='precomputed')
	db_score = davies_bouldin_score(valid_feature_matrix, valid_labels)
	ch_score = calinski_harabasz_score(valid_feature_matrix, valid_labels)
	return sil_score, db_score, ch_score


def _tsne_perplexity(n_samples: int) -> int:
	if n_samples < 5:
		return 2
	return max(2, min(30, n_samples // 10))


def detect_few_shot_clusters(
	cluster_labels: np.ndarray,
	method: str = 'avg_percent',
	n_percent: float = 50.0,
	threshold: int = 5,
) -> dict:
	"""Detect few-shot clusters from cluster size statistics (excluding noise cluster -1)."""
	labels = np.asarray(cluster_labels)
	if labels.size == 0:
		return {
			'method': method,
			'n_percent': float(n_percent),
			'threshold': int(threshold),
			'average_cluster_size': None,
			'few_shot_clusters': [],
		}

	unique_labels, counts = np.unique(labels, return_counts=True)
	valid_pairs = [(int(l), int(c)) for l, c in zip(unique_labels, counts) if int(l) != -1]
	if len(valid_pairs) == 0:
		return {
			'method': method,
			'n_percent': float(n_percent),
			'threshold': int(threshold),
			'average_cluster_size': None,
			'few_shot_clusters': [],
		}

	cluster_sizes = np.array([c for _, c in valid_pairs], dtype=np.float64)
	avg_cluster_size = float(np.mean(cluster_sizes))
	method_norm = str(method).lower()

	if method_norm in ('avg_percent', 'percent_avg', 'avg_ratio', 'ratio'):
		if n_percent < 0:
			raise ValueError(f'n_percent must be >= 0, got {n_percent}')
		cutoff = avg_cluster_size * (float(n_percent) / 100.0)
		few_shot_pairs = [(cid, cnt) for cid, cnt in valid_pairs if float(cnt) < cutoff]
	elif method_norm == 'threshold':
		if threshold < 0:
			raise ValueError(f'threshold must be >= 0, got {threshold}')
		few_shot_pairs = [(cid, cnt) for cid, cnt in valid_pairs if int(cnt) < int(threshold)]
	else:
		raise ValueError(
			f"Unsupported few_shot_detection method: {method}. "
			"Supported methods: avg_percent, threshold"
		)

	few_shot_clusters = [
		{'cluster_id': int(cid), 'sample_count': int(cnt)}
		for cid, cnt in sorted(few_shot_pairs, key=lambda x: (x[1], x[0]))
	]

	return {
		'method': method_norm,
		'n_percent': float(n_percent),
		'threshold': int(threshold),
		'average_cluster_size': avg_cluster_size,
		'few_shot_clusters': few_shot_clusters,
	}


def save_kmeans_scan_artifacts(
	scan_records: list,
	best_k: int,
	save_dir: str,
	data_path: Optional[str] = None,
	appliance_name: Optional[str] = None,
):
	"""Persist KMeans-scan metrics as JSON and line chart."""
	os.makedirs(save_dir, exist_ok=True)
	payload = {
		'scan_method': 'kmeans-scan',
		'best_n_clusters': int(best_k),
		'selection_rule': 'max_sci',
		'data_source': {
			'data_path': str(data_path) if data_path else None,
			'appliance_name': str(appliance_name) if appliance_name else None,
		},
		'records': scan_records,
	}
	json_path = os.path.join(save_dir, 'kmeans_scan_metrics.json')
	with open(json_path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)
	print(f"[TimeClustering] KMeans scan metrics saved to {json_path}")

	ks = [r['n_clusters'] for r in scan_records]
	sci_vals = [r['sci'] for r in scan_records]
	dbi_vals = [r['dbi'] for r in scan_records]
	chi_vals = [r['chi'] for r in scan_records]
	dbcv_vals = [np.nan if r.get('dbcv') is None else float(r['dbcv']) for r in scan_records]

	fig, axes = plt.subplots(4, 1, figsize=(10, 15), dpi=150)
	fig.suptitle('KMeans Scan Metrics', fontsize=14, fontweight='bold')

	axes[0].plot(ks, sci_vals, marker='o', color='tab:blue')
	axes[0].set_title('SCI (Silhouette) vs n_clusters')
	axes[0].set_xlabel('n_clusters')
	axes[0].set_ylabel('SCI (higher is better)')
	axes[0].grid(alpha=0.3)

	axes[1].plot(ks, dbi_vals, marker='o', color='tab:orange')
	axes[1].set_title('DBI vs n_clusters')
	axes[1].set_xlabel('n_clusters')
	axes[1].set_ylabel('DBI (lower is better)')
	axes[1].grid(alpha=0.3)

	axes[2].plot(ks, chi_vals, marker='o', color='tab:green')
	axes[2].set_title('CHI vs n_clusters')
	axes[2].set_xlabel('n_clusters')
	axes[2].set_ylabel('CHI (higher is better)')
	axes[2].grid(alpha=0.3)

	axes[3].plot(ks, dbcv_vals, marker='o', color='tab:red')
	axes[3].set_title('DBCV vs n_clusters')
	axes[3].set_xlabel('n_clusters')
	axes[3].set_ylabel('DBCV (higher is better)')
	axes[3].grid(alpha=0.3)

	plt.tight_layout()
	fig_path = os.path.join(save_dir, 'kmeans_scan_metrics.png')
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close(fig)
	print(f"[TimeClustering] KMeans scan plot saved to {fig_path}")
	return payload


def save_dbscan_scan_artifacts(
	scan_records: list,
	best_eps: float,
	save_dir: str,
	data_path: Optional[str] = None,
	appliance_name: Optional[str] = None,
):
	"""Persist DBSCAN-scan metrics as JSON and line charts."""
	os.makedirs(save_dir, exist_ok=True)
	payload = {
		'scan_method': 'dbscan-scan',
		'best_eps': float(best_eps),
		'selection_rule': 'max_sci',
		'data_source': {
			'data_path': str(data_path) if data_path else None,
			'appliance_name': str(appliance_name) if appliance_name else None,
		},
		'records': scan_records,
	}
	json_path = os.path.join(save_dir, 'dbscan_scan_metrics.json')
	with open(json_path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)
	print(f"[TimeClustering] DBSCAN scan metrics saved to {json_path}")

	x_vals = [float(r['eps']) for r in scan_records]
	sci_vals = [np.nan if r['sci'] is None else float(r['sci']) for r in scan_records]
	dbi_vals = [np.nan if r['dbi'] is None else float(r['dbi']) for r in scan_records]
	chi_vals = [np.nan if r['chi'] is None else float(r['chi']) for r in scan_records]
	dbcv_vals = [np.nan if r.get('dbcv') is None else float(r['dbcv']) for r in scan_records]
	n_noise_vals = [int(r['n_noise']) for r in scan_records]
	n_cluster_vals = [int(r['n_clusters']) for r in scan_records]

	fig, axes = plt.subplots(6, 1, figsize=(10, 22), dpi=150)
	fig.suptitle('DBSCAN Scan Metrics', fontsize=14, fontweight='bold')

	axes[0].plot(x_vals, sci_vals, marker='o', color='tab:blue')
	axes[0].set_title('SCI (Silhouette) vs eps')
	axes[0].set_xlabel('eps')
	axes[0].set_ylabel('SCI (higher is better)')
	axes[0].grid(alpha=0.3)

	axes[1].plot(x_vals, dbi_vals, marker='o', color='tab:orange')
	axes[1].set_title('DBI vs eps')
	axes[1].set_xlabel('eps')
	axes[1].set_ylabel('DBI (lower is better)')
	axes[1].grid(alpha=0.3)

	axes[2].plot(x_vals, chi_vals, marker='o', color='tab:green')
	axes[2].set_title('CHI vs eps')
	axes[2].set_xlabel('eps')
	axes[2].set_ylabel('CHI (higher is better)')
	axes[2].grid(alpha=0.3)

	axes[3].plot(x_vals, n_noise_vals, marker='o', color='tab:red')
	axes[3].set_title('n_noise vs eps')
	axes[3].set_xlabel('eps')
	axes[3].set_ylabel('n_noise')
	axes[3].grid(alpha=0.3)

	axes[4].plot(x_vals, n_cluster_vals, marker='o', color='tab:purple')
	axes[4].set_title('n_clusters vs eps')
	axes[4].set_xlabel('eps')
	axes[4].set_ylabel('n_clusters')
	axes[4].grid(alpha=0.3)

	axes[5].plot(x_vals, dbcv_vals, marker='o', color='tab:brown')
	axes[5].set_title('DBCV vs eps')
	axes[5].set_xlabel('eps')
	axes[5].set_ylabel('DBCV (higher is better)')
	axes[5].grid(alpha=0.3)

	plt.tight_layout()
	fig_path = os.path.join(save_dir, 'dbscan_scan_metrics.png')
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	plt.close(fig)
	print(f"[TimeClustering] DBSCAN scan plot saved to {fig_path}")
	return payload


def visualize_cluster_results(
	cluster_labels: np.ndarray,
	valid_labels: np.ndarray,
	valid_org_data: np.ndarray,
	feature_matrix: np.ndarray,
	org_data: np.ndarray,
	save_dir: Optional[str] = None,
	dist_method: str = 'dtw',
	col_index: int = 1,
	sampling_threshold: int = 200,
	cluster_stack_count: int = 50,
	visualize_noise: bool = True,
	language: str = 'zh',
	show: bool = True,
) -> None:
	"""Render cluster center, stacked series, and tSNE visualizations."""
	valid_org_data = valid_org_data[:, :, col_index]
	org_data = org_data[:, :, col_index]
	n_clusters = len(np.unique(valid_labels))
	texts = _i18n(language)

	if _normalize_language(language) == 'zh':
		setup_chinese_font()
	else:
		plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
	plt.rcParams['axes.unicode_minus'] = False
	cluster_colors = plt.cm.tab10(np.arange(n_clusters + 1))

	center_title = texts['center_dtw'] if dist_method == 'dtw' else texts['center_mean']

	figsize_height = max(8, n_clusters * 2)
	fig, axes = plt.subplots(n_clusters, 1, figsize=(12, figsize_height))
	fig.suptitle(center_title, fontsize=14, fontweight='bold')
	axes = [axes] if n_clusters == 1 else axes.flatten()

	for i, cluster_id in enumerate(np.unique(valid_labels)):
		cluster_seq = valid_org_data[valid_labels == cluster_id]
		if len(cluster_seq) > sampling_threshold:
			np.random.seed(42)
			sampled_indices = np.random.choice(len(cluster_seq), size=sampling_threshold, replace=False)
			cluster_seq = cluster_seq[sampled_indices]

		if len(cluster_seq) > 0:
			if dist_method == 'dtw':
				cluster_center = dtw_barycenter_averaging(cluster_seq)
			else:
				min_len = min(len(s) for s in cluster_seq)
				cluster_center = np.mean([s[:min_len] for s in cluster_seq], axis=0)

			axes[i].plot(
				cluster_center,
				color=cluster_colors[cluster_id % 10],
				linewidth=2.5,
				label=f"{texts['cluster_prefix']} {cluster_id} ({texts['sample_count']}: {len(valid_org_data[valid_labels == cluster_id])})",
			)
			axes[i].set_title(f"{texts['cluster_prefix']} {cluster_id} {texts['cluster_center_suffix']}", fontsize=12)
			axes[i].set_xlabel(texts['time_step'], fontsize=10)
			axes[i].set_ylabel(texts['series_value'], fontsize=10)
			axes[i].legend(fontsize=9)
			axes[i].grid(alpha=0.3, linestyle='--')

	plt.tight_layout()
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'cluster_center.png'), dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

	has_noise = np.any(cluster_labels == -1)
	show_noise = bool(visualize_noise and has_noise)
	stack_count = max(1, int(cluster_stack_count))
	total_plots = n_clusters + 1 if show_noise else n_clusters
	n_cols = min(3, total_plots)
	n_rows = (total_plots + n_cols - 1) // n_cols
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
	stacked_title = texts['stacked_with_noise'] if has_noise else texts['stacked_without_noise']
	fig.suptitle(stacked_title, fontsize=16, fontweight='bold')
	axes = [axes] if total_plots == 1 else axes.flatten()

	for i, cluster_id in enumerate(np.unique(valid_labels)):
		cluster_seq = valid_org_data[valid_labels == cluster_id]
		cluster_seq_subset = cluster_seq[:stack_count]
		for j, series in enumerate(cluster_seq_subset):
			axes[i].plot(series, alpha=0.6, label=f"{texts['series']} {j}" if j < 3 else '')
		first_n = texts['first_n_data'].format(n=len(cluster_seq_subset))
		axes[i].set_title(f"{texts['cluster_prefix']} {cluster_id} ({first_n})", fontsize=12)
		axes[i].set_xlabel(texts['time_step'], fontsize=10)
		axes[i].set_ylabel(texts['series_value'], fontsize=10)
		axes[i].grid(alpha=0.3, linestyle='--')

	if show_noise:
		noise_idx = cluster_labels == -1
		noise_ax = axes[n_clusters]
		noise_seq = org_data[noise_idx]
		noise_seq_subset = noise_seq[:stack_count]
		if len(noise_seq_subset) > 0:
			for j, series in enumerate(noise_seq_subset):
				noise_ax.plot(series, alpha=0.6, color='gray', label=f"{texts['noise']} {j}" if j < 3 else '')
			first_n = texts['first_n_data'].format(n=len(noise_seq_subset))
			noise_ax.set_title(texts['noise_points_with_n'].format(first_n=first_n), fontsize=12)
			noise_ax.set_xlabel(texts['time_step'], fontsize=10)
			noise_ax.set_ylabel(texts['series_value'], fontsize=10)
			noise_ax.grid(alpha=0.3, linestyle='--')

	for j in range(total_plots, len(axes)):
		axes[j].set_visible(False)

	plt.tight_layout()
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'clusters_stacked.png'), dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()

	tsne = TSNE(
		n_components=2,
		perplexity=_tsne_perplexity(len(feature_matrix)),
		random_state=42,
		n_jobs=-1,
		init='pca',
	)
	tsne_2d = tsne.fit_transform(feature_matrix)

	plt.figure(figsize=(10, 8))
	for cluster_id in np.unique(valid_labels):
		idx = (cluster_labels == cluster_id) & (cluster_labels != -1)
		plt.scatter(
			tsne_2d[idx, 0],
			tsne_2d[idx, 1],
			c=[cluster_colors[cluster_id % 10]],
			label=f"{texts['cluster_prefix']} {cluster_id}",
			s=70,
			alpha=0.8,
			edgecolors='white',
			linewidth=0.5,
		)

	noise_idx = cluster_labels == -1
	if show_noise:
		plt.scatter(tsne_2d[noise_idx, 0], tsne_2d[noise_idx, 1], c='black', marker='x', label=texts['noise'], s=90, alpha=0.8)

	plt.title(texts['tsne_title'], fontsize=14, fontweight='bold')
	plt.xlabel(texts['tsne_dim1'], fontsize=11)
	plt.ylabel(texts['tsne_dim2'], fontsize=11)
	plt.legend(fontsize=10, loc='best')
	plt.grid(alpha=0.2, linestyle='--')
	plt.tight_layout()
	if save_dir:
		plt.savefig(os.path.join(save_dir, 'tsne.png'), dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	plt.close()


def cluster_result_quantification(
	cluster_labels: np.ndarray,
	dist_matrix: np.ndarray,
	org_data: np.ndarray,
	feature_matrix: np.ndarray,
	save_dir: Optional[str] = None,
	seq_length: Optional[np.ndarray] = None,
	dist_method: str = 'dtw',
	cluster_method: str = 'unknown',
	cluster_hyperparams: Optional[dict] = None,
	language: str = 'zh',
	col_index: int = 1,
	visualize: bool = True,
	visualize_noise: bool = True,
	cluster_stack_count: int = 50,
	sampling_threshold: int = 200,
	data_path: Optional[str] = None,
	appliance_name: Optional[str] = None,
	few_shot_enabled: bool = False,
	few_shot_method: str = 'avg_percent',
	few_shot_n_percent: float = 50.0,
	few_shot_threshold: int = 5,
	return_metrics_payload: bool = False,
):
	"""Unified entry for clustering quantification, metrics persistence and visualization."""
	(
		valid_idx,
		valid_dist_matrix,
		valid_labels,
		valid_org_data,
		valid_feature_matrix,
		n_clusters,
	) = preprocess_cluster_data(
		cluster_labels=cluster_labels,
		dist_matrix=dist_matrix,
		org_data=org_data,
		feature_matrix=feature_matrix,
	)

	sil_score, db_score, ch_score = calculate_cluster_metrics(
		valid_dist_matrix=valid_dist_matrix,
		valid_labels=valid_labels,
		valid_feature_matrix=valid_feature_matrix,
		cluster_labels=cluster_labels,
	)
	dbcv_score = calculate_dbcv_score(dist_matrix=dist_matrix, cluster_labels=cluster_labels)

	metrics = {
		'clustering_method': str(cluster_method),
		'clustering_hyperparameters': cluster_hyperparams if isinstance(cluster_hyperparams, dict) else {},
		'distance_method_for_quantification': str(dist_method),
		'data_source': {
			'data_path': str(data_path) if data_path else None,
			'appliance_name': str(appliance_name) if appliance_name else None,
		},
		'cluster_distribution': {
			('noise' if int(label) == -1 else f'cluster_{int(label)}'): int(count)
			for label, count in zip(*np.unique(cluster_labels, return_counts=True))
		},
		'n_clusters': int(n_clusters),
		'n_noise': int(np.sum(cluster_labels == -1)),
	}
	if sil_score is not None:
		metrics['silhouette_score'] = float(sil_score)
	if db_score is not None:
		metrics['davies_bouldin_score'] = float(db_score)
	if ch_score is not None:
		metrics['calinski_harabasz_score'] = float(ch_score)
	if dbcv_score is not None:
		metrics['dbcv_score'] = float(dbcv_score)

	if few_shot_enabled:
		few_shot_result = detect_few_shot_clusters(
			cluster_labels=cluster_labels,
			method=few_shot_method,
			n_percent=few_shot_n_percent,
			threshold=few_shot_threshold,
		)
	else:
		few_shot_result = {
			'method': str(few_shot_method).lower(),
			'n_percent': float(few_shot_n_percent),
			'threshold': int(few_shot_threshold),
			'average_cluster_size': None,
			'few_shot_clusters': [],
		}

	metrics['few_shot_detection'] = {
		'enabled': bool(few_shot_enabled),
		'method': few_shot_result['method'],
		'n_percent': few_shot_result['n_percent'],
		'threshold': few_shot_result['threshold'],
		'average_cluster_size': few_shot_result['average_cluster_size'],
		'few_shot_clusters': few_shot_result['few_shot_clusters'],
	}

	if save_dir:
		os.makedirs(save_dir, exist_ok=True)
		metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
		with open(metrics_path, 'w', encoding='utf-8') as f:
			json.dump(metrics, f, ensure_ascii=False, indent=2)

	if visualize:
		if seq_length is None:
			seq_length = np.full((org_data.shape[0],), org_data.shape[1], dtype=np.int32)

		cluster_result_pic_save(
			data_array=org_data,
			seq_length=seq_length,
			cluster_result=cluster_labels,
			save_dir=save_dir if save_dir else './',
			threshold=sampling_threshold,
			col_index=col_index,
			language=language,
		)

		visualize_cluster_results(
			cluster_labels=cluster_labels,
			valid_labels=valid_labels,
			valid_org_data=valid_org_data,
			feature_matrix=feature_matrix,
			org_data=org_data,
			save_dir=save_dir,
			dist_method=dist_method,
			col_index=col_index,
			sampling_threshold=sampling_threshold,
			cluster_stack_count=cluster_stack_count,
			visualize_noise=visualize_noise,
			language=language,
			show=False,
		)

	if return_metrics_payload:
		return sil_score, db_score, ch_score, dbcv_score, metrics

	return sil_score, db_score, ch_score
