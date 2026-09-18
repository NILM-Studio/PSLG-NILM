from pathlib import Path
import numpy as np
from .nilm_common import NilmStep, strict_guard, require_artifact, read_json, write_json, signature, digest
from src.utils.state_activity_mapping import map_blocks, sequence_statistics


class StateSequenceStep(NilmStep):
    step_type = 'state_sequence'

    def run(self, context):
        strict_guard(context)
        activity_path = require_artifact(context, 'extract_active_data', 'activities')
        am = read_json(activity_path)
        tag = self.selection.get('cluster_tag')
        if not tag or not tag.endswith('_merged'):
            raise ValueError('state_sequence requires an explicit final merged cluster_tag')
        manifest = context['manifest']
        block_path = manifest.cluster_artifact_path(tag, 'blocks')
        if not block_path:
            raise ValueError('Run state_merge for selected tag first')
        blocks = read_json(block_path)
        paths = {key: manifest.cluster_artifact_path(tag, key) for key in ('blocks', 'labels', 'feature_matrix', 'indices', 'seq_len')}
        hashes = {str(Path(p).resolve()): digest(p) for p in [activity_path, *paths.values()]}
        if self.resolve(context, 'nilm_data', 'split_manifest'):
            for key in ('weights', 'normalization', 'model_config'):
                p = self.resolve(context, 'feature_extract', key)
                if p:
                    hashes[str(Path(p).resolve())] = digest(p)
        source = self.resolve(context, 'nilm_data', 'split_manifest')
        records = read_json(source)['records'] if source else [dict(id='discovery', file=self.resolve(context, 'extract_active_data', 'timeline'))]
        dt = self.cfg.get('data_protocol', {}).get('sample_seconds', int(round(1 / self.cfg['extract_active_data']['fs'])))
        k = self.cfg.get('nilm', {}).get('k', max(b['state_label'] for b in blocks) + 1)
        dictionary_id = signature(dict(inputs=hashes, source=digest(source) if source else 'discovery_only',
                                       appliance=context['appliance'], tag=tag))
        out = self.fresh_dir(context)
        entries, sequences_all, quality = [], [], []
        for ri, record in enumerate(records):
            path = Path(source).parent / record['file'] if source else Path(record['file'])
            with np.load(path) as z:
                t, y = z['timestamp'], z['target']
            aa = [a for a in am['activities'] if a['record_id'] == record['id']]
            fids = {a['csv_idx'] for a in aa}
            mapping, sequences = map_blocks(t, aa, [b for b in blocks if b['csv_idx'] in fids], dt)
            trans, durations = sequence_statistics(sequences, np.isfinite(y), mapping['context_conflict'], k, dt,
                                                   self.cfg.get('nilm', {}).get('window_length', 600))
            name = f'mapping_{ri}.npz'
            np.savez_compressed(out / name, timestamp=t, **mapping)
            entries.append(dict(record_id=record['id'], file=name, sha256=digest(out / name)))
            sequences_all.extend(sequences)
            write_json(out / f'transitions_{ri}.json', trans)
            write_json(out / f'durations_{ri}.json', durations)
            window_seconds = dt * self.cfg.get('nilm', {}).get('window_length', 600)
            quality.append(dict(record_id=record['id'], activities=len(aa), blocks=len(sequences),
                complete_observed_blocks=sum(b['complete_observed'] for b in sequences),
                activity_lengths_seconds=[a['core_end_exclusive']-a['core_start'] for a in aa],
                activities_longer_than_window=sum(a['core_end_exclusive']-a['core_start'] > window_seconds for a in aa),
                context_conflict_points=int(mapping['context_conflict'].sum()),
                teacher_interpolated_points=sum(a.get('interpolated_support', 0) for a in aa),
                missing_state_points_within_core=sum(int((mapping['state_full_merge'][a['core_start_index']:a['core_end_index']] < 0).sum()) for a in aa)))
        write_json(out / 'activity_state_sequences.json', sequences_all)
        write_json(out / 'sequence_quality.json', quality)
        write_json(out / 'state_to_activity.json', dict(dictionary_id=dictionary_id, k=k, records=entries,
                   input_hashes=hashes, source_only=bool(source), tag=tag))
        # Frozen original cluster scaler/centers: reconstruct from the actual source rows and labels.
        raw_tag = tag.removesuffix('_merged')
        raw = np.load(manifest.cluster_artifact_path(raw_tag, 'feature_matrix'))
        labels = np.load(manifest.cluster_artifact_path(raw_tag, 'labels')).ravel()
        if self.cfg['time_clustering'].get('normalization_method', 'zscore') != 'zscore':
            raise ValueError('The aligned NILM dictionary requires original zscore clustering')
        mean, scale = raw.mean(axis=0), raw.std(axis=0)
        scale[scale == 0] = 1
        norm = (raw - mean) / scale
        centers = np.stack([norm[labels == i].mean(axis=0) for i in range(k)])
        if not np.isfinite(centers).all():
            raise ValueError('Dictionary contains an empty class')
        np.savez_compressed(out / 'dictionary.npz', mean=mean, scale=scale, centers=centers,
                            dictionary_id=np.asarray(dictionary_id))
        return self.register(context, out, {'mapping': 'state_to_activity.json', 'sequences': 'activity_state_sequences.json',
            'quality': 'sequence_quality.json', 'dictionary': 'dictionary.npz'})


build = StateSequenceStep
