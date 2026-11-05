"""Prepare reproducible annotation partitions; no model training or predictions.

Run from repository root: uv run python -m scripts.golfdb_prepare --help
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import av
import numpy as np
import scipy

from backend.swingsync.research.golfdb import load_annotations, make_split_manifest, write_split_manifest
from backend.swingsync.research.timeline import inspect_clip_timeline


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(annotations_path, output_dir, *, seed=20261104, validation_fraction=.2,
            archive=None, clips_dir=None, clip_ids=(), all_clips=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('Output directory already exists; choose a new run directory')
    dataset = load_annotations(annotations_path)
    records = {r.clip_id: r for r in dataset.records}
    manifests = [make_split_manifest(dataset, fold, seed, validation_fraction) for fold in range(1, 5)]
    summary = {
        'schema_version': 1, 'evidence_type': 'data-preparation-not-model-performance',
        'dataset_sha256': dataset.sha256, 'annotations': len(records),
        'source_recordings': len({r.source_id for r in records.values()}),
        'official_fold_counts': dict(sorted(Counter(r.outer_split for r in records.values()).items())),
        'annotation_order_anomaly_clip_ids': [r.clip_id for r in records.values() if not r.event_order_valid],
        'seed': seed, 'numpy': np.__version__, 'scipy': scipy.__version__, 'pyav': av.__version__,
        'split_counts': {str(m['outer_fold']): {k: {'clips': len(v['clip_ids']), 'sources': len(v['source_ids'])}
                                              for k, v in m['partitions'].items()} for m in manifests},
        'media_checks': [], 'original_source_image_alignment_verified': False,
        'browser_playback_alignment_verified': False,
    }
    if archive:
        with ZipFile(archive) as zipped:
            media = [i for i in zipped.infolist() if not i.is_dir()]
            expected = {f'videos_160/{i}.mp4' for i in records}
            names = [i.filename for i in media]
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError('Archive must contain exactly one expected MP4 per annotation, no other files')
            if any(i.file_size > 150 * 1024 * 1024 for i in media):
                raise ValueError('Archive member exceeds expected media size budget')
            summary['archive'] = {'sha256': file_hash(archive), 'clips': len(names),
                                  'uncompressed_bytes': sum(i.file_size for i in media),
                                  'scope': 'Inventory and archive hash only; not all clips decoded or CRC checked'}
    if all_clips and clip_ids:
        raise ValueError('Choose all_clips or explicit clip_ids, not both')
    if all_clips:
        clip_ids = tuple(records)
    if clip_ids and not clips_dir:
        raise ValueError('clip_ids requires clips_dir')
    for clip_id in clip_ids:
        if clip_id not in records:
            raise ValueError(f'Unknown clip ID: {clip_id}')
        path = Path(clips_dir) / f'{clip_id}.mp4'
        timeline = inspect_clip_timeline(path, records[clip_id], origin='preprocessed')
        check = asdict(timeline)
        check.update(file_sha256=file_hash(path), frame_count=len(timeline.labels),
                     annotation_event_indices=records[clip_id].event_indices,
                     event_pts_seconds=timeline.event_timestamps())
        summary['media_checks'].append(check)
    # Compute and validate everything before reserving a new artifact directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    for manifest in manifests:
        path = output_dir / f"fold-{manifest['outer_fold']}.json"
        write_split_manifest(manifest, path)
    summary['manifest_sha256'] = {p.name: file_hash(p) for p in sorted(output_dir.glob('fold-*.json'))}
    summary = json.loads(json.dumps(summary, allow_nan=False))
    with (output_dir / 'summary.json').open('x') as output:
        json.dump(summary, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--clips-dir', type=Path)
    parser.add_argument('--clip-id', type=int, action='append', default=[])
    parser.add_argument('--all-clips', action='store_true', help='Decode every annotated preprocessed clip')
    args = parser.parse_args()
    try:
        summary = prepare(args.annotations, args.output_dir, seed=args.seed,
                          validation_fraction=args.validation_fraction, archive=args.archive,
                          clips_dir=args.clips_dir, clip_ids=args.clip_id, all_clips=args.all_clips)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Preparation failed: {exc}\n')
    print(json.dumps({'output_dir': str(args.output_dir), 'annotations': summary['annotations'],
                      'sources': summary['source_recordings'], 'split_counts': summary['split_counts'],
                      'annotation_order_anomaly_clip_ids': summary['annotation_order_anomaly_clip_ids'],
                      'evidence_type': summary['evidence_type']}, indent=2))


if __name__ == '__main__':
    main()
