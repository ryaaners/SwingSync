"""Synthetic preparation jobs test resume/integrity, not learned-model quality."""
import fcntl
from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np
import pytest
from test_golfdb import write_mat

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')

from backend.swingsync.research import cache_job  # noqa: E402
from backend.swingsync.research.cache_job import run_cache_job  # noqa: E402
from backend.swingsync.research.features import extract_clip_features  # noqa: E402
from backend.swingsync.research.golfdb import load_annotations, make_split_manifest  # noqa: E402


class TinyFrozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, images):
        return images.float().mean(dim=(-2, -1)) * self.scale


@pytest.fixture
def setup_job(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_job, '_git_base_commit', lambda: 'fixture-revision')
    annotations = write_mat(tmp_path / 'annotations.mat', lambda i: {'events': np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])})
    dataset = load_annotations(annotations)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(make_split_manifest(dataset, 1)))
    clips = tmp_path / 'clips'
    clips.mkdir()
    example = tmp_path / 'example.mp4'
    with av.open(str(example), 'w') as c:
        stream = c.add_stream('mpeg4', rate=25)
        stream.width = stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for i in range(9):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 15, np.uint8), format='rgb24')
            frame.pts, frame.time_base = i, Fraction(1, 25)
            for packet in stream.encode(frame):
                c.mux(packet)
        for packet in stream.encode():
            c.mux(packet)
    for record in dataset.records:
        (clips / f'{record.clip_id}.mp4').write_bytes(example.read_bytes())
    calls = []

    def extractor(*args, **kwargs):
        calls.append(args[1].clip_id)
        return extract_clip_features(*args, **kwargs)

    options = {'annotations': annotations, 'manifest': manifest, 'clips_dir': clips,
               'job_dir': tmp_path / 'job', 'cache_dir': tmp_path / 'cache',
               'backbone': TinyFrozen(), 'extractor': extractor, 'lock_path': tmp_path / 'active.lock'}
    return options, calls


def test_partial_resume_revalidates_without_duplicate_extraction(setup_job):
    options, calls = setup_job
    state = run_cache_job(**options, max_new_clips=2)
    assert state['state'] == 'partial'
    assert state['target_clip_count'] == 12
    assert state['completed_clip_count'] == 2
    assert state['git_base_commit'] == 'fixture-revision'
    assert not state['outer_test_selected']
    assert len(calls) == 2
    first_completed = set(state['completed'])
    state = run_cache_job(**options, max_new_clips=1, resume=True)
    assert state['completed_clip_count'] == 3
    assert state['attempts'][-1]['newly_extracted'] == 1
    assert first_completed <= set(state['completed'])
    assert len(calls) == len(set(calls)) == 3
    state = run_cache_job(**options, resume=True)
    assert state['state'] == 'completed'
    assert state['completed_clip_count'] == state['target_clip_count'] == 12
    assert len(calls) == len(set(calls)) == 12
    completed = run_cache_job(**options, resume=True, max_new_clips=0)
    assert completed['state'] == 'completed'
    assert completed['attempts'][-1]['newly_extracted'] == 0
    assert len(calls) == 12
    assert all(row['partition'] in {'train', 'validation'} for row in completed['completed'].values())


def test_existing_valid_shared_caches_recovered_without_ledger(setup_job):
    options, calls = setup_job
    first = run_cache_job(**options, max_new_clips=2)
    second_options = dict(options, job_dir=options['job_dir'].with_name('second-job'))
    second = run_cache_job(**second_options, max_new_clips=0)
    assert second['state'] == 'partial'
    assert second['completed_clip_count'] == 2
    assert second['attempts'][-1]['reused_cache'] == 2
    assert second['attempts'][-1]['newly_extracted'] == 0
    assert set(second['completed']) == set(first['completed'])
    assert len(calls) == 2


def test_corrupt_completed_cache_fails_closed_preserves_file(setup_job):
    options, calls = setup_job
    state = run_cache_job(**options, max_new_clips=1)
    entry = next(iter(state['completed'].values()))
    path = Path(entry['cache_path'])
    path.write_bytes(b'corrupt artifact')
    with pytest.raises(ValueError, match='corrupt'):
        run_cache_job(**options, resume=True)
    assert path.read_bytes() == b'corrupt artifact'
    assert len(calls) == 1
    assert json.loads((options['job_dir'] / 'job.json').read_text())['state'] == 'failed'


def test_changed_media_manifest_or_backbone_cannot_resume(setup_job):
    options, _ = setup_job
    state = run_cache_job(**options, max_new_clips=1)
    with pytest.raises(ValueError, match='provenance changed'):
        run_cache_job(**options, resume=True, batch_size=8)
    changed_backbone = TinyFrozen()
    changed_backbone.scale += 1
    with pytest.raises(ValueError, match='backbone'):
        run_cache_job(**dict(options, backbone=changed_backbone), resume=True)
    media = options['clips_dir'] / f"{state['config']['inventory'][0]['clip_id']}.mp4"
    media.write_bytes(media.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='provenance changed'):
        run_cache_job(**options, resume=True)


def test_manifest_tampering_and_outer_test_require_explicit_opt_in(setup_job):
    options, calls = setup_job
    with pytest.raises(ValueError, match='allow_test'):
        run_cache_job(**options, partitions=('test',))
    assert not options['job_dir'].exists()
    manifest = json.loads(options['manifest'].read_text())
    manifest['partitions']['train']['clip_ids'][0] = manifest['partitions']['test']['clip_ids'][0]
    options['manifest'].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest does not match'):
        run_cache_job(**options)
    assert not calls
    assert not options['job_dir'].exists()


def test_existing_state_and_global_lock_prevent_duplicate_job(setup_job):
    options, calls = setup_job
    with options['lock_path'].open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='shared expensive-work lock'):
            run_cache_job(**options)
    run_cache_job(**options, max_new_clips=0)
    with pytest.raises(ValueError, match='explicitly resume'):
        run_cache_job(**options)
    assert not calls


def test_interrupt_keeps_partial_progress_and_can_resume(setup_job):
    options, calls = setup_job
    fail = [True]

    def interrupted(*args, **kwargs):
        if len(calls) == 1 and fail[0]:
            fail[0] = False
            raise KeyboardInterrupt('software fixture interruption')
        calls.append(args[1].clip_id)
        return extract_clip_features(*args, **kwargs)

    options['extractor'] = interrupted
    with pytest.raises(KeyboardInterrupt):
        run_cache_job(**options)
    state = json.loads((options['job_dir'] / 'job.json').read_text())
    assert state['state'] == 'interrupted'
    assert state['completed_clip_count'] == 1
    run_cache_job(**options, resume=True)
    assert len(calls) == len(set(calls)) == 12


def test_resume_rejects_changed_completed_partition_metadata(setup_job):
    options, calls = setup_job
    state = run_cache_job(**options, max_new_clips=1)
    next(iter(state['completed'].values()))['partition'] = 'test'
    (options['job_dir'] / 'job.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='ledger semantics'):
        run_cache_job(**options, resume=True)
    assert len(calls) == 1


def test_queued_media_change_after_initial_inventory_rejected(setup_job):
    options, calls = setup_job
    manifest = json.loads(options['manifest'].read_text())
    target_ids = sorted(manifest['partitions']['train']['clip_ids'] + manifest['partitions']['validation']['clip_ids'])

    def mutate_later(*args, **kwargs):
        result = extract_clip_features(*args, **kwargs)
        calls.append(args[1].clip_id)
        if len(calls) == 1:
            later = options['clips_dir'] / f'{target_ids[1]}.mp4'
            later.write_bytes(later.read_bytes() + b'queued media modified')
        return result

    options['extractor'] = mutate_later
    with pytest.raises(ValueError, match='Queued media changed'):
        run_cache_job(**options)
    state = json.loads((options['job_dir'] / 'job.json').read_text())
    assert state['state'] == 'failed'
    assert state['completed_clip_count'] == 1
    assert calls == [target_ids[0]]


def test_running_record_requires_stopped_owner_before_resume(setup_job, monkeypatch):
    options, calls = setup_job
    state = run_cache_job(**options, max_new_clips=1)
    state['state'] = 'running'
    state['process_start_identity'] = None
    (options['job_dir'] / 'job.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='still alive or unverifiable'):
        run_cache_job(**options, resume=True)

    def owner_gone(*args):
        raise ProcessLookupError('fixture owner stopped')

    monkeypatch.setattr(cache_job.os, 'kill', owner_gone)
    resumed = run_cache_job(**options, resume=True, max_new_clips=0)
    assert resumed['state'] == 'partial'
    assert resumed['completed_clip_count'] == 1
    assert len(calls) == 1


def test_null_completed_entry_never_counts_as_finished(setup_job):
    options, calls = setup_job
    state = run_cache_job(**options, max_new_clips=1)
    first_key = next(iter(state['completed']))
    state['completed'][first_key] = None
    (options['job_dir'] / 'job.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='Completed ledger'):
        run_cache_job(**options, resume=True)
    assert len(calls) == 1
