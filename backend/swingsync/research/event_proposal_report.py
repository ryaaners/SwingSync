"""Authenticated offline event evidence; no inference, accuracy or model promotion."""
from dataclasses import dataclass
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

from backend.swingsync.frame_images import extract_frames_bounded
from .evaluation import EVENT_NAMES
from .event_input_contract import validate_metadata, validate_protocol

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_SHA = '4b4339a469dd5fe09f97b57ae4b650ce378ce1c3a3863f86b46249ad8d9d0008'
MAX_VIDEO = 20 * 1024 * 1024
DERIVATION_FILES = ('backend/swingsync/research/event_proposal_report.py', 'scripts/report_event_proposals.py',
                    'backend/swingsync/frame_images.py', 'backend/swingsync/research/evaluation.py',
                    'backend/swingsync/research/event_input_contract.py', 'uv.lock')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json(path, digest=None):
    path = Path(path)
    if not 0 < path.stat().st_size <= 8 * 1024 * 1024:
        raise ValueError('Event artifact JSON exceeds review budget')
    raw = path.read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Event artifact SHA256 mismatch')
    def unique(pairs):
        value = {}
        for k, v in pairs:
            if k in value:
                raise ValueError('Duplicate JSON key')
            value[k] = v
        return value
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError('Event artifact must be an object')
    return value


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Independently recorded lowercase SHA256 required')
    return value


def _inspect(video):
    code = ('import json,sys;from backend.swingsync.research.event_input_contract import inspect_input;'
            'print(json.dumps(inspect_input(sys.argv[1]),allow_nan=False))')
    try:
        p = subprocess.run([sys.executable, '-c', code, str(video)], cwd=ROOT, timeout=45,
                           capture_output=True, text=True, check=True)
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError('Bounded event source reinspection failed') from exc


def _provenance(run):
    identity = run['identity']
    contract_path = ROOT / 'experiments/heldout-v1.json'
    contract = _json(contract_path, CONTRACT_SHA)
    if (identity['heldout_contract_sha256'] != CONTRACT_SHA
            or identity['feature_pipeline'] != contract['feature_pipeline']
            or identity['backbone_weight_sha256'] != '7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f'
            or identity['context'] != 'reference-64-reset' or identity['batch_size'] != 16):
        raise ValueError('Event model/pipeline differs from fixed registered research artifacts')
    expected_paths = ('backend/swingsync/research/event_input.py',
                      'backend/swingsync/research/event_input_contract.py',
                      'scripts/propose_events.py', 'experiments/event-input-v1.json')
    validate_protocol(ROOT/'experiments/event-input-v1.json')
    if (identity['protocol_sha256'] != sha(ROOT/'experiments/event-input-v1.json')
            or identity['selected_heads'] != contract['heads']
            or identity['video_sha256'] != run['video_sha256']):
        raise ValueError('Event protocol/head/source registration differs')
    if identity['source_sha256'] != {p: sha(ROOT/p) for p in expected_paths}:
        raise ValueError('Event proposal source/protocol drift')
    admission = identity['selected_head_admission']
    if (admission['contract_sha256'] != CONTRACT_SHA
            or admission['dependency_lock_sha256'] != sha(ROOT/'uv.lock')
            or admission['dependency_lock_sha256'] != contract['dependency_lock_sha256']
            or admission['evaluator_sha256'] != sha(ROOT/'backend/swingsync/research/heldout.py')
            or admission['evaluation_cli_sha256'] != sha(ROOT/'scripts/evaluate_event_heads.py')):
        raise ValueError('Selected-head admission source/lock drift')
    expected_training = {'experiment.py', 'models.py', 'training.py', 'evaluation.py',
                         'features.py', 'golfdb.py', 'timeline.py'}
    if (set(admission['training_implementation_sha256']) != expected_training
            or admission['manifest_sha256'] != contract['manifest_sha256']
            or admission['dataset_sha256'] != contract['dataset_sha256']
            or identity['manifest_sha256'] != contract['manifest_sha256']
            or identity['annotations_sha256'] != contract['dataset_sha256']):
        raise ValueError('Training/data/split provenance coverage differs')
    for name, digest in admission['training_implementation_sha256'].items():
        if Path(name).name != name or sha(ROOT/'backend/swingsync/research'/name) != digest:
            raise ValueError('Frozen training source drift')
    c = run['context']
    expected = validate_metadata(c['camera_view'], c['handedness'], c['club'], c['mirrored'] is False,
                                c['single_swing_confirmed_by_operator'], c['rights_provenance'], c['timestamp_basis'])
    if c != expected:
        raise ValueError('Operator metadata/admission flags differ')


@dataclass(frozen=True)
class EventEvidence:
    run: dict
    result: dict
    run_path: Path
    video: Path
    run_sha256: str
    arrays_sha256: str


def assert_unchanged(evidence):
    run = _json(evidence.run_path, evidence.run_sha256)
    if (run != evidence.run or _json(evidence.run_path.with_name('result.json'), run['result_sha256']) != evidence.result
            or sha(evidence.run_path.with_name('arrays.npz')) != evidence.arrays_sha256
            or sha(evidence.video) != run['video_sha256']):
        raise ValueError('Event evidence snapshot/source drift')
    _provenance(run)


def load_event_evidence(run_path, *, expected_run_sha256, video):
    run_path, video = Path(run_path).resolve(), Path(video).resolve()
    run = _json(run_path, _digest(expected_run_sha256))
    if run.get('schema_version') != 1 or run.get('state') != 'completed':
        raise ValueError('Completed event observation ledger required')
    result = _json(run_path.with_name('result.json'), _digest(run['result_sha256']))
    if result.get('state') != 'completed' or result.get('schema_version') != 1:
        raise ValueError('Completed event observation result required')
    for key in ('job_id', 'context', 'identity', 'video_sha256', 'video', 'completed_at',
                'normal_model_available', 'validated_for_serving', 'accuracy_metrics_computed', 'advice'):
        if result.get(key) != run.get(key):
            raise ValueError('Event ledger/result mismatch')
    for doc in (run, result):
        if (doc.get('normal_model_available') is not False or doc.get('validated_for_serving') is not False
                or doc.get('accuracy_metrics_computed') is not False or doc.get('advice') != []):
            raise ValueError('Research artifact cannot claim model/accuracy/coaching availability')
    if (result.get('primary_head') != 'temporal' or result.get('temporal_context') != 'reference-64-reset'
            or result.get('order_constrained_decoding') is not False or result.get('scores_are_calibrated') is not False
            or result.get('domain_compatibility_validated') is not False
            or result.get('outcome_integration_permitted') is not False
            or result.get('physical_timing_validated') is not False):
        raise ValueError('Research proposal decoding/admission policy differs')
    _provenance(run)
    if sha(video) != run['video_sha256']:
        raise ValueError('Recording SHA256 differs')
    inspection = _inspect(video)
    if (inspection['frames'] != result['frames']
            or {k: v for k, v in inspection.items() if k != 'frames'} != result['video']):
        raise ValueError('Native source PTS/raster correspondence differs')
    for frame in result['frames']:
        if any(type(frame[k]) is not (float if k == 'pts_seconds' else int)
               for k in ('frame_index', 'pts', 'time_base_numerator', 'time_base_denominator', 'pts_seconds')):
            raise ValueError('Native frame identity types differ')
    n = len(result['frames'])
    array_path = run_path.with_name('arrays.npz')
    array_sha = _digest(run['arrays_sha256'])
    if array_path.stat().st_size > 32*1024*1024:
        raise ValueError('Event array archive exceeds compressed size budget')
    if result.get('arrays_file') != 'arrays.npz' or result.get('arrays_sha256') != array_sha or sha(array_path) != array_sha:
        raise ValueError('Saved event arrays hash differs')
    with zipfile.ZipFile(array_path) as archive:
        names = archive.namelist()
        if (len(names) != 4 or set(names) != {'features.npy', 'frame_probabilities.npy', 'temporal_probabilities.npy', 'pts_seconds.npy'}
                or sum(v.file_size for v in archive.infolist()) > 24*1024*1024):
            raise ValueError('Event array archive exceeds exact schema/budget')
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if (arrays['features'].dtype != np.float32 or arrays['features'].shape != (n, 1280)
            or not np.isfinite(arrays['features']).all() or arrays['pts_seconds'].dtype != np.float64
            or not np.array_equal(arrays['pts_seconds'], [f['pts_seconds'] for f in result['frames']])):
        raise ValueError('Saved features/PTS geometry differs')
    if set(result['heads']) != {'frame', 'temporal'}:
        raise ValueError('Both fixed selected heads required')
    for head in ('frame', 'temporal'):
        probs = arrays[head+'_probabilities']
        if (probs.dtype != np.float32 or probs.shape != (n, 9) or not np.isfinite(probs).all()
                or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1, rtol=0, atol=1e-6)):
            raise ValueError('Saved head probabilities malformed')
        peaks = np.argsort(probs[:, :8], axis=0)[-1]
        rows = result['heads'][head]['events']
        if len(rows) != 8:
            raise ValueError('Eight independently selected class peaks required')
        for label, (index, row) in enumerate(zip(peaks.tolist(), rows)):
            frame = result['frames'][index]
            expected = {'name': EVENT_NAMES[label], 'class_index': label, 'frame_index': index,
                        **{k: frame[k] for k in ('pts_seconds', 'pts', 'time_base_numerator', 'time_base_denominator')},
                        'score': float(probs[index, label]), 'background_probability': float(probs[index, 8])}
            if row != expected or type(row['frame_index']) is not int or type(row['class_index']) is not int:
                raise ValueError('Saved proposal/peak/timestamp arithmetic differs')
        saved = result['heads'][head]
        if (saved['strict_order'] is not all(a < b for a, b in zip(peaks, peaks[1:]))
                or saved['duplicate_peak_count'] != 8-len(set(peaks.tolist()))):
            raise ValueError('Saved peak-order diagnostics differ')
    evidence = EventEvidence(run, result, run_path, video, expected_run_sha256, array_sha)
    assert_unchanged(evidence)
    return evidence


def build_report(evidence, output):
    """Publish a local diagnostic timeline; browser seeks are explicitly approximate."""
    assert_unchanged(evidence)
    sources = {p: sha(ROOT/p) for p in DERIVATION_FILES}
    if not 0 < evidence.video.stat().st_size <= MAX_VIDEO:
        raise ValueError('Embedded diagnostic recording exceeds20MiB; use a smaller research copy')
    result = evidence.result
    events = result['heads']['temporal']['events']
    indices = sorted(set(e['frame_index'] for e in events))
    info = {'frame_count': len(result['frames']), 'source_timestamps_seconds': [f['pts_seconds'] for f in result['frames']],
            'sha256': result['video_sha256']}
    images = extract_frames_bounded(evidence.video, info, indices)
    media = evidence.video.read_bytes()
    if hashlib.sha256(media).hexdigest() != result['video_sha256']:
        raise ValueError('Video changed during embedding')
    mime = {'.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm', '.avi': 'video/x-msvideo'}[evidence.video.suffix.lower()]
    primary = result['heads']['temporal']
    body = '<h1>SwingSync · research event proposals</h1><p class="notice">Unvalidated proposals for review. These are eight forced independent class peaks, not confirmed events. No accuracy, timing measurements, pose, shot prediction or coaching advice.</p>'
    body += '<p>Class scores are uncalibrated. Background may exceed an event score. Peaks may duplicate or reverse order; no decoder repair is applied. Operator view/hand/mirror/club/single-swing declarations do not validate recording compatibility.</p>'
    body += '<video controls preload="metadata" id="video" src="data:'+mime+';base64,'+base64.b64encode(media).decode()+'"></video><p id="playback">Playback jumps are approximate and unverified; decoded canonical frames below are authoritative.</p><nav>'
    origin = result['frames'][0]['pts_seconds']
    for e in events:
        body += '<button data-time="'+str(e['pts_seconds']-origin)+'" data-target="event-'+str(e['class_index'])+'">'+html.escape(e['name'])+'</button> '
    body += '</nav><p>Strict event order: '+str(primary['strict_order'])+' · duplicate peaks: '+str(primary['duplicate_peak_count'])+'</p>'
    body += '<details><summary>Recording and preprocessing provenance</summary><pre>'+html.escape(json.dumps({'source_sha256':result['video_sha256'],'context':result['context'],'video':result['video'],'run_sha256':evidence.run_sha256},indent=2))+'</pre></details>'
    for e in events:
        row = str(e['frame_index'])
        body += '<section id="event-'+str(e['class_index'])+'"><h2>'+html.escape(e['name'])+' · proposal</h2>'
        body += '<p>Decoded frame '+str(e['frame_index']+1)+' (index '+row+') · native PTS '+format(e['pts_seconds'],'.9f')+'s</p>'
        body += '<p>Uncalibrated class score '+format(e['score'],'.4f')+' · background '+format(e['background_probability'],'.4f')+'</p>'
        body += '<img alt="Canonical '+html.escape(e['name'],quote=True)+' proposal frame" src="data:image/png;base64,'+images[row]+'"></section>'
    body += '<details><summary>Independent frame-head comparison (research diagnostic)</summary><pre>'+html.escape(json.dumps(result['heads']['frame'],indent=2))+'</pre></details>'
    script = '''<script>const v=document.getElementById('video'),status=document.getElementById('playback');document.querySelectorAll('button[data-time]').forEach(b=>b.onclick=()=>{const seek=()=>{v.currentTime=Number(b.dataset.time);v.pause();status.textContent='Approximate playback jump requested. Canonical decoded frame remains evidence.'};if(v.readyState)seek();else v.addEventListener('loadedmetadata',seek,{once:true});document.getElementById(b.dataset.target).scrollIntoView({behavior:'smooth'})});v.onerror=()=>{status.textContent='Browser cannot play this codec. Canonical decoded frames remain available.'};</script>'''
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src &#x27;none&#x27;; img-src data:; media-src data:; script-src &#x27;unsafe-inline&#x27;; style-src &#x27;unsafe-inline&#x27;; connect-src &#x27;none&#x27;; base-uri &#x27;none&#x27;; form-action &#x27;none&#x27;"><title>SwingSync · event evidence</title><style>body{background:#13232b;color:#e3ecec;font:16px/1.5 system-ui;margin:0}main{max-width:1000px;margin:auto;padding:24px}section{border:1px solid #58716a;border-radius:12px;padding:20px;margin:20px 0}.notice{border-left:4px solid #a7d9b7;background:#263b31;padding:16px}video{max-width:100%;max-height:440px;width:100%;background:#000}img{max-width:100%;height:auto}nav{display:flex;flex-wrap:wrap;gap:8px}button{font:inherit;background:#b1dfc5;color:#10251a;border:0;border-radius:7px;padding:8px;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}h1{font-size:30px}</style></head><body><main>'+body+'</main>'+script+'</body></html>'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output/'review.html').write_text(page)
        assert_unchanged(evidence)
        if sources != {p: sha(ROOT/p) for p in DERIVATION_FILES}:
            raise ValueError('Event report source drift')
        manifest = {'schema_version':1,'state':'completed','run_sha256':evidence.run_sha256,
                    'source_video_sha256':result['video_sha256'],'arrays_sha256':evidence.arrays_sha256,
                    'files':{'review.html':sha(output/'review.html')},'derivation_source_sha256':sources,
                    'primary_head':'temporal','event_count':8,'canonical_frame_indices':indices,
                    'browser_time_alignment_validated':False,'normal_model_available':False,
                    'accuracy_metrics_computed':False,'validated_for_serving':False,'advice':[]}
        with (output/'manifest.json').open('x') as f:
            json.dump(manifest,f,indent=2,allow_nan=False)
            f.write('\n')
        assert_unchanged(evidence)
        if sources != {p: sha(ROOT/p) for p in DERIVATION_FILES}:
            raise ValueError('Event report source drift during manifest publication')
        return manifest
    except BaseException:
        shutil.rmtree(output)
        raise
