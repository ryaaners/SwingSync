"""Local research pose evidence; no inference, accuracy claim, or product serving.

Only the clean reference page is suitable for a model-blinded manual annotation.
The diagnostic page intentionally shows predictions and must not be used to label
an independent reference. Both preserve coded pixels and actual source PTS.
"""
import base64
from dataclasses import asdict
import hashlib
import html
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
DERIVATION_FILES = (
    'backend/swingsync/research/pose_evidence.py',
    'scripts/build_pose_evidence.py',
    'backend/swingsync/movement/pose_artifacts.py',
    'backend/swingsync/research/pose_validation.py',
    'backend/swingsync/movement/measurements.py',
    'backend/swingsync/movement/contracts.py',
    'backend/swingsync/pose_runtime/contracts.py',
    'experiments/pose-validation-v1.json',
    'uv.lock',
)
MAX_SELECTED = 12
MAX_SELECTED_PIXELS = 12_000_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
REFERENCE_INDICES = (0, 37, 68, 87, 137)
JOINT_NAMES = ('nose', 'left eye inner', 'left eye', 'left eye outer', 'right eye inner', 'right eye',
               'right eye outer', 'left ear', 'right ear', 'mouth left', 'mouth right', 'left shoulder',
               'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left pinky',
               'right pinky', 'left index', 'right index', 'left thumb', 'right thumb', 'left hip',
               'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle', 'left heel',
               'right heel', 'left foot index', 'right foot index')
EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
         (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
         (28, 30), (30, 32), (0, 7), (0, 8))
LABEL_IDS = {0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24}
CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'"
STYLE = '''body{margin:0;background:#101c23;color:#e7edef;font:16px/1.5 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:32px;margin:0 0 12px}h2{font-size:23px}h3{font-size:18px}p{max-width:90ch}.muted{color:#b5c3cb}.notice{background:#293831;border-left:4px solid #9bd5ac;padding:16px;border-radius:6px}section,.card{background:#1a2a34;border:1px solid #3b4d58;border-radius:12px;padding:20px;margin:22px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{display:block;max-width:100%;height:auto;background:#000;image-rendering:auto}a{color:#a8ded1}nav{display:flex;flex-wrap:wrap;gap:12px}button,input,select,textarea{font:inherit}button{padding:9px 15px;background:#b5e6ce;border:0;border-radius:6px;color:#15251b;cursor:pointer}input,select,textarea{background:#101c23;color:#e7edef;border:1px solid #738792;border-radius:4px;padding:8px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #41515c;padding:5px;text-align:left}code{overflow-wrap:anywhere}.annotation{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:24px}canvas{width:100%;height:auto;display:block;background:#000;cursor:crosshair;touch-action:none}label{display:block;margin:12px 0}.controls{display:flex;gap:12px;flex-wrap:wrap}textarea{width:95%}@media(max-width:760px){main{padding:18px}.pair,.annotation{grid-template-columns:1fr}}'''


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _derivation_identity():
    """Bind this derived workbench to its source and prospective protocol bytes."""
    return {name: _sha(ROOT / name) for name in DERIVATION_FILES}


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c').replace('&', '\\u0026')


def _page(title, body, script=''):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" '
            f'content="{html.escape(CSP, quote=True)}"><title>{html.escape(title)}</title><style>{STYLE}'
            f'</style></head><body><main>{body}</main>{script}</body></html>')


def validate_indices(indices, frame_count, width, height):
    if (not isinstance(indices, (list, tuple)) or not 1 <= len(indices) <= MAX_SELECTED
            or any(type(i) is not int or not 0 <= i < frame_count for i in indices)
            or list(indices) != sorted(set(indices))):
        raise ValueError('Select one to12 distinct, increasing, zero-based decoded frame indices')
    if width * height * len(indices) > MAX_SELECTED_PIXELS:
        raise ValueError('Selected full-resolution images exceed the12-million-pixel review budget')
    return tuple(indices)


def _png(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def _overlay(image, observation):
    from PIL import ImageDraw
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    usable = {}
    if observation['status'] == 'single_pose':
        for index, row in enumerate(observation['landmarks']):
            if (len(row) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in row)
                    and 0 <= row[0] <= 1 and 0 <= row[1] <= 1 and row[3] >= .5):
                usable[index] = (row[0] * width, row[1] * height)
    for a, b in EDGES:
        if a in usable and b in usable:
            draw.line([usable[a], usable[b]], fill='#72ecda', width=max(1, round(width / 160)))
    radius = max(1, width / 160)
    for index, (x, y) in usable.items():
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill='#ffda83', outline='#14232a')
        if index in LABEL_IDS:
            draw.text((x+radius+1, y-radius-1), str(index), fill='#ffffff', stroke_width=1, stroke_fill='#101c23')
    return overlay


def _decode_request(request):
    """Trusted-local bounded full sequential decode, executed in a child process."""
    import av
    from PIL import Image
    path = Path(request['video'])
    frames, geometry = request['frames'], request['geometry']
    wanted = validate_indices(request['indices'], len(frames), geometry['width'], geometry['height'])
    if not 1 <= len(frames) <= 3600 or _sha(path) != request['video_sha256']:
        raise ValueError('Changed source bytes or frame budget')
    formats = {'.mp4': 'mov', '.mov': 'mov', '.webm': 'matroska', '.avi': 'avi'}
    if path.suffix.lower() not in formats:
        raise ValueError('Unsupported local video format')
    images, count = {}, 0
    with av.open(str(path), format=formats[path.suffix.lower()], options={'protocol_whitelist': 'file'}) as video:
        if len(video.streams.video) != 1:
            raise ValueError('Exactly one video stream required')
        stream = video.streams.video[0]
        stream.thread_count = 1
        for index, frame in enumerate(video.decode(stream)):
            if index >= len(frames):
                raise ValueError('Frame count changed')
            saved = frames[index]
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Native frame time is missing')
            if ((frame.pts, frame.time_base.numerator, frame.time_base.denominator)
                    != (saved['pts'], saved['time_base_numerator'], saved['time_base_denominator'])
                    or float(frame.pts * frame.time_base) != saved['pts_seconds']
                    or (frame.width, frame.height) != (geometry['width'], geometry['height'])):
                raise ValueError('Native PTS/raster changed')
            if index in wanted:
                image = Image.fromarray(frame.to_ndarray(format='rgb24'))
                images[str(index)] = {'clean': _png(image), 'overlay': _png(_overlay(image, saved))}
            count += 1
    if count != len(frames) or len(images) != len(wanted) or _sha(path) != request['video_sha256']:
        raise ValueError('Frame correspondence or source bytes changed')
    return images


def render_frames_bounded(artifact, indices):
    request = {'video': str(artifact.video_path), 'video_sha256': artifact.result['video_sha256'],
               'geometry': artifact.result['video'], 'frames': [
                   {k: frame[k] for k in ('pts', 'time_base_numerator', 'time_base_denominator', 'pts_seconds')}
                   | ({'status': frame['status'], 'landmarks': frame['landmarks']} if i in indices else {})
                   for i, frame in enumerate(artifact.result['frames'])], 'indices': list(indices)}
    try:
        result = subprocess.run([sys.executable, '-m', 'backend.swingsync.research.pose_evidence', '--decode'],
                                input=_json(request), capture_output=True, text=True, timeout=45,
                                check=True, cwd=Path(__file__).resolve().parents[3])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        raise ValueError('Canonical pose review decode failed or exceeded45seconds') from exc
    if len(result.stdout) > MAX_OUTPUT_BYTES:
        raise ValueError('Canonical image evidence exceeded the output budget')
    payload = json.loads(result.stdout)
    if set(payload) != {str(i) for i in indices}:
        raise ValueError('Canonical image evidence is incomplete')
    return payload


def _assert_current(artifact):
    from backend.swingsync.movement.pose_artifacts import assert_artifact_unchanged
    assert_artifact_unchanged(artifact)


def manual_measurements(artifact, document):
    """Manual event frame marks only; never infer or import benchmark event labels."""
    if document is None:
        return {'status': 'unavailable', 'reason': 'No manual event annotations supplied', 'measurements': [], 'advice': []}
    if (not isinstance(document, dict) or set(document) != {'schema_version', 'video_sha256', 'events'}
            or document['schema_version'] != 1 or document['video_sha256'] != artifact.result['video_sha256']
            or not isinstance(document['events'], list) or not 1 <= len(document['events']) <= 8):
        raise ValueError('Manual event document must bind this source SHA and contain one to8 manual marks')
    from backend.swingsync.movement.contracts import EventName, PoseEvent
    from backend.swingsync.movement.measurements import extract_measurements
    from backend.swingsync.pose_runtime.contracts import as_pose_sequence
    names, last, events = [n.value for n in EventName], -1, []
    previous_slot = -1
    for item in document['events']:
        if (not isinstance(item, dict) or set(item) != {'name', 'frame_index', 'source'}
                or item['source'] != 'manual' or item['name'] not in names
                or type(item['frame_index']) is not int or not last < item['frame_index'] < len(artifact.result['frames'])
                or names.index(item['name']) <= previous_slot):
            raise ValueError('Manual marks must be unique, canonical, increasing decoded indices with source=manual')
        last, previous_slot = item['frame_index'], names.index(item['name'])
        events.append(PoseEvent(item['name'], artifact.result['frames'][last]['pts_seconds'], source='manual'))
    try:
        report = extract_measurements(as_pose_sequence(artifact.result), events)
    except ValueError as exc:
        return {'status': 'unavailable', 'reason': str(exc), 'measurements': [], 'advice': []}
    return {'status': 'descriptive_unvalidated', **asdict(report)}


def _reference_template(artifact, indices):
    from backend.swingsync.research.pose_validation import make_reference_template
    return make_reference_template(artifact.result, run_sha256=artifact.run_sha256,
                                   result_sha256=artifact.result_sha256)


def annotation_page(template, images):
    """Model coordinates and overlays are intentionally absent from this page."""
    source_images = [images[str(f['frame_index'])]['clean'] for f in template['frames']]
    body = '''<h1>SwingSync · independent reference</h1><p class="notice">Annotate only what you can see. This page contains clean source frames and your own marks. Do not inspect the diagnostic overlay page before annotating. If you have seen model output, leave the blinded/independent declarations unchecked.</p><p class="muted">Research reference collection — no accuracy result or coaching advice. Landmarks refer to the golfer’s anatomy, not screen left/right. Coordinates use continuous image-edge pixels; centers run from (0.5,0.5) to (width−0.5,height−0.5).</p><div class="annotation"><div><div class="controls"><button id="prev">Previous frame</button><button id="next">Next frame</button><span id="progress"></span></div><h2 id="frame-title"></h2><canvas id="canvas"></canvas></div><aside class="card"><label>Landmark <select id="joint"></select></label><label>Status <select id="status"><option value="unreviewed">Unreviewed</option><option value="visible">Visible — click its location</option><option value="occluded">Occluded — no coordinate</option><option value="out_of_frame">Out of frame — no coordinate</option></select></label><p id="coordinate">No coordinate</p><button id="clear">Clear this landmark</button><p>Click a visible landmark in the image to save its location. Occluded and out-of-frame marks require no guessed point. Changes are kept in this tab only: export before closing or reloading.</p></aside></div><section><h2>Reviewer and export</h2><label>Reviewer ID <input id="reviewer" autocomplete="off"></label><label>Notes <textarea id="notes" rows="3"></textarea></label><label><input id="blinded" type="checkbox"> I have not inspected model coordinates or overlays for these frames.</label><label><input id="independent" type="checkbox"> These are independently placed manual reference marks.</label><button id="export">Export reference JSON</button><p id="export-note" class="muted">Drafts may be exported with unreviewed landmarks and unchecked declarations. They do not qualify for accuracy evaluation.</p></section>'''
    script = '<script>\nconst reference=' + _json(template) + ';\nconst cleanImages=' + _json(source_images) + ';\nconst names=' + _json(JOINT_NAMES) + r''';
+let framePosition=0, currentImage=null, loadToken=0;
+const byId=id=>document.getElementById(id), canvas=byId('canvas'), context=canvas.getContext('2d');
+names.forEach((name,i)=>{const option=document.createElement('option');option.value=i;option.textContent=i+' · '+name;byId('joint').append(option)});
+function row(){return reference.frames[framePosition].landmarks[Number(byId('joint').value)]}
+function draw(){if(!currentImage)return;context.clearRect(0,0,canvas.width,canvas.height);context.drawImage(currentImage,0,0);reference.frames[framePosition].landmarks.forEach(p=>{if(p.status==='visible'&&p.x_px!==null){context.beginPath();context.arc(p.x_px,p.y_px,Math.max(1,canvas.width/160),0,Math.PI*2);context.fillStyle='#ffe18b';context.fill();context.fillStyle='#17252b';context.font=Math.max(9,canvas.width/45)+'px sans-serif';context.fillText(String(p.landmark_id),p.x_px+3,p.y_px-3)}})}
+function refreshJoint(){const p=row();byId('status').value=p.status;byId('coordinate').textContent=p.x_px===null?'No coordinate':p.x_px.toFixed(2)+', '+p.y_px.toFixed(2)+' px';draw()}
+function showFrame(){const requestToken=++loadToken, frame=reference.frames[framePosition];byId('frame-title').textContent='Frame '+(frame.frame_index+1)+' · decoded index '+frame.frame_index+' · native '+frame.pts_seconds.toFixed(9)+' s';byId('progress').textContent=(framePosition+1)+' / '+reference.frames.length;byId('prev').disabled=framePosition===0;byId('next').disabled=framePosition===reference.frames.length-1;canvas.width=frame.width;canvas.height=frame.height;currentImage=null;const image=new Image();image.onload=()=>{if(requestToken!==loadToken)return;currentImage=image;refreshJoint()};image.src='data:image/png;base64,'+cleanImages[framePosition]}
+byId('prev').onclick=()=>{if(framePosition>0){framePosition--;showFrame()}};byId('next').onclick=()=>{if(framePosition+1<reference.frames.length){framePosition++;showFrame()}};
+byId('joint').onchange=refreshJoint;byId('status').onchange=()=>{const p=row();p.status=byId('status').value;if(p.status!=='visible'){p.x_px=null;p.y_px=null}refreshJoint()};
+canvas.addEventListener('pointerdown',event=>{if(!currentImage)return;const rect=canvas.getBoundingClientRect(),p=row();p.status='visible';p.x_px=Math.max(0,Math.min(canvas.width,(event.clientX-rect.left)*canvas.width/rect.width));p.y_px=Math.max(0,Math.min(canvas.height,(event.clientY-rect.top)*canvas.height/rect.height));refreshJoint()});
+byId('clear').onclick=()=>{Object.assign(row(),{status:'unreviewed',x_px:null,y_px:null});refreshJoint()};
+byId('export').onclick=()=>{if(reference.frames.some(f=>f.landmarks.some(p=>p.status==='visible'&&(p.x_px===null||p.y_px===null)))){byId('export-note').textContent='Every visible landmark needs a clicked coordinate. Mark it unreviewed if uncertain.';return}reference.reviewer={id:byId('reviewer').value.trim(),notes:byId('notes').value,model_blinded:byId('blinded').checked,independent:byId('independent').checked};const blob=new Blob([JSON.stringify(reference,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='swingsync-pose-reference.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);byId('export-note').textContent='Reference JSON exported. This is an annotation, not an accuracy result.'};showFrame();
+</script>'''.replace('\n+', '\n')
    return _page('SwingSync · clean reference annotation', body, script)


def review_page(artifact, indices, images, measurements):
    result = artifact.result
    navigation = ' '.join(f'<a href="#frame-{i}">Frame{i+1}</a>' for i in indices)
    body = ('<h1>SwingSync · pose evidence review</h1><p class="notice">Diagnostic model overlay — unvalidated. '
            'This is not an independent reference annotation. Use the separate clean reference page with a blinded reviewer. '
            'No coaching advice or shot prediction is produced.</p><nav>' + navigation + '</nav>'
            '<section><h2>Recording and limits</h2><p>Camera, handedness, mirroring and single-golfer identity are operator '
            'declarations, not verified model reliability. Single-pose returns do not prove identity continuity. '
            'Native recording PTS below are distinct from the rounded API clock. Coordinates describe the coded upright '
            'image only; no calibrated depth or 3D angles.</p><p>Source SHA256: <code>' + html.escape(result['video_sha256'])
            + '</code></p><p>Context: ' + html.escape(json.dumps(result['context'], sort_keys=True)) + '</p>'
            '<p>Overlay markers use filtered normalized x×width / y×height, visibility ≥0.5, and in-image coordinates. '
            'Presence/visibility filtering is an engineering policy, not validated accuracy. Missing/malformed/multiple '
            'pose rows produce no markers. Body lines are visual aids, not measured bones.</p></section>')
    if measurements['status'] == 'descriptive_unvalidated':
        body += '<section><h2>Manual-event descriptive measurements · unvalidated</h2>'
        for row in measurements['measurements']:
            value = str(row['value']) + ' ' + row['unit'] if row['status'] == 'measured' else 'Unavailable: ' + str(row['reason'])
            body += '<h3>' + html.escape(row['name']) + '</h3><p>' + html.escape(value) + '</p><p class="muted">' + html.escape(row['method']) + '</p>'
        body += '<p>No ideal thresholds, practice advice, causal interpretation or validated reliability.</p></section>'
    else:
        body += '<section><h2>Measurements</h2><p>' + html.escape(measurements['reason']) + '</p></section>'
    for i in indices:
        row, imgs = result['frames'][i], images[str(i)]
        body += f'<section id="frame-{i}"><h2>Frame {i+1} · decoded index {i}</h2>'
        body += f'<p>Native PTS {row["pts_seconds"]:.9f} s · API {row["api_timestamp_ms"]} ms · {html.escape(row["status"])}</p>'
        body += ('<div class="pair"><div><h3>Canonical source</h3><img alt="Canonical decoded frame" src="data:image/png;base64,' + imgs['clean']
                 + '"></div><div><h3>Unvalidated projected overlay</h3><img alt="Projected pose diagnostic" src="data:image/png;base64,' + imgs['overlay'] + '"></div></div>')
        body += '<details><summary>Raw normalized landmark scores · anatomical IDs</summary><table><tr><th>ID</th><th>Landmark</th><th>x</th><th>y</th><th>Visibility</th><th>Presence</th></tr>'
        for pose in row['raw_poses']:
            for j, point in enumerate(pose):
                label = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f'Unknown landmark index {j}'
                body += '<tr><td>' + str(j) + '</td><td>' + label + '</td>'
                body += ''.join('<td>' + html.escape(str(point.get(key))) + '</td>' for key in ('x', 'y', 'visibility', 'presence')) + '</tr>'
        body += '</table></details></section>'
    return _page('SwingSync · diagnostic pose evidence', body)


def build_pose_evidence(artifact, frame_indices, output, *, manual_events=None):
    """Publish canonical review pages from an admitted artifact; never rerun pose."""
    derivation = _derivation_identity()
    _assert_current(artifact)
    result = artifact.result
    indices = validate_indices(frame_indices, len(result['frames']), result['video']['width'], result['video']['height'])
    if indices != REFERENCE_INDICES:
        raise ValueError('Reference workbench v1 requires the prospectively fixed indices0,37,68,87,137')
    measurements = manual_measurements(artifact, manual_events)
    images = render_frames_bounded(artifact, indices)
    template = _reference_template(artifact, indices)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        payloads = {'review.html': review_page(artifact, indices, images, measurements),
                    'reference.html': annotation_page(template, images),
                    'reference-template.json': json.dumps(template, indent=2, allow_nan=False) + '\n'}
        for name, contents in payloads.items():
            (output / name).write_text(contents, encoding='utf-8')
        _assert_current(artifact)
        if _derivation_identity() != derivation:
            raise ValueError('Pose review derivation source/protocol drift during rendering')
        manifest = {'schema_version': 1, 'state': 'completed', 'study_id': 'pose-validation-v1',
                    'video_sha256': result['video_sha256'], 'run_sha256': artifact.run_sha256,
                    'result_sha256': artifact.result_sha256, 'selected_frame_indices': list(indices),
                    'coordinate_system': 'image-edge-pixels-v1',
                    'protocol_sha256': derivation['experiments/pose-validation-v1.json'],
                    'derivation_source_sha256': derivation,
                    'files': {name: _sha(output/name) for name in payloads},
                    'measurements': measurements, 'advice': [], 'validated_for_coaching': False,
                    'accuracy_evaluation_completed': False,
                    'limitations': ['Reference annotations are unreviewed templates, not ground truth',
                                    'Diagnostic overlays reveal model output; references must be independent and blinded',
                                    'Operator metadata/identity and golf coordinate accuracy remain unvalidated']}
        with (output/'manifest.json').open('x') as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return manifest
    except BaseException:
        shutil.rmtree(output)
        raise


if __name__ == '__main__':
    if sys.argv[1:] != ['--decode']:
        raise SystemExit('Use scripts/build_pose_evidence.py for verified evidence publication')
    try:
        request = json.loads(sys.stdin.read(8 * 1024 * 1024))
        print(json.dumps(_decode_request(request), allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
