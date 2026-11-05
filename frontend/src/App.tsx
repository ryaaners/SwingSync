import { useEffect, useRef, useState, type FormEvent } from 'react';

type Analysis = {
  id: string; status: 'queued' | 'processing' | 'completed' | 'failed' | 'interrupted'; progress: number;
  error?: string; created_at?: string;
  metadata?: { camera_view?: string; handedness?: string; club?: string; mirrored?: boolean; original_filename?: string };
  video?: { duration_seconds: number; frame_count: number; width: number; height: number; timestamps_seconds?: number[] };
  model?: { status: string; reason?: string };
};
const active = (item: Analysis) => ['queued', 'processing'].includes(item.status);
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Could not complete the request (${response.status}).`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
const label = (item: Analysis) => item.metadata?.original_filename || `Swing ${item.id.slice(0, 8)}`;

export default function App() {
  const [items, setItems] = useState<Analysis[]>([]);
  const [selected, setSelected] = useState<Analysis | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [connected, setConnected] = useState(false);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const selectionId = useRef<string | null>(null);
  const [view, setView] = useState('face-on');
  const [handedness, setHandedness] = useState('right');
  const [club, setClub] = useState('7 iron');
  const [mirrored, setMirrored] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      try {
        const result = await request<Analysis[]>('/api/analyses');
        if (cancelled) return;
        setConnected(true); setItems(result);
        if (selectionId.current) {
          const id = selectionId.current;
          const detail = await request<Analysis>(`/api/analyses/${encodeURIComponent(id)}`);
          if (!cancelled && selectionId.current === id) setSelected(detail);
        }
      } catch { if (!cancelled) setConnected(false); }
    };
    void refresh();
    const interval = window.setInterval(refresh, 1800);
    return () => { cancelled = true; window.clearInterval(interval); };
  }, []);

  async function choose(item: Analysis) {
    selectionId.current = item.id; setSelected(item); setError('');
    try {
      const detail = await request<Analysis>(`/api/analyses/${encodeURIComponent(item.id)}`);
      if (selectionId.current === item.id) setSelected(detail);
    } catch (e) { setError((e as Error).message); }
  }
  function acceptFile(value?: File) {
    if (!value) return;
    if (value.size > 150 * 1024 * 1024) { setError('Choose a video smaller than 150 MB.'); return; }
    setFile(value); setError('');
  }
  async function submit(event: FormEvent) {
    event.preventDefault(); if (!file || busy) return;
    setBusy(true); setError('');
    const body = new FormData();
    body.append('file', file); body.append('camera_view', view); body.append('handedness', handedness);
    body.append('club', club); body.append('mirrored', String(mirrored));
    try {
      const result = await request<Analysis>('/api/analyses', { method: 'POST', body });
      selectionId.current = result.id; setSelected(result); setItems(previous => [result, ...previous]);
      setFile(null); if (input.current) input.current.value = '';
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function remove(item: Analysis) {
    if (!window.confirm('Delete this recording and its local analysis permanently?')) return;
    setError('');
    try {
      await request(`/api/analyses/${encodeURIComponent(item.id)}`, { method: 'DELETE' });
      setItems(previous => previous.filter(value => value.id !== item.id));
      if (selectionId.current === item.id) { selectionId.current = null; setSelected(null); }
    } catch (e) { setError((e as Error).message); }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#"><span className="brand-mark" aria-hidden="true">S<span>↗</span></span>SwingSync</a>
      <p className="sidebar-label">YOUR WORKSPACE</p>
      <button className={`nav-item ${!selected ? 'current' : ''}`} onClick={() => { selectionId.current = null; setSelected(null); }}><span aria-hidden="true">＋</span> New recording</button>
      <div className="library-heading"><span>Local recordings</span><span>{items.length}</span></div>
      <div className="history">
        {items.length === 0 ? <p className="history-empty">Your recordings will appear here. Everything stays on this computer.</p> : items.map(item => <button key={item.id} className={`history-item ${selected?.id === item.id ? 'selected' : ''}`} onClick={() => void choose(item)}><span className="history-icon" aria-hidden="true">▷</span><span><strong>{label(item)}</strong><small>{item.status === 'completed' ? 'Video inspected' : item.status}</small></span></button>)}
      </div>
      <div className="privacy"><span aria-hidden="true">◎</span><div><strong>Private by design</strong><p>Local processing. No accounts.<br />Your video stays with you.</p></div></div>
    </aside>
    <main>
      <header className="topbar"><span>THE SWING STUDIO <span className="edition">/ EARLY BUILD</span></span><span className={`connection ${connected ? 'online' : ''}`}><i />{connected ? 'Local service connected' : 'Local service unavailable'}</span></header>
      <div className="page-content">
        <div className="page-heading"><div><p className="eyebrow">A CLOSER LOOK AT YOUR GAME</p><h1>{selected ? 'Your recording.' : 'Every swing tells a story.'}</h1><p className="subtitle">{selected ? 'Review your video and its recording details.' : 'Start with one swing. Keep the details in focus.'}</p></div><span className="version-tag">LOCAL PREVIEW <span>01</span></span></div>
        {!connected && <div className="notice warning" role="status">The local service is unavailable. Start the SwingSync backend, then this page will reconnect automatically.</div>}
        {error && <div className="notice error" role="alert">{error}</div>}
        {selected ? <section className="review-layout">
          <div className="card video-card"><div className="card-heading"><span className="step">01</span><h2>{label(selected)}</h2><span className="badge">{selected.status === 'completed' ? 'Inspected' : selected.status}</span></div>
            {selected.status === 'completed' ? <video key={selected.id} src={`/api/analyses/${encodeURIComponent(selected.id)}/video`} controls preload="metadata" aria-label="Uploaded swing recording" onError={() => setError('This browser cannot play the uploaded format. The recording may still be valid; try an MP4 encoded with H.264 for playback.')} /> : <div className="processing" role="status"><div className="processing-ring" aria-hidden="true">{active(selected) ? '↻' : '!'}</div><h3>{active(selected) ? 'Inspecting your recording' : 'Recording could not be inspected'}</h3><p>{selected.error || 'Checking video frames and preserving their recorded timestamps.'}</p>{active(selected) && <progress max="100" value={selected.progress || 0} />}</div>}
            <div className="video-caption"><span>Original recording · {selected.metadata?.camera_view?.replaceAll('-', ' ') || view}</span><span>{selected.metadata?.handedness || handedness}-handed</span></div>
          </div>
          <aside className="review-details"><section className="card details-card"><p className="eyebrow">RECORDING DETAILS</p><h2>What we can verify</h2>{selected.video ? <dl><div><dt>Duration</dt><dd>{selected.video.duration_seconds.toFixed(2)} s</dd></div><div><dt>Decoded frames</dt><dd>{selected.video.frame_count.toLocaleString()}</dd></div><div><dt>Resolution</dt><dd>{selected.video.width} × {selected.video.height}</dd></div><div><dt>Club</dt><dd>{selected.metadata?.club || 'Not specified'}</dd></div><div><dt>Mirrored recording</dt><dd>{selected.metadata?.mirrored ? 'Yes' : 'No'}</dd></div></dl> : <p className="muted">Details appear after the video is inspected.</p>}</section>
            <section className="capability"><span className="small-icon" aria-hidden="true">◷</span><h3>Swing analysis unavailable</h3><p>{selected.model?.reason || 'A validated swing-event model is not installed yet.'}</p><p>Event markers, pose measurements and coaching observations will appear only when their required models and checks are ready. No shot predictions are enabled.</p></section>
            <div className="review-actions">{selected.status === 'completed' && <a className="button primary" href={`/api/analyses/${encodeURIComponent(selected.id)}/report`} target="_blank" rel="noreferrer">Open local report <span aria-hidden="true">↗</span></a>}<button className="button subtle danger" disabled={active(selected)} title={active(selected) ? 'Wait for video inspection to finish before deleting.' : undefined} onClick={() => void remove(selected)}>Delete recording</button></div>
          </aside>
        </section> : <div className="upload-layout">
          <section className="card upload-card"><div className="card-heading"><span className="step">01</span><h2>Add your recording</h2><span className="badge">Single swing</span></div>
            <form onSubmit={submit}>
              <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]); }}>
                <div className="upload-symbol" aria-hidden="true">↥</div><h3>{file ? file.name : 'Bring your swing into focus'}</h3><p>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · ready to upload` : 'Drop a video here, or choose one from your computer.'}</p><button className="button secondary" type="button" onClick={() => input.current?.click()}>{file ? 'Choose another video' : 'Choose video'}<span aria-hidden="true">＋</span></button><input ref={input} type="file" accept="video/mp4,video/quicktime,video/webm,video/x-msvideo,.mp4,.mov,.webm,.avi" className="visually-hidden" tabIndex={-1} aria-label="Choose swing video" onChange={event => acceptFile(event.target.files?.[0])} /><small>Up to 150 MB · 30 seconds · MP4 recommended</small>
              </div>
              <div className="form-heading"><span className="step">02</span><h2>Set the scene</h2></div>
              <div className="form-grid"><fieldset><legend>Camera view</legend><div className="segmented"><button type="button" aria-pressed={view === 'face-on'} onClick={() => setView('face-on')}>Face-on</button><button type="button" aria-pressed={view === 'down-the-line'} onClick={() => setView('down-the-line')}>Down the line</button></div></fieldset><fieldset><legend>Handedness</legend><div className="segmented"><button type="button" aria-pressed={handedness === 'right'} onClick={() => setHandedness('right')}>Right</button><button type="button" aria-pressed={handedness === 'left'} onClick={() => setHandedness('left')}>Left</button></div></fieldset><label className="club-field">Club<select value={club} onChange={event => setClub(event.target.value)}>{['Driver', '3 wood', '5 wood', 'Hybrid', '3 iron', '4 iron', '5 iron', '6 iron', '7 iron', '8 iron', '9 iron', 'Pitching wedge', 'Sand wedge', 'Other'].map(value => <option key={value}>{value}</option>)}</select></label><label className="checkbox-field"><input type="checkbox" checked={mirrored} onChange={event => setMirrored(event.target.checked)} /><span>My video is mirrored<small>For a recording flipped by your camera.</small></span></label></div>
              <div className="upload-footer"><span>Saved only on this computer</span><button className="button primary" type="submit" disabled={!file || busy || !connected}>{busy ? 'Uploading…' : 'Inspect recording'}<span aria-hidden="true">→</span></button></div>
            </form>
          </section>
          <aside className="guidance"><div className="guide-art" aria-hidden="true"><div className="art-sun" /><svg viewBox="0 0 300 180"><path d="M0 150Q110 70 300 145V180H0Z" fill="#6b8261"/><path d="M0 165Q180 105 300 160V180H0Z" fill="#344f3d"/><path d="M170 143V46l37 12-37 12" fill="#e4dbb7" stroke="#e4dbb7" strokeWidth="2"/><ellipse cx="170" cy="148" rx="27" ry="7" fill="#233c30"/><circle cx="112" cy="145" r="4" fill="#fffcef"/></svg><span>BETTER RECORDING. CLEARER CONTEXT.</span></div><p className="eyebrow">BEFORE YOU PRESS RECORD</p><h2>A little setup goes a long way.</h2><ol className="tips"><li><span>01</span><div><strong>One complete swing</strong><p>Trim from just before address to after the finish. Keep it under 30 seconds.</p></div></li><li><span>02</span><div><strong>Keep your whole body in frame</strong><p>Use a steady camera in good light. Include your feet, head and full club path.</p></div></li><li><span>03</span><div><strong>Choose a clear view</strong><p>Face-on looks toward your chest. Down the line looks along the target line.</p></div></li></ol><div className="preview-note"><strong>A foundation, built carefully.</strong><p>This build checks and plays your video. Swing events, movement measurements and feedback are still being validated.</p></div></aside>
        </div>}
        <footer className="page-footer"><span>SWINGSYNC</span><p>Evidence first. Every observation should earn its place.</p><span>ON YOUR DEVICE</span></footer>
      </div>
    </main>
  </div>;
}
