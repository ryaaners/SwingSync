import { useEffect, useState } from 'react';
import { EVENT_NAMES, type Analysis, type EventName } from './analysisTypes.ts';
import { draftProblem, eventPayload, fromAnalysis, markFrame, receiveAnalysis } from './manualEvents.ts';

type Props = { analysis: Analysis; onSave: (payload: ReturnType<typeof eventPayload>) => Promise<Analysis>; onSeek: (seconds: number) => void; onReload: () => Promise<Analysis> };
const names: Record<string, string> = {
  backswing_duration_seconds: 'Address → Top', downswing_duration_seconds: 'Top → Impact', backswing_downswing_ratio: 'Backswing / downswing',
};
export default function EventReview({ analysis, onSave, onSeek, onReload }: Props) {
  const [draft, setDraft] = useState(() => fromAnalysis(analysis));
  const [frame, setFrame] = useState(analysis.events?.[0]?.frame_index ?? 0);
  const [eventName, setEventName] = useState<EventName>('Address');
  const [readyImage, setReadyImage] = useState('');
  const [imageError, setImageError] = useState('');
  const [imageAttempt, setImageAttempt] = useState(0);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');
  const [saveError, setSaveError] = useState('');
  useEffect(() => { setDraft(previous => receiveAnalysis(previous, analysis)); }, [analysis]);
  const count = analysis.video?.frame_count ?? 0;
  const times = analysis.video?.timestamps_seconds ?? [];
  const sourceTimes = analysis.video?.source_timestamps_seconds ?? [];
  const imageUrl = `/api/analyses/${encodeURIComponent(analysis.id)}/frames/${frame}?attempt=${imageAttempt}`;
  const problem = draftProblem(draft, count);
  const marks = eventPayload(draft).events;
  function selectFrame(index: number) {
    if (!Number.isInteger(index) || index < 0 || index >= count) return;
    setFrame(index); setImageError('');
  }
  async function save() {
    if (saving || problem || !draft.dirty) return;
    setSaving(true); setSaveError(''); setSaveMessage('');
    try {
      const result = await onSave(eventPayload(draft));
      setDraft(fromAnalysis(result));
      setSaveMessage(`Saved revision ${result.annotation_revision ?? 0}. Timing was recomputed from your marked frames.`);
    } catch (error) { setSaveError((error as Error).message); }
    finally { setSaving(false); }
  }
  async function reloadSaved() {
    if (saving) return;
    setSaving(true); setSaveError('');
    try {
      const result = await onReload();
      setDraft(fromAnalysis(result));
      setSaveMessage('Reloaded saved events. Your unsaved marks were discarded.');
    } catch (error) { setSaveError((error as Error).message); }
    finally { setSaving(false); }
  }
  if (!analysis.video?.sha256) return <section className="manual-review card"><h2>Manual event review</h2><p>This older upload has no recording fingerprint. Upload it again to mark frames and export their evidence.</p></section>;
  return <section className="manual-review card" aria-label="Manual swing event review">
    <div className="manual-heading"><div><p className="eyebrow">YOUR MARKED FRAMES</p><h2>Build a swing timeline</h2></div><span className="badge">Manual · {marks.length}/8 marked</span></div>
    <p className="manual-intro">Find an event in the decoded frame below, then mark it. These are your annotations, not automatic detections.</p>
    <div className="canonical-frame">
      <img key={imageUrl} src={imageUrl} alt={`Decoded frame ${frame + 1} of your recording`} onLoad={() => { setReadyImage(imageUrl); setImageError(''); }} onError={() => { setReadyImage(''); setImageError('This decoded frame could not be verified. Reload the recording, or upload it again.'); }} />
      <div><strong>Frame {frame + 1} of {count}</strong><span>{times[frame]?.toFixed(6)} s from first frame</span></div>
    </div>
    <p className="frame-limitation">Decoded image in its stored orientation; a camera rotation can make the video player look different. Preview seeking is approximate. Use this frame panel to mark events.</p>
    {imageError && <p className="event-error" role="alert">{imageError} <button type="button" onClick={() => { setImageAttempt(value => value + 1); setImageError(''); }}>Retry frame</button></p>}
    <fieldset className="frame-controls" disabled={saving}>
      <label htmlFor={`frame-${analysis.id}`}>Review frame</label>
      <input id={`frame-${analysis.id}`} type="range" min="0" max={Math.max(0, count - 1)} step="1" value={frame} onChange={event => selectFrame(Number(event.target.value))} aria-valuetext={`Frame ${frame + 1}, ${times[frame]?.toFixed(3)} seconds`} />
      <div className="frame-navigation"><button type="button" className="button subtle" disabled={frame === 0} onClick={() => selectFrame(frame - 1)}>← Previous</button><label>Frame number<input type="number" min="1" max={count} step="1" value={frame + 1} onChange={event => selectFrame(Number(event.target.value) - 1)} /></label><button type="button" className="button subtle" disabled={frame === count - 1} onClick={() => selectFrame(frame + 1)}>Next →</button></div>
      <div className="mark-controls"><label>Event<select value={eventName} onChange={event => setEventName(event.target.value as EventName)}>{EVENT_NAMES.map(name => <option key={name}>{name}</option>)}</select></label><button type="button" className="button primary" disabled={readyImage !== imageUrl || !!imageError || draft.conflict} onClick={() => { setDraft(previous => markFrame(previous, eventName, frame)); setSaveMessage(''); }}>Mark {eventName}</button></div>
    </fieldset>
    <ol className="event-markers" aria-label="Swing event markers">{EVENT_NAMES.map(name => {
      const index = draft.frames[name];
      return <li key={name} className={index === undefined ? '' : 'marked'}><button type="button" disabled={saving || index === undefined} onClick={() => { if (index !== undefined) { selectFrame(index); onSeek(times[index]); } }}><strong>{name}</strong><span>{index === undefined ? 'Not marked' : `Frame ${index + 1} · ${times[index]?.toFixed(3)} s`}</span></button>{index !== undefined && <button type="button" className="clear-event" disabled={saving || draft.conflict} aria-label={`Clear ${name}`} onClick={() => { setDraft(previous => markFrame(previous, name, null)); setSaveMessage(''); }}>×</button>}</li>;
    })}</ol>
    <div className="event-save"><button type="button" className="button primary" disabled={!draft.dirty || !!problem || saving} onClick={() => void save()}>{saving ? 'Saving…' : 'Save marked events'}</button><span>{draft.dirty ? 'Unsaved changes' : `Saved revision ${draft.revision}`}</span>{draft.conflict && <button type="button" className="button subtle" disabled={saving} onClick={() => void reloadSaved()}>Reload saved events</button>}</div>
    {(problem || saveError) && <p className="event-error" role="alert">{saveError || problem} {saveError && <button type="button" disabled={saving} onClick={() => void reloadSaved()}>Discard draft and reload saved events</button>}</p>}
    {saveMessage && <p className="save-message" role="status">{saveMessage}</p>}
    <div className="manual-measurements"><p className="eyebrow">RECORDING-TIME MEASUREMENTS</p><h3>Timing from saved marks</h3>{(analysis.measurements ?? []).length ? <dl>{analysis.measurements?.map(value => <div key={value.name}><dt>{names[value.name] ?? value.name}</dt><dd>{value.value.toFixed(3)} {value.unit === 'seconds' ? 's' : 'ratio'}</dd></div>)}</dl> : <p>Save Address, Top and Impact to calculate backswing time, downswing time and their ratio.</p>}<p>Manual event timing is unvalidated. Slow motion or speed ramps can change recording time and tempo. No ideal ratio or practice advice is inferred.</p></div>
    <details className="annotation-provenance"><summary>Annotation evidence · revision {analysis.annotation_revision ?? 0}</summary><p>Time values come from decoded presentation timestamps, not estimated frame rate. Current frame native timestamp: {sourceTimes[frame]?.toFixed(6) ?? 'Unavailable'} s.</p><p>{analysis.annotation_history?.length ?? 0} saved revision(s). Changes remain marked as manual in the local report.</p></details>
  </section>;
}
