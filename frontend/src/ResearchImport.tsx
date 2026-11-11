import { useEffect, useRef, useState } from 'react';
import { applyResearchPreview, previewMatchesDraft, type Draft, type DraftIdentity, type ResearchPreview } from './manualEvents.ts';

type Props = { draft: Draft; count: number; disabled: boolean; onApply: (draft: Draft) => void };
export default function ResearchImport({ draft, count, disabled, onApply }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [checksum, setChecksum] = useState('');
  const [preview, setPreview] = useState<{ value: ResearchPreview; identity: DraftIdentity } | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState('');
  const requestVersion = useRef(0);
  const currentDraft = useRef(draft);
  currentDraft.current = draft;
  useEffect(() => {
    requestVersion.current += 1; setPreview(null); setChecking(false); setError('');
  }, [draft.id, draft.revision, draft.editVersion, draft.conflict]);
  useEffect(() => () => { requestVersion.current += 1; }, []);
  function invalidate() { requestVersion.current += 1; setPreview(null); setChecking(false); setError(''); }
  async function inspect() {
    if (!file || disabled || checking || draft.conflict) return;
    const identity = { id: draft.id, revision: draft.revision, editVersion: draft.editVersion };
    if (file.size > 64 * 1024) { setError('Choose a research draft smaller than 64 KiB.'); return; }
    const version = ++requestVersion.current;
    setChecking(true); setPreview(null); setError('');
    try {
      const body = new FormData(); body.append('file', file); body.append('expected_sha256', checksum.trim());
      const response = await fetch(`/api/analyses/${encodeURIComponent(identity.id)}/research-draft`, { method: 'POST', body });
      const result = await response.json();
      if (requestVersion.current !== version || !previewMatchesDraft(currentDraft.current, identity)) return;
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The research draft could not be checked.');
      setPreview({ value: result as ResearchPreview, identity });
    } catch (cause) {
      if (requestVersion.current === version && previewMatchesDraft(currentDraft.current, identity)) setError((cause as Error).message);
    } finally { if (requestVersion.current === version) setChecking(false); }
  }
  function apply() {
    if (!preview || disabled) return;
    try { onApply(applyResearchPreview(currentDraft.current, preview.identity, preview.value, count)); setPreview(null); }
    catch (cause) { setError((cause as Error).message); }
  }
  return <details className="research-import"><summary>Import research proposals · optional</summary>
    <p>Use a draft exported by the local research tool for this exact recording. These are unvalidated proposals, not confirmed events. Importing does not save marks or calculate timing.</p>
    <p>Copy the file checksum from the independently trusted export output. A matching checksum checks file identity; it is not a digital signature or proof of accuracy.</p>
    <fieldset disabled={disabled || draft.conflict}>
      <label>Research draft file<input type="file" accept=".json,application/json" onChange={event => { invalidate(); setFile(event.target.files?.[0] ?? null); }} /></label>
      <label>Expected file SHA-256<input type="text" value={checksum} maxLength={64} spellCheck={false} autoComplete="off" onChange={event => { invalidate(); setChecksum(event.target.value); }} placeholder="64 lowercase hexadecimal characters" /></label>
      <button type="button" className="button subtle" disabled={!file || !/^[0-9a-f]{64}$/.test(checksum.trim()) || checking} onClick={() => void inspect()}>{checking ? 'Checking draft…' : 'Check research draft'}</button>
    </fieldset>
    {error && <p className="event-error" role="alert">{error}</p>}
    {preview && <div className="research-preview" role="status"><strong>Checked against this recording. Your marks have not changed.</strong>
      <p>{preview.value.warning}</p><p>Scores are uncalibrated; background may exceed an event score. No coaching or accuracy claim is supported.</p>
      <p>Event order: {preview.value.strict_order ? 'strictly increasing' : 'out of order — correct or remove conflicting proposals before saving'}. Duplicate peaks: {preview.value.duplicate_peak_count}.</p>
      <ol>{preview.value.events.map(event => <li key={event.name}>{event.name}: frame {event.frame_index + 1}</li>)}</ol>
      <button type="button" className="button subtle" disabled={disabled || !previewMatchesDraft(draft, preview.identity)} onClick={apply}>Replace unsaved marks with these proposals</button>
    </div>}
  </details>;
}
