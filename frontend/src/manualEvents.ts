import { EVENT_NAMES, type Analysis, type EventName } from './analysisTypes.ts';
export type ResearchPreview = {
  document_json: string; expected_sha256: string; events: { name: EventName; frame_index: number }[];
  strict_order: boolean; duplicate_peak_count: number; warning: string;
};
export type DraftIdentity = { id: string; revision: number; editVersion: number };
export type Draft = DraftIdentity & {
  frames: Partial<Record<EventName, number>>; dirty: boolean; conflict: boolean;
  research?: Omit<ResearchPreview, 'events'> & { confirmed: boolean };
};
export function fromAnalysis(analysis: Analysis): Draft {
  return { id: analysis.id, revision: analysis.annotation_revision ?? 0,
    frames: Object.fromEntries((analysis.events ?? []).map(event => [event.name, event.frame_index])), dirty: false, conflict: false, editVersion: 0 };
}
export function receiveAnalysis(draft: Draft, analysis: Analysis): Draft {
  if (draft.id !== analysis.id) return fromAnalysis(analysis);
  const revision = analysis.annotation_revision ?? 0;
  if (revision <= draft.revision) return draft;
  if (draft.dirty) return draft.conflict ? draft : { ...draft, conflict: true };
  return fromAnalysis(analysis);
}
export function markFrame(draft: Draft, name: EventName, frame: number | null): Draft {
  const frames = { ...draft.frames };
  if (frame === null) delete frames[name]; else frames[name] = frame;
  return { ...draft, frames, dirty: true, editVersion: draft.editVersion + 1,
    ...(draft.research ? { research: { ...draft.research, confirmed: false } } : {}) };
}
export function eventPayload(draft: Draft) {
  return { expected_revision: draft.revision, events: EVENT_NAMES.flatMap(name => {
    const frame = draft.frames[name];
    return frame === undefined ? [] : [{ name, frame_index: frame }];
  }), ...(draft.research ? { research_draft: { document_json: draft.research.document_json, expected_sha256: draft.research.expected_sha256, confirmed: draft.research.confirmed } } : {}) };
}
export function draftProblem(draft: Draft, count: number): string | null {
  if (draft.conflict) return 'Saved events changed in another window. Reload saved events before editing.';
  let previous = -1;
  for (const { name, frame_index } of eventPayload(draft).events) {
    if (!Number.isInteger(frame_index) || frame_index < 0 || frame_index >= count) return `${name} is outside this recording.`;
    if (frame_index <= previous) return 'Marked events must use different frames in swing order.';
    previous = frame_index;
  }
  if (draft.research && !draft.research.confirmed) return 'Review and correct the proposed frames, then confirm your review before saving.';
  return null;
}
export function newerAnalysis(previous: Analysis | null, incoming: Analysis): Analysis {
  return previous?.id === incoming.id && (previous.annotation_revision ?? 0) > (incoming.annotation_revision ?? 0) ? previous : incoming;
}

// Preview responses never change marks. Applying one is a separate explicit action.
export function previewMatchesDraft(draft: Draft, identity: DraftIdentity): boolean {
  return !draft.conflict && draft.id === identity.id && draft.revision === identity.revision && draft.editVersion === identity.editVersion;
}
export function applyResearchPreview(draft: Draft, identity: DraftIdentity, preview: ResearchPreview, count: number): Draft {
  if (!previewMatchesDraft(draft, identity)) throw new Error('Your draft changed. Check the research file again before replacing marks.');
  const names = new Set(preview.events.map(event => event.name));
  if (preview.events.length !== 8 || names.size !== 8 || preview.events.some(event => !EVENT_NAMES.includes(event.name) || !Number.isInteger(event.frame_index) || event.frame_index < 0 || event.frame_index >= count)) {
    throw new Error('The research preview contains invalid frame proposals.');
  }
  const { events, ...research } = preview;
  return { ...draft, frames: Object.fromEntries(events.map(event => [event.name, event.frame_index])), dirty: true,
    editVersion: draft.editVersion + 1, research: { ...research, confirmed: false } };
}
export function confirmResearchReview(draft: Draft, confirmed: boolean): Draft {
  return draft.research ? { ...draft, research: { ...draft.research, confirmed } } : draft;
}
