import { EVENT_NAMES, type Analysis, type EventName } from './analysisTypes.ts';
export type Draft = { id: string; revision: number; frames: Partial<Record<EventName, number>>; dirty: boolean; conflict: boolean };
export function fromAnalysis(analysis: Analysis): Draft {
  return { id: analysis.id, revision: analysis.annotation_revision ?? 0,
    frames: Object.fromEntries((analysis.events ?? []).map(event => [event.name, event.frame_index])), dirty: false, conflict: false };
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
  return { ...draft, frames, dirty: true };
}
export function eventPayload(draft: Draft) {
  return { expected_revision: draft.revision, events: EVENT_NAMES.flatMap(name => {
    const frame = draft.frames[name];
    return frame === undefined ? [] : [{ name, frame_index: frame }];
  }) };
}
export function draftProblem(draft: Draft, count: number): string | null {
  if (draft.conflict) return 'Saved events changed in another window. Reload saved events before editing.';
  let previous = -1;
  for (const { name, frame_index } of eventPayload(draft).events) {
    if (!Number.isInteger(frame_index) || frame_index < 0 || frame_index >= count) return `${name} is outside this recording.`;
    if (frame_index <= previous) return 'Marked events must use different frames in swing order.';
    previous = frame_index;
  }
  return null;
}
export function newerAnalysis(previous: Analysis | null, incoming: Analysis): Analysis {
  return previous?.id === incoming.id && (previous.annotation_revision ?? 0) > (incoming.annotation_revision ?? 0) ? previous : incoming;
}
