import test from 'node:test';
import assert from 'node:assert/strict';
import { fromAnalysis, receiveAnalysis, markFrame, eventPayload, draftProblem, newerAnalysis, applyResearchPreview, confirmResearchReview, previewMatchesDraft } from './manualEvents.ts';
const analysis = (revision = 0, events = []) => ({ id: 'one', status: 'completed', progress: 100, annotation_revision: revision, events });
test('draft edits survive background polling and stale responses', () => {
  const draft = markFrame(fromAnalysis(analysis(2)), 'Top', 3);
  assert.equal(receiveAnalysis(draft, analysis(2)), draft);
  assert.equal(receiveAnalysis(draft, analysis(1)), draft);
  const conflict = receiveAnalysis(draft, analysis(3));
  assert.equal(conflict.frames.Top, 3);
  assert.equal(conflict.revision, 2);
  assert.equal(conflict.conflict, true);
  assert.match(draftProblem(conflict, 8), /another window/);
});
test('saved updates replace a clean draft, different recordings reset, old polls cannot undo saves', () => {
  const saved = analysis(3, [{ name: 'Top', frame_index: 5 }]);
  assert.equal(receiveAnalysis(fromAnalysis(analysis(2)), saved).frames.Top, 5);
  assert.equal(receiveAnalysis(fromAnalysis(saved), { ...analysis(), id: 'two' }).id, 'two');
  assert.equal(newerAnalysis(saved, analysis(2)), saved);
  assert.equal(newerAnalysis(saved, { ...analysis(), id: 'two' }).id, 'two');
});
test('partial replacement is canonical and frame based, no guessed seconds or confidence', () => {
  let draft = markFrame(fromAnalysis(analysis(4)), 'Impact', 7);
  draft = markFrame(draft, 'Address', 1);
  assert.deepEqual(eventPayload(draft), { expected_revision: 4, events: [{ name: 'Address', frame_index: 1 }, { name: 'Impact', frame_index: 7 }] });
  assert.equal(draftProblem(draft, 8), null);
  draft = markFrame(draft, 'Address', null);
  assert.deepEqual(eventPayload(draft).events, [{ name: 'Impact', frame_index: 7 }]);
});
test('duplicate, reversed, fractional and out-of-range frame marks are blocked', () => {
  let draft = markFrame(markFrame(fromAnalysis(analysis()), 'Address', 3), 'Top', 3);
  assert.match(draftProblem(draft, 8), /swing order/);
  draft = markFrame(draft, 'Top', 2);
  assert.match(draftProblem(draft, 8), /swing order/);
  for (const value of [-1, 8, 2.5, NaN, Infinity]) assert.match(draftProblem(markFrame(fromAnalysis(analysis()), 'Top', value), 8), /outside/);
});

const research = (events = ['Address', 'Toe-up', 'Mid-backswing', 'Top', 'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish'].map((name, frame_index) => ({ name, frame_index }))) => ({
  document_json: '{"opaque":"exact bytes\n"}', expected_sha256: 'a'.repeat(64), events, strict_order: true, duplicate_peak_count: 0, warning: 'Unvalidated forced peaks.'
});
test('research preview only replaces marks explicitly and requires confirmation before save', () => {
  const original = markFrame(fromAnalysis(analysis(4)), 'Top', 6);
  const preview = research();
  assert.equal(original.frames.Top, 6);
  const draft = applyResearchPreview(original, original, preview, 10);
  assert.equal(draft.frames.Top, 3);
  assert.equal(draft.revision, 4);
  assert.equal(draft.dirty, true);
  assert.match(draftProblem(draft, 10), /confirm your review/);
  const confirmed = confirmResearchReview(draft, true);
  assert.equal(draftProblem(confirmed, 10), null);
  assert.deepEqual(eventPayload(confirmed).research_draft, { document_json: preview.document_json, expected_sha256: preview.expected_sha256, confirmed: true });
  assert.equal(eventPayload(draft).research_draft.confirmed, false);
});
test('research edits reset review confirmation and retain original byte provenance', () => {
  const imported = applyResearchPreview(fromAnalysis(analysis()), fromAnalysis(analysis()), research(), 12);
  const confirmed = confirmResearchReview(imported, true);
  const corrected = markFrame(confirmed, 'Finish', 9);
  assert.equal(corrected.research.confirmed, false);
  assert.equal(corrected.research.document_json, imported.research.document_json);
  assert.match(draftProblem(corrected, 12), /confirm your review/);
  assert.equal(markFrame(confirmed, 'Finish', null).research.confirmed, false);
  assert.equal(fromAnalysis(analysis(2)).research, undefined);
});
test('out of order or duplicate research peaks remain unrepaired and unsaveable after confirmation', () => {
  const base = fromAnalysis(analysis());
  const preview = research(); preview.events[0].frame_index = 6; preview.strict_order = false;
  const draft = confirmResearchReview(applyResearchPreview(base, base, preview, 10), true);
  assert.equal(draft.frames.Address, 6);
  assert.match(draftProblem(draft, 10), /swing order/);
  const corrected = markFrame(draft, 'Address', 0);
  assert.match(draftProblem(corrected, 10), /confirm your review/);
  assert.equal(draftProblem(confirmResearchReview(corrected, true), 10), null);
});
test('stale imports cannot replace a different recording, revision, conflict or edited draft', () => {
  const base = fromAnalysis(analysis());
  const variants = [markFrame(base, 'Top', 4), { ...base, id: 'two' }, { ...base, revision: 1 }, { ...base, conflict: true }];
  for (const current of variants) {
    assert.equal(previewMatchesDraft(current, base), false);
    assert.throws(() => applyResearchPreview(current, base, research(), 10), /draft changed/);
  }
  const imported = applyResearchPreview(base, base, research(), 10);
  const conflicted = receiveAnalysis(imported, analysis(1));
  assert.equal(conflicted.conflict, true);
  assert.equal(conflicted.research.document_json, imported.research.document_json);
  assert.match(draftProblem(conflicted, 10), /another window/);
});
test('malformed research preview cannot introduce invalid or missing event frames', () => {
  const base = fromAnalysis(analysis());
  for (const events of [research().events.slice(1), [...research().events.slice(0,7), { name: 'Top', frame_index: 9 }], [...research().events.slice(0,7), { name: 'Finish', frame_index: 10 }], [...research().events.slice(0,7), { name: 'Finish', frame_index: 1.5 }]]) {
    assert.throws(() => applyResearchPreview(base, base, research(events), 10), /invalid frame/);
  }
});
