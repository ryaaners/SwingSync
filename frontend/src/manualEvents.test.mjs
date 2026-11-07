import test from 'node:test';
import assert from 'node:assert/strict';
import { fromAnalysis, receiveAnalysis, markFrame, eventPayload, draftProblem, newerAnalysis } from './manualEvents.ts';
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
