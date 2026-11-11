# Independent review 013 — research proposals to manual review

Date: 2026-11-11. Scope: 003g3 backend draft/export/annotation/report changes and frontend opt-in preview, replacement, correction and confirmation. Reviewer read AGENTS.md, CURRENT_STATUS.md, backlog.json and autonomy/roles/reviewer.md. No implementation edits, Git operations, training or inference were performed. Process inspection was denied by the review sandbox; job inspection belongs to coordinator evidence.

## Disposition

No remaining material software finding in the reviewed snapshot. One reviewer P2 finding was reproduced, fixed by the backend owner and independently verified. Coordinator also identified and resolved exporter publication source coverage; the reviewer independently verified both source-drift rejection branches. This is software and saved-artifact review only. Coordinator supplied actual historical-artifact export/API evidence, independently audited in the addendum below. Browser interaction and aggregate completion claims remain coordinator evidence; the reviewer did not operate the browser.

## Resolved P2 — malformed proposal numbers raised HTTP 500

Location: backend/swingsync/proposal_handoff.py, numeric admission in parse_document; final fix is _finite at line 62 and its use at line 121.

Reproduction before the fix:

1. Start with an otherwise valid supported draft.
2. Set events[0].score to the JSON integer 10**309, serialize it and calculate its matching SHA-256. The reproduction document was only 2726 bytes. The same failure occurred for source_timestamp_seconds and background_probability.
3. Pass those bytes/hash to parse_document, POST them to /api/analyses/{id}/research-draft, or PUT them in a confirmed research_draft on /events.
4. math.isfinite attempted to convert the integer to float and raised OverflowError. The routes caught ValueError, so malformed input escaped as a server error.

Action taken by backend owner: _finite rejects non-number types and catches OverflowError; all three numeric fields use it. Independent final API tests now return 422 for all three fields at both preview and confirmed-save endpoints and assert the saved analysis remains byte-equivalent as JSON. The relevant parameterized regression is test_api_huge_numeric_values_return422_without_mutation.

## Independently checked behavior

- The offline CLI calls the existing known-ledger-SHA evidence loader; export uses saved results and performs no model forward. It writes exclusively, validates its own bounded schema, retains both derivation source hashes and rechecks evidence/sources before and after publication. In-memory mocks independently simulated either proposal_handoff.py or export_event_draft.py changing after output creation; both raised ValueError and removed the incomplete output. Existing output remains protected by exclusive creation.
- Raw UTF-8 bytes and independently supplied exporter checksum are preserved through preview/save. Draft JSON rejects duplicate keys, nonfinite values, excess nesting, unknown fields, unsupported types and byte sizes above 64 KiB. Semantic event names, indices, timestamps, scores, original order diagnostics and every closed admission flag are checked. An additional 468 value/container substitutions produced no unexpected exception; accepted substitutions retained valid semantics.
- Source byte SHA, frame count, exact full native-seconds-sequence digest, recording metadata and each proposal index/time correspondence are rebound to the current inspected recording. The app hashes current source bytes at preview/save. The offline evidence loader freshly checks rational PTS before export; the app's stored inspection representation is float64 native seconds. This review does not claim the app separately reconstructs rational tick metadata at import.
- Preview performs no DB write or timing computation. Saving requires strict explicit confirmation, validates final manual frame order and applies the existing transactional revision CAS. Original proposed frames and document/model/run hashes remain separate from confirmed/corrected frames across subsequent edits, clearing and restart. Reports disclose research origin instead of describing imported proposals as wholly independent manual labels. Normal model status remains unavailable, observations/advice remain empty, and timing remains descriptive/unvalidated.
- Frontend preview checks use request versions plus current analysis ID, revision, edit version and conflict state. File/checksum changes, draft edits and unmount invalidate responses. Applying proposals is a separate replacement action. Every mark/removal resets confirmation; imported duplicate/reversed phases stay unrepaired and cannot save until corrected or removed. Revision conflicts preserve the draft and require explicit reload. Reviewed final UI also discloses saved research provenance.
- A checksum establishes identity only relative to an independently trusted local exporter output; it is not a signature or scientific validation. The UI, module and workflow documentation disclose that limitation. Copying a hash from an untrusted file does not establish its origin.

## Independent verification

- `.venv/bin/python -m pytest tests/test_proposal_handoff.py tests/test_manual_annotations.py tests/test_application.py tests/test_event_proposal_report.py -q -p no:cacheprovider --basetemp=/private/tmp/swingsync-review013-final`: **94 passed**, one existing AnyIO deprecation warning. This run included 43 handoff and 51 existing regression tests, using synthetic media only. After two publication-drift regressions were added, the reviewer separately reran the final handoff module: **45 passed**. Thus 96 distinct focused backend tests passed across the recorded runs.
- `frontend: npm test`: **9 passed**. `npm run typecheck`: passed. React/browser interaction itself is reserved for coordinator QA; the reviewer did not infer browser success from these state-function tests.
- Ruff over the changed backend modules, CLI and handoff tests: passed.
- Independent in-memory post-publication drift mocks: module drift rejected/output removed; CLI drift rejected/output removed.

## Reviewed source fingerprints

SHA-256 at software-review completion:

| File | SHA-256 |
| --- | --- |
| backend/swingsync/proposal_handoff.py | e8bd00bb7fa68a93c4bf6b82792e3e20c1e79a4ff39cbea5213d9ba32c1aa58b |
| backend/swingsync/annotations.py | fd301724b7fec55e2aca3d344486eb81efb4b17b46f588a8644157bee933401a |
| backend/swingsync/app.py | 2218b0406dc8e3f442f84cc1f5b0db0ee7f25285588e73139ac6b9ffaa2cc36a |
| scripts/export_event_draft.py | 8217659788d0218db4c87a2cb5c69a225526a45af77e2f5755e04dc03fc7e546 |
| tests/test_proposal_handoff.py | edf45cc94255b8fcb568fd4e09aa43086b564a7f381c44ae53ccbda05395af46 |
| frontend/src/ResearchImport.tsx | ffd9ed45f4d2a44cad529a20b9c763e4ac97d1612dac44fa68baa860e6dc6cc9 |
| frontend/src/EventReview.tsx | 884ebd8787118211b66bb27e5b1f2211e9d8046121529e1c988f3629dd9a18d8 |
| frontend/src/manualEvents.ts | 7cc5351a5360012ba97202cb2373494f7313464b0ce3f4b58ce559bb2e187c0d |
| frontend/src/analysisTypes.ts | 9bf591e46973c9914e04e8ba26c0a112925fa1e015258133153e75636c6ed76a |
| frontend/src/manualEvents.test.mjs | 7bbbe95e749ddd2dbd65dd8ee272eabd8bbced29f8f66fb55019120087d07ad2 |
| frontend/src/style.css | 5024b869bf05bee0de0f280cf562874de99279a0f9fe54883ad8d5d3d59feb9e |
| docs/research-draft-handoff.md | 347161284cf2cf3fcb87cbe8fadf4e5589950d3cc268caa17936b216f5954b81 |


## Actual saved-artifact audit addendum

Coordinator exported the existing completed clip0/clip1 artifacts without new inference. Reviewer independently audited them using only standard-library JSON, SHA-256, ZIP and NPY float32 parsing; the audit did not import model code, fit, infer or alter the runtime environment.

- Authenticated both original known ledger hashes (clip0 `425d1729b966f45d7d1041658f41d9d7adc5e7133c726de4ca5ea47be1858141`, clip1 `b63bfdb1d7779048940f2e0f406caa3247fe4cea6b8bcca70885b70db0c8138d`), each result/array hash, actual source-video byte hash, completed state, exporter source identities, complete native-seconds digest and draft byte hashes.
- Clip0 draft `dc24eadba4b6e75bf2a5837a187b67791d970b0902404fc83c94c107f5ca7689`: 138 frames; independently recomputed temporal peaks `[47,65,70,83,87,90,93,106]`, strict order true.
- Clip1 draft `0b2adb610fc46588cfaff2ce0ab486d2297a6513dd59c70e2544909a94184e75`: 324 frames; independently recomputed temporal peaks `[196,104,119,134,191,208,215,268]`, strict order false. No chronology repair occurred.
- All 16 class maxima were unique. Every exported event index, native timestamp, float32 class score, background probability, duplicate count and strict-order diagnostic matched the saved probability arrays and original research results exactly.
- `artifacts/handoff-013/clip0-saved.json` revision 1 retains original draft events, final confirmed events, draft/run/model hashes and immutable history correspondence. All eight saved events are manual. Model status is unavailable and observations/advice are empty.
- The saved report hash `34d2f51a2e0ffab862d23efde0d425f9da05f905053100e37916828d12857b30` matches API QA evidence. It contains eight embedded canonical PNGs, research-assisted provenance and the draft hash; it does not claim imported marks were independent manual estimates. Reviewer checked presence/count, not a fresh pixel decode of those eight images.

Coordinator's `api-qa.json` records unchanged previews, confirmation refusal, stale-revision refusal, nonchronological clip1 save refusal and restart persistence. The reviewer audited the saved documents/report and their internal correspondence; the endpoint interactions in that file were performed by the coordinator. They are automated software QA, not independent human event labels or accuracy validation. Read-only coordinator logs additionally show base 533 passed/13 skipped and full ML 631 passed/1 skipped; these full-suite commands were not run by this reviewer.

## Final aggregate and documentation acceptance

Reviewed final CURRENT_STATUS.md, backlog.json, validation-013.md, research-handoff-013.json, README, architecture/model-card/experiments/data-contract documentation, remaining-work and the new workflow instructions. No material completion-claim inconsistency was found. Backlog 003g3 is completed; empirical 003g2 remains blocked and clean reproduction 007a is pending. The documents retain unavailable ordinary automatic models/advice, separate software handoff completion from empirical admission, identify one source/golfer, and describe confirmations as automated QA rather than independent human labels. New remote CI success is not claimed.

Independently checked aggregate SHA-256 `8166775891c226a0bcbf24ffb638bd606c81c49930a1bbfbfc62fe58e89bc9de`: both draft paths/hashes/event indices/order/provenance match the raw drafts; all derivation source hashes match current files; API evidence hash and copied checks match the source JSON. Browser saved-state hash `06851f758f468fcd20d4d94aba4f88b34191c13f02d7d1b785e5ee05fdc1cb97` and screenshot hash `91c8982220c334e12723d5a74adbdf2e10f807232879655c6c764636c0678897` match the local files.

The initial API artifact/report remain revision 1. The later browser artifact is revision 2, has exactly two history entries, preserves revision 1 unchanged and retains original draft provenance separately from explicit confirmation. Its three saved measurements independently equal native-PTS arithmetic: backswing 1.200880645806925 s, downswing 0.23350457001801317 s, ratio 5.142857142857143. Model is unavailable and observations/advice remain empty. The inspected screenshot visibly shows saved revision 2 and rounded 1.201 s / 0.234 s / 5.143 ratio with the unvalidated recording-time limitation. A final screenshot alone cannot establish the preceding checkbox/reset interactions; those remain coordinator-observed browser QA, supported separately by reviewed state logic/tests. The aggregate accurately keeps revision-1 API evidence distinct from revision-2 browser evidence.

Final disposition remains no unresolved material findings. No implementation changes, Git operations or model execution were performed for this final claims review.
