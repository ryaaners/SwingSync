# Research proposals to manual review

This opt-in workflow moves an existing authenticated local research artifact into an unsaved review draft. It performs no new model inference and does not enable automatic detection or coaching. Use the same recording bytes and matching camera view, handedness, club and mirror declarations. The normal upload/manual-mark workflow remains available without research files.

## Export locally

Use the pinned project environment and the original completed research run, its independently recorded ledger SHA and recording:

```sh
uv run python scripts/export_event_draft.py \
  --run artifacts/event-input/clip0-v1/run.json \
  --run-sha256 425d1729b966f45d7d1041658f41d9d7adc5e7133c726de4ca5ea47be1858141 \
  --video data/golfdb/videos_160/0.mp4 \
  --output artifacts/event-input/clip0-draft-new.json
```

Use a fresh output filename. The exporter authenticates the completed run, selected model/protocol/source identities, arrays and freshly decoded native timestamps through the existing evidence loader. It then writes a bounded draft and prints its SHA-256. Keep the draft and hash locally; neither media nor predictions belong in Git. This is an integrity checksum, not a digital signature. Obtain the expected hash independently from the trusted local export output; copying a hash supplied with an untrusted document does not establish its origin.

## Review and save

Upload the exact same recording to the app and choose matching recording options. Expand the research import section, select the draft file and supply the independently obtained hash. Validation checks exact UTF-8 bytes, schema, recording SHA, full native timestamp mapping, metadata, event indices/timestamps, peak diagnostics and closed admission flags. Preview alone changes no saved events, revision, measurements or history.

Explicitly replace the unsaved review draft, inspect canonical decoded frame images, correct or remove proposals as needed, and confirm that you reviewed the selected frames before saving. Any edit requires confirmation again. Duplicate or reversed phases are retained on import and prevent saving until manually corrected; no sorting or decoder repair is applied. Existing revision conflicts still require an explicit reload. Browser playback remains approximate and codec dependent; canonical decoded images are the frame evidence.

The final save revalidates the imported document and recording and uses the existing transactional revision check. Saved marks remain manual, with research origin retained alongside original proposed frames and final confirmed frames in correction history and the exported report. Subsequent changes and clearing must preserve prior history. A research proposal is not an independently labeled event or validated coaching evidence. Confirmation is a workflow declaration; the software cannot establish reviewer expertise or independent ground truth.

No measurements are recomputed from proposals before explicit save. Saved descriptive timing is recording-time arithmetic on reviewed marks, with the existing slow-motion and unvalidated-manual-reliability limitations. No pose observations, outcome predictions, ideal thresholds, practice suggestions or automatic-serving admission follow. Empirical event/crop reliability and independent pose reference gates remain unchanged.
