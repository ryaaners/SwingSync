# Offline event proposal evidence

The report CLI authenticates a completed research run using an independently recorded ledger SHA, its bound result/arrays, frozen checkpoint/pipeline/protocol metadata and source bytes. It independently verifies all native frame timestamps and class-peak arithmetic, then renders eight primary temporal proposal cards, canonical decoded images, uncalibrated class/background values, order/duplicate diagnostics and a separate frame-head comparison. It performs no model forward or accuracy calculation.

```sh
uv run python scripts/report_event_proposals.py \
  --run artifacts/event-input/my-run/run.json \
  --run-sha256 THE_INDEPENDENTLY_RECORDED_COMPLETED_LEDGER_SHA256 \
  --video path/to/the/same-recording.mp4 \
  --output artifacts/event-input/my-report
```

Use a new output directory. The self-contained HTML embeds a recording of at most20MiB and up to eight canonical640px PNGs. Larger research inputs can be processed by the runner, but require a smaller separately identified recording to produce this embedded report; do not silently transcode or replace a hash-bound recording. Existing output is not overwritten. A source/protocol/report-code change during publication removes incomplete derived output. Review/source/model videos remain ignored and must not be redistributed just because local research access exists.

The timeline buttons request **approximate** browser seeks using native PTS minus the first decoded PTS. Container/audio/display conventions and codec support can differ. Canonical frame-index/native-PTS images are the evidence; the report does not claim browser synchronization. Unsupported browser codecs leave images available. No browser screenshot or currentTime is treated as an exact event annotation.

Eight independent peaks are forced even when background dominates or phases are duplicated/reversed. Scores are not calibrated correctness probabilities. The primary temporal head and reference64-reset context are fixed before review; the report does not choose a better-looking head, change a peak, repair chronology, invent swing labels or derive tempo/advice from predictions. Operator camera/hand/club/mirror/single-swing declarations and same-pixel parity cannot establish raw-upload distribution compatibility, physical tempo or accuracy. The normal upload app's model status remains unavailable.

The runner is documented in event-proposals.md; exact admission/crop/timestamp policy and parity scope are in event-input-contract.md. The protocol and model/lock/source identities remain fixed. Empirical event/crop/reliability evidence and independently reviewed permitted media are separate gates, as are pose validation and CaddieSet outcome equivalence. No outcome prediction or coaching is promoted by this report.
