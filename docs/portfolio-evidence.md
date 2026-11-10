# Supported portfolio evidence — through execution 011

The full product is still in development. The following are draft statements about completed work only; retain dataset/fold/protocol limits when using them. Do not claim validated coaching, user testing, validated uploaded-video outcome prediction or a finished automatic video-to-coaching pipeline.

- Trained frame-independent and bidirectional-LSTM swing-event heads over frozen ImageNet MobileNetV2 features, achieving 58.8% PCE versus 45.1% baseline on a preregistered 350-clip GolfDB fold-1 test set with disjoint source recordings.
- Built a local React/TypeScript and FastAPI video-review application with decoded-frame timestamp evidence, revision-safe manual event correction and downloadable reports containing event images and provenance.
- Implemented hash-bound feature caches, resumable CPU training, source-grouped inner selection and audited held-out evaluation; an independent replay reproduced all 1,400 per-clip/head/context results and aggregate metrics.

The first bullet uses the **reference 64-frame-reset evaluation**, not the higher separately registered full-clip result 64.4%. “Trained” applies to the event heads, not the pretrained frozen image backbone. PCE is the official variable-tolerance Percentage of Correct Events, not swing quality or generic classification accuracy. Only one outer fold was run; source separation does not prove unseen-golfer generalization. No significance/real-time/minimum-hardware claim is supported. Address/Finish reliability and uploaded-video/pose/coaching gates remain open.

Evidence: [machine results](../autonomy/reports/heldout-results-008.json), [independent audit/replay](../autonomy/reports/review-008.md), [experiment details](experiments.md), [model card](model-card.md) and the committed fixed contract. Raw data/caches/checkpoints/run/replay witnesses remain ignored and hash-bound. Synthetic UI clips prove software behavior only. Before a final portfolio release, complete permitted real-golf demonstration, local pose/input validation, setup reproduction or explicitly state those remaining limits.

## Additional completed outcome research — 011

Supported scope: implemented a fixed eight-golfer CaddieSet evaluation with whole-golfer outer splits, grouped inner model selection and training-only imputation/scaling/encoding; compared club-conditioned means, Ridge and modest histogram gradient boosting separately by camera view. Neither learned model improved pooled MAE; small DTL secondary gains and large FACEON Ridge errors are disclosed. Evidence: caddieset-results-011.json, caddieset-experiments.md and independent review-011.md.

Possible résumé wording: “Built leakage-resistant golf-outcome experiments with nested golfer-grouped validation; compared three baselines/models and documented failure to consistently beat club-conditioned means, withholding unvalidated video predictions.” Do not claim improved shot accuracy, slice/hook classification, causal coaching effectiveness or a working video-to-outcome pipeline. This does not alter the separately evidenced event-model results.
