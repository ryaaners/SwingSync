# Additional official-fold results — execution019

All three additional official GolfDB folds completed the registered comparison:350outer clips each,1050clips/295221frames/4200head-context variants. Heads were trained in017/018;019 performed no training. Fresh review, pushed code6a6a370 and all6CI44864587822 passed before real forwards. Independent replay reproduced every saved metric and aggregate from full float32 probabilities and native float64 PTS. Completed resume left all1052files byte-identical, added no attempt, and made zero model calls.

Primary reference-64-reset results follow. Each head uses the same frozen upstream ImageNet MobileNetV2 features and its own inner-selected best_model. These are this project's trained heads, with a pretrained image backbone. PCE is the official variable-tolerance event metric.

| Fold | Frame PCE % | Temporal PCE % | Frame MAE frames / ms | Temporal MAE frames / ms |
| --- | ---: | ---: | ---: | ---: |
| 2 | 43.2500 | 55.1429 | 23.980 / 800.083 | 13.983 / 466.535 |
| 3 | 42.7500 | 54.8214 | 23.298 / 777.288 | 13.923 / 464.514 |
| 4 | 44.3571 | 58.1071 | 30.261 / 1009.666 | 14.242 / 475.173 |
| additional-three-folds | 43.4524 | 56.0238 | 25.847 / 862.346 | 14.049 / 468.740 |

Separately registered full-clip context; do not substitute this for the primary benchmark:

| Fold | Frame PCE % | Temporal PCE % | Temporal MAE frames / ms |
| --- | ---: | ---: | ---: |
| 2 | 43.2500 | 62.1429 | 8.509 / 283.907 |
| 3 | 42.7500 | 63.2857 | 7.562 / 252.313 |
| 4 | 44.3571 | 64.2143 | 7.064 / 235.692 |
| additional-three-folds | 43.4524 | 63.2143 | 7.712 / 257.304 |

Primary per-event results pooled across the additional1050clips:

| Official event slot | Frame PCE % | Temporal PCE % | Frame MAE frames / ms | Temporal MAE frames / ms |
| --- | ---: | ---: | ---: | ---: |
| Address | 5.7143 | 10.7619 | 56.890 / 1898.114 | 31.009 / 1034.568 |
| Toe-up | 50.4762 | 60.2857 | 26.258 / 876.070 | 10.568 / 352.581 |
| Mid-backswing | 49.2381 | 62.6667 | 22.045 / 735.503 | 12.212 / 407.446 |
| Top | 34.9524 | 49.4286 | 14.332 / 478.168 | 13.686 / 456.609 |
| Mid-downswing | 59.7143 | 81.9048 | 16.756 / 559.055 | 9.052 / 302.030 |
| Impact | 66.4762 | 84.4762 | 31.425 / 1048.472 | 8.590 / 286.592 |
| Mid-follow-through | 70.6667 | 82.6667 | 15.611 / 520.865 | 7.927 / 264.464 |
| Finish | 10.3810 | 16.0000 | 23.454 / 782.518 | 19.351 / 645.633 |

The temporal primary fold PCE range is54.8214–58.1071%, descriptive population SD1.479percentage points; these are three folds, not independent population confidence bounds. Primary temporal timelines are strictly ordered for770/1050clips; full-clip temporal943/1050. Address10.7619% and Finish16.0% primary PCE remain weak. Peaks are forced independent event maxima, uncalibrated and unrepaired.

The raw class-order annotation anomaly677 is preserved and flagged. Official target slots are chronological occupied label frames; raw class indices remain available. Inclusive PCE tolerance is max(round((target[5]-target[0])/30),1), with NumPy default argsort-last peaks. Millisecond errors use actual cached reencoded PTS, not reconstructed FPS or original-source clock equivalence. All per-fold/per-event errors and descriptive variation are in the [small machine-readable result](../autonomy/reports/additional-folds-results-019.json).

## Scoped CPU timing

The registered one-shot timing used TRAIN IDs fold2=0/1,fold3=6/7,fold4=0/1,16warmup frames per fold,batch16,one repetition,one Torch CPU thread. Six observations cover only four unique clips. MPS/CUDA detection does not mean they were benchmarked.

| Fold | TRAIN clips | Timed frames | Shared extraction ms / frame | Mean whole-clip temporal head ms (two calls), reference64 / fullclip |
| --- | --- | ---: | ---: | ---: |
| 2 | [0, 1] | 462 | 60.920 | 8.261 / 5.459 |
| 3 | [6, 7] | 415 | 60.915 | 6.172 / 5.179 |
| 4 | [0, 1] | 462 | 62.949 | 10.844 / 5.589 |

Measured hardware: Apple M1, 8 GiB RAM, 8 logical CPUs, with one Torch CPU thread. MPS was detected and CUDA was unavailable; only CPU was timed. Full decode/preprocessing/backbone/head stage counts, timings and provenance are retained in the machine-readable result. Feature extraction is shared across the four head/context variants; composed pipeline time adds a head call to that shared extraction and is not a separately measured pipeline. Warmup/admission/integrity checks are excluded from stage timing. Per-frame/batch timing vectors are not retained; independent review checked their summary consistency and bounds, while recomputing the saved whole-clip head-call quantiles. One repetition is descriptive; it does not establish minimum hardware, app latency or population bounds.

## Scope and remaining gates

Historicalfold1 already had observed results when this fixed extension was registered, and additional-fold clips appeared in its training/inner data. The additional heads initialize fresh and use only their own TRAIN/inner partitions, but this is not a wholly prospective fourfold claim. Source grouping is not verified unseen-golfer generalization. No settings were retuned against these outer results. Frozen code/config/seed/registration/runtime/checkpoints and raw run hashes are recorded.

Normal automatic events, spatial pose measurements, supervised outcomes and coaching advice remain unavailable. Uploaded-domain crop/view/error reliability, independent pose references/identity and a permitted integrated demonstration remain separate gates. Reference comparisons do not define universal correct technique. Raw videos, caches, checkpoints, ledgers and probability arrays stay ignored. See [independent review](../autonomy/reports/review-019.md), [evaluation guide](additional-fold-evaluation.md), [timing guide](additional-fold-latency.md), [model card](model-card.md), and [remaining work](remaining-work.md).
