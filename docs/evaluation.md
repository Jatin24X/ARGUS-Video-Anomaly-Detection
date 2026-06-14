# Evaluation

## Primary Contract

The primary claim is the frozen Avenue report at:

`outputs/reports/avenue_stream_a_best_test.json`

| Metric | Value |
|---|---:|
| Frame micro AUC | 0.8451 |
| Frame macro AUC | 0.8514 |
| Clip AUC | 0.8400 |

Micro AUC pools all test frames. Macro AUC computes per-video AUC and averages the valid video scores. Clip AUC evaluates the temporal clip surface.

## Baseline And Improvement

The initial Avenue frame-centric bring-up achieved `0.7738` micro AUC. The frozen improved recipe achieved `0.8451`, an absolute gain of `0.0713`.

## Diagnostic Result

`outputs/reports/avenue_stream_a_best_test_high_micro_diag.json` records `0.8466` micro AUC. It is retained as a post-hoc diagnostic and is not substituted for the primary benchmark-safe claim.

## Comparison Rule

Object-centric methods crop and model detected objects, while this system scores full-frame temporal features. Results from those protocols are useful context but are not treated as directly comparable.
