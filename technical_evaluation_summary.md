# Stream A Evaluation Summary

## System

ARGUS Stream A is a full-frame, frame-level video anomaly detection system. It uses frozen VideoMAE-v2 clip embeddings and MULDE-style density scoring to rank abnormal temporal behavior.

## Evaluation Contract

- Training uses normal videos only.
- Frame labels are reserved for validation and final benchmark reporting.
- Avenue is treated as the primary frame-level benchmark path.
- UBnormal is retained as a reference profile.
- Full-frame Stream A results are kept separate from object-centric Avenue results.

## Primary Avenue Result

| Metric | Value |
|---|---:|
| Frame micro AUC | `0.8451` |
| Frame macro AUC | `0.8514` |
| Clip AUC | `0.8400` |

Primary report:

```text
outputs/reports/avenue_stream_a_best_test.json
```

## Reference UBnormal Result

| Metric | Value |
|---|---:|
| Frame micro AUC | `0.7394` |
| Frame macro AUC | `0.8410` |
| Clip AUC | `0.7309` |

Reference report:

```text
outputs/reports/stream_a_frozen_baseline.json
```

## Avenue Pipeline

The Avenue path includes:

- dataset metadata support
- frame-label import and validation
- VideoMAE feature extraction
- normal-only holdout checkpoint selection
- log-density plus GMM score calibration
- temporal smoothing
- frame-level and clip-level reporting

## Protocol Note

MULDE's Avenue headline is object-centric. ARGUS Stream A reports a full-frame, frame-centric result. These are different evaluation surfaces and should be discussed separately.

## Deployment Surface

- Vercel frontend
- Modal T4 GPU backend
- FastAPI inference service
- sample-video gallery
- bounded custom uploads
- anomaly timeline visualization
- highest-scoring frame evidence

## Current Validation

- Python deployment contract tests pass.
- Next.js production build passes.
- Modal health endpoint reports cached Avenue and UBnormal profiles.
- The deployed frontend is connected to the Modal inference API.
