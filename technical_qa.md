# Technical Q&A

## What is Stream A?

Stream A is the full-frame anomaly detection path in ARGUS. It uses VideoMAE-v2 clip embeddings and a MULDE-style density scorer to rank abnormal moments over time.

## Is it supervised?

No. It is one-class anomaly detection. Normal videos fit the scorer; labels are reserved for validation, checkpoint selection, and final metrics.

## What does the demo show?

The demo takes a video, extracts temporal embeddings, scores anomaly over time, plots the frame-level timeline, and displays the highest-scoring frames.

## What is the primary benchmark result?

The primary Avenue result is:

- micro AUC: `0.8451`
- macro AUC: `0.8514`
- clip AUC: `0.8400`

## Why not directly compare to object-centric Avenue scores?

Object-centric methods score cropped object tracks or regions. ARGUS Stream A scores the full frame. The input representation and scoring granularity are different, so direct comparison would be misleading without that protocol note.

## Why use VideoMAE-v2?

VideoMAE-v2 provides strong temporal video representations. Keeping it frozen makes the system reproducible and keeps the anomaly modeling focused on the density scorer.

## What is the anomaly scorer?

The scorer is MULDE-style density modeling. Low-likelihood clip embeddings receive higher anomaly scores, and the scores are smoothed into a frame-level curve.

## What changes matter most?

- Avenue metadata and feature support.
- Normal-only holdout checkpoint selection.
- Log-density and GMM score calibration.
- Temporal smoothing for frame-level reporting.
- Stable model-sidecar loading for deployment.

## What is the deployment architecture?

The frontend runs on Vercel. The backend is a FastAPI service on Modal T4 GPU with cached model artifacts, sample-video endpoints, upload validation, and frame evidence generation.

## What should improve next?

The most direct next branch is region-aware scoring for small anomalies while preserving the full-frame Stream A path as the benchmark reference.
