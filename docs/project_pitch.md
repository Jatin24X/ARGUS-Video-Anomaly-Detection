# Project Brief

## Short Description

ARGUS Stream A is a frame-level video anomaly detection system for ranking abnormal moments in surveillance-style video. It uses frozen VideoMAE-v2 temporal embeddings, MULDE-style density scoring, Avenue benchmark reporting, and a deployed Vercel + Modal inference stack.

## Technical Summary

- One-class training: normal videos fit the scorer; frame labels are used for validation and reporting.
- Frozen backbone: VideoMAE-v2 produces temporal clip embeddings without fine-tuning.
- Density scoring: low-likelihood clips are ranked as anomalous and projected to frame-level timelines.
- Avenue path: metadata validation, feature extraction, checkpoint selection, GMM score calibration, smoothing, and benchmark-safe reporting.
- Deployment: FastAPI inference on Modal T4 with a Vercel frontend for sample analysis, uploads, timelines, and visual evidence.

## Benchmark Snapshot

| Dataset | Frame micro AUC | Frame macro AUC | Clip AUC |
|---|---:|---:|---:|
| Avenue | 84.51% | 85.14% | 84.00% |
| UBnormal | 73.94% | 84.10% | 73.09% |

The Avenue result is full-frame and frame-centric. Object-centric Avenue results use a different input representation and should be discussed separately.

## Design Rationale

### Why one-class anomaly detection?

Real anomaly categories are open-ended and difficult to enumerate. Normal-only training keeps the detector focused on modeling expected temporal behavior, then ranks deviations at inference time.

### Why a frozen backbone?

Frozen VideoMAE features reduce compute cost, improve reproducibility, and isolate the anomaly-scoring layer from expensive backbone training.

### Why Modal and Vercel?

Vercel provides a fast frontend deployment path. Modal provides Python-native GPU inference with persistent model caching and scale-to-zero cost control.

### What would improve the system next?

The highest-impact next branch would add region-aware localization for small anomalies while preserving the current full-frame benchmark path as the reference system.
