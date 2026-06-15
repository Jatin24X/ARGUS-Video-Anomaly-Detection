# ARGUS 

Frame-level video anomaly detection with frozen VideoMAE features, MULDE-style density scoring, Avenue benchmark reporting, and a deployed Vercel + Modal inference stack.

**Live demo:** https://anamolydetect.vercel.app  
**GPU API:** https://jatinsheoran2412--argus-stream-a-api-fastapi-app.modal.run  
**Primary benchmark:** Avenue frame-level evaluation

## Overview

ARGUS Stream A detects abnormal moments in surveillance-style video by learning the distribution of normal temporal behavior. The system extracts clip embeddings with a frozen VideoMAE-v2 backbone, fits a one-class density scorer, and converts clip-level anomaly evidence into a frame-level timeline with visual evidence frames.

The repository contains the complete Stream A workflow: dataset metadata, feature extraction, model selection, evaluation scripts, saved reports, FastAPI inference, Modal GPU deployment, a Next.js frontend, sample videos, and tests.

## Results

| Dataset | Role | Frame micro AUC | Frame macro AUC | Clip AUC |
|---|---|---:|---:|---:|
| Avenue | Primary frame-level benchmark | **84.51%** | **85.14%** | **84.00%** |
| UBnormal | Reference profile | **73.94%** | **84.10%** | **73.09%** |

Primary Avenue report:

```text
outputs/reports/avenue_stream_a_best_test.json
```

The Avenue path is reported as a full-frame, frame-centric pipeline. MULDE's Avenue headline result is object-centric, so the two protocols are documented separately instead of being mixed into a single leaderboard.

## System Architecture

```text
Video
  -> adaptive frame sampling
  -> frozen VideoMAE-v2 clip embeddings
  -> MULDE-style density scoring
  -> temporal smoothing
  -> frame-level anomaly timeline
  -> top-frame visual evidence
```

Training is one-class. Normal videos fit the scorer, while ground-truth frame labels are used for validation, checkpoint selection, and final benchmark reporting.

## Key Capabilities

- Full-frame anomaly scoring for Avenue and UBnormal profiles.
- Frozen VideoMAE-v2 backbone for reproducible temporal feature extraction.
- MULDE-style density scoring with log-density and GMM calibration.
- Normal-only holdout selection for model checkpointing.
- Frame-level micro/macro AUC reporting and clip-level diagnostics.
- Modal T4 GPU backend with model/profile preload and bounded uploads.
- Vercel + Next.js frontend with sample gallery, upload flow, anomaly timeline, and frame evidence.
- Contract tests for deployment endpoints and runtime behavior.

## Repository Layout

```text
configs/                 Dataset and training recipes
data/                    Metadata and compact feature artifacts
deployment/              FastAPI, Modal, and Vercel/Next.js apps
docs/                    Architecture, methodology, evaluation, deployment notes
outputs/                 Frozen checkpoints and benchmark reports
scripts/                 Training, extraction, selection, and evaluation CLIs
src/                     Data, model, evaluation, training, and inference modules
test_videos/             Prepared demo clips used by the live gallery
tests/                   API and deployment contract tests
```

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the local Gradio demo:

```bash
python demo.py
```

Run the FastAPI backend locally:

```bash
python deployment/app.py
```

Run the Vercel frontend locally:

```bash
cd deployment/vercel_app
npm install
npm run dev
```

## Evaluation

Reproduce the main Avenue frame-level report:

```bash
python scripts/eval_frame_level.py \
  --dataset avenue_stream_a \
  --checkpoint outputs/avenue_stream_a_ld_gmm1_beta01_lr4e5_run1/checkpoints/stream_a/best_holdout.pt \
  --split test \
  --output-json outputs/reports/avenue_stream_a_reproduced_test.json
```

Run the test suite:

```bash
pytest tests -q
cd deployment/vercel_app
npm run build
```

## Deployment

Deploy the Modal GPU API:

```bash
modal run deployment/modal_app.py --prime-cache
modal deploy deployment/modal_app.py
```

Deploy the Next.js frontend:

```bash
cd deployment/vercel_app
npx vercel --prod --yes
```

Production frontend environment variable:

```text
NEXT_PUBLIC_ARGUS_API_URL=https://jatinsheoran2412--argus-stream-a-api-fastapi-app.modal.run
```

Modal runs with `min_containers=0` to keep idle cost low. The first request after an idle period can take longer while the container starts and loads the model.

## API Surface

```text
GET  /health
GET  /profiles
GET  /samples
GET  /samples/{sample_id}/thumbnail
GET  /samples/{sample_id}/video
POST /samples/{sample_id}/analyze
POST /analyze
```

## Current Validation

- Python deployment contract tests pass.
- Next.js production build passes.
- Modal health endpoint reports a CUDA-ready service with Avenue and UBnormal profiles cached.
- The live frontend can load prepared samples and send analysis requests to the GPU API.

## Limitations

- Full-frame features are less spatially localized than object-centric pipelines.
- Avenue and UBnormal are different datasets and should be evaluated separately.
- AUC measures ranking quality; deployment thresholds still need calibration for a target operating environment.
- Cold starts are expected when Modal scales to zero after idle time.

## Tech Stack

Python, PyTorch, VideoMAE, scikit-learn, OpenCV, FastAPI, Modal, Next.js, Vercel, TypeScript.
