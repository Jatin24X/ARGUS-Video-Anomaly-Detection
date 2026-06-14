# ARGUS Stream A: Video Anomaly Detection

ARGUS Stream A is a production-style, unsupervised video anomaly detection system. It learns normal temporal behavior from video clips, scores unusual moments with density modeling, and exposes the result through a live Vercel + Modal GPU demo.

**Live demo:** https://anamolydetect.vercel.app  
**Backend:** https://jatinsheoran2412--argus-stream-a-api-fastapi-app.modal.run  
**Primary benchmark:** Avenue frame-level evaluation

## Why This Project Matters

Most anomaly-detection demos stop at notebooks or offline scores. ARGUS Stream A is built as an end-to-end ML system:

- data preparation and metadata validation
- VideoMAE feature extraction
- one-class anomaly scoring with a MULDE-style density model
- checkpoint selection and frame-level evaluation
- FastAPI inference service
- Modal T4 GPU deployment with model preload
- Vercel frontend with sample gallery, upload flow, timeline visualization, and frame evidence

The project is intentionally honest about benchmark protocol. The MULDE paper's Avenue headline is object-centric; this project reports a full-frame, frame-centric Stream A pipeline and documents that difference clearly.

## Results

| Dataset | Role | Frame micro AUC | Frame macro AUC | Clip AUC |
|---|---|---:|---:|---:|
| Avenue | Primary benchmark | **84.51%** | **85.14%** | **84.00%** |
| UBnormal | Reference profile | **73.94%** | **84.10%** | **73.09%** |

The Avenue frame-centric bring-up baseline was `77.38%` micro AUC. The final benchmark-safe recipe reaches `84.51%`, a `+7.13` point absolute improvement.

Primary report:

```text
outputs/reports/avenue_stream_a_best_test.json
```

## What I Improved

The main contribution was not just running an existing repo. I built and improved a missing frame-centric Avenue path:

- imported and validated Avenue labels
- added Avenue feature extraction and metadata support
- implemented normal-only holdout checkpoint selection
- moved the scoring surface to `log_density + GMM`
- tuned temporal smoothing for frame-level AUC
- fixed numerical and portability issues in GMM sidecar loading
- packaged the final system as a deployed GPU application

## Architecture

```text
Video input
  -> adaptive frame sampling
  -> frozen VideoMAE-v2 clip embeddings
  -> MULDE-style density scoring
  -> temporal smoothing
  -> anomaly timeline + top-frame evidence
```

Training is one-class: only normal videos are used for fitting the scorer. Ground-truth frame labels are used for validation and final benchmarking, not as supervised anomaly classes.

## Deployed System

Frontend:

- Next.js 16
- Vercel production deployment
- sample video gallery from `test_videos`
- upload flow for custom short clips
- anomaly timeline SVG
- top-scoring frame evidence
- profile switcher for Avenue and UBnormal

Backend:

- FastAPI
- Modal T4 GPU
- persistent Hugging Face cache
- model/profile preload
- bounded upload validation
- sample-video endpoints
- thumbnail generation with OpenCV and FFmpeg fallback

Key API endpoints:

```text
GET  /health
GET  /profiles
GET  /samples
GET  /samples/{sample_id}/thumbnail
GET  /samples/{sample_id}/video
POST /samples/{sample_id}/analyze
POST /analyze
```

## Repository Structure

```text
configs/                 Dataset and training recipes
data/                    Metadata and compact feature artifacts
deployment/              FastAPI, Modal, and Vercel/Next.js apps
docs/                    Architecture, methodology, evaluation, deployment notes
outputs/                 Frozen checkpoints and benchmark reports
scripts/                 Training, feature extraction, selection, evaluation CLIs
src/                     Data, model, evaluation, training, inference modules
test_videos/             Prepared demo clips used by the live gallery
tests/                   API and deployment contract tests
```

## Quick Start

```bash
pip install -r requirements.txt
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

Run checks:

```bash
pytest tests -q
cd deployment/vercel_app
npm run build
```

## Reproduce The Main Avenue Evaluation

```bash
python scripts/eval_frame_level.py \
  --dataset avenue_stream_a \
  --checkpoint outputs/avenue_stream_a_ld_gmm1_beta01_lr4e5_run1/checkpoints/stream_a/best_holdout.pt \
  --split test \
  --output-json outputs/reports/avenue_stream_a_reproduced_test.json
```

## Deployment

Modal:

```bash
modal run deployment/modal_app.py --prime-cache
modal deploy deployment/modal_app.py
```

Vercel:

```bash
cd deployment/vercel_app
npx vercel --prod --yes
```

Production environment variable:

```text
NEXT_PUBLIC_ARGUS_API_URL=https://jatinsheoran2412--argus-stream-a-api-fastapi-app.modal.run
```

## Validation

Current verified checks:

- `pytest tests -q` passes
- `npm run build` passes on Next.js 16
- Modal `/health` reports CUDA-ready service with cached Avenue and UBnormal profiles
- `/samples` returns seven prepared demo clips
- real deployed sample analysis completes successfully on `avenue-1`

## Limitations

- Full-frame features are less localized than object-centric pipelines.
- Avenue and UBnormal are different datasets and should not be merged into one leaderboard.
- AUC measures ranking quality; a production alerting system would still need threshold calibration.
- Modal `min_containers=0` reduces cost but introduces cold-start latency after idle periods.

## Interview Summary

> I built a full-stack unsupervised video anomaly detection system. The core ML pipeline uses frozen VideoMAE features and MULDE-style density scoring. My main contribution was bringing up and improving the missing frame-centric Avenue path, moving micro AUC from 77.38% to 84.51%, then deploying the model as a real GPU-backed web application with sample-video analysis, upload validation, timelines, and visual evidence.

## Tech Stack

Python, PyTorch, VideoMAE, scikit-learn, OpenCV, FastAPI, Modal, Next.js, Vercel, TypeScript.
