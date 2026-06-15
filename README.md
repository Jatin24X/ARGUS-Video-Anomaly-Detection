# ARGUS 

**Unsupervised Video Anomaly Detection Platform**  
Frame-level exception tracking in surveillance video using frozen **VideoMAE-v2** features, multiscale **MULDE** density estimation, GMM score calibration, and an interactive Grafana/Datadog-grade developer console.

[![CI Status](https://github.com/Jatin24X/ARGUS---Video-Anomaly-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Jatin24X/ARGUS---Video-Anomaly-Detection/actions)
[![FastAPI](https://img.shields.safe/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Modal](https://img.shields.safe/badge/Serverless-Modal-blue.svg)](https://modal.com)
[![Next.js](https://img.shields.safe/badge/Frontend-Next.js-black.svg)](https://nextjs.org)

* **Live Console Dashboard:** https://anamolydetect.vercel.app  
* **Serverless GPU API:** https://jatinsheoran2412--argus-stream-a-api-fastapi-app.modal.run  
* **Primary Benchmark:** Avenue frame-level evaluation

---

## ⚡ Phase 2 Dashboard & ML Optimizations

In Phase 2, we upgraded the frontend from a basic interface to a high-density, professional developer console dashboard, introducing **12 key features**:

1. **Interactive SVG Node-Based Architecture Map**: Visual flowchart representing the video inference pipeline. Clicking nodes expands technical details, input/output tensor shapes, and model design choices.
2. **Strict Anti-AI-Slop Styling**: Matte-charcoal panels, glowing telemetry readouts, thin borders, and mouse-reactive radial gradient lighting mimicking enterprise-grade cloud consoles.
3. **0ms Client-Side Demo Caching**: Pre-cached full-timeline analysis results and visual scene thumbnails for gallery clips. Demo videos load immediately in under 10ms, avoiding serverless T4 GPU cold-starts for recruiters.
4. **Interactive Anomaly Sensitivity Slider**: Dynamic client-side percentile thresholding (50th-99th) that recalculates anomaly intervals and redraws charts in real-time.
5. **Real-time Playhead & Hover Guide**: A moving vertical guide synchronized with HTML5 video progress, alongside cursor-guided hover tooltips detailing timestamps and anomaly scores.
6. **🚨 Active Surveillance Siren HUD**: A warning banner that flashes red during anomalous frame intervals.
7. **Chronological Surveillance Event Log**: Structured list of exception intervals. Clicking an entry seeks the video player to the start of the anomalous event.
8. **Interactive Visual Evidence Cards**: Highest-scoring anomaly frame thumbnails that scale up on hover. Clicking seeks the player to that exact frame.
9. **Exportable Inspection Reports**: Allows downloading timeline scores, settings, and detected events in a standardized JSON schema.
10. **Spatial ROI Sector Selector**: Supports frame cropping (`full`, `center`, `left`, `right`) before ViT feature extraction to isolate pedestrian lanes and ignore background noise (e.g. swaying trees).
11. **Telemetry & Latency Profiler HUD**: Telemetry indicators displaying hardware devices, pipeline latency, and inference speed (FPS).
12. **DevOps Ruff Style Guide**: Configured a root `pyproject.toml` file to enforce python code formatting and rules.

---

## Technical Results

| Dataset | Metric Role | Frame Micro AUC | Frame Macro AUC | Clip AUC |
|---|---|---:|---:|---:|
| **Avenue** | Primary frame-level evaluation | **84.51%** | **85.14%** | **84.00%** |
| **UBnormal** | Reference comparison profile | **73.94%** | **84.10%** | **73.09%** |

* **Validation Protocol:** Unsupervised One-Class Learning. Only normal sequences are used during training. Frame labels are kept strictly for validation and final benchmarking to maintain real-world deployment conditions.
* **Backbone Backbone:** Frozen `VideoMAEv2-Base` (86.2M parameters) extracts spatio-temporal representations (`[B, 3, 16, 224, 224]` -> `[B, 768]`), preventing overfitting and ensuring robust zero-shot feature transfer.

---

## Inference Pipeline Architecture

```text
[Video Stream]
      │
      ▼
[Video Decoder (OpenCV FFmpeg)] ──► Optional Spatial ROI Sector Slicing (Left, Center, Right)
      │
      ▼
[Adaptive Frame Sampler] ────────► Downsamples native FPS to 12.0 FPS (Capped at 720 frames)
      │
      ▼
[VideoMAE-v2 ViT Backbone] ──────► Extracts 768-dim temporal feature vectors per clip
      │
      ▼
[MULDE Density Core] ────────────► Estimates multiscale log-likelihood density scores
      │
      ▼
[GMM Calibrator] ────────────────► Scales raw scores through a 1-component Gaussian Mixture Model
      │
      ▼
[Gaussian Smoothing & Norm] ─────► Applies 1D temporal Gaussian kernel (sigma=13/20) + MinMax scale
      │
      ▼
[Dynamic Threshold Filter] ──────► Draggable percentile slider (50th-99th) flags anomalous regions
```

---

## Repository Layout

```text
configs/                 Dataset configurations and GMM/density settings
data/                    Dataset metadata and compact feature embeddings
deployment/              FastAPI backend, Modal serverless files, and Next.js app
docs/                    Architecture, evaluation methodology, and deployment notes
outputs/                 Pretrained scoring models and benchmark reports
scripts/                 CLI commands for feature extraction, training, and evaluation
test_videos/             Short MP4 videos utilized by the gallery dashboard
tests/                   FastAPI and deployment contract verification suites
pyproject.toml           Ruff linter and style configurations
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run local Python demos
Run the Gradio demonstration dashboard:
```bash
python demo.py
```

Or start the FastAPI backend server:
```bash
python deployment/app.py
```

### 3. Run the Next.js Frontend locally
```bash
cd deployment/vercel_app
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the console dashboard.

---

## Verification & Tests

Ensure the python backend contract tests pass:
```bash
pytest tests/ -v
```

Ensure the frontend builds without compiler errors:
```bash
cd deployment/vercel_app
npm run build
```

---

## Cloud Deployment

### Deploys the Modal GPU Service
```bash
# Warm model safetensors weight cache on Modal Volume
modal run deployment/modal_app.py --prime-cache

# Deploy class-based ASGI service on T4 GPU
modal deploy deployment/modal_app.py
```

### Deploys the Vercel Frontend
```bash
cd deployment/vercel_app
npx vercel --prod --yes
```
*Make sure your Vercel project has `NEXT_PUBLIC_ARGUS_API_URL` set to your Modal endpoint.*
