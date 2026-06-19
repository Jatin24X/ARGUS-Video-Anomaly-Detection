# ARGUS: Unsupervised Video Anomaly Detection Platform

[![CI Status](https://github.com/Jatin24X/ARGUS---Video-Anomaly-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Jatin24X/ARGUS---Video-Anomaly-Detection/actions)
[![FastAPI](https://img.shields.shields.safe/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Modal](https://img.shields.safe/badge/Serverless-Modal-blue.svg)](https://modal.com)
[![Next.js](https://img.shields.safe/badge/Frontend-Next.js-black.svg)](https://nextjs.org)

ARGUS is a frame-level video anomaly detection platform designed for automated exception tracking in surveillance streams. The system leverages a frozen **VideoMAE-v2** self-supervised Vision Transformer (ViT) backbone, multiscale **MULDE** density estimation, and 1-component **GMM** score calibration to rank and visualize temporal deviations with high mathematical precision.

* **Live Developer Console:** [https://anamolydetect.vercel.app](https://anamolydetect.vercel.app)
* **Serverless GPU API:** [https://jatinsheoran2412--argus-stream-a-api-argusapi-fastapi-app.modal.run](https://jatinsheoran2412--argus-stream-a-api-argusapi-fastapi-app.modal.run)
* **Primary Evaluation Benchmark:** Avenue frame-level validation protocol

---

## ⚡ Inference Pipeline Architecture

```text
                                    SURVEILLANCE INFERENCE PIPELINE
                                    
  [Raw Video File]
          │
          ▼
   [Video Decoder]    ──► Optional Spatial ROI Sector Slicing (Left, Center, Right cropping)
  (OpenCV & FFmpeg)
          │
          ▼
  [Adaptive Sampler]  ──► Downsamples to uniform 12.0 FPS; caps at 720 frames (avoids GPU OOM)
          │
          ▼
   [VideoMAE-v2]      ──► Auto-permutes input batch shape [B, C, T, H, W]
   (Frozen Backbone)      Extracts 768-dim mean-pooled spatiotemporal embeddings per clip
          │
          ▼
    [MULDE Scorer]    ──► Evaluates multiscale log-likelihood density under normal-only training distribution
   (Density Core)
          │
          ▼
   [GMM Calibrator]   ──► Scales raw densities to unified outlier probabilities P(anomaly|x) in [0, 1]
          │
          ▼
  [Gaussian Smoother] ──► Applies 1D temporal Gaussian kernel (sigma=13/20) + MinMax normalization
          │
          ▼
 [Dynamic Threshold]  ──► Client-side percentile slider (50th-99th) isolates alert intervals instantly
```

---

## 🔬 Core Machine Learning Design

### 1. Unsupervised One-Class Density Estimation
surveillance anomalies are structurally diverse and impossible to fully catalog during training. ARGUS treats anomaly detection as an **unsupervised one-class density estimation** problem:
* **Training Protocol**: The model is trained strictly on normal, non-anomalous sequences to map expected human and environmental behaviors.
* **Inference Mechanism**: Clips with low-likelihood density under the normal distribution are flagged as anomalies. Frame-level annotations are reserved exclusively for validation splits and final metrics reporting.

### 2. Spatiotemporal Feature Extraction (VideoMAE-v2)
We employ a frozen **VideoMAEv2-Base** (86.2M parameters) model as our core feature extractor. By passing overlapping 16-frame clips (sliding window stride of 4 raw frames), the backbone captures high-density temporal dependencies.
* **FP16 GPU Load Optimization**: To minimize serverless container cold-starts, we bypass the standard HuggingFace `from_pretrained` meta-tensor initialization. We instantiate the config-defined architecture on the target device directly in `float16` precision, downloading and mapping safetensors parameters directly to GPU VRAM. This cuts initialization latency and CPU-to-GPU memory copying overhead.

### 3. Log-Density & GMM Calibration
Raw density scores vary dramatically depending on the video's lighting, background movement, and frame depth. 
* To ensure standard decision thresholds are statistically valid across different camera angles and datasets, we calibrate the raw outputs using a **1-component Gaussian Mixture Model (GMM)**. The GMM projects raw density values into a standardized probability scale $P(\text{anomaly}|x) \in [0, 1]$.
* Anomaly scores are smoothed temporally with a 1D Gaussian kernel to enforce continuity and prevent high-frequency false positives (e.g., compression artifacts).

---

## 📊 Academic Evaluation Results

| Dataset | Metric Protocol | Frame Micro AUC | Frame Macro AUC | Clip AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Avenue** | Primary full-frame validation | **84.51%** | **85.14%** | **84.00%** |
| **UBnormal** | Reference benchmark baseline | **73.94%** | **84.10%** | **73.09%** |

> [!IMPORTANT]
> **Evaluation Protocol Note**: Standard MULDE and Avenue baselines are frequently evaluated using *object-centric* annotations (focusing on localized bounding-box regions). ARGUS Stream A reports a *full-frame, frame-centric* result. Because the spatial granularity and input representations differ, these metrics should be analyzed separately.

---

## 💻 Interactive Developer Console (Next.js)

The Vercel-deployed frontend implements a high-density, premium developer dashboard designed to show deep system visibility:
1. **Interactive SVG Architecture Map**: Visualizes the 7-stage ML pipeline. Clicking any node expands its input/output tensor shapes, engineering rationales, and provides direct clickable links to the underlying python source lines in the GitHub repository.
2. **Instant Demo Default (Cost-Saving)**: Defaults to cached edge delivery of analysis results (0ms compute) for instant sample playbacks, ensuring 0ms cold-starts and strict backend cost control.
3. **Live GPU Worker Switcher**: Operators can toggle the mode to "Live GPU Worker" to wake up/ping the serverless GPU worker (active health checks start only when live is selected).
4. **Resilient Cached Backup**: If the serverless GPU times out or is offline during a live run, the frontend automatically falls back to local static pre-cached JSONs to ensure zero-downtime recruiter inspection.
5. **Active Visibility Heartbeat**: Maintains container warmth with a 10s silent heartbeat only while the tab is active **and** Live GPU mode is toggled, scaling to zero 15s after the user leaves or switches tabs to minimize serverless costs.
6. **Dynamic Percentile Slider**: Recalculates anomaly alert thresholds and redraws timelines on the client side in real-time, separating threshold selection from GPU compute.
7. **Active Siren HUD & Captured Keyframes**: Warning banners flash red during anomalous intervals, and Captured Keyframe cards show cropped frames at peak anomaly moments. Click-seeking on cards snaps the video directly to the event timestamp.

---

## 🛠️ Repository Layout

```text
configs/                 Dataset configurations and density/GMM hyperparameters
data/                    Dataset metadata files (Avenue/UBnormal index sheets)
deployment/              FastAPI backend, Modal serverless configuration, Next.js code
  ├── vercel_app/        Next.js dashboard web application
  ├── app.py             FastAPI backend serving inference and cache endpoints
  └── modal_app.py       Modal serverless wrapper configuring T4/L4 GPU scale-to-zero
docs/                    Detailed methodology, audits, and deployment specifications
outputs/                 Pretrained scoring model checkpoints and benchmark reports
scripts/                 CLI commands for feature extraction, GMM calibration, and evaluation
test_videos/             Surveillance clips used by the interactive gallery console
tests/                   Pytest backend validation and integration contract suite
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Local Gradio Dashboard
Test the pipeline interactively on your local machine using the Gradio UI:
```bash
python demo.py
```

### 3. Run FastAPI Backend locally
To start the API server locally:
```bash
python deployment/app.py
```

### 4. Run Next.js Frontend locally
Navigate to the vercel application and start the development server:
```bash
cd deployment/vercel_app
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the console dashboard.

---

## 🧪 Verification & Cloud Deployment

### Run Unit & Integration Tests
Ensure the python backend contracts and pipeline operations pass validation:
```bash
pytest tests/ -v
```

### Deploy Modal Serverless API
Modal provides scale-to-zero T4 GPU hosting. Deploy your API cluster with these commands:
```bash
# Setup credentials
modal setup

# Prime HuggingFace safetensors weights cache volume
modal run deployment/modal_app.py --prime-cache

# Deploy class-based ASGI service
modal deploy deployment/modal_app.py
```

### Deploy Vercel Frontend
```bash
cd deployment/vercel_app
npx vercel --prod --yes
```
*Note: Make sure your Vercel project's `NEXT_PUBLIC_ARGUS_API_URL` environment variable is configured to your active Modal endpoint.*
