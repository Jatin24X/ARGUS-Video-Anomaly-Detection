# Architecture

## System Boundary

ARGUS Stream A separates offline model development from online inference:

1. Training and evaluation scripts produce frozen scorer checkpoints and JSON reports.
2. `src/inference` exposes the stable runtime contract used by deployment code.
3. FastAPI validates requests, resolves profiles, and invokes the inference engine.
4. Modal packages the model, persistent Hugging Face cache, T4 GPU, and sample videos.
5. The Vercel application handles discovery, upload, progress feedback, and visualization.

## Inference Path

The engine decodes a video, adaptively samples frames, forms temporal clips, and extracts embeddings with a frozen VideoMAE-v2 Base backbone. A MULDE-style scorer estimates the density of each clip representation. Lower-density observations become higher anomaly scores, which are aligned to frames and temporally smoothed.

## Deployment Path

Prepared samples live with the Modal image. The browser requests thumbnails and previews from the API, then calls a sample-specific analysis endpoint. This avoids uploading a bundled sample back to the server. User uploads are streamed to a bounded temporary file and deleted after inference.

The service validates file type and size before model execution. Modal keeps the Hugging Face model cache in a persistent volume while allowing the T4 container to scale to zero.
