# Deployment

This folder contains the new `Vercel + Modal FastAPI` deployment path for `ARGUS Stream A`.

## Layout

- `app.py` - FastAPI backend entrypoint
- `modal_app.py` - Modal deployment entrypoint for the backend
- `requirements.txt` - pinned production-only dependency set
- `run_api.bat` - runs the FastAPI backend locally
- `prime_modal_cache.bat` - warms the persistent Hugging Face cache on Modal
- `deploy_modal.bat` - deploys the Modal backend
- `vercel_app/` - Next.js frontend for Vercel

## Backend: local run

From the project root:

```bash
pip install -r requirements.txt
python deployment/app.py
```

The backend serves:

- `GET /health`
- `GET /profiles`
- `GET /samples`
- `GET /samples/{sample_id}/thumbnail`
- `GET /samples/{sample_id}/video`
- `POST /samples/{sample_id}/analyze`
- `POST /analyze`

## Modal deployment

1. Make sure Modal is configured:

```bash
modal setup
```

2. Prime the VideoMAE Hugging Face cache:

```bash
modal run deployment/modal_app.py --prime-cache
```

3. Deploy the FastAPI backend:

```bash
modal deploy deployment/modal_app.py
```

### Runtime choices

- GPU: `T4`
- `min_containers=0`
- `max_containers=1`
- persistent HF cache volume for VideoMAE downloads
- eager profile preload at container startup

Because `min_containers=0`, the first request after idle will cold start. After the container is warm, analysis requests should be much faster.

The `test_videos/` folder is packaged into the Modal image. Sample analysis runs
directly against those server-local files and does not re-upload them through
the browser.

## Vercel frontend

The frontend lives in `vercel_app/`.

1. Set the Vercel project root to `deployment/vercel_app`
2. Configure:

```bash
NEXT_PUBLIC_ARGUS_API_URL=https://your-modal-fastapi-url.modal.run
```

3. Deploy the Vercel project
