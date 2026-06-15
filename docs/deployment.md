# Deployment

## Production Topology

- Vercel hosts the Next.js interface.
- Modal hosts the FastAPI inference service on an NVIDIA T4.
- A Modal volume persists downloaded VideoMAE artifacts.
- `min_containers=0` removes idle GPU cost.
- `max_containers=1` bounds GPU concurrency and keeps runtime cost predictable.

## API

- `GET /health`
- `GET /profiles`
- `GET /samples`
- `GET /samples/{id}/thumbnail`
- `GET /samples/{id}/video`
- `POST /samples/{id}/analyze`
- `POST /analyze`

Uploads are limited to 50 MB by default and accept MP4, AVI, MOV, MKV, or WebM. Override the limit with `ARGUS_STREAM_A_MAX_UPLOAD_MB`.

## Deploy

```bash
modal run deployment/modal_app.py --prime-cache
modal deploy deployment/modal_app.py
cd deployment/vercel_app
vercel --prod
```

Set `NEXT_PUBLIC_ARGUS_API_URL` in Vercel. In production, set `ARGUS_STREAM_A_CORS_ORIGINS` to the exact Vercel origin.

## Operational Tradeoff

Scaling to zero introduces a cold start. The UI reports staged progress so users understand that the GPU is waking and the model is loading. Keeping one warm T4 would reduce latency but incur continuous cost.
