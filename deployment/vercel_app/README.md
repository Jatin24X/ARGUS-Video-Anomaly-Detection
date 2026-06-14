# Vercel Frontend

This Next.js app is the frontend for the `ARGUS Stream A` Modal FastAPI backend.

## Environment

Create a `.env.local` file from `.env.example`:

```bash
NEXT_PUBLIC_ARGUS_API_URL=https://your-modal-fastapi-url.modal.run
```

## Local Run

```bash
npm install
npm run dev
```

The app expects the FastAPI backend to expose:

- `GET /profiles`
- `GET /samples`
- `POST /samples/{sample_id}/analyze`
- `POST /analyze`
- `GET /health`

## Vercel Deployment

1. Import this `vercel_app` folder as the Vercel project root.
2. Set `NEXT_PUBLIC_ARGUS_API_URL` in Vercel project settings.
3. Deploy.
