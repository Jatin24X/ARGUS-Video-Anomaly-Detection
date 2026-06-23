from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Final

import cv2
import uvicorn
import uuid
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = PROJECT_ROOT / "test_videos"
os.environ.setdefault("ARGUS_STREAM_A_ROOT", str(PROJECT_ROOT))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ENGINE, PROFILES, preload, profile_payload  # noqa: E402

LOGGER = logging.getLogger("argus.stream_a.api")
logging.basicConfig(
    level=os.environ.get("ARGUS_STREAM_A_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

ALLOWED_VIDEO_SUFFIXES: Final = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES: Final = int(
    float(os.environ.get("ARGUS_STREAM_A_MAX_UPLOAD_MB", "50")) * 1024 * 1024
)

JOBS: dict[str, dict[str, object]] = {}


def run_async_analysis(
    job_id: str,
    video_path: Path,
    profile_label: str,
    roi_sector: str,
    bypass_cache: bool = False,
    delete_source: bool = False,
) -> None:
    JOBS[job_id] = {
        "status": "processing",
        "progress_pct": 5,
        "step": "Opening video",
        "result": None,
        "error": None,
    }

    def progress_cb(fraction: float, desc: str) -> None:
        JOBS[job_id]["progress_pct"] = int(fraction * 100)
        JOBS[job_id]["step"] = desc

    try:
        payload = ENGINE.analyze_payload(
            video_path,
            profile_label,
            roi_sector=roi_sector,
            bypass_cache=bypass_cache,
            progress_callback=progress_cb,
        )
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress_pct"] = 100
        JOBS[job_id]["step"] = "Analysis finished"
        JOBS[job_id]["result"] = payload
    except Exception as exc:
        LOGGER.exception("Error during async analysis job %s", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)
    finally:
        if delete_source and video_path.exists():
            try:
                video_path.unlink(missing_ok=True)
            except Exception as e:
                LOGGER.warning("Could not delete temp file %s: %s", video_path, e)


def _sample_catalog() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    if not SAMPLE_ROOT.exists():
        return samples

    for path in sorted(SAMPLE_ROOT.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_VIDEO_SUFFIXES:
            continue

        dataset = "Avenue" if path.stem.lower().startswith("avenue") else "UBnormal"
        sample_id = path.stem.lower().replace("_", "-").replace(" ", "-")
        samples.append(
            {
                "id": sample_id,
                "title": path.stem.replace("-", " ").replace("_", " "),
                "dataset": dataset,
                "profile": dataset,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "video_url": f"/samples/{sample_id}/video",
                "thumbnail_url": f"/samples/{sample_id}/thumbnail",
            }
        )
    return samples


def _sample_path(sample_id: str) -> Path:
    for sample in _sample_catalog():
        if sample["id"] == sample_id:
            return SAMPLE_ROOT / str(sample["filename"])
    raise HTTPException(status_code=404, detail="Sample video not found.")


def _sample_payload(sample: dict[str, object]) -> dict[str, object]:
    base_url = os.environ.get("ARGUS_STREAM_A_PUBLIC_API_URL", "").rstrip("/")
    payload = dict(sample)
    if base_url:
        payload["video_url"] = f"{base_url}{sample['video_url']}"
        payload["thumbnail_url"] = f"{base_url}{sample['thumbnail_url']}"
    return payload


def _resolve_profile_label(profile_value: str) -> str:
    if profile_value in PROFILES:
        return profile_value

    for label, profile in PROFILES.items():
        if profile.key == profile_value or profile.dataset_name == profile_value:
            return label

    raise HTTPException(status_code=400, detail=f"Unknown profile: {profile_value}")


def preload_profiles(*, include_extractor: bool = True) -> None:
    preload(include_extractor=include_extractor)


def _cors_origins() -> tuple[list[str], bool]:
    raw = os.environ.get("ARGUS_STREAM_A_CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"], True
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins, False


def create_fastapi_app(*, preload: bool = False) -> FastAPI:
    if preload:
        preload_profiles()

    app = FastAPI(title="ARGUS Stream A API", version="1.1.0")

    origins, allow_all = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, object]:
        return {
            "service": "ARGUS Stream A API",
            "status": "ok",
            "endpoints": [
                "/health",
                "/profiles",
                "/samples",
                "/samples/{sample_id}/analyze",
                "/analyze",
            ],
        }

    @app.get("/health")
    def health() -> dict[str, object]:
        ready = ENGINE.extractor is not None and len(ENGINE.scorers) == len(PROFILES)
        return {
            "status": "ready" if ready else "warming",
            "version": app.version,
            "device": ENGINE.device,
            "cached_profiles": sorted(ENGINE.scorers.keys()),
            "extractor_loaded": ENGINE.extractor is not None,
            "sample_count": len(_sample_catalog()),
            "max_upload_mb": round(MAX_UPLOAD_BYTES / (1024 * 1024)),
        }

    @app.get("/profiles")
    def profiles() -> dict[str, object]:
        return {
            "profiles": [
                profile_payload(profile)
                for profile in (PROFILES[label] for label in PROFILES)
            ]
        }

    @app.get("/samples")
    def samples() -> dict[str, object]:
        return {"samples": [_sample_payload(sample) for sample in _sample_catalog()]}

    @app.get("/samples/{sample_id}/video")
    def sample_video(sample_id: str) -> FileResponse:
        path = _sample_path(sample_id)
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/samples/{sample_id}/thumbnail")
    def sample_thumbnail(sample_id: str) -> Response:
        path = _sample_path(sample_id)
        capture = cv2.VideoCapture(str(path))
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 3))
            ok, frame = capture.read()
        finally:
            capture.release()

        encoded_bytes: bytes | None = None
        if ok and frame is not None:
            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 82],
            )
            if success:
                encoded_bytes = encoded.tobytes()

        if encoded_bytes is None:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    "1",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-",
                ],
                check=False,
                capture_output=True,
            )
            if result.returncode != 0 or not result.stdout:
                raise HTTPException(
                    status_code=422,
                    detail="Could not decode sample thumbnail.",
                )
            encoded_bytes = result.stdout

        return Response(
            content=encoded_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/samples/{sample_id}/analyze", status_code=202)
    def analyze_sample(
        sample_id: str,
        background_tasks: BackgroundTasks,
        profile: str | None = None,
        roi_sector: str = "full",
        bypass_cache: bool = False,
    ) -> dict[str, object]:
        path = _sample_path(sample_id)
        catalog_entry = next(
            sample for sample in _sample_catalog() if sample["id"] == sample_id
        )
        profile_label = _resolve_profile_label(profile or str(catalog_entry["profile"]))
        
        from src.inference.engine import generate_video_hash
        video_hash = generate_video_hash(path)
        LOGGER.info(f"Analyzing sample video {sample_id} (bypass_cache={bypass_cache}) with content hash: {video_hash}")
        
        job_id = f"job-{uuid.uuid4()}"
        background_tasks.add_task(
            run_async_analysis,
            job_id,
            path,
            profile_label,
            roi_sector,
            bypass_cache=bypass_cache,
            delete_source=False,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.post("/analyze", status_code=202)
    async def analyze(
        background_tasks: BackgroundTasks,
        profile: str = Form(...),
        roi_sector: str = Form("full"),
        bypass_cache: bool = Form(False),
        video: UploadFile = File(...),
    ) -> dict[str, object]:
        profile_label = _resolve_profile_label(profile)
        filename = video.filename or "upload.mp4"
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=(
                    "Unsupported video format. Allowed: "
                    f"{', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}"
                ),
            )
        if video.content_type and not video.content_type.startswith("video/"):
            raise HTTPException(status_code=415, detail="Uploaded file must be a video.")

        temp_path: Path | None = None
        uploaded_bytes = 0
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = Path(temp_file.name)
                while True:
                    chunk = await video.read(1024 * 1024)
                    if not chunk:
                        break
                    uploaded_bytes += len(chunk)
                    if uploaded_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Video exceeds the "
                                f"{round(MAX_UPLOAD_BYTES / (1024 * 1024))} MB upload limit."
                            ),
                        )
                    temp_file.write(chunk)
        except Exception as exc:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise exc
        finally:
            await video.close()

        from src.inference.engine import generate_video_hash
        video_hash = generate_video_hash(temp_path)
        LOGGER.info(f"Analyzing uploaded video with content hash: {video_hash}")

        job_id = f"job-{uuid.uuid4()}"
        background_tasks.add_task(
            run_async_analysis,
            job_id,
            temp_path,
            profile_label,
            roi_sector,
            bypass_cache=bypass_cache,
            delete_source=True,
        )
        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs/{job_id}")
    def get_job_status(job_id: str) -> dict[str, object]:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    return app


app = create_fastapi_app(preload=False)


def main() -> None:
    uvicorn.run(
        "deployment.app:app",
        host=os.environ.get("ARGUS_STREAM_A_HOST", "127.0.0.1"),
        port=int(os.environ.get("ARGUS_STREAM_A_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
