from __future__ import annotations

import os
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/argus_stream_a"
CACHE_DIR = "/cache/huggingface"
APP_NAME = "argus-stream-a-api"

hf_cache_volume = modal.Volume.from_name("argus-stream-a-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(LOCAL_ROOT / "deployment" / "requirements.txt"))
    .pip_install("hf_transfer")
    .env(
        {
            "ARGUS_STREAM_A_ROOT": REMOTE_ROOT,
            "HF_HOME": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "ARGUS_STREAM_A_DEVICE": "cuda",
            "ARGUS_STREAM_A_BATCH_SIZE": "6",
            "ARGUS_STREAM_A_PUBLIC_API_URL": (
                "https://jatinsheoran2412--argus-stream-a-api-argusapi-fastapi-app.modal.run"
            ),
        }
    )
    .add_local_dir(
        LOCAL_ROOT,
        remote_path=REMOTE_ROOT,
        copy=True,
        ignore=[
            "**/__pycache__",
            "**/*.pyc",
            "data/**",
            "docs/**",
            "*.md",
            "*.txt",
            "*.bat",
            "**/node_modules",
            "**/.next",
            "**/.vercel",
            "**/.git",
        ],
    )
    .workdir(REMOTE_ROOT)
)

app = modal.App(APP_NAME, image=image)


@app.function(
    timeout=20 * 60,
    volumes={CACHE_DIR: hf_cache_volume},
)
def prime_backbone_cache() -> str:
    os.environ["ARGUS_STREAM_A_DEVICE"] = "cpu"
    from deployment.app import preload_profiles

    preload_profiles(include_extractor=True)
    hf_cache_volume.commit()
    return "Backbone cache warmed"


@app.cls(
    gpu="T4",
    timeout=30 * 60,
    max_containers=1,
    min_containers=0,
    scaledown_window=120,  # Scaledown after 2 minutes of idle time to save costs
    volumes={CACHE_DIR: hf_cache_volume},
)
class ArgusAPI:
    @modal.enter()
    def startup(self):
        os.environ.setdefault("ARGUS_STREAM_A_ROOT", REMOTE_ROOT)
        os.environ.setdefault("HF_HOME", CACHE_DIR)
        os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("ARGUS_STREAM_A_DEVICE", "cuda")
        os.environ.setdefault("ARGUS_STREAM_A_BATCH_SIZE", "6")

        from deployment.app import preload_profiles
        preload_profiles(include_extractor=True)

    @modal.asgi_app()
    def fastapi_app(self):
        from deployment.app import create_fastapi_app
        return create_fastapi_app(preload=False)


@app.local_entrypoint()
def main(prime_cache: bool = False) -> None:
    if prime_cache:
        print(prime_backbone_cache.remote())
    else:
        print("Use `modal deploy deployment/modal_app.py` to deploy the ARGUS Stream A API.")
