from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from deployment import app as api


def test_sample_catalog_matches_video_folder() -> None:
    expected = {
        path.name
        for path in (Path(__file__).parents[1] / "test_videos").iterdir()
        if path.suffix.lower() in api.ALLOWED_VIDEO_SUFFIXES
    }
    catalog = api._sample_catalog()

    assert {str(item["filename"]) for item in catalog} == expected
    assert all(item["dataset"] in {"Avenue", "UBnormal"} for item in catalog)
    assert all(float(item["size_mb"]) > 0 for item in catalog)


def test_public_routes_do_not_require_model_preload() -> None:
    client = TestClient(api.create_fastapi_app(preload=False))

    assert client.get("/").status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["sample_count"] == len(api._sample_catalog())

    samples = client.get("/samples")
    assert samples.status_code == 200
    assert len(samples.json()["samples"]) == len(api._sample_catalog())


def test_profile_aliases_accept_dataset_display_name() -> None:
    assert api._resolve_profile_label("Avenue") in api.PROFILES
    assert api._resolve_profile_label("UBnormal") in api.PROFILES


def test_unknown_sample_is_404() -> None:
    client = TestClient(api.create_fastapi_app(preload=False))
    response = client.get("/samples/not-a-real-sample/video")
    assert response.status_code == 404


def test_non_video_upload_is_rejected_before_inference() -> None:
    client = TestClient(api.create_fastapi_app(preload=False))
    profile_label = next(iter(api.PROFILES))
    response = client.post(
        "/analyze",
        data={"profile": profile_label},
        files={"video": ("notes.txt", b"not a video", "text/plain")},
    )
    assert response.status_code == 415
