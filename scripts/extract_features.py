"""Standalone Stream A feature extraction for arbitrary video folders."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbones.videomae import VideoMAEFeatureExtractor
from src.utils.io import save_features
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _extract_frames(video_path: Path, frame_dir: Path, max_frames: int | None = None) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    paths: list[Path] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and index >= max_frames:
            break
        frame_path = frame_dir / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        paths.append(frame_path)
        index += 1
    cap.release()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Stream A VideoMAE features from videos")
    parser.add_argument("--video-dir", required=True, help="Folder of input videos")
    parser.add_argument("--output-dir", required=True, help="Folder where .npy features will be written")
    parser.add_argument("--extensions", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Optional prefix for saved feature names, e.g. train or test",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [
        path
        for path in sorted(video_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {ext.lower() for ext in args.extensions}
    ]
    if not videos:
        raise FileNotFoundError(f"No videos found under {video_dir}")

    extractor = VideoMAEFeatureExtractor(device=device)
    logger.info("Found %s videos", len(videos))
    prefix = str(args.name_prefix).strip()

    for index, video_path in enumerate(videos, start=1):
        logger.info("[%s/%s] Extracting %s", index, len(videos), video_path.name)
        tmp_dir = Path(tempfile.mkdtemp(prefix="stream_a_extract_"))
        try:
            frame_paths = _extract_frames(video_path, tmp_dir, max_frames=args.max_frames)
            features = extractor.extract_single_video(
                frame_paths,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            output_name = f"{prefix}_{video_path.stem}" if prefix else video_path.stem
            save_features(features, output_dir / f"{output_name}.npy")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
