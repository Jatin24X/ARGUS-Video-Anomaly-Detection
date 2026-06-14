"""ARGUS Stream A demo with Avenue and UBnormal analysis profiles.

This demo is intentionally faster than the benchmark scripts:
- adaptive frame thinning for long videos
- in-memory VideoMAE extraction (no temp JPEG roundtrip)
- per-video feature and score caching across profile switches

The benchmark numbers shown in the UI come from the offline reports bundled
with this standalone package. Demo analysis is for presentation and inspection,
not an exact replacement for the full benchmark pipeline.
"""

from __future__ import annotations

import base64
import html
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(os.environ.get("ARGUS_STREAM_A_ROOT", Path(__file__).resolve().parent)).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import gradio as gr
    import plotly.graph_objects as go
except ImportError:
    print("Missing demo dependencies. Install with: pip install gradio plotly")
    raise

from src.evaluation.metrics import gaussian_smooth, minmax_normalize
from src.models.backbones.videomae import (
    CLIP_LENGTH,
    FRAME_SIZE,
    TEMPORAL_STRIDE,
    VideoMAEFeatureExtractor,
)
from src.models.scorers.mulde import MULDEScorer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float override for %s=%r; using %s", name, value, default)
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int override for %s=%r; using %s", name, value, default)
        return default


TARGET_ANALYSIS_FPS = _env_float("ARGUS_STREAM_A_TARGET_ANALYSIS_FPS", 12.0)
MAX_ANALYSIS_FRAMES = _env_int("ARGUS_STREAM_A_MAX_ANALYSIS_FRAMES", 720)
DISPLAY_MAX_EDGE = _env_int("ARGUS_STREAM_A_DISPLAY_MAX_EDGE", 720)
CACHE_SIZE = _env_int("ARGUS_STREAM_A_CACHE_SIZE", 2)


@dataclass(frozen=True)
class DemoProfile:
    key: str
    label: str
    dataset_name: str
    checkpoint_path: Path
    benchmark_report: str
    benchmark_micro: float
    benchmark_macro: float
    benchmark_clip: float
    scoring_mode: str
    signal_kind: str
    smoothing_sigma: float
    gmm_components: int = 0
    single_sigma_index: int = 0
    percentile: float = 85.0
    headline: str = ""
    note: str = ""
    accent: str = "#0f766e"
    accent_soft: str = "#ccfbf1"
    badge: str = ""

    def score_description(self) -> str:
        if self.scoring_mode == "gmm":
            return (
                f"{self.signal_kind} + GMM({self.gmm_components}), "
                f"smoothing={self.smoothing_sigma:g}"
            )
        return (
            f"{self.signal_kind} @ sigma_index={self.single_sigma_index}, "
            f"smoothing={self.smoothing_sigma:g}"
        )


AVENUE_PROFILE = DemoProfile(
    key="avenue",
    label="Avenue profile",
    dataset_name="Avenue",
    checkpoint_path=PROJECT_ROOT
    / "outputs"
    / "avenue_stream_a_ld_gmm1_beta01_lr4e5_run1"
    / "checkpoints"
    / "stream_a"
    / "best_holdout.pt",
    benchmark_report="outputs/reports/avenue_stream_a_best_test.json",
    benchmark_micro=0.8451,
    benchmark_macro=0.8514,
    benchmark_clip=0.8400,
    scoring_mode="gmm",
    signal_kind="log_density",
    smoothing_sigma=13.0,
    gmm_components=1,
    headline="Avenue analysis profile",
    note="Main saved Avenue profile for the standalone Stream A demo.",
    accent="#0f766e",
    accent_soft="rgba(20, 184, 166, 0.16)",
    badge="Saved profile",
)

UBNORMAL_PROFILE = DemoProfile(
    key="ubnormal",
    label="UBnormal profile",
    dataset_name="UBnormal",
    checkpoint_path=PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "stream_a_locked_videomae_beta1_score_norm_sigma0.pt",
    benchmark_report="outputs/reports/stream_a_frozen_baseline.json",
    benchmark_micro=0.7394,
    benchmark_macro=0.8410,
    benchmark_clip=0.7309,
    scoring_mode="multiscale",
    signal_kind="score_norm",
    smoothing_sigma=20.0,
    single_sigma_index=0,
    headline="UBnormal analysis profile",
    note="Locked Stream A profile kept in the demo for comparison.",
    accent="#b45309",
    accent_soft="rgba(245, 158, 11, 0.16)",
    badge="Saved profile",
)

PROFILES: Dict[str, DemoProfile] = {
    AVENUE_PROFILE.label: AVENUE_PROFILE,
    UBNORMAL_PROFILE.label: UBNORMAL_PROFILE,
}


def _recommended_batch_size() -> int:
    if not torch.cuda.is_available():
        return 2
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if total_gb >= 20:
        return 12
    if total_gb >= 10:
        return 8
    if total_gb >= 6:
        return 4
    return 2


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _resize_for_display(frame_rgb: np.ndarray) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, DISPLAY_MAX_EDGE / max(h, w))
    if scale >= 1.0:
        return frame_rgb
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _resolve_video_path(video_value: object) -> str | None:
    if video_value is None:
        return None
    if isinstance(video_value, Path):
        return str(video_value)
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        for key in ("path", "name", "video"):
            value = video_value.get(key)
            if isinstance(value, (str, Path)):
                return str(value)
    raise ValueError(f"Unsupported video input type: {type(video_value)!r}")


def _empty_plot() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=360,
        template="plotly_dark",
        margin=dict(l=30, r=30, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.26)",
        font=dict(color="#dbe6f5"),
        title="Upload a video and choose a profile to see the anomaly timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized anomaly score",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zeroline=False)
    return fig


def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    diff = np.diff(mask.astype(np.int32))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask.size and mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask.size and mask[-1]:
        ends = np.append(ends, mask.size)
    return list(zip(starts.tolist(), ends.tolist()))


def _build_timeline(
    scores: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    anomaly_mask: np.ndarray,
    profile: DemoProfile,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=scores,
            mode="lines",
            fill="tozeroy",
            fillcolor=profile.accent_soft,
            line=dict(color=profile.accent, width=3),
            name="Anomaly score",
        )
    )
    fig.add_hline(
        y=threshold,
        line_dash="dot",
        line_color="rgba(226, 232, 240, 0.78)",
        annotation_text="highlight cutoff",
        annotation_position="top left",
    )

    for start, end in _contiguous_regions(anomaly_mask):
        x0 = float(timestamps[start])
        x1 = float(timestamps[min(end - 1, len(timestamps) - 1)])
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(239,68,68,0.12)",
            line_width=0,
        )

    fig.update_layout(
        height=380,
        template="plotly_dark",
        title=f"{profile.dataset_name} demo timeline",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(15,23,42,0.28)",
        font=dict(color="#dbe6f5"),
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis_title="Time (seconds)",
        yaxis_title="Normalized anomaly score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    return fig


def _select_gallery_indices(
    scores: np.ndarray,
    *,
    max_items: int = 4,
    min_gap: int = 8,
) -> List[int]:
    if scores.size == 0:
        return []

    selected: List[int] = []
    for idx in np.argsort(scores)[::-1]:
        idx_i = int(idx)
        if any(abs(idx_i - prev) < min_gap for prev in selected):
            continue
        selected.append(idx_i)
        if len(selected) >= max_items:
            break
    return selected


def _build_gallery(
    frames_rgb: List[np.ndarray],
    scores: np.ndarray,
    timestamps: np.ndarray,
    max_items: int = 4,
    min_gap: int = 8,
) -> List[Tuple[np.ndarray, str]]:
    if not frames_rgb:
        return []

    gallery = []
    for idx in _select_gallery_indices(scores, max_items=max_items, min_gap=min_gap):
        caption = f"{timestamps[idx]:.2f}s  |  score {scores[idx]:.3f}"
        gallery.append((frames_rgb[idx], caption))
    return gallery


def _encode_frame_data_uri(frame_rgb: np.ndarray, *, jpeg_quality: int = 90) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError("Failed to encode gallery frame for API response.")
    encoded_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_b64}"


def _gallery_payload(
    frames_rgb: List[np.ndarray],
    scores: np.ndarray,
    timestamps: np.ndarray,
    *,
    max_items: int = 4,
    min_gap: int = 8,
) -> List[dict[str, object]]:
    payload: List[dict[str, object]] = []
    for idx in _select_gallery_indices(scores, max_items=max_items, min_gap=min_gap):
        payload.append(
            {
                "index": int(idx),
                "timestamp_sec": float(timestamps[idx]),
                "score": float(scores[idx]),
                "caption": f"{timestamps[idx]:.2f}s  |  score {scores[idx]:.3f}",
                "image_data_url": _encode_frame_data_uri(frames_rgb[idx]),
            }
        )
    return payload


def _anomaly_regions_payload(
    timestamps: np.ndarray,
    anomaly_mask: np.ndarray,
) -> List[dict[str, float]]:
    if timestamps.size == 0:
        return []

    payload: List[dict[str, float]] = []
    for start, end in _contiguous_regions(anomaly_mask):
        end_idx = min(end - 1, len(timestamps) - 1)
        payload.append(
            {
                "start_time_sec": float(timestamps[start]),
                "end_time_sec": float(timestamps[end_idx]),
                "start_index": int(start),
                "end_index": int(end_idx),
            }
        )
    return payload


def _profile_payload(profile: DemoProfile) -> dict[str, object]:
    return {
        "key": profile.key,
        "label": profile.label,
        "dataset_name": profile.dataset_name,
        "headline": profile.headline,
        "note": profile.note,
        "badge": profile.badge,
        "accent": profile.accent,
        "benchmark_micro_auc": float(profile.benchmark_micro),
        "benchmark_macro_auc": float(profile.benchmark_macro),
        "benchmark_clip_auc": float(profile.benchmark_clip),
        "benchmark_micro_auc_pct": _pct(profile.benchmark_micro),
        "benchmark_macro_auc_pct": _pct(profile.benchmark_macro),
        "benchmark_clip_auc_pct": _pct(profile.benchmark_clip),
        "benchmark_report": profile.benchmark_report,
    }


APP_CSS = """
:root {
  --hero-page-gradient:
    radial-gradient(circle at top right, rgba(59, 130, 246, 0.42), transparent 36%),
    linear-gradient(135deg, #0b1220 0%, #111827 55%, #172554 100%);
}
html,
body {
  min-height: 100%;
  background: var(--hero-page-gradient) fixed !important;
  background-color: #0b1220 !important;
  overflow-x: hidden;
  background-repeat: no-repeat !important;
  background-size: cover !important;
}
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  background: var(--hero-page-gradient);
  background-repeat: no-repeat;
  background-size: cover;
}
.gradio-container,
.gradio-container > .main,
.gradio-container .main,
.gradio-container .contain,
.gradio-container .wrap {
  background: transparent !important;
}
.gradio-container {
  max-width: 1320px !important;
  margin: 0 auto;
  min-height: 100vh;
  padding: 18px 28px 42px !important;
  background:
    radial-gradient(circle at top right, rgba(59, 130, 246, 0.32), transparent 34%),
    linear-gradient(135deg, #0b1220 0%, #111827 55%, #172554 100%) !important;
  border-radius: 34px;
  box-shadow: 0 32px 120px rgba(2, 6, 23, 0.28);
  color: #e5eefb !important;
}
.app-shell { color: #e5eefb; }
.hero-shell {
  background:
    linear-gradient(180deg, rgba(8, 15, 32, 0.18), rgba(8, 15, 32, 0.18)),
    var(--hero-page-gradient);
  border-radius: 30px;
  padding: 30px 32px;
  color: #f8fafc;
  border: 1px solid rgba(148, 163, 184, 0.16);
  box-shadow: 0 28px 90px rgba(2, 6, 23, 0.48);
}
.hero-title {
  font-size: 2.35rem;
  line-height: 1.05;
  font-weight: 800;
  margin: 0 0 10px 0;
  letter-spacing: -0.03em;
}
.hero-subtitle {
  font-size: 1.04rem;
  color: rgba(226, 232, 240, 0.88);
  margin: 0 0 16px 0;
  max-width: 880px;
}
.contribution-shell {
  margin-top: 14px;
  background: linear-gradient(180deg, rgba(8, 47, 73, 0.88), rgba(10, 37, 64, 0.84));
  border: 1px solid rgba(56, 189, 248, 0.18);
  border-radius: 20px;
  padding: 16px 18px;
  box-shadow: 0 14px 38px rgba(2, 6, 23, 0.30);
}
.contribution-kicker {
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: #7dd3fc;
  font-weight: 800;
  margin-bottom: 8px;
}
.contribution-copy {
  color: #e5eefb;
  font-size: 1rem;
  line-height: 1.6;
}
.benchmark-strip {
  display: grid;
  grid-template-columns: 1.2fr 1fr 1fr;
  gap: 14px;
  margin-top: 14px;
}
.benchmark-tile {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.94), rgba(15, 23, 42, 0.92));
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 22px;
  padding: 18px;
  box-shadow: 0 18px 42px rgba(2, 6, 23, 0.30);
}
.benchmark-label {
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.10em;
  color: #7dd3fc;
  font-weight: 800;
  margin-bottom: 10px;
}
.benchmark-title {
  color: #f8fafc;
  font-size: 1.05rem;
  font-weight: 700;
  margin-bottom: 6px;
}
.benchmark-value {
  color: #f8fafc;
  font-size: 2rem;
  font-weight: 800;
  line-height: 1;
}
.benchmark-sub {
  color: #a9b8d0;
  margin-top: 8px;
  line-height: 1.5;
}
.badge-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}
.badge-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 0.92rem;
  font-weight: 600;
  color: #f8fafc;
  background: rgba(15, 23, 42, 0.42);
  border: 1px solid rgba(148, 163, 184, 0.22);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.pipeline-shell {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.92), rgba(15, 23, 42, 0.90));
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 22px;
  padding: 16px 18px;
  margin-top: 14px;
  box-shadow: 0 18px 42px rgba(2, 6, 23, 0.34);
}
.pipeline-title {
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #93c5fd;
  margin-bottom: 10px;
  font-weight: 700;
}
.pipeline-flow {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.pipeline-step {
  padding: 10px 14px;
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.88);
  border: 1px solid rgba(148, 163, 184, 0.16);
  font-weight: 600;
  color: #e2e8f0;
}
.pipeline-arrow {
  color: #38bdf8;
  font-weight: 800;
}
.panel-card {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.94), rgba(15, 23, 42, 0.92));
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 24px;
  padding: 18px;
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.34);
}
.panel-card h3 {
  margin: 0 0 8px 0;
  font-size: 1.1rem;
  color: #f8fafc;
}
.panel-card p {
  margin: 0;
  color: #b6c2d9;
}
.section-header {
  margin: 6px 0 10px 0;
  color: #f8fafc;
}
.section-header .section-kicker {
  margin-bottom: 6px;
}
.section-header .section-title {
  font-size: 1.06rem;
  font-weight: 800;
  color: #f8fafc;
}
.section-header .section-sub {
  margin-top: 4px;
  color: #94a3b8;
  line-height: 1.5;
}
.profile-shell {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.96), rgba(15, 23, 42, 0.94));
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 24px;
  padding: 18px;
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.34);
}
.section-kicker {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.78rem;
  font-weight: 800;
  color: #7dd3fc;
  margin-bottom: 10px;
}
.profile-topline {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: start;
}
.profile-title {
  font-size: 1.28rem;
  font-weight: 800;
  margin: 0;
  color: #f8fafc;
}
.profile-badge {
  display: inline-flex;
  padding: 8px 12px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--accent);
  background: rgba(15, 23, 42, 0.72);
  border: 1px solid rgba(148,163,184,0.18);
}
.profile-meta {
  margin-top: 10px;
  color: #b6c2d9;
  line-height: 1.55;
}
.profile-summary {
  margin-top: 10px;
  color: #dbe6f5;
  line-height: 1.6;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.metric-card {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.94));
  border: 1px solid rgba(148,163,184,0.12);
  border-radius: 20px;
  padding: 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.metric-label {
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #7dd3fc;
  margin-bottom: 8px;
  font-weight: 800;
}
.metric-value {
  font-size: 1.48rem;
  font-weight: 800;
  color: #f8fafc;
  line-height: 1;
}
.metric-foot {
  font-size: 0.84rem;
  color: #94a3b8;
  margin-top: 8px;
}
.profile-note {
  margin-top: 16px;
  padding: 14px;
  border-radius: 18px;
  background: rgba(15, 23, 42, 0.82);
  border: 1px solid rgba(148,163,184,0.16);
  color: #cbd5e1;
  line-height: 1.55;
}
.summary-shell {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.96), rgba(15, 23, 42, 0.94));
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 24px;
  padding: 18px;
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.34);
}
.summary-title {
  font-size: 1.24rem;
  font-weight: 800;
  margin: 0;
  color: #f8fafc;
}
.summary-sub {
  margin-top: 8px;
  color: #cbd5e1;
  line-height: 1.55;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.summary-card {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.94));
  border: 1px solid rgba(148,163,184,0.12);
  border-radius: 18px;
  padding: 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.summary-card .label {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.74rem;
  color: #7dd3fc;
  font-weight: 800;
  margin-bottom: 8px;
}
.summary-card .value {
  font-size: 1.18rem;
  font-weight: 800;
  color: #f8fafc;
}
.summary-card .subvalue {
  margin-top: 8px;
  color: #94a3b8;
  font-size: 0.88rem;
}
.summary-list {
  margin-top: 16px;
  padding-left: 18px;
  color: #dbe6f5;
  line-height: 1.6;
}
.gr-button-primary {
  background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
  border: none !important;
  color: #eff6ff !important;
  box-shadow: 0 12px 28px rgba(37, 99, 235, 0.32) !important;
}
.gr-button-primary:hover {
  filter: brightness(1.03);
}
.cta-button button {
  min-height: 56px !important;
  font-size: 1.06rem !important;
  font-weight: 800 !important;
  border-radius: 18px !important;
  background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%) !important;
  color: #eff6ff !important;
  border: none !important;
  box-shadow: 0 14px 32px rgba(37, 99, 235, 0.34) !important;
}
.card-shell {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.96), rgba(15, 23, 42, 0.94));
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 24px;
  padding: 8px;
  box-shadow: 0 18px 48px rgba(2, 6, 23, 0.34);
}
.card-shell > .wrap,
.card-shell .block {
  background: transparent !important;
}
.card-shell .label-wrap,
.gradio-container .label-wrap,
.gradio-container .block-label,
.gradio-container label,
.gradio-container .label-text,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong {
  color: #e5eefb !important;
}
.gradio-container .label-wrap,
.gradio-container .block-label {
  background: rgba(15, 23, 42, 0.92) !important;
  border: 1px solid rgba(148,163,184,0.14) !important;
  border-radius: 12px !important;
  box-shadow: 0 8px 24px rgba(2, 6, 23, 0.22) !important;
}
.gradio-container .prose code,
.profile-note code,
.summary-shell code {
  background: rgba(15, 23, 42, 0.78);
  color: #93c5fd;
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 8px;
  padding: 2px 6px;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap,
.gradio-container .container,
.gradio-container .form,
.gradio-container .form > * {
  color: #e5eefb;
}
.gradio-container .upload-container,
.gradio-container .empty,
.gradio-container .video-container,
.gradio-container .image-container,
.gradio-container .gallery-item,
.gradio-container .grid-wrap,
.gradio-container .inner,
.gradio-container .preview,
.gradio-container .wrap.svelte-12cmxck {
  background: rgba(15, 23, 42, 0.88) !important;
  border-color: rgba(148,163,184,0.16) !important;
  color: #e5eefb !important;
}
.gradio-container .upload-container:hover,
.gradio-container .gallery-item:hover {
  border-color: rgba(56,189,248,0.38) !important;
}
.gradio-container .gallery-item figcaption,
.gradio-container figcaption {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(17, 24, 39, 0.94)) !important;
  color: #dbe6f5 !important;
}
.gradio-container .tabs,
.gradio-container .tabitem,
.gradio-container .form {
  background: transparent !important;
}
.gradio-container .radio-group,
.gradio-container .radio,
.gradio-container .wrap.svelte-1ipelgc,
.gradio-container .wrap.svelte-1ipelgc label {
  color: #e5eefb !important;
}
.gradio-container .radio label,
.gradio-container .checkbox label {
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(17, 24, 39, 0.92)) !important;
  border: 1px solid rgba(148,163,184,0.16) !important;
  border-radius: 14px !important;
}
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {
  accent-color: #38bdf8 !important;
}
.gradio-container input[type="radio"] {
  appearance: none !important;
  -webkit-appearance: none !important;
  width: 18px !important;
  height: 18px !important;
  border-radius: 999px !important;
  border: 2px solid rgba(148, 163, 184, 0.42) !important;
  background: transparent !important;
  display: inline-grid !important;
  place-content: center !important;
  margin-right: 8px !important;
}
.gradio-container input[type="radio"]::before {
  content: "" !important;
  width: 8px !important;
  height: 8px !important;
  border-radius: 999px !important;
  transform: scale(0) !important;
  transition: transform 120ms ease-in-out !important;
  box-shadow: inset 1em 1em #38bdf8 !important;
}
.gradio-container input[type="radio"]:checked {
  border-color: #38bdf8 !important;
  background: rgba(14, 165, 233, 0.12) !important;
}
.gradio-container input[type="radio"]:checked::before {
  transform: scale(1) !important;
}
.gradio-container .radio label:has(input:checked),
.gradio-container .checkbox label:has(input:checked) {
  background: linear-gradient(135deg, rgba(8, 47, 73, 0.96), rgba(30, 64, 175, 0.34)) !important;
  border-color: rgba(56,189,248,0.42) !important;
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.03),
    0 0 0 1px rgba(56,189,248,0.12) !important;
}
.gradio-container .radio label:has(input:checked) span,
.gradio-container .checkbox label:has(input:checked) span {
  color: #f8fafc !important;
}
.profile-radio {
  background: linear-gradient(180deg, rgba(11, 18, 32, 0.96), rgba(15, 23, 42, 0.94));
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 18px;
  padding: 12px;
}
.gradio-container .radio input:checked + span,
.gradio-container .checkbox input:checked + span {
  color: #7dd3fc !important;
}
.gradio-container .plot-container,
.gradio-container .plotly {
  background: transparent !important;
}
.gradio-container .modebar {
  display: none !important;
}
.gradio-container .modebar-btn path {
  fill: #cbd5e1 !important;
}
.gradio-container footer {
  display: none !important;
}
@media (max-width: 900px) {
  .benchmark-strip,
  .metric-grid, .summary-grid {
    grid-template-columns: 1fr;
  }
  .hero-title {
    font-size: 1.8rem;
  }
}
"""


def _hero_html() -> str:
    return """
<div class="app-shell">
  <div class="hero-shell">
    <div class="hero-title">ARGUS Stream A</div>
    <div class="hero-subtitle">
      Standalone frame-level video anomaly detection demo built with a frozen
      VideoMAE backbone and MULDE scoring. Upload a short clip and analyze it
      using the saved Avenue or UBnormal profile.
    </div>
    <div class="badge-row">
      <span class="badge-chip">VideoMAE-v2 Base</span>
      <span class="badge-chip">MULDE</span>
      <span class="badge-chip">Frame-centric</span>
      <span class="badge-chip">Avenue + UBnormal</span>
      <span class="badge-chip">Standalone demo</span>
    </div>
  </div>
</div>
"""


def _section_html(kicker: str, title: str, subtitle: str = "") -> str:
    subtitle_html = (
        f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    )
    return f"""
<div class="section-header">
  <div class="section-kicker">{html.escape(kicker)}</div>
  <div class="section-title">{html.escape(title)}</div>
  {subtitle_html}
</div>
"""


def _pipeline_html() -> str:
    return """
<div class="pipeline-shell">
  <div class="pipeline-title">How this demo works</div>
  <div class="pipeline-flow">
    <div class="pipeline-step">Upload video</div>
    <div class="pipeline-arrow">&rarr;</div>
    <div class="pipeline-step">Adaptive frame sampling</div>
    <div class="pipeline-arrow">&rarr;</div>
    <div class="pipeline-step">VideoMAE clip embeddings</div>
    <div class="pipeline-arrow">&rarr;</div>
    <div class="pipeline-step">MULDE scoring</div>
    <div class="pipeline-arrow">&rarr;</div>
    <div class="pipeline-step">Interactive anomaly timeline</div>
  </div>
</div>
"""


def _profile_info_html(profile: DemoProfile) -> str:
    note = html.escape(profile.note)
    return f"""
<div class="profile-shell" style="--accent:{profile.accent}; --accent-soft:{profile.accent_soft};">
  <div class="section-kicker">Selected profile</div>
  <div class="profile-topline">
    <div>
      <div class="profile-title">{html.escape(profile.dataset_name)}</div>
      <div class="profile-meta">
        {html.escape(profile.headline)}
      </div>
    </div>
    <div class="profile-badge">{html.escape(profile.badge)}</div>
  </div>
  <div class="profile-summary">
    {note}
  </div>
  <div class="metric-grid">
    <div class="metric-card">
      <div class="metric-label">Saved micro AUC</div>
      <div class="metric-value">{_pct(profile.benchmark_micro)}</div>
      <div class="metric-foot">Saved evaluation result</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Saved macro AUC</div>
      <div class="metric-value">{_pct(profile.benchmark_macro)}</div>
      <div class="metric-foot">Per-video averaged AUC</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Saved clip AUC</div>
      <div class="metric-value">{_pct(profile.benchmark_clip)}</div>
      <div class="metric-foot">Offline report metric</div>
    </div>
  </div>
</div>
"""


def _empty_summary_html() -> str:
    return """
<div class="summary-shell">
  <div class="section-kicker">Analysis summary</div>
  <div class="summary-title">Live analysis summary</div>
  <div class="summary-sub">
    Upload a video, choose a profile, and run the analysis. The timeline and
    frame gallery below summarize the uploaded clip under the selected saved profile.
  </div>
  <ul class="summary-list">
    <li>The cards above show the saved metrics for the selected profile.</li>
    <li>The timeline and frame gallery summarize the uploaded clip only.</li>
    <li>Use the same uploaded video to compare the two saved profiles.</li>
  </ul>
</div>
"""


def _build_summary(
    profile: DemoProfile,
    raw_frame_count: int,
    sampled_frame_count: int,
    sample_step: int,
    source_fps: float,
    timestamps: np.ndarray,
    clip_count: int,
    scores: np.ndarray,
    threshold: float,
    elapsed: float,
    cache_hit: bool,
) -> str:
    peak_idx = int(np.argmax(scores)) if len(scores) else 0
    peak_time = float(timestamps[peak_idx]) if len(timestamps) else 0.0
    analyzed_duration = float(timestamps[-1]) if len(timestamps) else 0.0

    return f"""
<div class="summary-shell" style="--accent:{profile.accent}; --accent-soft:{profile.accent_soft};">
  <div class="section-kicker">Analysis summary</div>
  <div class="summary-title">Live analysis summary</div>
  <div class="summary-sub">
    Uploaded clip analyzed under the <strong>{html.escape(profile.dataset_name)}</strong>
    saved profile. The cards below summarize the live demo pass, not the benchmark run.
  </div>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="label">Profile</div>
      <div class="value">{html.escape(profile.dataset_name)}</div>
      <div class="subvalue">{html.escape(profile.badge)}</div>
    </div>
    <div class="summary-card">
      <div class="label">Clip duration</div>
      <div class="value">{analyzed_duration:.2f}s</div>
      <div class="subvalue">uploaded video span analyzed in the demo</div>
    </div>
    <div class="summary-card">
      <div class="label">Peak anomaly</div>
      <div class="value">{peak_time:.2f}s</div>
      <div class="subvalue">peak normalized score {scores[peak_idx]:.3f}</div>
    </div>
    <div class="summary-card">
      <div class="label">Demo runtime</div>
      <div class="value">{elapsed:.2f}s</div>
      <div class="subvalue">{"reused cached embeddings" if cache_hit else "fresh live analysis pass"}</div>
    </div>
  </div>
  <ul class="summary-list">
    <li>The highlighted region marks the highest-anomaly portion of the uploaded clip.</li>
    <li>The frame gallery shows the top-scoring moments from the live analysis.</li>
    <li>The saved benchmark metrics for this profile are shown in the benchmark cards above.</li>
  </ul>
</div>
"""


class ARGUSDemoEngine:
    """Interactive engine for the standalone Stream A demo."""

    def __init__(self) -> None:
        self.device = os.environ.get(
            "ARGUS_STREAM_A_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.batch_size = _env_int("ARGUS_STREAM_A_BATCH_SIZE", _recommended_batch_size())
        self.extractor: VideoMAEFeatureExtractor | None = None
        self.scorers: Dict[str, MULDEScorer] = {}
        self.cache: OrderedDict[Tuple[str, int, int], dict] = OrderedDict()

    def preload(
        self,
        *,
        include_extractor: bool = True,
        profile_labels: List[str] | None = None,
    ) -> None:
        logger.info(
            "Preloading ARGUS Stream A assets on %s (extractor=%s)",
            self.device,
            include_extractor,
        )
        if include_extractor:
            self._get_extractor()

        labels = profile_labels or list(PROFILES.keys())
        for label in labels:
            profile = PROFILES[label]
            self._get_scorer(profile)

    def _get_extractor(self) -> VideoMAEFeatureExtractor:
        if self.extractor is None:
            logger.info("Loading VideoMAE extractor on %s", self.device)
            self.extractor = VideoMAEFeatureExtractor(device=self.device)
        return self.extractor

    def _get_scorer(self, profile: DemoProfile) -> MULDEScorer:
        scorer = self.scorers.get(profile.key)
        if scorer is not None:
            return scorer

        if not profile.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {profile.checkpoint_path}")

        logger.info("Loading scorer for profile %s", profile.key)
        scorer = MULDEScorer.load_checkpoint(profile.checkpoint_path, device=self.device)
        scorer.eval()
        self.scorers[profile.key] = scorer
        return scorer

    @staticmethod
    def _cache_key(video_path: str) -> Tuple[str, int, int]:
        path = Path(video_path)
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))

    def _cache_get(self, key: Tuple[str, int, int]) -> dict | None:
        item = self.cache.get(key)
        if item is None:
            return None
        self.cache.move_to_end(key)
        return item

    def _cache_put(self, key: Tuple[str, int, int], value: dict) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def _decode_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0:
            source_fps = 30.0
        raw_frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        sample_step = max(1, int(round(source_fps / TARGET_ANALYSIS_FPS)))
        if raw_frame_total > 0:
            sample_step = max(sample_step, int(np.ceil(raw_frame_total / MAX_ANALYSIS_FRAMES)))

        display_frames: List[np.ndarray] = []
        model_frames: List[np.ndarray] = []
        sampled_indices: List[int] = []

        frame_idx = 0
        while len(model_frames) < MAX_ANALYSIS_FRAMES:
            ok = cap.grab()
            if not ok:
                break

            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue

            ok, frame_bgr = cap.retrieve()
            if not ok:
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            display_frames.append(_resize_for_display(frame_rgb))
            model_frames.append(
                cv2.resize(
                    frame_rgb,
                    (FRAME_SIZE, FRAME_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            sampled_indices.append(frame_idx)
            frame_idx += 1

        cap.release()

        if raw_frame_total <= 0:
            raw_frame_total = frame_idx

        timestamps = (
            np.asarray(sampled_indices, dtype=np.float64) / source_fps
            if sampled_indices
            else np.empty((0,), dtype=np.float64)
        )

        return {
            "raw_frame_total": raw_frame_total,
            "source_fps": source_fps,
            "sample_step": sample_step,
            "display_frames": display_frames,
            "model_frames": model_frames,
            "timestamps": timestamps,
        }

    def _score_clips(self, cached_video: dict, profile: DemoProfile) -> np.ndarray:
        score_cache: Dict[str, np.ndarray] = cached_video["score_cache"]
        cached_scores = score_cache.get(profile.key)
        if cached_scores is not None:
            return cached_scores

        features = cached_video["features"]
        if features.size == 0:
            return np.empty((0,), dtype=np.float64)

        scorer = self._get_scorer(profile)
        feat_t = torch.from_numpy(features.astype(np.float32)).to(self.device)

        if profile.scoring_mode == "gmm":
            with torch.inference_mode():
                clip_scores = scorer.score_anomaly(feat_t)
        else:
            signal = scorer.compute_multiscale_signal(feat_t, signal_kind=profile.signal_kind)
            clip_scores = signal[:, profile.single_sigma_index]

        clip_scores = np.asarray(clip_scores, dtype=np.float64)
        score_cache[profile.key] = clip_scores
        return clip_scores

    def _run_analysis(
        self,
        video_path: object,
        profile_label: str,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Any]:
        resolved_video_path = _resolve_video_path(video_path)
        if resolved_video_path is None:
            raise ValueError("Upload a video to start the analysis.")

        try:
            profile = PROFILES[profile_label]
        except KeyError as exc:
            raise ValueError(f"Unknown analysis profile: {profile_label}") from exc

        progress_callback = progress_callback or (lambda _fraction, _desc: None)
        cache_hit = False
        started = time.time()

        progress_callback(0.05, "Opening video")
        key = self._cache_key(resolved_video_path)
        cached_video = self._cache_get(key)

        if cached_video is None:
            decoded = self._decode_video(resolved_video_path)
            sampled_frame_count = len(decoded["model_frames"])
            if sampled_frame_count < CLIP_LENGTH:
                raise ValueError(
                    "Video too short for Stream A analysis. "
                    f"Need at least {CLIP_LENGTH} sampled frames and only found "
                    f"{sampled_frame_count}."
                )

            progress_callback(0.28, "Loading VideoMAE")
            extractor = self._get_extractor()

            progress_callback(0.42, "Extracting clip embeddings")
            features = extractor.extract_from_frames(
                decoded["model_frames"],
                batch_size=self.batch_size,
            )

            cached_video = {
                **decoded,
                "features": features,
                "score_cache": {},
            }
            self._cache_put(key, cached_video)
        else:
            cache_hit = True

        progress_callback(0.72, f"Scoring with {profile.dataset_name}")
        clip_scores = self._score_clips(cached_video, profile)

        timestamps = cached_video["timestamps"]
        frame_count = len(cached_video["display_frames"])
        clip_count = int(len(clip_scores))
        if clip_count == 0 or frame_count == 0:
            raise ValueError("No valid clips were extracted from this video.")

        progress_callback(0.84, "Reconstructing frame-level scores")
        clip_starts = np.arange(
            0,
            frame_count - CLIP_LENGTH + 1,
            TEMPORAL_STRIDE,
            dtype=np.int32,
        )[:clip_count]
        center_offset = (CLIP_LENGTH // 2) * TEMPORAL_STRIDE
        centers = np.minimum(clip_starts + center_offset, frame_count - 1)

        if clip_count == 1:
            frame_scores = np.full((frame_count,), float(clip_scores[0]), dtype=np.float64)
        else:
            frame_scores = np.interp(
                np.arange(frame_count, dtype=np.float64),
                centers.astype(np.float64),
                clip_scores.astype(np.float64),
            )

        smoothed = gaussian_smooth(frame_scores, sigma=profile.smoothing_sigma)
        normalized = minmax_normalize(smoothed)
        threshold = float(np.percentile(normalized, profile.percentile))
        anomaly_mask = normalized >= threshold
        elapsed = time.time() - started

        return {
            "profile": profile,
            "resolved_video_path": resolved_video_path,
            "cache_hit": cache_hit,
            "cached_video": cached_video,
            "timestamps": timestamps,
            "frame_count": frame_count,
            "clip_count": clip_count,
            "scores": normalized,
            "threshold": threshold,
            "anomaly_mask": anomaly_mask,
            "elapsed": elapsed,
        }

    def analyze_payload(self, video_path: object, profile_label: str) -> dict[str, Any]:
        analysis = self._run_analysis(video_path, profile_label)
        profile: DemoProfile = analysis["profile"]
        cached_video = analysis["cached_video"]
        timestamps: np.ndarray = analysis["timestamps"]
        scores: np.ndarray = analysis["scores"]
        anomaly_mask: np.ndarray = analysis["anomaly_mask"]

        peak_idx = int(np.argmax(scores)) if scores.size else 0
        analyzed_duration = float(timestamps[-1]) if timestamps.size else 0.0

        return {
            "profile": _profile_payload(profile),
            "analysis": {
                "video_name": Path(str(analysis["resolved_video_path"])).name,
                "cache_hit": bool(analysis["cache_hit"]),
                "runtime_sec": float(analysis["elapsed"]),
                "timeline": {
                    "timestamps_sec": [float(value) for value in timestamps.tolist()],
                    "scores": [float(value) for value in scores.tolist()],
                    "threshold": float(analysis["threshold"]),
                    "threshold_label": "highlight cutoff",
                    "anomaly_regions": _anomaly_regions_payload(timestamps, anomaly_mask),
                },
                "summary": {
                    "duration_sec": analyzed_duration,
                    "peak_time_sec": float(timestamps[peak_idx]) if timestamps.size else 0.0,
                    "peak_score": float(scores[peak_idx]) if scores.size else 0.0,
                    "raw_frame_count": int(cached_video["raw_frame_total"]),
                    "sampled_frame_count": int(analysis["frame_count"]),
                    "sample_step": int(cached_video["sample_step"]),
                    "source_fps": float(cached_video["source_fps"]),
                    "clip_count": int(analysis["clip_count"]),
                    "profile_label": profile.label,
                    "profile_dataset": profile.dataset_name,
                },
                "frames": _gallery_payload(
                    cached_video["display_frames"],
                    scores,
                    timestamps,
                ),
            },
        }

    def analyze(
        self,
        video_path: object,
        profile_label: str,
        progress: gr.Progress = gr.Progress(track_tqdm=False),
    ) -> Tuple[go.Figure, List[Tuple[np.ndarray, str]], str]:
        try:
            analysis = self._run_analysis(
                video_path,
                profile_label,
                progress_callback=lambda fraction, desc="": progress(fraction, desc=desc),
            )
        except ValueError as exc:
            return _empty_plot(), [], f"### {exc}"

        profile: DemoProfile = analysis["profile"]
        cached_video = analysis["cached_video"]
        timestamps: np.ndarray = analysis["timestamps"]
        normalized: np.ndarray = analysis["scores"]
        anomaly_mask: np.ndarray = analysis["anomaly_mask"]

        progress(0.94, desc="Building visuals")
        fig = _build_timeline(normalized, timestamps, float(analysis["threshold"]), anomaly_mask, profile)
        gallery = _build_gallery(cached_video["display_frames"], normalized, timestamps)
        summary = _build_summary(
            profile=profile,
            raw_frame_count=int(cached_video["raw_frame_total"]),
            sampled_frame_count=int(analysis["frame_count"]),
            sample_step=int(cached_video["sample_step"]),
            source_fps=float(cached_video["source_fps"]),
            timestamps=timestamps,
            clip_count=int(analysis["clip_count"]),
            scores=normalized,
            threshold=float(analysis["threshold"]),
            elapsed=float(analysis["elapsed"]),
            cache_hit=bool(analysis["cache_hit"]),
        )
        progress(1.0, desc="Done")
        return fig, gallery, summary


ENGINE = ARGUSDemoEngine()


def _render_profile_panel(profile_label: str) -> str:
    return _profile_info_html(PROFILES[profile_label])


def _reset_summary_for_profile(profile_label: str) -> str:
    profile = PROFILES[profile_label]
    return f"""
<div class="summary-shell">
  <div class="section-kicker">Analysis summary</div>
  <div class="summary-title">Live analysis summary</div>
  <div class="summary-sub">
    Upload a video and run the analysis under the
    <strong>{html.escape(profile.label)}</strong>.
  </div>
  <ul class="summary-list">
    <li><strong>Saved benchmark:</strong> {_pct(profile.benchmark_micro)} micro / {_pct(profile.benchmark_macro)} macro / {_pct(profile.benchmark_clip)} clip</li>
    <li><strong>Profile role:</strong> {html.escape(profile.badge)}</li>
    <li><strong>Context:</strong> {html.escape(profile.note)}</li>
  </ul>
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ARGUS Stream A Demo") as app:
        gr.HTML(_hero_html())
        gr.HTML(_pipeline_html())

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(
                    _section_html(
                        "Input",
                        "Upload video and choose profile",
                        "Use the same uploaded clip to compare the Avenue and UBnormal saved profiles.",
                    )
                )
                profile_input = gr.Radio(
                    choices=list(PROFILES.keys()),
                    value=AVENUE_PROFILE.label,
                    label="Analysis profile",
                    elem_classes=["profile-radio"],
                )
                video_input = gr.Video(
                    label=None,
                    sources=["upload"],
                    elem_classes=["card-shell"],
                    height=320,
                )
                run_button = gr.Button(
                    "Run live analysis",
                    variant="primary",
                    elem_classes=["cta-button"],
                )
            with gr.Column(scale=1):
                gr.HTML(
                    _section_html(
                        "Saved profile metrics",
                        "Selected profile",
                        "These metrics come from the saved offline evaluation for the chosen profile.",
                    )
                )
                profile_overview = gr.HTML(_profile_info_html(AVENUE_PROFILE))

        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML(
                    _section_html(
                        "Live result",
                        "Anomaly timeline",
                        "Timeline generated from the uploaded video under the selected saved profile.",
                    )
                )
                timeline_output = gr.Plot(
                    label=None,
                    value=_empty_plot(),
                    elem_classes=["card-shell"],
                )
            with gr.Column(scale=4):
                summary_output = gr.HTML(
                    _empty_summary_html(),
                    elem_classes=["card-shell"],
                )

        gr.HTML(
            _section_html(
                "Frame evidence",
                "Highest-scoring frames",
                "Top anomalous moments extracted from the uploaded clip.",
            )
        )
        gallery_output = gr.Gallery(
            label=None,
            columns=4,
            rows=1,
            object_fit="contain",
            height=360,
            elem_classes=["card-shell"],
        )

        profile_input.change(
            fn=lambda label: (_render_profile_panel(label), _reset_summary_for_profile(label)),
            inputs=[profile_input],
            outputs=[profile_overview, summary_output],
        )
        run_button.click(
            fn=ENGINE.analyze,
            inputs=[video_input, profile_input],
            outputs=[timeline_output, gallery_output, summary_output],
        )

    return app


def main() -> None:
    app = build_app()
    app.launch(css=APP_CSS)


if __name__ == "__main__":
    main()
