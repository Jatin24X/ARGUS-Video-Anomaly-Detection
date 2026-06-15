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


from src.inference.engine import (
    AVENUE_PROFILE,
    UBNORMAL_PROFILE,
    PROFILES,
    InferenceProfile,
    InferenceEngine,
    _pct,
    _contiguous_regions,
    _select_gallery_indices,
)

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


def _build_timeline(
    scores: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    anomaly_mask: np.ndarray,
    profile: InferenceProfile,
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


def _profile_info_html(profile: InferenceProfile) -> str:
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
    profile: InferenceProfile,
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


class ARGUSDemoEngine(InferenceEngine):
    """Interactive engine for the standalone Stream A demo."""

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

        profile: InferenceProfile = analysis["profile"]
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
