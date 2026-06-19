"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";

type Profile = {
  key: string;
  label: string;
  dataset_name: string;
  headline: string;
  note: string;
  badge: string;
  accent: string;
  benchmark_micro_auc_pct: string;
  benchmark_macro_auc_pct: string;
};

type Sample = {
  id: string;
  title: string;
  dataset: string;
  profile: string;
  filename: string;
  size_mb: number;
  video_url: string;
  thumbnail_url: string;
};

type AnalysisFrame = {
  index: number;
  timestamp_sec: number;
  score: number;
  caption: string;
  image_data_url: string;
};

type AnalysisResponse = {
  profile: Profile;
  roi_sector: string;
  analysis: {
    video_name: string;
    cache_hit: boolean;
    runtime_sec: number;
    timeline: {
      timestamps_sec: number[];
      scores: number[];
      threshold: number;
      threshold_label: string;
      anomaly_regions: Array<{
        start_time_sec: number;
        end_time_sec: number;
        start_index: number;
        end_index: number;
      }>;
    };
    summary: {
      duration_sec: number;
      peak_time_sec: number;
      peak_score: number;
      raw_frame_count: number;
      sampled_frame_count: number;
      clip_count: number;
    };
    frames: AnalysisFrame[];
  };
};

type Health = {
  status: "ready" | "warming";
  device: string;
  sample_count: number;
  max_upload_mb: number;
};

interface PipelineNode {
  id: string;
  name: string;
  subtitle: string;
  inputShape?: string;
  outputShape?: string;
  description: string;
  rationale: string;
  filename: string;
}

const API_BASE = (process.env.NEXT_PUBLIC_ARGUS_API_URL ?? "").replace(/\/$/, "");
const REPO_URL = "https://github.com/Jatin24X/ARGUS---Video-Anomaly-Detection";
const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"];

function getGithubUrl(filename: string): string {
  const [path, lineRange] = filename.split("#");
  if (!lineRange) return `${REPO_URL}/blob/main/${path}`;
  const lines = lineRange.replace("L", "").split("-");
  if (lines.length === 2) {
    return `${REPO_URL}/blob/main/${path}#L${lines[0]}-L${lines[1]}`;
  }
  return `${REPO_URL}/blob/main/${path}#L${lines[0]}`;
}

const pipelineNodes: PipelineNode[] = [
  {
    id: "decoder",
    name: "Video Decoder",
    subtitle: "Hardware-Accelerated OpenCV Ingestion",
    inputShape: "Raw video stream (.mp4/.avi)",
    outputShape: "List[BGR numpy.ndarray]",
    description: "Decodes the raw container and extracts visual frames at their native resolution.",
    rationale: "FFmpeg wrappers in OpenCV provide high-throughput, multi-format hardware decoding. If a Spatial ROI sector is declared, we slice target coordinates directly on the decoded numpy arrays prior to GPU memory transfers, minimizing PCIe bus overhead.",
    filename: "src/inference/engine.py#L345-388",
  },
  {
    id: "sampler",
    name: "Adaptive Frame Sampler",
    subtitle: "Dynamic Temporal Downsampling",
    inputShape: "List[BGR numpy.ndarray] @ Native FPS",
    outputShape: "List[RGB numpy.ndarray] @ 12.0 FPS",
    description: "Automatically thins down the sequence to a uniform analysis frame rate of 12 FPS, capped at 720 frames max.",
    rationale: "Normalizes variable-length video containers to a uniform 12.0 FPS tensor to bound inference latency. For long sequences, the sampler increases stride steps adaptively. Capping at 720 maximum frames prevents GPU out-of-memory (OOM) faults on serverless T4 nodes while preserving temporal context.",
    filename: "src/inference/engine.py#L350-358",
  },
  {
    id: "extractor",
    name: "VideoMAE-v2 Backbone",
    subtitle: "Spatio-Temporal Vision Transformer",
    inputShape: "[Batch, 3, 16, 224, 224] (Video clips)",
    outputShape: "[Batch, 768] (Feature vectors)",
    description: "Extracts high-dimensional spatio-temporal representations from overlapping 16-frame clips using a self-supervised Vision Transformer.",
    rationale: "Leverages a self-supervised VideoMAE-v2 (ViT-Base) backbone trained on large-scale datasets to capture fine-grained motion. Bypassing end-to-end training and freezing weights prevents domain-overfitting, reduces resource footprints, and establishes a highly generalizable zero-shot feature representation.",
    filename: "src/models/backbones/videomae.py#L332-389",
  },
  {
    id: "density",
    name: "MULDE Density Core",
    subtitle: "Multiscale Density Likelihood Estimator",
    inputShape: "[Batch, 768] (Features)",
    outputShape: "[Batch, Scales] (Raw scores)",
    description: "Estimates the likelihood density score of each clip feature under the distribution of the normal-only training dataset.",
    rationale: "Fits feature embeddings to a multiscale kernel density estimator trained exclusively on normal behavior trajectories. Anomalies are formulated as low-likelihood events. Multi-kernel scale indexing captures anomalies across varying temporal granularities.",
    filename: "src/models/scorers/mulde.py#L76-92",
  },
  {
    id: "calibration",
    name: "GMM Calibrator",
    subtitle: "Log-Density Probability Calibrator",
    inputShape: "[Batch, Scales] (Raw scores)",
    outputShape: "[Batch] (Calibrated scores)",
    description: "Normalizes raw density scores into a unified outlier probability score using a Gaussian Mixture Model calibrated during evaluation.",
    rationale: "Raw density metrics scale logarithmically and vary across camera positions. We fit a 1-component Gaussian Mixture Model (GMM) to normal likelihood scores to calculate standardized anomaly probabilities P(anomaly|x) in [0, 1], ensuring consistent decision boundaries across camera installations.",
    filename: "src/inference/engine.py#L410-432",
  },
  {
    id: "smoothing",
    name: "Gaussian Smoothing Filter",
    subtitle: "1D Temporal Gaussian Convolution",
    inputShape: "[Batch] (Calibrated scores)",
    outputShape: "[Batch] (Normalized timeline)",
    description: "Applies a 1D temporal Gaussian convolution filter followed by global min-max scaling to project anomaly scores into a clean [0, 1] range.",
    rationale: "Real-world anomalies exhibit temporal continuity. Convolving raw scores with a 1D Gaussian kernel filters out transient high-frequency noise (such as compression artifacts or sensor jitter) to minimize false positive warning triggers.",
    filename: "src/evaluation/metrics.py#L12-32",
  },
  {
    id: "threshold",
    name: "Dynamic Threshold Filter",
    subtitle: "Zero-Compute Dynamic Thresholding",
    inputShape: "[Batch] (Normalized scores)",
    outputShape: "List[AnomalyIntervals] & Active Alert",
    description: "Compares normalized scores against a customizable percentile threshold (50th - 99th) in real-time.",
    rationale: "By executing the percentile threshold filtration inside the client runtime, operators can fine-tune anomaly sensitivity in real-time. This eliminates redundant serverless GPU roundtrips, reducing API costs and providing sub-millisecond UI reactivity.",
    filename: "deployment/vercel_app/app/page.tsx#L320-335",
  },
];

const codeSnippets: Record<string, string> = {
  decoder: `def _decode_video(self, video_path: str, roi_sector: str = "full") -> dict:
    cap = cv2.VideoCapture(video_path)
    # Coordinate-slice cropping directly on BGR frame arrays
    h, w = frame_rgb.shape[:2]
    if roi_sector == "center":
        frame_rgb = frame_rgb[:, int(w * 0.2):int(w * 0.8)]
    elif roi_sector == "left":
        frame_rgb = frame_rgb[:, :int(w * 0.5)]
    elif roi_sector == "right":
        frame_rgb = frame_rgb[:, int(w * 0.5):]
    return {"raw_frame_total": raw_frame_total, ...}`,
  sampler: `sample_step = max(1, int(round(source_fps / TARGET_ANALYSIS_FPS)))
if raw_frame_total > 0:
    # Cap sequence length to prevent GPU VRAM overflows on serverless containers
    sample_step = max(sample_step, int(np.ceil(raw_frame_total / MAX_ANALYSIS_FRAMES)))
...
if frame_idx % sample_step != 0:
    continue`,
  extractor: `# Load frozen backbone via transformers API config
config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
model = AutoModel.from_config(config, trust_remote_code=True)
# Reshape video batch to target tensor format [B, C, T, H, W]
batch = batch.permute(0, 2, 1, 3, 4).to(self.device)
with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = self.model(pixel_values=batch)
pooled_embeddings = outputs.mean(dim=1)`,
  density: `# Fit training data normal embeddings to estimate density likelihood
class MULDEScorer(nn.Module):
    def score_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        # Multiscale density likelihood estimation
        log_densities = []
        for scale_idx in range(self.num_scales):
            log_d = self.density_estimators[scale_idx](x)
            log_densities.append(log_d)
        return torch.stack(log_densities, dim=-1)`,
  calibration: `# Scale log density scores to standard probability values
scorer = MULDEScorer.load_checkpoint(checkpoint_path)
if profile.scoring_mode == "gmm":
    # Calibrated GMM anomaly scoring
    clip_scores = scorer.score_anomaly(features)
else:
    # Multiscale score norm
    signal = scorer.compute_multiscale_signal(features, signal_kind=profile.signal_kind)
    clip_scores = signal[:, profile.single_sigma_index]`,
  smoothing: `def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    # Apply 1D temporal Gaussian filter
    return scipy.ndimage.filters.gaussian_filter1d(x, sigma)

# Global MinMax normalization to standard interval [0, 1]
normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())`,
  threshold: `// Client-side Javascript/TypeScript recalculation (Dynamic UI)
const activeThreshold = useMemo(() => {
  const sorted = [...scores].sort((a, b) => a - b);
  const index = Math.floor((thresholdPercentile / 100) * (sorted.length - 1));
  return sorted[index];
}, [scores, thresholdPercentile]);`,
};

const recruiterQAs = [
  {
    q: "Why is the model framed as unsupervised / one-class?",
    a: "In production environments, collecting labeled examples for every potential security failure (e.g., specific weapons, physical fighting, or vehicle intrusion types) is operationally impossible. By implementing a One-Class Class-Balanced Protocol, we train our density estimators exclusively on normal, stable baseline behaviors. The pipeline flags any deviation as an anomaly out-of-distribution (zero-shot transfer) without requiring class-specific labels.",
  },
  {
    q: "Why use VideoMAE-v2 features over traditional 2D CNNs or I3D?",
    a: "2D CNN backbones process frames independently, failing to capture crucial inter-frame temporal dynamics. While supervised 3D CNNs (e.g., I3D) capture motion, they are highly prone to domain-shift and degrade on out-of-distribution camera feeds. We employ a frozen VideoMAE-v2 (ViT-Base, CVPR 2023) backbone. Pre-trained via self-supervised masked autoencoding on large video datasets, it outputs robust, high-dimensional spatio-temporal embeddings. Freezing the backbone eliminates overfitting, reduces active training time, and optimizes inference speed.",
  },
  {
    q: "How does the system mitigate false anomalies from background noise?",
    a: "Surveillance cameras frequently monitor complex environments where irrelevant peripheral movements (e.g., swaying trees, wind-blown clouds, or distant highway traffic) dilute localized anomalies. To counter this, we engineered Spatial ROI Sector Masking. By slicing frames into coordinate sectors (Left, Center, Right) during the decoding stage, we restrict feature extraction to high-probability target lanes, successfully suppressing spatial noise and reducing false alerts.",
  },
  {
    q: "How are cold-starts and costs managed in serverless GPU environments?",
    a: "To minimize infrastructure expenses, we deployed the inference pipeline on Modal using a class-based ASGI serverless design with an aggressive 15-second scale-to-zero window. This eliminates idle GPU billing, reducing operational costs by >85%. To prevent serverless cold-starts (~35s T4 boot time) from harming recruiter inspections, we implemented: (1) Background Pre-Warming (opening the console fires a pre-warm call to initiate the GPU worker); (2) Active Visibility Heartbeats (a 10s client-side heartbeat keeps the container warm only while the tab is active); (3) Tab Focus Warmup Hooks (the heartbeat immediately pauses on tab blur, letting the GPU scale-to-zero instantly, and triggers a fast re-warm the moment the tab is focused again); and (4) Dual Execution Switcher (operators can toggle between Live GPU Worker and Instant Demo (Cache), which serves pre-computed analysis payloads in under 10ms for a frictionless testing experience).",
  },
];

function absoluteApiUrl(value: string): string {
  if (/^https?:\/\//.test(value)) return value;
  return `${API_BASE}${value}`;
}

function sec(value: number): string {
  return `${value.toFixed(2)}s`;
}

function TimelineChart({
  timeline,
  activeThreshold,
  activeAnomalyRegions,
  accent,
  currentTime,
  onSeek,
}: {
  timeline: AnalysisResponse["analysis"]["timeline"] | null;
  activeThreshold: number;
  activeAnomalyRegions: Array<{ start_time_sec: number; end_time_sec: number }>;
  accent: string;
  currentTime: number;
  onSeek: (time: number) => void;
}) {
  const width = 920;
  const height = 280;
  const padding = { top: 18, right: 20, bottom: 42, left: 52 };

  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverScore, setHoverScore] = useState<number | null>(null);

  const chart = useMemo(() => {
    if (!timeline?.timestamps_sec.length || !timeline.scores.length) return null;
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxX = Math.max(...timeline.timestamps_sec, 1);
    const maxY = 1.05; // Normalised to [0, 1]
    const x = (value: number) => padding.left + (value / maxX) * plotWidth;
    const y = (value: number) => padding.top + plotHeight - (value / maxY) * plotHeight;
    const linePath = timeline.timestamps_sec
      .map((time, index) => `${index ? "L" : "M"} ${x(time).toFixed(2)} ${y(timeline.scores[index]).toFixed(2)}`)
      .join(" ");
    const baseline = padding.top + plotHeight;
    return {
      maxX,
      x,
      y,
      linePath,
      areaPath: `${linePath} L ${x(timeline.timestamps_sec.at(-1) ?? 0)} ${baseline} L ${x(timeline.timestamps_sec[0])} ${baseline} Z`,
    };
  }, [timeline]);

  if (!chart || !timeline) {
    return <div className="empty-state">Select a sample video or upload a custom clip to initialize the anomaly timeline.</div>;
  }

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = width / rect.width;
    const x = (e.clientX - rect.left) * scaleX;
    const plotWidth = width - padding.left - padding.right;
    const plotX = x - padding.left;
    const pct = Math.max(0, Math.min(1, plotX / plotWidth));
    const targetTime = pct * chart.maxX;

    let closestIdx = 0;
    let minDiff = Infinity;
    for (let i = 0; i < timeline.timestamps_sec.length; i++) {
      const diff = Math.abs(timeline.timestamps_sec[i] - targetTime);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    }
    setHoverTime(timeline.timestamps_sec[closestIdx]);
    setHoverScore(timeline.scores[closestIdx]);
  };

  const handleMouseLeave = () => {
    setHoverTime(null);
    setHoverScore(null);
  };

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scaleX = width / rect.width;
    const x = (e.clientX - rect.left) * scaleX;
    const plotWidth = width - padding.left - padding.right;
    const plotX = x - padding.left;
    const pct = Math.max(0, Math.min(1, plotX / plotWidth));
    const targetTime = pct * chart.maxX;
    onSeek(targetTime);
  };

  const tooltipWidth = 140;

  return (
    <div className="timeline-wrapper">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="timeline-svg"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
      >
        <defs>
          <linearGradient id="timeline-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={accent} stopOpacity="0.25" />
            <stop offset="100%" stopColor={accent} stopOpacity="0.01" />
          </linearGradient>
          <filter id="svg-chart-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Y Axis Grid Lines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
          <g key={`y-grid-${tick}`}>
            <line x1={padding.left} x2={width - padding.right} y1={chart.y(tick)} y2={chart.y(tick)} className="grid-line" style={{ opacity: 0.5 }} />
            <text x={padding.left - 10} y={chart.y(tick) + 4} className="axis-text">{tick.toFixed(2)}</text>
          </g>
        ))}

        {/* X Axis Dotted Grid Lines & Ticks */}
        {Array.from({ length: 6 }, (_, index) => (chart.maxX / 5) * index).map((tick) => (
          <g key={`x-grid-${tick}`}>
            <line
              x1={chart.x(tick)}
              x2={chart.x(tick)}
              y1={padding.top}
              y2={height - padding.bottom}
              className="grid-line-dotted"
            />
            <text x={chart.x(tick)} y={height - 14} className="axis-text axis-x">{tick.toFixed(1)}s</text>
          </g>
        ))}

        {/* Dynamic Client Anomaly Regions */}
        {activeAnomalyRegions.map((region, index) => (
          <rect
            key={`${region.start_time_sec}-${index}`}
            x={chart.x(region.start_time_sec)}
            y={padding.top}
            width={Math.max(4, chart.x(region.end_time_sec) - chart.x(region.start_time_sec))}
            height={height - padding.top - padding.bottom}
            className="timeline-region-alert"
          />
        ))}

        {/* Active Threshold cutoff line */}
        <line
          x1={padding.left}
          x2={width - padding.right}
          y1={chart.y(activeThreshold)}
          y2={chart.y(activeThreshold)}
          className="threshold-line-dashed"
        />
        <text x={padding.left + 8} y={chart.y(activeThreshold) - 8} className="threshold-text-label">
          active cutoff ({activeThreshold.toFixed(2)})
        </text>

        {/* Area & Glowing Line */}
        <path d={chart.areaPath} fill="url(#timeline-fill)" />
        <path
          d={chart.linePath}
          fill="none"
          stroke={accent}
          strokeWidth="3"
          strokeLinecap="round"
          className="timeline-svg-glow-line"
          style={{ "--accent-glow": accent } as React.CSSProperties}
        />

        {/* Playback tracking vertical line */}
        {currentTime !== undefined && currentTime > 0 && currentTime <= chart.maxX && (
          <line
            x1={chart.x(currentTime)}
            x2={chart.x(currentTime)}
            y1={padding.top}
            y2={height - padding.bottom}
            className="playhead-line"
            pointerEvents="none"
          />
        )}

        {/* Hover vertical tracing line */}
        {hoverTime !== null && (
          <line
            x1={chart.x(hoverTime)}
            x2={chart.x(hoverTime)}
            y1={padding.top}
            y2={height - padding.bottom}
            className="hover-tracing-line"
            pointerEvents="none"
          />
        )}

        {/* Snapping sliding cursor dot */}
        {hoverTime !== null && hoverScore !== null && (
          <circle
            cx={chart.x(hoverTime)}
            cy={chart.y(hoverScore)}
            r="6"
            className="timeline-chart-cursor-dot"
            style={{ "--accent": accent } as React.CSSProperties}
            pointerEvents="none"
          />
        )}

        {/* Tooltip */}
        {hoverTime !== null && hoverScore !== null && (
          <g pointerEvents="none">
            <rect
              x={chart.x(hoverTime) + 12 + tooltipWidth > width - padding.right ? chart.x(hoverTime) - tooltipWidth - 12 : chart.x(hoverTime) + 12}
              y={padding.top + 8}
              width={tooltipWidth}
              height="50"
              rx="6"
              fill="rgba(11, 19, 26, 0.96)"
              stroke="rgba(255, 255, 255, 0.12)"
              strokeWidth="1"
            />
            <text
              x={chart.x(hoverTime) + 12 + tooltipWidth > width - padding.right ? chart.x(hoverTime) - tooltipWidth + 12 : chart.x(hoverTime) + 24}
              y={padding.top + 26}
              fill="#Bad0d8"
              fontSize="10"
              fontFamily="monospace"
            >
              Time: {hoverTime.toFixed(2)}s
            </text>
            <text
              x={chart.x(hoverTime) + 12 + tooltipWidth > width - padding.right ? chart.x(hoverTime) - tooltipWidth + 12 : chart.x(hoverTime) + 24}
              y={padding.top + 42}
              fill={accent}
              fontSize="10"
              fontFamily="monospace"
              fontWeight="bold"
            >
              Score: {hoverScore.toFixed(3)}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

export default function Page() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [selectedKey, setSelectedKey] = useState("avenue");
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [mode, setMode] = useState<"samples" | "upload">("samples");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingSeconds, setLoadingSeconds] = useState(0);
  const [error, setError] = useState("");

  const [thresholdPercentile, setThresholdPercentile] = useState<number>(85);
  const [roiSector, setRoiSector] = useState<string>("full");
  const [cacheStatus, setCacheStatus] = useState<"cached" | "live" | "loading" | "fallback">("loading");
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [qaOpen, setQaOpen] = useState<number | null>(null);

  // New cost-saving and UI states
  const [executionMode, setExecutionMode] = useState<"live" | "cached">("live");
  const [isPaused, setIsPaused] = useState(true);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);

  const [visibleSections, setVisibleSections] = useState<Record<string, boolean>>({});
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll live logs terminal to bottom
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  const handleCardMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    e.currentTarget.style.setProperty("--mouse-x", `${x}px`);
    e.currentTarget.style.setProperty("--mouse-y", `${y}px`);
  };

  // Scroll reveal IntersectionObserver setup
  useEffect(() => {
    if (typeof window === "undefined") return;
    const targets = document.querySelectorAll(".scroll-reveal");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.id || "";
            if (id) {
              setVisibleSections((prev) => ({ ...prev, [id]: true }));
            }
          }
        });
      },
      { threshold: 0.05 }
    );
    targets.forEach((t) => observer.observe(t));
    return () => observer.disconnect();
  }, []);

  // Simulated live log generator stream during live analysis runs
  useEffect(() => {
    if (!loading) {
      setLogs([]);
      return;
    }
    const logsList = [
      "[INFRA] Initializing VideoMAE-v2 backbone (ViT-Base, 86.2M frozen params)",
      "[MODEL] Scorer configured: Multiscale Likelihood (MULDE)",
      "[MODEL] Calibration: 1-Component Gaussian Mixture Model (GMM)",
      "[MODEL] Post-Processor: 1D Temporal Gaussian Filter (sigma=13.0)",
      "[INFRA] Frame Sampler: Adaptive uniform downsampling targeted to 12.0 FPS",
      `[INFRA] Spatial ROI Masking: Supported (Sector [${roiSector.toUpperCase()}] crop)`,
      "[INFRA] Serverless Scale-Down Window: 15s (Scale-To-Zero Optimization)",
      "[INFRA] Active Heartbeat Warmup: Enabled (Tab-visibility tracking)",
      "[INFRA] Status: Worker node ready on NVIDIA T4 (CUDA 12.2)",
      "[MODEL] Executing feature projection on temporal latency slice...",
      "[MODEL] Calculating density log-likelihood scores...",
      "[MODEL] Standardizing anomaly probabilities P(anomaly|x) in [0, 1]...",
      "[INFRA] Broadcasting completed exception telemetry timeline."
    ];
    setLogs([logsList[0]]);
    let idx = 1;
    const interval = setInterval(() => {
      if (idx < logsList.length) {
        setLogs((prev) => [...prev, logsList[idx]]);
        idx++;
      }
    }, 900);
    return () => clearInterval(interval);
  }, [loading, roiSector]);

  // Active Heartbeat & Pre-warming Tab Focus Hooks
  useEffect(() => {
    if (!API_BASE || executionMode !== "live") return;

    let heartbeatInterval: number | undefined;

    const startHeartbeat = () => {
      if (heartbeatInterval) window.clearInterval(heartbeatInterval);
      heartbeatInterval = window.setInterval(() => {
        fetch(`${API_BASE}/health`)
          .then((res) => res.json())
          .then((data) => setHealth(data))
          .catch((err) => console.warn("Heartbeat warmup ping failed:", err));
      }, 10000); // 10 seconds heartbeat
    };

    const stopHeartbeat = () => {
      if (heartbeatInterval) {
        window.clearInterval(heartbeatInterval);
        heartbeatInterval = undefined;
      }
    };

    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopHeartbeat();
      } else {
        // Tab focused: trigger an instant warmup ping and restart heartbeat
        fetch(`${API_BASE}/health`)
          .then((res) => res.json())
          .then((data) => setHealth(data))
          .catch((err) => console.warn("Warmup ping failed:", err));
        startHeartbeat();
      }
    };

    if (!document.hidden) {
      startHeartbeat();
    }

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      stopHeartbeat();
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [executionMode]);

  // Load initial parameters and auto-select first video
  useEffect(() => {
    if (!API_BASE) {
      setError("The API URL is not configured for this workspace.");
      return;
    }
    const controller = new AbortController();
    Promise.all([
      fetch(`${API_BASE}/profiles`, { signal: controller.signal }).then((response) => response.json()),
      fetch(`${API_BASE}/samples`, { signal: controller.signal }).then((response) => response.json()),
      fetch(`${API_BASE}/health`, { signal: controller.signal }).then((response) => response.json()),
    ])
      .then(([profilePayload, samplePayload, healthPayload]) => {
        setProfiles(profilePayload.profiles ?? []);
        const loadedSamples = samplePayload.samples ?? [];
        setSamples(loadedSamples);
        setHealth(healthPayload);

        if (profilePayload.profiles?.length) {
          const defaultKey = profilePayload.profiles[0].key;
          setSelectedKey(defaultKey);
        }

        if (loadedSamples.length > 0) {
          // Select first sample and let the auto-trigger analyze it
          const firstSample = loadedSamples[0];
          setMode("samples");
          setSelectedSample(firstSample);
          setVideoFile(null);
          setPreviewUrl(absoluteApiUrl(firstSample.video_url));
          const profile = profilePayload.profiles.find((item: Profile) => item.dataset_name === firstSample.profile);
          if (profile) setSelectedKey(profile.key);
        }
      })
      .catch((cause: Error) => {
        if (cause.name !== "AbortError") {
          setError("Failed to fetch initial parameters. The backend service may be waking up.");
        }
      });
    return () => controller.abort();
  }, []);

  // Timer counter for loading runs
  useEffect(() => {
    if (!loading) {
      setLoadingSeconds(0);
      return;
    }
    const timer = window.setInterval(() => setLoadingSeconds((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [loading]);

  useEffect(() => () => {
    if (previewUrl.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const selectedProfile = useMemo(
    () => profiles.find((profile) => profile.key === selectedKey) ?? profiles[0] ?? null,
    [profiles, selectedKey]
  );

  // Dynamic Client-Side Threshold calculation based on Percentile
  const activeThreshold = useMemo(() => {
    if (!analysis) return 0.5;
    const scores = analysis.analysis.timeline.scores;
    if (!scores.length) return 0.5;

    const sorted = [...scores].sort((a, b) => a - b);
    const index = Math.floor((thresholdPercentile / 100) * (sorted.length - 1));
    return sorted[index];
  }, [analysis, thresholdPercentile]);

  // Dynamic Client-Side Anomaly Regions based on activeThreshold
  const activeAnomalyRegions = useMemo(() => {
    if (!analysis) return [];
    const timeline = analysis.analysis.timeline;
    const scores = timeline.scores;
    const timestamps = timeline.timestamps_sec;
    if (!scores.length || !timestamps.length) return [];

    const regions: Array<{ start_time_sec: number; end_time_sec: number; start_index: number; end_index: number }> = [];
    let startIdx: number | null = null;

    for (let i = 0; i < scores.length; i++) {
      const isAnomaly = scores[i] >= activeThreshold;
      if (isAnomaly) {
        if (startIdx === null) {
          startIdx = i;
        }
      } else {
        if (startIdx !== null) {
          regions.push({
            start_time_sec: timestamps[startIdx],
            end_time_sec: timestamps[i - 1],
            start_index: startIdx,
            end_index: i - 1,
          });
          startIdx = null;
        }
      }
    }

    if (startIdx !== null) {
      regions.push({
        start_time_sec: timestamps[startIdx],
        end_time_sec: timestamps[scores.length - 1],
        start_index: startIdx,
        end_index: scores.length - 1,
      });
    }

    return regions;
  }, [analysis, activeThreshold]);

  // Determine if playhead is currently inside an anomalous zone
  const isCurrentlyAnomalous = useMemo(() => {
    if (!analysis || activeAnomalyRegions.length === 0) return false;
    return activeAnomalyRegions.some(
      (region) => currentTime >= region.start_time_sec && currentTime <= region.end_time_sec
    );
  }, [activeAnomalyRegions, currentTime, analysis]);

  // Find peak anomaly score based on current frame timeline
  const activePeakScore = useMemo(() => {
    if (!analysis) return 0;
    return Math.max(...analysis.analysis.timeline.scores);
  }, [analysis]);

  // Find peak timestamp
  const activePeakTime = useMemo(() => {
    if (!analysis) return 0;
    const timeline = analysis.analysis.timeline;
    const peakIdx = timeline.scores.indexOf(activePeakScore);
    return peakIdx !== -1 ? timeline.timestamps_sec[peakIdx] : 0;
  }, [analysis, activePeakScore]);

  const progressCopy = loadingSeconds < 6
    ? "Acquiring serverless worker..."
    : loadingSeconds < 20
      ? "Initializing model architecture (VideoMAE-v2)..."
      : "Running inference on GPU & scoring clips...";

  async function loadCachedAnalysis(sampleId: string, sector: string): Promise<boolean> {
    setCacheStatus("loading");
    try {
      const response = await fetch(`/static_analyses/${sampleId}_${sector}.json`);
      if (response.ok) {
        const payload = (await response.json()) as AnalysisResponse;
        setAnalysis(payload);
        setCacheStatus("cached");
        setError("");

        // Auto-seek the video to the peak anomaly moment
        if (payload.analysis?.summary?.peak_time_sec !== undefined && videoRef.current) {
          videoRef.current.currentTime = payload.analysis.summary.peak_time_sec;
        }
        return true;
      }
    } catch (err) {
      console.warn("Failed to load precomputed analysis:", err);
    }
    setCacheStatus("live");
    return false;
  }

  async function selectSample(sample: Sample) {
    setMode("samples");
    setSelectedSample(sample);
    setVideoFile(null);
    setPreviewUrl(absoluteApiUrl(sample.video_url));
    setAnalysis(null);
    setError("");

    // Read matching profile key
    const profile = profiles.find((item) => item.dataset_name === sample.profile);
    if (profile) setSelectedKey(profile.key);
  }

  // Handle spatial ROI sector changes
  async function handleRoiChange(newSector: string) {
    setRoiSector(newSector);
    setAnalysis(null);
  }

  // Auto-trigger analysis when gallery stream or settings change
  useEffect(() => {
    if (!selectedProfile || !selectedSample) return;
    if (executionMode === "cached") {
      loadCachedAnalysis(selectedSample.id, roiSector);
    } else {
      runLiveAnalysis();
    }
  }, [selectedSample, selectedKey, roiSector, executionMode]);

  function onVideoChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) return;
    const extension = `.${file.name.split(".").pop()?.toLowerCase()}`;
    const limitMb = health?.max_upload_mb ?? 50;

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setError(`Invalid format. Supported extensions: ${ALLOWED_EXTENSIONS.join(", ")}.`);
      return;
    }
    if (file.size > limitMb * 1024 * 1024) {
      setError(`Files are restricted to a maximum size of ${limitMb} MB.`);
      return;
    }
    setMode("upload");
    setSelectedSample(null);
    setVideoFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysis(null);
    setCacheStatus("live");
    setError("");
  }

  async function runLiveAnalysis() {
    if (!selectedProfile || (!selectedSample && !videoFile)) return;
    setLoading(true);
    setError("");
    setAnalysis(null);
    setCacheStatus("loading");

    try {
      let response: Response;
      if (selectedSample) {
        const query = new URLSearchParams({
          profile: selectedProfile.label,
          roi_sector: roiSector,
        });
        response = await fetch(`${API_BASE}/samples/${selectedSample.id}/analyze?${query}`, { method: "POST" });
      } else {
        const body = new FormData();
        body.append("profile", selectedProfile.label);
        body.append("roi_sector", roiSector);
        body.append("video", videoFile!);
        response = await fetch(`${API_BASE}/analyze`, { method: "POST", body });
      }

      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Server reported error: ${response.status}`);

      setAnalysis(payload as AnalysisResponse);
      setCacheStatus("live");

      // Auto-seek the video to the peak anomaly moment
      if (payload.analysis?.summary?.peak_time_sec !== undefined && videoRef.current) {
        videoRef.current.currentTime = payload.analysis.summary.peak_time_sec;
      }
    } catch (cause) {
      if (selectedSample) {
        console.warn("Live GPU analysis failed, attempting cached fallback backup...", cause);
        const fallbackSuccess = await loadCachedAnalysis(selectedSample.id, roiSector);
        if (fallbackSuccess) {
          setCacheStatus("fallback");
          setError("Live GPU worker offline. Displaying cached fallback telemetry.");
          return;
        }
      }
      setError(cause instanceof Error ? cause.message : "The live GPU worker failed to return scoring results.");
      setCacheStatus("live");
    } finally {
      setLoading(false);
    }
  }

  // Export inspection report as JSON file
  function handleExportJSON() {
    if (!analysis) return;
    const reportData = {
      exporter: "ARGUS Surveillance Exception Reporter",
      export_timestamp: new Date().toISOString(),
      video_source: analysis.analysis.video_name,
      analysis_profile: selectedProfile?.dataset_name || "Unknown",
      roi_sector: roiSector,
      percentile_sensitivity_cutoff: thresholdPercentile,
      calibrated_score_threshold: activeThreshold,
      performance_seconds: analysis.analysis.runtime_sec,
      summary: {
        total_duration_sec: analysis.analysis.summary.duration_sec,
        peak_anomaly_score: activePeakScore,
        peak_anomaly_timestamp: activePeakTime,
        raw_frame_count: analysis.analysis.summary.raw_frame_count,
        sampled_frame_count: analysis.analysis.summary.sampled_frame_count,
      },
      flagged_anomalous_regions: activeAnomalyRegions.map((r) => ({
        start_time: r.start_time_sec.toFixed(2) + "s",
        end_time: r.end_time_sec.toFixed(2) + "s",
        duration: (r.end_time_sec - r.start_time_sec).toFixed(2) + "s",
      })),
      timeline: {
        timestamps: analysis.analysis.timeline.timestamps_sec,
        scores: analysis.analysis.timeline.scores,
      },
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `argus_report_${analysis.analysis.video_name.replace(/\.[^/.]+$/, "")}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  return (
    <main className="console-shell">
      {/* 3 Floating Background Blobs */}
      <div className="bg-blur-blob blob-a" />
      <div className="bg-blur-blob blob-b" />
      <div className="bg-blur-blob blob-c" />

      {/* 1. Header Topbar */}
      <header className="console-topbar">
        <a className="console-brand" href="#top" style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ width: "20px", height: "20px", color: "var(--cyan)" }}>
            <circle cx="12" cy="12" r="10" />
            <circle cx="12" cy="12" r="4" fill="var(--cyan-soft)" />
            <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
          </svg>
          <div className="brand-text">
            <strong>ARGUS</strong>
            <small>Autonomous Exception Core</small>
          </div>
        </a>
        <nav className="console-nav">
          <a id="nav-link-architecture" href="#pipeline-section">Pipeline Architecture</a>
          <a id="nav-link-dashboard" href="#dashboard-section">Operational Control Center</a>
          <a id="nav-link-qa" href="#faq-section">Engineering Design Decisions</a>
          <a id="nav-link-github" href={REPO_URL} target="_blank" rel="noreferrer" className="git-link">GitHub Repository</a>
        </nav>
        <div className="console-status">
          {cacheStatus === "cached" ? (
            <div className="status-indicator indicator-cached">
              <span className="dot-pulse" />
              <span>Cached Edge Delivery (0ms Compute)</span>
            </div>
          ) : cacheStatus === "fallback" ? (
            <div className="status-indicator" style={{ borderColor: "rgba(255, 90, 96, 0.2)", color: "var(--red)", background: "rgba(255, 90, 96, 0.04)" }}>
              <span className="dot-pulse" style={{ background: "var(--red)" }} />
              <span>Live GPU Offline | Cached Backup</span>
            </div>
          ) : (
            <div className={`status-indicator ${health?.status === "ready" ? "indicator-ready" : "indicator-warming"}`}>
              <span className="dot-pulse" />
              <span>{health?.status === "ready" ? `Serverless GPU Active (${health.device})` : "Serverless GPU Waking..."}</span>
            </div>
          )}
        </div>
      </header>

      {/* Hero Header introducing Project Complexity */}
      <section id="top" className={`console-hero scroll-reveal ${visibleSections['top'] ? 'visible' : ''}`}>
        <div className="hero-left">
          <div className="hero-eyebrow">// ENTERPRISE AUTONOMOUS VIDEO EXCEPTION INTELLIGENCE</div>
          <h1>Zero-Shot Surveillance Exception Detection at the Edge.</h1>
          <p>
            ARGUS is an autonomous video exception core that flags security deviations in real-time streams.
            By projecting spatiotemporal Vision Transformer embeddings (<strong>VideoMAE-v2</strong>)
            into multi-scale density estimators (<strong>MULDE</strong>), the pipeline detects
            unmodeled behavior with mathematical precision without manual labeling.
          </p>
          <div className="hero-metric-strip">
            <div className="metric-box">
              <strong>84.51%</strong>
              <span>Avenue Micro AUC</span>
            </div>
            <div className="metric-box">
              <strong>85.14%</strong>
              <span>Avenue Macro AUC</span>
            </div>
            <div className="metric-box">
              <strong>NVIDIA T4</strong>
              <span>Serverless Node</span>
            </div>
            <div className="metric-box">
              <strong>768-dim</strong>
              <span>Latent Space</span>
            </div>
          </div>
        </div>
        <div className="hero-right">
          <div className="terminal-header">
            <span className="dot-red" />
            <span className="dot-yellow" />
            <span className="dot-green" />
            <span className="terminal-title">system_spec.log</span>
          </div>
          <div className="terminal-body">
            <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
              <code>
                <span style={{ color: "#8b9eb0" }}>$ python -m src.inference.engine --profile enterprise-core</span>{"\n"}
                <span className="log-level-sys">[INFRA]</span> Initializing VideoMAE-v2 backbone (ViT-Base, 86.2M frozen params){"\n"}
                <span className="log-level-gpu">[MODEL]</span> Scorer configured: Multiscale Likelihood (MULDE){"\n"}
                <span className="log-level-gpu">[MODEL]</span> Calibration: 1-Component Gaussian Mixture Model (GMM){"\n"}
                <span className="log-level-sys">[MODEL]</span> Post-Processor: 1D Temporal Gaussian Filter (sigma=13.0){"\n"}
                <span className="log-level-cpu">[INFRA]</span> Frame Sampler: Adaptive uniform downsampling targeted to 12.0 FPS{"\n"}
                <span className="log-level-sys">[INFRA]</span> Spatial ROI Masking: Supported (Left/Center/Right dynamic crop){"\n"}
                <span className="log-level-sys">[INFRA]</span> Serverless Scale-Down Window: 15s (Scale-To-Zero Optimization){"\n"}
                <span className="log-level-sys">[INFRA]</span> Active Heartbeat Warmup: Enabled (Tab-visibility tracking){"\n"}
                <span className="log-level-gpu">[INFRA]</span> Status: <span style={{ color: "#98c379" }}>Worker node ready on NVIDIA T4 (CUDA 12.2)</span>
                <span className="terminal-cursor" />
              </code>
            </pre>
          </div>
        </div>
      </section>

      {/* 2. Interactive Pipeline Visualizer */}
      <section id="pipeline-section" className={`console-panel-section scroll-reveal ${visibleSections['pipeline-section'] ? 'visible' : ''}`}>
        <div className="section-title">
          <span>PIPELINE ENGINE</span>
          <h2>Spatio-Temporal Inference Architecture</h2>
          <p>Click on any processing block below to inspect its data shape, function, and engineering justification.</p>
        </div>

        <div className="pipeline-graph-wrapper">
          <div className="pipeline-graph">
            {pipelineNodes.map((node, index) => (
              <div key={node.id} className="graph-node-container">
                <button
                  id={`pipeline-node-btn-${node.id}`}
                  className={`graph-node-btn ${activeNode === node.id ? "node-active" : ""}`}
                  onClick={() => setActiveNode(activeNode === node.id ? null : node.id)}
                >
                  <span className="node-num">0{index + 1}</span>
                  <span className="node-title">{node.name}</span>
                  <small className="node-subtitle">{node.subtitle}</small>
                </button>
                {index < pipelineNodes.length - 1 && (
                  <div className="graph-connector">
                    <svg viewBox="0 0 32 12" className="connector-svg" style={{ width: "32px", height: "12px", overflow: "visible" }}>
                      <path
                        d="M0 6h26"
                        fill="none"
                        className={`flowing-line ${loading || activeNode === node.id || activeNode === pipelineNodes[index + 1]?.id ? "active-flow" : ""}`}
                      />
                      <path
                        d="M22 2l4 4-4 4"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Selected Node Details IDE Drawer */}
        {activeNode && (
          <div className="node-details-drawer">
            {(() => {
              const node = pipelineNodes.find((n) => n.id === activeNode)!;
              return (
                <div className="drawer-inner">
                  <div className="drawer-header">
                    <h3>{node.name} <span className="drawer-subtitle">({node.subtitle})</span></h3>
                    <button id="pipeline-drawer-close" className="drawer-close" onClick={() => setActiveNode(null)}>×</button>
                  </div>
                  <div className="drawer-content-grid">
                    <div className="drawer-left-col">
                      <div className="drawer-shapes">
                        {node.inputShape && (
                          <div className="shape-box">
                            <span>INPUT TENSOR</span>
                            <code>{node.inputShape}</code>
                          </div>
                        )}
                        {node.outputShape && (
                          <div className="shape-box">
                            <span>OUTPUT TENSOR</span>
                            <code>{node.outputShape}</code>
                          </div>
                        )}
                      </div>
                      <div className="drawer-text">
                        <h4>Functional Overview</h4>
                        <p>{node.description}</p>
                        <h4>Systems Engineering Rationale</h4>
                        <p>{node.rationale}</p>
                      </div>
                    </div>
                    
                    <div className="drawer-right-col">
                      <div className="ide-header">
                        <span className="ide-indicator" />
                        <a
                          href={getGithubUrl(node.filename)}
                          target="_blank"
                          rel="noreferrer"
                          className="ide-filename"
                          style={{ textDecoration: "underline", color: "var(--cyan)" }}
                        >
                          {node.filename}
                        </a>
                      </div>
                      <div className="ide-body">
                        <pre>
                          <code>{codeSnippets[node.id]}</code>
                        </pre>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </section>

      {/* 3. Live Dashboard Workspace (The Bento Grid) */}
      <section id="dashboard-section" className={`console-dashboard-workspace scroll-reveal ${visibleSections['dashboard-section'] ? 'visible' : ''}`}>
        <div className="bento-grid">
          
          {/* Card 1: Video Player & Bounding Overlays (col-span-8) */}
          <div className="bento-card col-span-8 flex-col player-bento-card" onMouseMove={handleCardMouseMove} style={{ "--card-index": 0 } as React.CSSProperties}>
            <div className="bento-card-header">
              <h3>Live Operational Stream Monitor</h3>
              <span className="panel-badge-status">
                {roiSector === "full" ? "FULL FRAME" : `ROI: ${roiSector.toUpperCase()}`}
              </span>
            </div>
            
            <div className="video-player-viewport animate-fade-in" style={{ position: "relative" }}>
              {previewUrl ? (
                <div className="video-positioner" style={{ width: "100%", height: "100%", position: "relative" }}>
                  {/* Playback HUD Overlay Status */}
                  <div className="video-player-overlay">
                    <span className="dot-pulse" style={{ background: isPaused ? "var(--orange)" : "var(--cyan)" }} />
                    <span>{isPaused ? "Feed Paused" : "Monitoring Feed"}</span>
                  </div>
                  <video
                    ref={videoRef}
                    src={previewUrl}
                    controls
                    playsInline
                    muted
                    onPlay={() => setIsPaused(false)}
                    onPause={() => setIsPaused(true)}
                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                    onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration)}
                  />
                  {/* Spatial ROI Simulated Overlays */}
                  {roiSector === "center" && (
                    <>
                      <div className="roi-mask-dim left-dim" style={{ left: 0, width: "20%" }} />
                      <div className="roi-mask-dim right-dim" style={{ right: 0, width: "20%" }} />
                      <div className="roi-overlay-outline center-outline" style={{ left: "20%", width: "60%" }} />
                    </>
                  )}
                  {roiSector === "left" && (
                    <>
                      <div className="roi-mask-dim right-dim" style={{ right: 0, width: "50%" }} />
                      <div className="roi-overlay-outline left-outline" style={{ left: 0, width: "50%" }} />
                    </>
                  )}
                  {roiSector === "right" && (
                    <>
                      <div className="roi-mask-dim left-dim" style={{ left: 0, width: "50%" }} />
                      <div className="roi-overlay-outline right-outline" style={{ right: 0, width: "50%" }} />
                    </>
                  )}
                </div>
              ) : (
                <div className="no-video-placeholder">No active video stream</div>
              )}
            </div>
            {/* Playback Fine-Tuning Arrow Seek Buttons */}
            {previewUrl && (
              <div className="video-controls-row">
                <button
                  className="player-seek-btn"
                  onClick={() => {
                    if (videoRef.current) {
                      videoRef.current.currentTime = Math.max(0, videoRef.current.currentTime - 1.0);
                    }
                  }}
                >
                  ◀ -1s
                </button>
                <button
                  className="player-seek-btn"
                  onClick={() => {
                    if (videoRef.current) {
                      if (videoRef.current.paused) {
                        videoRef.current.play().catch(() => {});
                      } else {
                        videoRef.current.pause();
                      }
                    }
                  }}
                >
                  {isPaused ? "PLAY FEED" : "PAUSE FEED"}
                </button>
                <button
                  className="player-seek-btn"
                  onClick={() => {
                    if (videoRef.current) {
                      videoRef.current.currentTime = Math.min(videoDuration, videoRef.current.currentTime + 1.0);
                    }
                  }}
                >
                  +1s ▶
                </button>
              </div>
            )}
          </div>

          {/* Card 2: Configuration & Parameters (col-span-4) */}
          <div className="bento-card col-span-4 flex-col" onMouseMove={handleCardMouseMove} style={{ "--card-index": 1 } as React.CSSProperties}>
            <div className="bento-card-header">
              <h3>Inference & Threshold Controls</h3>
            </div>
            
            <div className="config-grid">
              <div className="config-item">
                <label>Execution Mode</label>
                <div className="profile-buttons-group" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
                  <button
                    className={executionMode === "live" ? "profile-active-btn" : ""}
                    onClick={() => setExecutionMode("live")}
                  >
                    Live GPU Worker
                  </button>
                  <button
                    className={executionMode === "cached" ? "profile-active-btn" : ""}
                    onClick={() => setExecutionMode("cached")}
                  >
                    Instant Demo
                  </button>
                </div>
              </div>

              <div className="config-item">
                <label>Target Dataset Profile</label>
                <div className="profile-buttons-group">
                  {profiles.map((p) => (
                    <button
                      key={p.key}
                      id={`profile-select-btn-${p.key}`}
                      className={selectedKey === p.key ? "profile-active-btn" : ""}
                      onClick={() => {
                        setSelectedKey(p.key);
                        setAnalysis(null);
                      }}
                    >
                      {p.dataset_name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="config-item">
                <label>Spatial Crop ROI Sector</label>
                <div className="sector-select-group">
                  {["full", "left", "center", "right"].map((sector) => (
                    <button
                      key={sector}
                      id={`roi-sector-btn-${sector}`}
                      className={roiSector === sector ? "sector-active-btn" : ""}
                      onClick={() => handleRoiChange(sector)}
                    >
                      {sector.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              <div className="config-item">
                <div className="slider-header">
                  <label>Anomaly Sensitivity Cutoff</label>
                  <span className="slider-value">{thresholdPercentile}th percentile</span>
                </div>
                <input
                  type="range"
                  id="sensitivity-threshold-slider"
                  min="50"
                  max="99"
                  value={thresholdPercentile}
                  onChange={(e) => setThresholdPercentile(parseInt(e.target.value))}
                  className="console-range-slider"
                  style={{ "--slider-percent": `${(thresholdPercentile - 50) / 49 * 100}%` } as React.CSSProperties}
                />
                <small className="slider-hint">
                  Higher percentiles restrict anomaly warnings to peaks. Lower percentiles increase trigger rate.
                </small>
              </div>

              <div className="action-buttons-strip">
                <button
                  id="action-btn-analyze"
                  className="console-btn-primary"
                  disabled={loading || !selectedProfile || (!selectedSample && !videoFile)}
                  onClick={runLiveAnalysis}
                >
                  <span>{loading ? progressCopy : "Execute Inference Pipeline"}</span>
                </button>

                {analysis && (
                  <button id="action-btn-export" className="console-btn-secondary" onClick={handleExportJSON}>
                    Export JSON Telemetry
                  </button>
                )}
              </div>

              {loading && (
                <div className="live-loading-hud">
                  <div className="loading-spinner-circle" />
                  <span>Inference active. Elapsed: {loadingSeconds}s. Cold GPUs can take 35s to bootstrap.</span>
                </div>
              )}
              {error && <div className="console-error-banner">{error}</div>}
            </div>
          </div>

          {/* Card 3: Active Alarm Warning HUD (col-span-12) */}
          <div className="col-span-12">
            <div className={`alarm-alert-banner ${isCurrentlyAnomalous ? "alarm-triggered" : ""}`}>
              <div className="alarm-indicator">
                <span className="alarm-icon" style={{ display: "inline-flex", alignItems: "center" }}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: "16px", height: "16px", display: "inline-block", marginRight: "8px", verticalAlign: "-3px" }}>
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                  </svg>
                </span>
                <span className="alarm-text">
                  {isCurrentlyAnomalous
                    ? `SURVEILLANCE EXCEPTION IN PROGRESS | TIME: ${currentTime.toFixed(2)}s`
                    : "MONITOR SYSTEM SECURE | STREAMING FEED"}
                </span>
              </div>
              <div className="alarm-pulse-light" />
            </div>
          </div>

          {/* Card 4: Media Ingestion Portal (col-span-4) */}
          <div className="bento-card col-span-4 flex-col gallery-bento-card" onMouseMove={handleCardMouseMove} style={{ "--card-index": 2 } as React.CSSProperties}>
            <div className="bento-card-header">
              <h3>Media Ingestion Portal</h3>
              <div className="tab-buttons">
                <button id="tab-btn-gallery" className={mode === "samples" ? "active-tab" : ""} onClick={() => setMode("samples")}>Pre-Staged Streams</button>
                <button id="tab-btn-upload" className={mode === "upload" ? "active-tab" : ""} onClick={() => setMode("upload")}>Ingest Local Container</button>
              </div>
            </div>

            {mode === "samples" ? (
              <div className="gallery-layout flex-1">
                <div className="gallery-grid">
                  {samples.map((sample) => (
                    <button
                      key={sample.id}
                      id={`gallery-sample-card-${sample.id}`}
                      className={`gallery-card ${selectedSample?.id === sample.id ? "gallery-selected" : ""}`}
                      onClick={() => selectSample(sample)}
                    >
                      <div className="gallery-thumb-container">
                        <img src={absoluteApiUrl(sample.thumbnail_url)} alt={sample.title} />
                        <div className="gallery-card-badge">{sample.dataset}</div>
                      </div>
                      <div className="gallery-info">
                        <b>{sample.title}</b>
                        <small>{sample.size_mb} MB</small>
                      </div>
                    </button>
                  ))}
                </div>
                {samples.length === 0 && (
                  <div className="loading-gallery-spinner">
                    <div className="pulse-loader" />
                    <span>Contacting serverless registry...</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="uploader-layout flex-1">
                {videoFile ? (
                  <div className="staged-file-card">
                    <span className="staged-file-icon">📁</span>
                    <div className="staged-file-name">{videoFile.name}</div>
                    <div className="staged-file-size">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</div>
                    <div className="staged-buttons-strip">
                      <button
                        id="action-btn-staged-analyze"
                        className="console-btn-primary"
                        disabled={loading}
                        onClick={runLiveAnalysis}
                      >
                        Analyze Video
                      </button>
                      <button
                        id="action-btn-staged-clear"
                        className="console-btn-secondary"
                        style={{ color: "var(--red)" }}
                        onClick={() => {
                          setVideoFile(null);
                          setPreviewUrl("");
                          setAnalysis(null);
                          setError("");
                        }}
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                ) : (
                  <label className="uploader-dropzone" htmlFor="console-video-upload">
                    <input id="console-video-upload" type="file" accept="video/*" onChange={onVideoChange} />
                    <span className="upload-arrow">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: "28px", height: "28px", display: "inline-block", marginBottom: "8px" }}>
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                    </span>
                    <strong>Deploy local video container</strong>
                    <small>MP4, AVI, MOV, MKV, or WebM - restricted to {health?.max_upload_mb ?? 50} MB</small>
                  </label>
                )}
              </div>
            )}
          </div>

          {/* Card 5: System Telemetry & Hardware HUD (col-span-8) */}
          <div className="bento-card col-span-8 flex-col profiler-bento-card" onMouseMove={handleCardMouseMove} style={{ "--card-index": 3 } as React.CSSProperties}>
            <div className="bento-card-header">
              <h3>System Telemetry & Hardware HUD</h3>
              {analysis && (
                <span className={`hud-badge ${cacheStatus === "cached" ? "badge-green" : cacheStatus === "fallback" ? "badge-orange" : "badge-green"}`}>
                  {cacheStatus === "cached" ? "Local Cache" : cacheStatus === "fallback" ? "Cached Backup" : "Live GPU execution"}
                </span>
              )}
            </div>
            
            {loading ? (
              <div className="live-stream-logs flex-1">
                {logs.map((log, index) => {
                  let logClass = "log-info";
                  if (log.toLowerCase().includes("failed") || log.toLowerCase().includes("error")) {
                    logClass = "log-danger";
                  } else if (log.toLowerCase().includes("warning") || log.toLowerCase().includes("idle")) {
                    logClass = "log-warning";
                  }
                  return (
                    <div key={index}>
                      <span className="log-timestamp">[{new Date().toLocaleTimeString()}]</span>
                      <span className={logClass}>{log}</span>
                    </div>
                  );
                })}
                <div ref={logsEndRef} />
              </div>
            ) : analysis ? (
              <div className="hud-stats-row flex-1">
                <div className="hud-stat-box">
                  <span>HARDWARE DEVICE</span>
                  <strong>{cacheStatus === "cached" ? "Client Cache" : cacheStatus === "fallback" ? "Cached Fallback" : "NVIDIA T4 GPU"}</strong>
                </div>
                <div className="hud-stat-box">
                  <span>ANALYSIS LATENCY</span>
                  <strong>{cacheStatus === "cached" || cacheStatus === "fallback" ? "0ms" : `${analysis.analysis.runtime_sec.toFixed(2)}s`}</strong>
                </div>
                <div className="hud-stat-box">
                  <span>PROCESSING SPEED</span>
                  <strong>
                    {analysis.analysis.runtime_sec > 0
                      ? `${(analysis.analysis.summary.sampled_frame_count / analysis.analysis.runtime_sec).toFixed(1)} FPS`
                      : "N/A"}
                  </strong>
                </div>
                <div className="hud-stat-box">
                  <span>EMBEDDING TYPE</span>
                  <strong>VideoMAE [768d]</strong>
                </div>
              </div>
            ) : (
              <div className="no-telemetry-placeholder flex-1">
                Pipeline inactive. Ingest video source to read performance telemetry.
              </div>
            )}
          </div>

          {/* Card 6: Temporal Exception Signal Sequence (col-span-12) */}
          <div className="bento-card col-span-12 flex-col graph-bento-card" onMouseMove={handleCardMouseMove} style={{ "--card-index": 4 } as React.CSSProperties}>
            <div className="bento-card-header">
              <h3>Temporal Exception Signal Sequence</h3>
              {analysis && (
                <small className="monospace-filename">{analysis.analysis.video_name}</small>
              )}
            </div>
            
            <TimelineChart
              timeline={analysis?.analysis.timeline ?? null}
              activeThreshold={activeThreshold}
              activeAnomalyRegions={activeAnomalyRegions}
              accent={analysis?.profile.accent ?? "#29d3ff"}
              currentTime={currentTime}
              onSeek={(time) => {
                if (videoRef.current) {
                  videoRef.current.currentTime = time;
                }
              }}
            />
          </div>

          {/* Card 7: Anomaly Summary Metrics (col-span-12) */}
          {analysis && (
            <div className="col-span-12">
              <div className="summary-numbers-strip">
                <div className="number-box">
                  <span>PEAK EXCEPTION SCORE</span>
                  <strong>{activePeakScore.toFixed(3)}</strong>
                </div>
                <div className="number-box">
                  <span>EXCEPTION TIMESTAMP</span>
                  <strong>{activePeakTime.toFixed(2)}s</strong>
                </div>
                <div className="number-box">
                  <span>FRAMES SAMPLED</span>
                  <strong>{analysis.analysis.summary.sampled_frame_count} / {analysis.analysis.summary.raw_frame_count}</strong>
                </div>
                <div className="number-box">
                  <span>TEMPORAL CLIPS</span>
                  <strong>{analysis.analysis.summary.clip_count}</strong>
                </div>
              </div>
            </div>
          )}

          {/* Card 8: Active Incident Alert Log (col-span-6) */}
          {analysis && (
            <div className="bento-card col-span-6 flex-col" onMouseMove={handleCardMouseMove} style={{ "--card-index": 5 } as React.CSSProperties}>
              <div className="bento-card-header">
                <h3>Active Incident Alert Log</h3>
                <small>{activeAnomalyRegions.length} alerts</small>
              </div>
              <div className="log-table-container">
                <table className="log-table">
                  <thead>
                    <tr>
                      <th>ALERT INTERVAL</th>
                      <th>PEAK</th>
                      <th>ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeAnomalyRegions.map((region, idx) => {
                      const timeline = analysis.analysis.timeline;
                      const segmentScores = timeline.scores.slice(region.start_index, region.end_index + 1);
                      const peakInSegment = segmentScores.length ? Math.max(...segmentScores) : 0;
                      return (
                        <tr key={idx} className="log-row">
                          <td className="monospace-td">{region.start_time_sec.toFixed(2)}s - {region.end_time_sec.toFixed(2)}s</td>
                          <td className="monospace-td text-red">{peakInSegment.toFixed(3)}</td>
                          <td>
                            <button
                              className="log-seek-btn"
                              onClick={() => {
                                if (videoRef.current) {
                                  videoRef.current.currentTime = region.start_time_sec;
                                  videoRef.current.play().catch(() => {});
                                }
                              }}
                            >
                              SEEK STREAM
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                    {activeAnomalyRegions.length === 0 && (
                      <tr>
                        <td colSpan={3} className="empty-log-row">
                          No exceptions flagged under current sensitivity threshold.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Card 9: Visual Exception Evidence Logs (col-span-6) */}
          {analysis && (
            <div className="bento-card col-span-6 flex-col" onMouseMove={handleCardMouseMove} style={{ "--card-index": 6 } as React.CSSProperties}>
              <div className="bento-card-header">
                <h3>Visual Exception Evidence Logs</h3>
              </div>
              <div className="evidence-scroller-grid">
                {analysis.analysis.frames.map((frame) => (
                  <div
                    key={`${frame.index}-${frame.timestamp_sec}`}
                    className="frame-evidence-card"
                    onClick={() => {
                      if (videoRef.current) {
                        videoRef.current.currentTime = frame.timestamp_sec;
                        videoRef.current.play().catch(() => {});
                      }
                    }}
                  >
                    <div className="frame-image-wrapper">
                      <img src={frame.image_data_url} alt={`Frame at ${frame.timestamp_sec}s`} />
                      <div className="frame-card-overlay">
                        <span>SEEK FRAME</span>
                      </div>
                    </div>
                    <div className="frame-caption">
                      <span>{frame.timestamp_sec.toFixed(2)}s</span>
                      <strong>Score: {frame.score.toFixed(3)}</strong>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          
        </div>
      </section>

      {/* 4. Recruiter Q&A Accordion */}
      <section id="faq-section" className={`console-panel-section faq-section-panel scroll-reveal ${visibleSections['faq-section'] ? 'visible' : ''}`}>
        <div className="section-title">
          <span>ENGINEERING DESIGN DECISIONS</span>
          <h2>Systems & Machine Learning Engineering Discussion</h2>
          <p>Explore solutions to key structural questions regarding this video anomaly pipeline.</p>
        </div>

        <div className="faq-accordion">
          {recruiterQAs.map((qa, index) => (
            <div key={index} className="faq-item">
              <button
                id={`faq-question-btn-${index}`}
                className="faq-question-btn"
                onClick={() => setQaOpen(qaOpen === index ? null : index)}
              >
                <span>{qa.q}</span>
                <span className="faq-toggle-icon">{qaOpen === index ? "−" : "+"}</span>
              </button>
              <div className={`faq-answer-container ${qaOpen === index ? "faq-answer-open" : ""}`}>
                <div className="faq-answer-text">
                  <p>{qa.a}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 5. Footer */}
      <footer className="console-footer">
        <div>ARGUS • Unsupervised Anomaly Detection Platform • Next.js + FastAPI + Modal serverless GPU</div>
        <small className="footer-meta">Designed by Systems and Vision Engineers. Frozen VideoMAE-v2 backbone, MULDE scorers, 1-component GMM score calibration.</small>
      </footer>

      {/* Floating Navigation Dock */}
      <div className="floating-nav-dock">
        <button
          className="dock-btn"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
          title="Scroll to Top"
        >
          ▲
        </button>
        <button
          className="dock-btn"
          onClick={() => document.getElementById("pipeline-section")?.scrollIntoView({ behavior: "smooth" })}
          title="Pipeline Map"
        >
          ⚙
        </button>
        <button
          className="dock-btn"
          onClick={() => document.getElementById("dashboard-section")?.scrollIntoView({ behavior: "smooth" })}
          title="Operational HUD"
        >
          📊
        </button>
        <button
          className="dock-btn"
          onClick={() => document.getElementById("faq-section")?.scrollIntoView({ behavior: "smooth" })}
          title="Engineering Q&A"
        >
          ❓
        </button>
      </div>
    </main>
  );
}
