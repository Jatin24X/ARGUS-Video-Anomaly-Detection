"use client";

import { ChangeEvent, useEffect, useMemo, useState } from "react";

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

const API_BASE = (process.env.NEXT_PUBLIC_ARGUS_API_URL ?? "").replace(/\/$/, "");
const REPO_URL = "https://github.com/Jatin24X/ARGUS---Video-Anomaly-Detection";
const ALLOWED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"];

const projectStats = [
  ["84.51%", "Avenue micro AUC"],
  ["85.14%", "Avenue macro AUC"],
  ["T4", "live GPU inference"],
  ["7", "prepared video samples"],
];

const engineeringHighlights = [
  "Normal-only training: labels are reserved for validation and final benchmark reporting.",
  "Frozen VideoMAE-v2 embeddings provide stable temporal features without backbone fine-tuning.",
  "MULDE-style density scoring ranks low-likelihood clips as anomalous frame evidence.",
  "FastAPI, Modal, and Vercel serve the same scorer through sample and upload workflows.",
];

const pipelineSteps = [
  "Decode video",
  "Adaptive frame sampling",
  "VideoMAE clip embeddings",
  "MULDE density scoring",
  "Timeline and evidence",
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
  accent,
}: {
  timeline: AnalysisResponse["analysis"]["timeline"] | null;
  accent: string;
}) {
  const width = 920;
  const height = 300;
  const padding = { top: 18, right: 20, bottom: 42, left: 52 };

  const chart = useMemo(() => {
    if (!timeline?.timestamps_sec.length || !timeline.scores.length) return null;
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxX = Math.max(...timeline.timestamps_sec, 1);
    const maxY = Math.max(...timeline.scores, timeline.threshold, 1);
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
    return <div className="empty-state">Choose a sample or upload a clip to generate its anomaly timeline.</div>;
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="timeline-svg" role="img" aria-label="Frame anomaly timeline">
      <defs>
        <linearGradient id="timeline-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={accent} stopOpacity="0.34" />
          <stop offset="100%" stopColor={accent} stopOpacity="0.03" />
        </linearGradient>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
        <g key={tick}>
          <line x1={padding.left} x2={width - padding.right} y1={chart.y(tick)} y2={chart.y(tick)} className="grid-line" />
          <text x={padding.left - 10} y={chart.y(tick) + 4} className="axis-text">{tick.toFixed(2)}</text>
        </g>
      ))}
      {Array.from({ length: 6 }, (_, index) => (chart.maxX / 5) * index).map((tick) => (
        <text key={tick} x={chart.x(tick)} y={height - 14} className="axis-text axis-x">{tick.toFixed(1)}s</text>
      ))}
      {timeline.anomaly_regions.map((region, index) => (
        <rect
          key={`${region.start_time_sec}-${index}`}
          x={chart.x(region.start_time_sec)}
          y={padding.top}
          width={Math.max(5, chart.x(region.end_time_sec) - chart.x(region.start_time_sec))}
          height={height - padding.top - padding.bottom}
          className="timeline-region"
        />
      ))}
      <line x1={padding.left} x2={width - padding.right} y1={chart.y(timeline.threshold)} y2={chart.y(timeline.threshold)} className="threshold-line" />
      <text x={padding.left + 8} y={chart.y(timeline.threshold) - 8} className="threshold-text">{timeline.threshold_label}</text>
      <path d={chart.areaPath} fill="url(#timeline-fill)" />
      <path d={chart.linePath} fill="none" stroke={accent} strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

export default function Page() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [selectedKey, setSelectedKey] = useState("");
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [mode, setMode] = useState<"samples" | "upload">("samples");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingSeconds, setLoadingSeconds] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!API_BASE) {
      setError("The API URL is not configured for this deployment.");
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
        setSamples(samplePayload.samples ?? []);
        setHealth(healthPayload);
        if (profilePayload.profiles?.length) setSelectedKey(profilePayload.profiles[0].key);
      })
      .catch((cause: Error) => {
        if (cause.name !== "AbortError") setError("The GPU service is waking up. Please retry in a moment.");
      });
    return () => controller.abort();
  }, []);

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

  const progressCopy = loadingSeconds < 8
    ? "Connecting to the GPU service"
    : loadingSeconds < 25
      ? "Loading the model and extracting video features"
      : "Scoring frames and preparing visual evidence";

  function selectSample(sample: Sample) {
    setMode("samples");
    setSelectedSample(sample);
    setVideoFile(null);
    setPreviewUrl(absoluteApiUrl(sample.video_url));
    setAnalysis(null);
    setError("");
    const profile = profiles.find((item) => item.dataset_name === sample.profile);
    if (profile) setSelectedKey(profile.key);
  }

  function onVideoChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) return;
    const extension = `.${file.name.split(".").pop()?.toLowerCase()}`;
    const limitMb = health?.max_upload_mb ?? 50;
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setError(`Unsupported file type. Use ${ALLOWED_EXTENSIONS.join(", ")}.`);
      return;
    }
    if (file.size > limitMb * 1024 * 1024) {
      setError(`The upload limit is ${limitMb} MB.`);
      return;
    }
    setMode("upload");
    setSelectedSample(null);
    setVideoFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysis(null);
    setError("");
  }

  async function runAnalysis() {
    if (!selectedProfile || (!selectedSample && !videoFile)) return;
    setLoading(true);
    setError("");
    setAnalysis(null);
    try {
      let response: Response;
      if (selectedSample) {
        const query = new URLSearchParams({ profile: selectedProfile.label });
        response = await fetch(`${API_BASE}/samples/${selectedSample.id}/analyze?${query}`, { method: "POST" });
      } else {
        const body = new FormData();
        body.append("profile", selectedProfile.label);
        body.append("video", videoFile!);
        response = await fetch(`${API_BASE}/analyze`, { method: "POST", body });
      }
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `Analysis failed (${response.status})`);
      setAnalysis(payload as AnalysisResponse);
      document.getElementById("results")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Analysis failed. Please retry.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="ARGUS home">
          <span className="brand-mark">A</span>
          <span>
            <strong>ARGUS Stream A</strong>
            <small>Video anomaly detection</small>
          </span>
        </a>
        <nav className="topnav" aria-label="Project sections">
          <a href="#demo">Demo</a>
          <a href="#engineering">Engineering</a>
          <a href="#results">Results</a>
          <a href={REPO_URL} target="_blank" rel="noreferrer">GitHub</a>
        </nav>
        <div className={`service-status ${health?.status === "ready" ? "ready" : ""}`}>
          <i /> {health?.status === "ready" ? `GPU ready - ${health.device}` : "GPU wakes on demand"}
        </div>
      </header>

      <section id="top" className="hero">
        <div className="hero-copy">
          <div className="eyebrow">Video anomaly intelligence</div>
          <h1>Find abnormal moments in video with a deployed density pipeline.</h1>
          <p>
            ARGUS Stream A ranks frame-level anomaly evidence from surveillance-style video. The system
            combines frozen VideoMAE features, MULDE-style density scoring, calibrated Avenue evaluation,
            and live GPU inference behind a polished web interface.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#demo">Run live demo</a>
            <a className="button secondary" href={REPO_URL} target="_blank" rel="noreferrer">View GitHub</a>
          </div>
        </div>
        <div className="hero-panel">
          <div className="panel-label">Avenue benchmark path</div>
          <h2>84.51% frame micro AUC on Avenue.</h2>
          <p>
            The Avenue path includes metadata validation, feature extraction, normal-only model selection,
            GMM score calibration, temporal smoothing, and benchmark-safe reporting.
          </p>
        </div>
      </section>

      <section className="stat-grid" aria-label="Project metrics">
        {projectStats.map(([value, label]) => (
          <div className="stat-card" key={label}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </section>

      <section className="story-grid">
        <article className="story-card">
          <span className="section-tag">Problem</span>
          <h2>Detect abnormal moments without learning anomaly classes.</h2>
          <p>
            Stream A is one-class: only normal videos train the density scorer. Frame labels are kept for
            evaluation, which makes the setup closer to real surveillance deployment constraints.
          </p>
        </article>
        <article className="story-card accent-card">
          <span className="section-tag">Fair comparison</span>
          <h2>Clear protocol boundary between full-frame and object-centric evaluation.</h2>
          <p>
            MULDE's Avenue headline is object-centric. ARGUS Stream A reports the full-frame path separately,
            so the evaluation surface stays precise and comparable within its own protocol.
          </p>
        </article>
      </section>

      <section className="pipeline">
        <div className="section-heading wide">
          <div>
            <span>System pipeline</span>
            <h2>From raw video to ranked evidence</h2>
          </div>
        </div>
        <div className="pipeline-steps">
          {pipelineSteps.map((step, index) => (
            <div className="pipeline-step" key={step}>
              <small>0{index + 1}</small>
              <strong>{step}</strong>
            </div>
          ))}
        </div>
      </section>

      <section id="demo" className="demo-header">
        <div>
          <span className="section-tag">Live inference</span>
          <h2>Run a prepared sample or upload a short clip.</h2>
        </div>
        <p>
          The sample gallery runs directly on server-side test videos. Uploaded clips are validated,
          analyzed on the GPU service, and converted into a timeline with visual evidence.
        </p>
      </section>

      <nav className="mode-tabs" aria-label="Video source">
        <button className={mode === "samples" ? "active" : ""} onClick={() => setMode("samples")}>Sample gallery</button>
        <button className={mode === "upload" ? "active" : ""} onClick={() => setMode("upload")}>Upload video</button>
      </nav>

      <section className="workspace">
        <div className="source-panel">
          {mode === "samples" ? (
            <>
              <div className="section-heading">
                <div><span>Prepared examples</span><h2>Choose a test video</h2></div>
                <small>{samples.length || health?.sample_count || 0} videos</small>
              </div>
              <div className="sample-grid">
                {samples.map((sample) => (
                  <button
                    key={sample.id}
                    className={`sample-card ${selectedSample?.id === sample.id ? "selected" : ""}`}
                    onClick={() => selectSample(sample)}
                  >
                    <img src={absoluteApiUrl(sample.thumbnail_url)} alt="" loading="lazy" />
                    <span className="sample-overlay"><b>{sample.title}</b><small>{sample.dataset} - {sample.size_mb} MB</small></span>
                  </button>
                ))}
              </div>
              {!samples.length && <div className="loading-box">Loading sample gallery from the Modal GPU service...</div>}
            </>
          ) : (
            <label className="upload-zone" htmlFor="video-upload">
              <input id="video-upload" type="file" accept="video/*" onChange={onVideoChange} />
              <span className="upload-icon">UP</span>
              <strong>Drop a short video here</strong>
              <small>MP4, AVI, MOV, MKV or WebM - up to {health?.max_upload_mb ?? 50} MB</small>
            </label>
          )}
        </div>

        <aside className="control-panel">
          <div className="preview">
            {previewUrl ? <video src={previewUrl} controls playsInline /> : <div>Select a sample to preview</div>}
          </div>
          <label>Analysis profile</label>
          <div className="profile-switch">
            {profiles.map((profile) => (
              <button key={profile.key} className={selectedKey === profile.key ? "active" : ""} onClick={() => setSelectedKey(profile.key)}>
                {profile.dataset_name}
              </button>
            ))}
          </div>
          {selectedProfile && (
            <div className="profile-metrics">
              <div><span>Saved micro AUC</span><strong>{selectedProfile.benchmark_micro_auc_pct}</strong></div>
              <div><span>Saved macro AUC</span><strong>{selectedProfile.benchmark_macro_auc_pct}</strong></div>
            </div>
          )}
          <button className="run-button" disabled={loading || !selectedProfile || (!selectedSample && !videoFile)} onClick={runAnalysis}>
            <span>{loading ? progressCopy : "Analyze video"}</span>
          </button>
          {loading && <div className="progress"><i /><span>{loadingSeconds}s elapsed. A cold T4 can take about a minute.</span></div>}
          {error && <div className="error-banner">{error}</div>}
        </aside>
      </section>

      <section id="results" className="results">
        <div className="result-heading">
          <div><span>Model output</span><h2>Anomaly timeline</h2></div>
          {analysis && <small>{analysis.analysis.video_name} - {sec(analysis.analysis.runtime_sec)}</small>}
        </div>
        <TimelineChart timeline={analysis?.analysis.timeline ?? null} accent={analysis?.profile.accent ?? "#23b5d3"} />

        {analysis && (
          <>
            <div className="summary-strip">
              <div><span>Peak anomaly</span><strong>{sec(analysis.analysis.summary.peak_time_sec)}</strong></div>
              <div><span>Peak score</span><strong>{analysis.analysis.summary.peak_score.toFixed(3)}</strong></div>
              <div><span>Frames sampled</span><strong>{analysis.analysis.summary.sampled_frame_count} / {analysis.analysis.summary.raw_frame_count}</strong></div>
              <div><span>Clip embeddings</span><strong>{analysis.analysis.summary.clip_count}</strong></div>
            </div>
            <div className="evidence-heading"><span>Visual evidence</span><h2>Highest-scoring frames</h2></div>
            <div className="frame-grid">
              {analysis.analysis.frames.map((frame) => (
                <figure key={`${frame.index}-${frame.timestamp_sec}`}>
                  <img src={frame.image_data_url} alt={frame.caption} />
                  <figcaption>{frame.caption}</figcaption>
                </figure>
              ))}
            </div>
          </>
        )}
      </section>

      <section id="engineering" className="engineering">
        <div className="section-heading wide">
          <div>
            <span>Engineering surface</span>
            <h2>Model, API, deployment, and evidence UI in one system.</h2>
          </div>
        </div>
        <div className="highlight-grid">
          {engineeringHighlights.map((item) => (
            <article className="highlight-card" key={item}>
              <div className="dot" />
              <p>{item}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="repo-banner">
        <div>
          <span className="section-tag">Repository</span>
          <h2>Source, configs, reports, deployment scripts, tests, and selected artifacts are organized for reproduction.</h2>
        </div>
        <a className="button primary" href={REPO_URL} target="_blank" rel="noreferrer">Open GitHub repository</a>
      </section>

      <footer>ARGUS Stream A - Unsupervised frame-level anomaly detection - Vercel + Modal</footer>
    </main>
  );
}
