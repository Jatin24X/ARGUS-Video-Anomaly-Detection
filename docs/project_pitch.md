# Project Pitch

## Thirty Seconds

ARGUS Stream A is an unsupervised video anomaly detection system that learns normal temporal behavior and highlights unusual moments without requiring anomaly examples during training. I built the complete Avenue frame-centric path, improved frame micro AUC from 77.38% to 84.51%, and deployed GPU inference using a Vercel frontend and a scale-to-zero Modal FastAPI backend.

## Resume Bullets

- Built and deployed an unsupervised video anomaly detection system using frozen VideoMAE features and MULDE density scoring, achieving **84.51% frame micro AUC** and **85.14% macro AUC** on Avenue.
- Improved the Avenue frame-centric baseline by **7.13 AUC points** through scoring-surface correction, normal-only checkpoint selection, temporal calibration, and numerically stable GMM fitting.
- Shipped a production-style Vercel and Modal application with T4 inference, persistent model caching, bounded uploads, sample-video analysis, anomaly timelines, and frame-level visual evidence.

## Interview Questions

### Why is the system unsupervised?

The scorer is fitted only on normal training videos. Frame labels are held out for validation and test metrics, so the model does not learn anomaly classes directly.

### Why use a frozen backbone?

It lowers compute cost, reduces overfitting on a small surveillance dataset, and isolates whether density modeling can separate abnormal temporal representations.

### Why did macro and micro AUC differ?

Micro AUC is dominated by videos with more frames. Macro AUC gives each valid video equal weight, so difficult short scenes have more influence.

### What was the main engineering lesson?

The largest improvement came from fixing the statistical scoring and model-selection contract, not from blindly increasing model size.

### Why Modal and Vercel?

Vercel gives a fast, cacheable frontend while Modal provides an on-demand GPU Python runtime. Scaling the GPU to zero is appropriate for a portfolio demo with bursty traffic.

### What would you improve next?

I would add a region-aware branch for small anomalies, run a strict ablation against the frozen frame-centric baseline, and add calibration metrics for deployment thresholds.
