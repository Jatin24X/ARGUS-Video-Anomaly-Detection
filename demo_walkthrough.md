# Demo Walkthrough

## 30-Second Version

ARGUS Stream A is a standalone frame-level video anomaly detection system. It extracts frozen VideoMAE-v2 clip embeddings, scores low-likelihood temporal behavior with a MULDE-style density model, and displays the result as an anomaly timeline with top-frame evidence.

The main Avenue result is `0.8451` frame micro AUC and `0.8514` frame macro AUC. The Avenue path is reported as a full-frame, frame-centric setup, while object-centric Avenue numbers are treated as a separate protocol.

## 1-Minute Version

This system takes a video, samples frames, extracts temporal clip embeddings with VideoMAE-v2, and fits a normal-only density scorer. During inference, clips that are unlikely under the normal-video distribution receive higher anomaly scores. Those scores are smoothed and projected back to a frame-level timeline.

The deployed app supports two saved profiles: Avenue as the primary frame-level benchmark profile and UBnormal as a reference profile. The interface can run prepared sample videos or user uploads, then returns the anomaly curve, peak timestamp, runtime metadata, and highest-scoring frames.

## 2-Minute Version

The Stream A package is organized as a complete ML system: data metadata, feature extraction, scorer training, checkpoint selection, frame-level evaluation, API deployment, and frontend visualization.

The Avenue path includes real frame-label metadata, normal-only holdout selection, log-density plus GMM scoring, temporal smoothing, and saved benchmark reports. The primary Avenue test report is:

```text
outputs/reports/avenue_stream_a_best_test.json
```

The live deployment uses Vercel for the frontend and Modal for the FastAPI GPU backend. Modal runs on a T4 with cached model artifacts and `min_containers=0`, which keeps idle cost low while allowing the first request after idle to take longer.

## Protocol Note

MULDE's Avenue headline is object-centric. ARGUS Stream A reports a full-frame, frame-centric path. Those protocols use different input representations and should not be collapsed into one leaderboard number.

## If Asked About The Strongest Engineering Choice

The strongest engineering choice is the strict evaluation contract: normal-only fitting, fixed feature backbone, saved reports, clear full-frame protocol, and separate handling of object-centric numbers.
