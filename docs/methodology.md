# Methodology

## Learning Setting

Stream A is unsupervised, one-class anomaly detection. Training uses normal videos only. Ground-truth frame labels are not optimization targets; they are used for validation and final reporting.

## Feature Representation

VideoMAE-v2 Base is frozen and used as a temporal feature extractor. This keeps the representation stable and reduces the training cost to fitting the anomaly scorer.

## Density Scoring

The Avenue recipe uses log-density scoring with a Gaussian mixture calibration surface. Anomalies are observations that receive low probability under the learned normal distribution.

## Targeted Improvements

- Added a complete frame-centric Avenue data and evaluation path.
- Used a normal-only holdout for checkpoint selection.
- Moved from the naive scoring surface to `log_density + GMM`.
- Tuned `beta=0.1`, learning rate `4e-5`, and temporal smoothing.
- Added numerical fallbacks for stable GMM fitting and portable checkpoint loading.

These changes were selected from failure analysis rather than an unrestricted test-set search.
