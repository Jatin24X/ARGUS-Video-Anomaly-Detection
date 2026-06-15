# Presentation Script

## 30-Second Version

We built a standalone ARGUS Stream A anomaly detection pipeline using VideoMAE and MULDE.  
The main research gap we solved is that the famous MULDE Avenue number is object-centric, while our system is frame-centric.  
So we implemented and improved a frame-centric Avenue path end-to-end.  
Our Avenue bring-up baseline was `0.7738` micro AUC, and after targeted fixes we reached `0.8451` micro AUC with `0.8514` macro AUC.

## 1-Minute Version

This standalone package isolates Stream A from ARGUS as an end-to-end video anomaly detection system.  
It uses VideoMAE-v2 Base to extract clip embeddings and MULDE to score how abnormal those embeddings are relative to normal training data.

The important point is that the MULDE paper's well-known Avenue score, `94.3`, is from an object-centric setup. Our Stream A package is frame-centric, so that number is not the right direct comparison.

Our contribution was to bring up the missing frame-centric Avenue path, make it benchmarkable with real labels, fix the validation and scoring pipeline, and improve the result from `0.7738` micro AUC to `0.8451` micro AUC. We also reached `0.8514` macro AUC.

## 2-Minute Version

We approached the project in two layers.

First, we built a clean standalone Stream A system so it can be trained, evaluated, and demonstrated independently from the rest of ARGUS. The pipeline is: raw video to VideoMAE embeddings, then MULDE anomaly scoring, then frame-level anomaly timelines and benchmark metrics.

Second, we focused on the Avenue dataset. The key subtlety is that the MULDE paper's headline Avenue score is object-centric, meaning it uses a different feature pipeline than ours. Since our standalone Stream A is frame-centric, the correct research question became: can we make a frame-centric Avenue version work well?

We implemented that missing Avenue path end-to-end. That included importing and validating real Avenue frame labels, building a proper normal-only holdout selection path, stabilizing GMM-based scoring, and tuning the Avenue-specific MULDE evaluation surface.

Our initial Avenue bring-up baseline was `0.7738` micro AUC. After these targeted fixes, the main reported result became `0.8451` micro AUC, `0.8514` macro AUC, and `0.8400` clip AUC.

So the contribution is not that we matched the object-centric paper number. The contribution is that we implemented and improved a clean frame-centric Avenue version of the method and made it competitive.

## Exact Gap Statement

> The paper highlights Avenue in an object-centric setup. We solved the missing practical gap of implementing and improving a frame-centric Avenue MULDE pipeline in a clean standalone system.

## If Asked Why Not Compare Directly To 94.3

> Because `94.3` is object-centric and ours is frame-centric. The detector inputs, feature types, and scoring granularity are different, so that is not an apples-to-apples comparison.

## If Asked What The Main Improvement Was

> The highest-impact change was moving Avenue to the correct MULDE scoring surface, specifically log-density scoring with GMM aggregation, then fixing checkpoint selection and numerical stability for that setup.
