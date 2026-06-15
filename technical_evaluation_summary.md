# Stream A Lab Evaluation Summary

## Project In One Line

We built a standalone **ARGUS Stream A** video anomaly detection pipeline using:

- backbone: `VideoMAE-v2 Base`
- scorer: `MULDE`
- setting: **unsupervised / one-class anomaly detection**

Training uses only normal videos. Frame labels are used only for validation and benchmarking.

## The Gap We Solved

The MULDE paper's famous Avenue number, `94.3 / 96.1`, is from the **object-centric** setup.

Our Stream A package is a **frame-centric** pipeline. The paper does not report a directly matching standalone Avenue frame-centric path in the same style as this package.

So the gap we solved is:

- bring up MULDE-style **frame-centric Avenue** end-to-end
- make it benchmarkable with real labels
- identify why the naive Avenue setup underperformed
- improve it with targeted ML engineering

## What We Implemented

We added full Avenue support to the standalone Stream A package:

- imported and validated real Avenue frame labels
- extracted Avenue VideoMAE features
- implemented normal-only holdout checkpoint selection for Avenue
- stabilized GMM fitting for Avenue scoring
- tuned the Avenue Stream A recipe for the correct scoring surface

## Main Results

### Avenue Bring-Up Baseline

From:

- `outputs/reports/avenue_stream_a_run1_test.json`

Metrics:

- micro AUC: `0.7738`
- macro AUC: `0.7724`
- clip AUC: `0.7728`

### Main Reported Result

From:

- `outputs/reports/avenue_stream_a_best_test.json`

Checkpoint:

- `outputs/avenue_stream_a_ld_gmm1_beta01_lr4e5_run1/checkpoints/stream_a/best_holdout.pt`

Metrics:

- micro AUC: `0.8451`
- macro AUC: `0.8514`
- clip AUC: `0.8400`

This is the **benchmark-safe main result** we should report tomorrow.

### Best Observed Micro Diagnostic

From:

- `outputs/reports/avenue_stream_a_best_test_high_micro_diag.json`

Eval-only diagnostic surface:

- `signal_kind = log_density`
- `sigma_strategy = gmm`
- `gmm_components = 2`
- `smoothing_sigma = 12`

Metrics:

- micro AUC: `0.8466`
- macro AUC: `0.8514`
- clip AUC: `0.8411`

Important:

- this is a **post-hoc test-set eval refinement**
- it is fine to mention as "best observed micro"
- it should **not** replace the benchmark-safe main reported result

## What Changed To Improve Avenue

The biggest gains did not come from random tuning. They came from targeted fixes:

- moved Avenue to the correct MULDE scoring surface:
  - `log_density + GMM`
- tuned the regularization and optimization recipe:
  - `beta = 0.1`
  - `learning_rate = 4e-5`
- tuned Avenue smoothing
- fixed numerical instability in GMM fitting
- implemented holdout-based checkpoint selection for a normal-only Avenue validation setup

## Fair Comparison Context

We should **not** compare our frame-centric result directly to MULDE's `94.3`, because that result is object-centric.

The fair comparison is:

1. our baseline vs our improved result
2. our result vs other **frame-level, non-object-centric** Avenue methods

### Internal Comparison

| Setting | Micro AUC | Macro AUC | Clip AUC |
|---|---:|---:|---:|
| Avenue bring-up baseline | 0.7738 | 0.7724 | 0.7728 |
| Main reported result | 0.8451 | 0.8514 | 0.8400 |
| Best observed micro diagnostic | 0.8466 | 0.8514 | 0.8411 |

### Frame-Level Avenue Context

These are useful frame-level Avenue reference points for context:

| Method | Setup note | Frame-level AUC |
|---|---|---:|
| AMFCFBMem-Net | frame-level | 88.8 |
| SSMTL frame-level branch | frame-level only | 86.9 |
| STPR-net | frame-level | 86.5 |
| integrated appearance + motion prediction | frame-level | 86.4 |
| AnomalyNet | frame-level | 86.1 |
| **Our Stream A** | **frame-centric** | **84.5** |

Use this table only as **context**, not as a strict apples-to-apples leaderboard.

### Non-Comparable Upper Bound

| Method | Setup | Avenue AUC |
|---|---|---:|
| MULDE paper | object-centric | 94.3 |

This should be described as:

- a different setup
- a future-work ceiling
- not a direct benchmark claim for our current Stream A path

## Exact Contribution Statement

Best short version:

> We implemented the missing frame-centric Avenue path for a standalone MULDE-style pipeline and improved it from `0.7738` to `0.8451` frame-level micro AUC using targeted scoring, selection, and numerical-stability fixes.

Short viva version:

> The paper's Avenue headline is object-centric. Our contribution was to build and improve a frame-centric Avenue version of the method, then benchmark it honestly.

## What To Show Tomorrow

1. `run_demo.bat`
2. `lab_evaluation_summary.md`
3. `outputs/reports/avenue_stream_a_best_test.json`
4. `presentation_script.md`
5. `viva_qa.md`

## Safe Reporting Rules

- Lead with the **main reported result**: `0.8451 / 0.8514 / 0.8400`
- Mention the `0.8466` result only as a small eval-only diagnostic improvement
- Do **not** claim direct comparability with MULDE's `94.3`
- Do **not** mix object-centric and frame-centric numbers in one leaderboard without labeling them
