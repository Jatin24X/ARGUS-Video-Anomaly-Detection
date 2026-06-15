# Stream A Standalone Context

This folder is a self-contained **ARGUS Stream A** package focused on frame-level video anomaly detection, benchmark evaluation, and deployment.

## Scope

- Included:
  - VideoMAE feature extraction
  - MULDE training
  - frame-level evaluation
  - checkpoint ranking
  - evaluation sweep
  - interactive demo
- Excluded:
  - Stream B
  - Stream C
  - fusion
  - active benchmark-max research code outside Stream A

## Official Frozen Recipe

- Dataset: `ubnormal`
- Backbone: `VideoMAE-v2 Base`
- Scorer: `MULDE`
- Training: `beta=1.0`, `EMA=false`
- Eval surface:
  - `signal_kind=score_norm`
  - `sigma_strategy=single_sigma`
  - `single_sigma_index=0`
  - `smoothing_sigma=20`

## Primary Entry Points

- Demo:
  - `python demo.py`
  - or `run_demo.bat`
  - includes:
    - `Avenue (main reported frame-centric result)`
    - `UBnormal (locked frozen baseline)`
- Frozen benchmark eval:
  - `python scripts/eval_frame_level.py --dataset stream_a_locked --checkpoint outputs/checkpoints/stream_a_locked_videomae_beta1_score_norm_sigma0.pt --split test`
  - or `run_eval_frozen.bat`
- Retrain locked recipe:
  - `run_train_locked.bat`

## Avenue Status

This standalone folder now supports Avenue end to end:

- metadata loading is dataset-name based
- feature lookup supports either scene folders or flat video feature files
- `configs/avenue_stream_a.yaml` is the active Avenue config
- `outputs/reports/avenue_stream_a_best_test.json` is the benchmark-safe main Avenue report
- `outputs/reports/avenue_stream_a_best_test_high_micro_diag.json` is the small post-hoc diagnostic high-micro variant

## Notes

- Treat this folder as the standalone Stream A package.
- The parent ARGUS repo remains the full research workspace.
