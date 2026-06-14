# ARGUS Stream A Frozen Baseline

> Status: official current Stream A baseline as of April 1, 2026.
> This replaces earlier “current benchmark” summaries that still show `0.7110 / 0.8249`.

## Locked recipe

- Backbone: `VideoMAE`
- Training: `beta=1.0`, `EMA=false`
- Checkpoint selection: frame-level `val`
- Eval surface: `score_norm`, `single_sigma`, `single_sigma_index=0`, `smoothing_sigma=20`

## Official carried-forward result

- Test micro AUC: `0.7394`
- Test macro AUC: `0.8410`
- Clip AUC: `0.7309`

## Historical comparison points

- Post-Phase-1 baseline: `0.7110 / 0.8249`
- Default-surface `beta=1.0`: `0.7429 / 0.8377`
- Locked-surface frozen baseline: `0.7394 / 0.8410`
- `beta=1.0 + EMA` regression: `0.6915 / 0.8405`

## Reproduction runbook

Use the locked preset so the eval surface is not implicit.

```bash
python scripts/train.py --stream a --dataset stream_a_locked --output-dir outputs/stream_a_locked_repro
python scripts/select_stream_a_checkpoint.py --dataset stream_a_locked --checkpoint-dir outputs/stream_a_locked_repro/checkpoints/stream_a --promote-best --output-json outputs/reports/stream_a_locked_repro_rank_val.json --output-csv outputs/reports/stream_a_locked_repro_rank_val.csv
python scripts/eval_frame_level.py --dataset stream_a_locked --checkpoint outputs/stream_a_locked_repro/checkpoints/stream_a/best_frame.pt --split test --output-json outputs/reports/stream_a_locked_repro_test.json
```

## Freeze/archive command

After syncing the winning Stream A outputs locally from Lightning AI:

```bash
python scripts/freeze_stream_a_baseline.py
```

That copies the frozen checkpoint to:

`outputs/checkpoints/stream_a_locked_videomae_beta1_score_norm_sigma0.pt`

and writes a manifest plus a pointer JSON for future sessions.

## Protocol note

- The `test` split was used during Stream A development.
- From this freeze point onward, use `val` for selection.
- Use `test` only for a locked milestone report.
- This is the carried-forward Stream A baseline for progression and reproducibility, not a permanent closure.
- Stream A may be reopened later with explicit user approval, for paper-parity follow-up, or if downstream fusion underperforms.
