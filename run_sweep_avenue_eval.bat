@echo off
setlocal
cd /d "%~dp0"

echo Running holdout-aware Avenue evaluation sweep...
python scripts\sweep_stream_a_eval.py ^
  --dataset avenue_stream_a ^
  --checkpoint outputs\avenue_stream_a_run1\checkpoints\stream_a\best_holdout.pt ^
  --output-json outputs\reports\avenue_stream_a_run1_sweep_v2.json ^
  --output-csv outputs\reports\avenue_stream_a_run1_sweep_v2.csv ^
  --run-test-on-best

endlocal
