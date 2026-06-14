@echo off
setlocal
cd /d "%~dp0"

echo Evaluating current Avenue Stream A checkpoint...
python scripts\eval_frame_level.py ^
  --dataset avenue_stream_a ^
  --checkpoint outputs\avenue_stream_a_ld_gmm1_beta01_lr4e5_run1\checkpoints\stream_a\best_holdout.pt ^
  --split test ^
  --output-json outputs\reports\avenue_stream_a_best_test.json

endlocal
