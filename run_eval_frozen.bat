@echo off
setlocal
cd /d "%~dp0"

echo Evaluating frozen Stream A checkpoint on UBnormal test...
python scripts\eval_frame_level.py ^
  --dataset stream_a_locked ^
  --checkpoint outputs\checkpoints\stream_a_locked_videomae_beta1_score_norm_sigma0.pt ^
  --split test ^
  --output-json outputs\reports\standalone_stream_a_test.json

endlocal
