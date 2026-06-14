@echo off
setlocal
cd /d "%~dp0"

echo Training standalone Stream A on Avenue...
python scripts\train.py --dataset avenue_stream_a --output-dir outputs\avenue_stream_a

echo Ranking checkpoints...
python scripts\select_stream_a_checkpoint.py ^
  --dataset avenue_stream_a ^
  --checkpoint-dir outputs\avenue_stream_a\checkpoints\stream_a ^
  --ranking-mode auto ^
  --output-json outputs\reports\avenue_stream_a_rank_val.json ^
  --output-csv outputs\reports\avenue_stream_a_rank_val.csv

echo Evaluating selected best_holdout checkpoint on Avenue test...
python scripts\eval_frame_level.py ^
  --dataset avenue_stream_a ^
  --checkpoint outputs\avenue_stream_a\checkpoints\stream_a\best_holdout.pt ^
  --split test ^
  --output-json outputs\reports\avenue_stream_a_test.json

endlocal
