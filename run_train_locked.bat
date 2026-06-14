@echo off
setlocal
cd /d "%~dp0"

echo Training standalone Stream A locked recipe...
python scripts\train.py --dataset stream_a_locked --output-dir outputs\stream_a_locked_repro

echo Ranking checkpoints...
python scripts\select_stream_a_checkpoint.py ^
  --dataset stream_a_locked ^
  --checkpoint-dir outputs\stream_a_locked_repro\checkpoints\stream_a ^
  --promote-best ^
  --output-json outputs\reports\stream_a_locked_repro_rank_val.json ^
  --output-csv outputs\reports\stream_a_locked_repro_rank_val.csv

echo Evaluating promoted best_frame checkpoint on test...
python scripts\eval_frame_level.py ^
  --dataset stream_a_locked ^
  --checkpoint outputs\stream_a_locked_repro\checkpoints\stream_a\best_frame.pt ^
  --split test ^
  --output-json outputs\reports\stream_a_locked_repro_test.json

endlocal
