@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set GRADIO_ANALYTICS_ENABLED=False

echo Starting ARGUS Stream A demo...
python demo.py

endlocal
