@echo off
setlocal

cd /d "%~dp0.."
modal run deployment\modal_app.py --prime-cache
