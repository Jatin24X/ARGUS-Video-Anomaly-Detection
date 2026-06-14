@echo off
setlocal

cd /d "%~dp0.."
python deployment\app.py
