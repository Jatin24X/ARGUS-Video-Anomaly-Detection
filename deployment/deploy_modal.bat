@echo off
setlocal

cd /d "%~dp0.."
modal deploy deployment\modal_app.py
