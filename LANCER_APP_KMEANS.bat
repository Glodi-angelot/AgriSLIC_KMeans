@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Environnement .venv introuvable.
    pause
    exit /b 1
)

.venv\Scripts\python -m streamlit run app.py
