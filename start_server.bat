@echo off
rem GAI NPC server — uses the CUDA venv on D: (system Python has no torch anymore)
cd /d "%~dp0"
D:\venvs\gai\Scripts\python.exe -m uvicorn backend.server:app --host 127.0.0.1 --port 8000
pause
