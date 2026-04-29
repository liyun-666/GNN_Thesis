@echo off
cd /d D:\GNN_Thesis
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --app-dir D:\GNN_Thesis
if errorlevel 1 (
  echo.
  echo API start failed. Press any key to close...
  pause >nul
)
