@echo off
cd /d D:\GNN_Thesis
start "STGNN API" cmd /k "python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --app-dir D:\GNN_Thesis"
start "STGNN Mobile" cmd /k "cd /d D:\GNN_Thesis\mobile_app_pwa && python -m http.server 5173"
timeout /t 2 > nul
start "" http://127.0.0.1:5173
