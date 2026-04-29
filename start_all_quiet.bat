@echo off
cd /d D:\GNN_Thesis
powershell -WindowStyle Hidden -Command "Start-Process python -ArgumentList '-m','uvicorn','api_server:app','--host','127.0.0.1','--port','8000','--app-dir','D:\\GNN_Thesis' -WindowStyle Hidden"
powershell -WindowStyle Hidden -Command "Start-Process python -ArgumentList '-m','http.server','5173' -WorkingDirectory 'D:\\GNN_Thesis\\mobile_app_pwa' -WindowStyle Hidden"
timeout /t 2 > nul
start "" http://127.0.0.1:5173
