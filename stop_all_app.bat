@echo off
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do taskkill /PID %%a /F >nul 2>nul
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173') do taskkill /PID %%a /F >nul 2>nul
taskkill /IM cloudflared.exe /F >nul 2>nul
echo ST-GNN app services stopped.
timeout /t 1 > nul
