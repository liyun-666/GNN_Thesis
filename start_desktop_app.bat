@echo off
chcp 65001 > nul
cd /d D:\GNN_Thesis
where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw "D:\GNN_Thesis\desktop_app_v2.py"
) else (
  start "" python "D:\GNN_Thesis\desktop_app_v2.py"
)
