@echo off
cd /d D:\GNN_Thesis
call stop_all_app.bat
python launch_public_tunnel.py
pause
