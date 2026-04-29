@echo off
cd /d D:\GNN_Thesis
python -m streamlit run app.py
if errorlevel 1 (
  echo.
  echo Streamlit start failed. Press any key to close...
  pause >nul
)
