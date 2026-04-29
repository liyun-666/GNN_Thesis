Set sh = CreateObject("WScript.Shell")
sh.Run "cmd /c cd /d D:\GNN_Thesis && python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --app-dir D:\GNN_Thesis", 0, False
sh.Run "cmd /c cd /d D:\GNN_Thesis\mobile_app_pwa && python -m http.server 5173", 0, False
WScript.Sleep 2500
sh.Run "http://127.0.0.1:5173", 1, False
