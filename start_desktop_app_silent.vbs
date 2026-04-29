Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "D:\GNN_Thesis"
WshShell.Run "pythonw ""D:\GNN_Thesis\desktop_app_v2.py""", 0, False
