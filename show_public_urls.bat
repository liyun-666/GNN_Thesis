@echo off
cd /d D:\GNN_Thesis
python - <<PY
from pathlib import Path
import re
base=Path(r"D:\\GNN_Thesis\\artifacts\\tunnel_logs")
pat=re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")
p=base/"app_tunnel.log"
t=p.read_text(encoding="utf-8",errors="ignore") if p.exists() else ""
m=pat.findall(t)
u=m[-1] if m else "(not found)"
print("PUBLIC_BASE:",u)
if u!="(not found)":
 print("APP_URL:",u+"/web/index.html")
 print("API_HEALTH:",u+"/health")
PY
pause
