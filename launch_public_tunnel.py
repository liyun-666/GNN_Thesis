import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path

BASE = Path(r"D:\GNN_Thesis")
LOG_DIR = BASE / "artifacts" / "tunnel_logs"
PY = sys.executable

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
FLAGS = CREATE_NO_WINDOW | DETACHED_PROCESS

URL_RE = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")


def _can_run(exe: str) -> bool:
    try:
        r = subprocess.run(
            [exe, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=6,
        )
        return r.returncode == 0
    except Exception:
        return False


def find_cloudflared() -> str | None:
    cand = []
    hardcoded = r"C:\Users\ASUS\AppData\Local\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe"
    cand.append(hardcoded)
    env_path = os.environ.get("CLOUDFLARED_PATH")
    if env_path:
        cand.append(env_path)

    w = shutil.which("cloudflared")
    if w:
        cand.append(w)

    local = Path(os.environ.get("LOCALAPPDATA", ""))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))

    cand.extend(
        [
            str(program_files / "Cloudflare" / "cloudflared" / "cloudflared.exe"),
            str(program_files / "cloudflared" / "cloudflared.exe"),
            str(local / "cloudflared" / "cloudflared.exe"),
            str(local / "Programs" / "cloudflared" / "cloudflared.exe"),
            str(local / "Microsoft" / "WinGet" / "Links" / "cloudflared.exe"),
        ]
    )

    for p in cand:
        if p and _can_run(p):
            return p

    pkg_root = local / "Microsoft" / "WinGet" / "Packages"
    try:
        if pkg_root.exists():
            for p in pkg_root.rglob("cloudflared*.exe"):
                if _can_run(str(p)):
                    return str(p)
    except Exception:
        pass

    # Fallback: return hardcoded path for local Windows setup.
    # In some constrained shells path probing may fail even though execution works in user session.
    return hardcoded

    # return None


def spawn_hidden(args, cwd, stdout_path=None):
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    if stdout_path:
        f = open(stdout_path, "a", encoding="utf-8", errors="ignore")
        stdout = f
        stderr = f
    else:
        f = None

    p = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        creationflags=FLAGS,
        close_fds=True,
    )
    if f:
        f.close()
    return p


def read_url(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = URL_RE.findall(text)
    return m[-1] if m else ""


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "app_tunnel.log").write_text("", encoding="utf-8")

    cloudflared = find_cloudflared()
    if not cloudflared:
        print("[ERROR] cloudflared not found. Please set CLOUDFLARED_PATH or install cloudflared.")
        return

    # Local service: API + static web served under /web
    spawn_hidden([PY, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000", "--app-dir", str(BASE)], BASE)

    # Single tunnel
    spawn_hidden([cloudflared, "tunnel", "--url", "http://127.0.0.1:8000"], BASE, LOG_DIR / "app_tunnel.log")

    print("[INFO] Starting local service and public tunnel...")
    public_url = ""
    for _ in range(30):
        time.sleep(1)
        public_url = read_url(LOG_DIR / "app_tunnel.log") or public_url
        if public_url:
            break

    print("\n========== PUBLIC URLS ==========")
    print(f"PUBLIC_BASE: {public_url or '(pending)'}")
    if public_url:
        print(f"APP_URL: {public_url}/web/index.html")
        print(f"API_HEALTH: {public_url}/health")
    print("=================================")
    print(f"Logs: {LOG_DIR}")
    print("\nShare APP_URL directly. No extra API URL setup is required.")


if __name__ == "__main__":
    main()
