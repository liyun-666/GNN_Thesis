import os
import sys
import time
import webbrowser
import subprocess

BASE = r"D:\GNN_Thesis"
PY = sys.executable

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008
FLAGS = CREATE_NO_WINDOW | DETACHED_PROCESS


def spawn(args, cwd):
    return subprocess.Popen(
        args,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=FLAGS,
        close_fds=True,
    )


def main():
    spawn([PY, "-m", "uvicorn", "api_server:app", "--host", "127.0.0.1", "--port", "8000", "--app-dir", BASE], BASE)
    spawn([PY, "-m", "http.server", "5173"], os.path.join(BASE, "mobile_app_pwa"))
    time.sleep(2.5)
    webbrowser.open("http://127.0.0.1:5173")


if __name__ == "__main__":
    main()
