import hashlib
import json
import os
import sqlite3
import shutil
import sys
import time
import traceback
import tempfile
import webbrowser
from pathlib import Path
from datetime import datetime
from tkinter import Tk, StringVar, IntVar, BooleanVar, messagebox, filedialog, colorchooser, Toplevel
from tkinter import ttk

import pandas as pd

from qa_tool import run_batch_diagnostics, diagnose_item_across_users, export_inspector_results
from recommender_engine import STGNNPipeline, TrainConfig


def app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resolve_resource_path(rel_path: str) -> str:
    rel = rel_path.replace("/", os.sep).replace("\\", os.sep)
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(getattr(sys, "_MEIPASS")))
    base = Path(app_base_dir())
    candidates.extend([base, base / "_internal", Path.cwd(), Path.cwd() / "_internal"])
    for c in candidates:
        p = c / rel
        if p.exists():
            return str(p)
    # fallback: keep previous behavior path for clearer error message
    return str(base / rel)


def _pick_writable_dir(candidates: list[str], leaf: str) -> str:
    for base in candidates:
        if not base:
            continue
        p = os.path.join(base, leaf)
        try:
            os.makedirs(p, exist_ok=True)
            probe = os.path.join(p, ".write_probe")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return p
        except Exception:
            continue
    # last resort in current working directory
    p = os.path.join(os.getcwd(), leaf)
    os.makedirs(p, exist_ok=True)
    return p


def app_user_data_dir() -> str:
    home = os.path.expanduser("~")
    candidates = [
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("APPDATA"),
        os.path.join(home, "Documents"),
        r"C:\Users\Public\Documents",
        tempfile.gettempdir(),
        home,
    ]
    return _pick_writable_dir(candidates, "STGNNDesktop")


BASE_DIR = app_base_dir()
USER_DATA_DIR = app_user_data_dir()
DEMO_DB_SRC = resolve_resource_path("rec_system.db")
DEMO_DB = os.path.join(USER_DATA_DIR, "rec_system_runtime.db")
DEMO_ARTIFACT = resolve_resource_path(os.path.join("artifacts", "stgnn_artifact_v2.pt"))
PAPER_MAP = resolve_resource_path("paper_alignment.json")
EXPORT_DIR = os.path.join(USER_DATA_DIR, "inspector_exports")
ACCOUNT_DB = os.path.join(USER_DATA_DIR, "app_accounts.db")
SETTINGS_JSON = os.path.join(USER_DATA_DIR, "ui_settings.json")
DEFAULT_MAX_HISTORY = 300
MAX_HISTORY_HARD_LIMIT = 2000
TOPK_HARD_LIMIT = 200
INSPECTOR_SAMPLE_HARD_LIMIT = 2000


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _fallback_user_dir() -> str:
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Documents"),
        r"C:\Users\Public\Documents",
        tempfile.gettempdir(),
        home,
    ]
    return _pick_writable_dir(candidates, "STGNNDesktop")


def _probe_sqlite_path(path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA user_version")
        conn.close()
        return True
    except Exception:
        return False


def init_account_db():
    global USER_DATA_DIR, ACCOUNT_DB, EXPORT_DIR, DEMO_DB
    if not _probe_sqlite_path(ACCOUNT_DB):
        fb = _fallback_user_dir()
        USER_DATA_DIR = fb
        ACCOUNT_DB = os.path.join(fb, "app_accounts.db")
        EXPORT_DIR = os.path.join(fb, "inspector_exports")
        DEMO_DB = os.path.join(fb, "rec_system_runtime.db")
    os.makedirs(os.path.dirname(ACCOUNT_DB), exist_ok=True)
    conn = sqlite3.connect(ACCOUNT_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              created_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


class UnifiedApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("ST-GNN Unified Desktop App")
        self.root.geometry("1280x800")

        self.user_var = StringVar(value="")
        self.pw_var = StringVar(value="")
        self.login_state = StringVar(value="Not logged in")
        self.logged_in = False

        self.demo_uid = IntVar(value=10009)
        self.demo_topk = IntVar(value=10)
        self.demo_include = BooleanVar(value=False)
        self.demo_item = IntVar(value=460466)
        self.demo_beh = StringVar(value="Buy")
        self.demo_max_hist = IntVar(value=DEFAULT_MAX_HISTORY)

        self.custom_db = StringVar(value="")
        self.custom_table = StringVar(value="user_behavior_logs")
        self.custom_uid = IntVar(value=1)
        self.custom_topk = IntVar(value=10)
        self.custom_include = BooleanVar(value=False)
        self.custom_item = IntVar(value=1)
        self.custom_beh = StringVar(value="Buy")
        self.custom_epochs = IntVar(value=2)
        self.custom_max_hist = IntVar(value=DEFAULT_MAX_HISTORY)
        self.custom_note = StringVar(value="No custom DB loaded.")

        self.view_ws = StringVar(value="Demo")
        self.view_uid = IntVar(value=10009)
        self.view_topn = IntVar(value=20)
        self.view_summary = StringVar(value="No user visualization yet.")

        self.ins_ws = StringVar(value="Demo")
        self.ins_sample = IntVar(value=30)
        self.ins_topk = IntVar(value=10)
        self.ins_item = IntVar(value=460466)
        self.ins_summary = StringVar(value="No diagnostics yet.")

        self.status = StringVar(value="Ready")
        self.profile_text = StringVar(value="User: Guest")
        self.ui_language = StringVar(value="Chinese")
        self.ui_bg = StringVar(value="#f5f7fb")
        self.ui_name = StringVar(value="Guest")
        self.ui_avatar = StringVar(value="User")

        self.demo_pipe = None
        self.demo_df = None
        self.custom_pipe = None
        self.custom_df = None
        self.batch_df = None
        self.batch_summary = None

        init_account_db()
        self._load_ui_settings()
        self._ensure_demo_runtime_db()
        self._build_ui()
        self._apply_ui_settings()
        try:
            self._load_demo()
        except Exception as e:
            self._set_status(f"Demo load failed: {e}")
            messagebox.showerror("Startup", f"Failed to load demo workspace:\n{e}")

    def _set_status(self, s: str):
        self.status.set(s)
        self.root.update_idletasks()

    def _require_login(self) -> bool:
        if self.logged_in:
            return True
        messagebox.showwarning("Login required", "Please login first.")
        return False

    def _load_ui_settings(self):
        if not os.path.exists(SETTINGS_JSON):
            return
        try:
            data = json.load(open(SETTINGS_JSON, "r", encoding="utf-8"))
            self.ui_language.set(str(data.get("language", self.ui_language.get())))
            self.ui_bg.set(str(data.get("background", self.ui_bg.get())))
            self.ui_name.set(str(data.get("display_name", self.ui_name.get())))
            self.ui_avatar.set(str(data.get("avatar", self.ui_avatar.get())))
        except Exception:
            pass

    def _save_ui_settings(self):
        os.makedirs(os.path.dirname(SETTINGS_JSON), exist_ok=True)
        data = {
            "language": self.ui_language.get(),
            "background": self.ui_bg.get(),
            "display_name": self.ui_name.get(),
            "avatar": self.ui_avatar.get(),
        }
        json.dump(data, open(SETTINGS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def _refresh_profile_text(self):
        name = self.ui_name.get().strip() or "Guest"
        avatar = self.ui_avatar.get().strip() or "User"
        self.profile_text.set(f"{avatar} {name}")

    def _apply_ui_settings(self):
        bg = self.ui_bg.get().strip() or "#f5f7fb"
        style = ttk.Style()
        try:
            style.configure("TFrame", background=bg)
            style.configure("TLabelframe", background=bg)
            style.configure("TLabelframe.Label", background=bg)
            style.configure("TLabel", background=bg)
            self.root.configure(bg=bg)
        except Exception:
            pass
        self.root.title("ST-GNN Recommendation Desktop")
        self._refresh_profile_text()

    def on_open_settings(self):
        w = Toplevel(self.root)
        w.title("Settings")
        w.geometry("420x300")
        frm = ttk.Frame(w, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Language").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(frm, textvariable=self.ui_language, values=["Chinese", "English"], state="readonly", width=18).grid(row=0, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frm, text="Background").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(frm, textvariable=self.ui_bg, width=20).grid(row=1, column=1, sticky="w", padx=6, pady=6)
        ttk.Button(frm, text="Pick", command=self.on_pick_bg_color).grid(row=1, column=2, sticky="w", padx=6, pady=6)

        ttk.Label(frm, text="Display Name").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(frm, textvariable=self.ui_name, width=20).grid(row=2, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(frm, text="Avatar Label").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            frm,
            textvariable=self.ui_avatar,
            values=["User", "Student", "Researcher", "Analyst"],
            state="readonly",
            width=18,
        ).grid(row=3, column=1, sticky="w", padx=6, pady=6)

        tip = "Changes apply immediately and are saved for next launch."
        ttk.Label(frm, text=tip).grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=10)

        ttk.Button(frm, text="Save", command=lambda: self.on_save_settings(w)).grid(row=5, column=1, sticky="e", padx=6, pady=8)
        ttk.Button(frm, text="Cancel", command=w.destroy).grid(row=5, column=2, sticky="w", padx=6, pady=8)

    def on_pick_bg_color(self):
        c = colorchooser.askcolor(title="Choose background")
        if c and c[1]:
            self.ui_bg.set(c[1])
            # Apply and persist immediately after user confirms the picker.
            try:
                self._apply_ui_settings()
                self._save_ui_settings()
                self._set_status("Background updated")
            except Exception:
                pass

    def on_save_settings(self, win):
        try:
            self._save_ui_settings()
            self._apply_ui_settings()
            self._set_status("Settings saved")
        except Exception as e:
            messagebox.showerror("Settings", f"Save failed: {e}")
            return
        win.destroy()

    def _load_demo(self):
        if self.demo_pipe is not None and self.demo_df is not None:
            return
        if not os.path.exists(DEMO_ARTIFACT):
            raise FileNotFoundError(f"Missing model artifact: {DEMO_ARTIFACT}")
        p = STGNNPipeline(TrainConfig())
        p.load_artifact(DEMO_ARTIFACT)
        self.demo_pipe = p
        conn = sqlite3.connect(DEMO_DB)
        try:
            self.demo_df = pd.read_sql_query("select u,i,b,t from user_behavior_logs", conn)
        finally:
            conn.close()

    def _ensure_demo_runtime_db(self):
        global DEMO_DB
        if os.path.exists(DEMO_DB):
            return
        def _init_empty_demo_db(db_path: str):
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_behavior_logs (
                        u INTEGER NOT NULL,
                        i INTEGER NOT NULL,
                        b INTEGER NOT NULL,
                        t INTEGER NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_demo_u_t ON user_behavior_logs(u,t)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_demo_u_i_t ON user_behavior_logs(u,i,t)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_demo_i_b ON user_behavior_logs(i,b)")
                conn.commit()
            finally:
                conn.close()
        try:
            os.makedirs(os.path.dirname(DEMO_DB), exist_ok=True)
            if os.path.exists(DEMO_DB_SRC):
                shutil.copy2(DEMO_DB_SRC, DEMO_DB)
            else:
                _init_empty_demo_db(DEMO_DB)
        except Exception:
            fb = _fallback_user_dir()
            DEMO_DB = os.path.join(fb, "rec_system_runtime.db")
            os.makedirs(os.path.dirname(DEMO_DB), exist_ok=True)
            if os.path.exists(DEMO_DB_SRC):
                shutil.copy2(DEMO_DB_SRC, DEMO_DB)
            else:
                _init_empty_demo_db(DEMO_DB)

    def _beh_id(self, x: str) -> int:
        return {"Click": 0, "Favorite": 1, "Cart": 2, "Buy": 3}.get(x, 3)

    def _safe_table_name(self, table: str) -> str:
        t = (table or "").strip()
        if not t or not all(ch.isalnum() or ch == "_" for ch in t):
            raise ValueError("Invalid table name. Use letters, numbers, underscore.")
        return t

    def _max_hist_value(self, x: int) -> int:
        v = max(20, min(int(x), MAX_HISTORY_HARD_LIMIT))
        return v

    def _safe_topk(self, x: int) -> int:
        return max(1, min(int(x), TOPK_HARD_LIMIT))

    def _safe_sample_size(self, x: int) -> int:
        return max(5, min(int(x), INSPECTOR_SAMPLE_HARD_LIMIT))

    def _ensure_indexes(self, conn: sqlite3.Connection, table: str):
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_u_t ON {table}(u,t)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_u_i_t ON {table}(u,i,t)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_i_b ON {table}(i,b)")

    def _append_with_eviction(self, conn: sqlite3.Connection, table: str, u: int, i: int, b: int, t: int, max_hist: int):
        conn.execute(f"INSERT INTO {table} (u,i,b,t) VALUES (?,?,?,?)", (u, i, b, t))
        conn.execute(
            f"""
            DELETE FROM {table}
            WHERE rowid IN (
              SELECT rowid FROM {table}
              WHERE u=?
              ORDER BY t DESC, rowid DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (u, max_hist),
        )

    def _fill_rec_tree(self, tree, rec: pd.DataFrame):
        for r in tree.get_children():
            tree.delete(r)
        for _, it in rec.iterrows():
            tree.insert("", "end", values=(int(it["rank"]), int(it["item_id"]), f"{float(it['score']):.4f}", str(it["reason"])))

    def _active_workspace(self):
        if self.ins_ws.get() == "Custom" and self.custom_pipe is not None and self.custom_df is not None:
            return self.custom_pipe, self.custom_df
        return self.demo_pipe, self.demo_df

    # ---------- account ----------
    def on_register(self):
        win = Toplevel(self.root)
        win.title("Register")
        win.geometry("360x220")
        frm = ttk.Frame(win, padding=12)
        frm.pack(fill="both", expand=True)

        reg_user = StringVar(value=self.user_var.get().strip())
        reg_pw = StringVar(value="")
        reg_pw2 = StringVar(value="")

        ttk.Label(frm, text="Username").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(frm, textvariable=reg_user, width=24).grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(frm, text="Password").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(frm, textvariable=reg_pw, show="*", width=24).grid(row=1, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(frm, text="Confirm").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(frm, textvariable=reg_pw2, show="*", width=24).grid(row=2, column=1, sticky="w", padx=6, pady=6)

        def do_register():
            u = reg_user.get().strip()
            p = reg_pw.get()
            p2 = reg_pw2.get()
            if len(u) < 3:
                messagebox.showwarning("Register", "Username length must be >= 3")
                return
            if len(p) < 6:
                messagebox.showwarning("Register", "Password length must be >= 6")
                return
            if p != p2:
                messagebox.showerror("Register", "Password and Confirm Password do not match")
                return
            self.user_var.set(u)
            self.pw_var.set(p)
            ok = self._register_account(u, p)
            if ok:
                win.destroy()

        ttk.Button(frm, text="Confirm", command=do_register).grid(row=3, column=1, sticky="e", padx=6, pady=10)
        ttk.Button(frm, text="Cancel", command=win.destroy).grid(row=3, column=0, sticky="w", padx=6, pady=10)

    def _register_account(self, u: str, p: str) -> bool:
        conn = sqlite3.connect(ACCOUNT_DB)
        try:
            conn.execute("INSERT INTO users(username,password_hash,created_at) VALUES (?,?,?)", (u, hash_pw(p), int(time.time())))
            conn.commit()
            self._set_status("Register success")
            messagebox.showinfo("Register", "Register success")
            return True
        except sqlite3.IntegrityError:
            messagebox.showerror("Register", "Username exists")
            return False
        finally:
            conn.close()

    def on_login(self):
        u, p = self.user_var.get().strip(), self.pw_var.get()
        conn = sqlite3.connect(ACCOUNT_DB)
        try:
            row = conn.execute("SELECT id FROM users WHERE username=? AND password_hash=?", (u, hash_pw(p))).fetchone()
        finally:
            conn.close()
        if row:
            self.logged_in = True
            self.login_state.set(f"Logged in: {u}")
            if not self.ui_name.get().strip() or self.ui_name.get().strip().lower() == "guest":
                self.ui_name.set(u)
            self._refresh_profile_text()
            self._set_status("Login success")
        else:
            self.logged_in = False
            self.login_state.set("Not logged in")
            messagebox.showerror("Login", "Invalid username/password")

    def on_logout(self):
        self.logged_in = False
        self.login_state.set("Not logged in")
        self._refresh_profile_text()
        self._set_status("Logged out")

    # ---------- demo ----------
    def on_demo_recommend(self):
        if not self._require_login():
            return
        rec = self.demo_pipe.recommend_for_raw_user(
            int(self.demo_uid.get()),
            top_k=self._safe_topk(self.demo_topk.get()),
            filter_seen=not bool(self.demo_include.get()),
        )
        self._fill_rec_tree(self.demo_tree, rec)
        self._set_status(f"Demo recommendation loaded: {len(rec)}")

    def on_demo_interact(self):
        if not self._require_login():
            return
        u, i, b, t = int(self.demo_uid.get()), int(self.demo_item.get()), self._beh_id(self.demo_beh.get()), int(time.time())
        keep_n = self._max_hist_value(self.demo_max_hist.get())
        conn = sqlite3.connect(DEMO_DB)
        try:
            self._ensure_indexes(conn, "user_behavior_logs")
            self._append_with_eviction(conn, "user_behavior_logs", u, i, b, t, keep_n)
            conn.commit()
            self.demo_df = pd.read_sql_query("select u,i,b,t from user_behavior_logs", conn)
        finally:
            conn.close()
        self.demo_pipe.append_interaction(u, i, b, t)
        self._set_status(f"Demo interaction saved (keep latest {keep_n}/user)")
        self.on_demo_recommend()

    # ---------- custom ----------
    def on_pick_custom_db(self):
        p = filedialog.askopenfilename(title="Select sqlite db", filetypes=[("SQLite", "*.db *.sqlite *.sqlite3"), ("All", "*.*")])
        if p:
            self.custom_db.set(p)

    def on_validate_custom(self):
        if not self._require_login():
            return
        db = self.custom_db.get().strip()
        if not db or not os.path.exists(db):
            messagebox.showerror("Validate", "Select valid db file")
            return
        try:
            table = self._safe_table_name(self.custom_table.get())
        except ValueError as e:
            messagebox.showerror("Validate", str(e))
            return
        conn = sqlite3.connect(db)
        try:
            self._ensure_indexes(conn, table)
            df = pd.read_sql_query(f"select u,i,b,t from {table}", conn)
        finally:
            conn.close()
        if not set(["u", "i", "b", "t"]).issubset(df.columns):
            self.custom_note.set("Missing required columns u,i,b,t")
            return
        x = df[["u", "i", "b", "t"]].dropna()
        x = x.astype({"u": int, "i": int, "b": int, "t": int})
        if not x["b"].isin([0, 1, 2, 3]).all():
            self.custom_note.set("Invalid b values. Must be 0/1/2/3")
            return
        pu = x.groupby("u").size()
        self.custom_note.set(
            "Validation pass | "
            f"rows={len(x)} users={x.u.nunique()} items={x.i.nunique()} "
            f"min/user={int(pu.min())} p50/user={float(pu.quantile(0.5)):.1f} max/user={int(pu.max())} "
            f"| policy keep<= {self._max_hist_value(self.custom_max_hist.get())}/user (hard cap {MAX_HISTORY_HARD_LIMIT})"
        )

    def on_load_custom(self):
        if not self._require_login():
            return
        db = self.custom_db.get().strip()
        if not db or not os.path.exists(db):
            messagebox.showerror("Custom", "Select valid db file")
            return
        try:
            table = self._safe_table_name(self.custom_table.get())
        except ValueError as e:
            messagebox.showerror("Custom", str(e))
            return
        conn = sqlite3.connect(db)
        try:
            df = pd.read_sql_query(f"select u,i,b,t from {table}", conn)
        finally:
            conn.close()
        x = df[["u", "i", "b", "t"]].dropna().astype({"u": int, "i": int, "b": int, "t": int})
        cfg = TrainConfig(epochs=max(1, min(int(self.custom_epochs.get()), 20)), embed_dim=64, batch_size=256)
        p = STGNNPipeline(cfg)
        p.prepare_from_df(x)
        p.train(verbose=False)
        self.custom_pipe = p
        self.custom_df = x
        self.custom_note.set(f"Custom loaded & trained | rows={len(x)} users={x.u.nunique()} items={x.i.nunique()}")
        self._set_status("Custom workspace ready")

    def on_custom_recommend(self):
        if not self._require_login():
            return
        if self.custom_pipe is None:
            messagebox.showwarning("Custom", "Load custom DB first")
            return
        rec = self.custom_pipe.recommend_for_raw_user(
            int(self.custom_uid.get()),
            top_k=self._safe_topk(self.custom_topk.get()),
            filter_seen=not bool(self.custom_include.get()),
        )
        self._fill_rec_tree(self.custom_tree, rec)
        self._set_status(f"Custom recommendation loaded: {len(rec)}")

    def on_custom_interact(self):
        if not self._require_login():
            return
        if self.custom_pipe is None:
            messagebox.showwarning("Custom", "Load custom DB first")
            return
        db = self.custom_db.get().strip()
        try:
            table = self._safe_table_name(self.custom_table.get())
        except ValueError as e:
            messagebox.showerror("Custom", str(e))
            return
        u, i, b, t = int(self.custom_uid.get()), int(self.custom_item.get()), self._beh_id(self.custom_beh.get()), int(time.time())
        keep_n = self._max_hist_value(self.custom_max_hist.get())
        conn = sqlite3.connect(db)
        try:
            self._ensure_indexes(conn, table)
            self._append_with_eviction(conn, table, u, i, b, t, keep_n)
            conn.commit()
            self.custom_df = pd.read_sql_query(f"select u,i,b,t from {table}", conn)
        finally:
            conn.close()
        self.custom_pipe.append_interaction(u, i, b, t)
        self._set_status(f"Custom interaction saved (keep latest {keep_n}/user)")
        self.on_custom_recommend()

    # ---------- user visualization ----------
    def on_visualize_user(self):
        if not self._require_login():
            return
        if self.view_ws.get() == "Custom":
            _, df = self._active_workspace()
        else:
            df = self.demo_df
        if df is None or len(df) == 0:
            messagebox.showwarning("User View", "Workspace data is not ready")
            return

        uid = int(self.view_uid.get())
        topn = max(5, min(int(self.view_topn.get()), 100))
        x = df[df["u"].astype(int) == uid].copy()
        for r in self.user_tree.get_children():
            self.user_tree.delete(r)
        if len(x) == 0:
            self.view_summary.set(f"user={uid} has no records in current workspace")
            return

        x["b"] = x["b"].astype(int)
        agg = (
            x.groupby("i")
            .agg(
                total=("i", "size"),
                click=("b", lambda s: int((s == 0).sum())),
                favorite=("b", lambda s: int((s == 1).sum())),
                cart=("b", lambda s: int((s == 2).sum())),
                buy=("b", lambda s: int((s == 3).sum())),
                last_ts=("t", "max"),
            )
            .reset_index()
            .sort_values(["buy", "cart", "favorite", "click", "total", "last_ts"], ascending=False)
            .head(topn)
        )
        for _, r in agg.iterrows():
            dt = datetime.fromtimestamp(int(r["last_ts"])).strftime("%Y-%m-%d %H:%M:%S")
            self.user_tree.insert(
                "",
                "end",
                values=(int(r["i"]), int(r["total"]), int(r["click"]), int(r["favorite"]), int(r["cart"]), int(r["buy"]), dt),
            )

        cnt = x["b"].value_counts().to_dict()
        self.view_summary.set(
            f"user={uid} | total={len(x)} items={x['i'].nunique()} | "
            f"click={cnt.get(0,0)} favorite={cnt.get(1,0)} cart={cnt.get(2,0)} buy={cnt.get(3,0)} | topN={topn}"
        )

    # ---------- inspector ----------
    def on_run_batch(self):
        if not self._require_login():
            return
        pipe, df = self._active_workspace()
        if pipe is None or df is None:
            messagebox.showwarning("Inspector", "Workspace not ready")
            return
        d, s = run_batch_diagnostics(
            pipe,
            df,
            sample_size=self._safe_sample_size(self.ins_sample.get()),
            top_k=self._safe_topk(self.ins_topk.get()),
            random_seed=42,
        )
        self.batch_df, self.batch_summary = d, s
        self.ins_summary.set(f"cases={s.get('cases',0)} pass={s.get('pass_rate',0)*100:.1f}% avgQ={s.get('avg_quality_score',0):.1f}")
        self._fill_inspector(d)

    def on_export_batch(self):
        if self.batch_df is None or self.batch_summary is None or len(self.batch_df) == 0:
            messagebox.showwarning("Export", "Run batch first")
            return
        os.makedirs(EXPORT_DIR, exist_ok=True)
        paths = export_inspector_results(self.batch_df, self.batch_summary, EXPORT_DIR, prefix="batch_diag")
        messagebox.showinfo("Export", f"CSV: {paths.get('csv')}\nPNG: {paths.get('plot')}\nSummary: {paths.get('summary')}")

    def on_diag_item(self):
        if not self._require_login():
            return
        pipe, df = self._active_workspace()
        if pipe is None or df is None:
            messagebox.showwarning("Inspector", "Workspace not ready")
            return
        d, s = diagnose_item_across_users(
            pipe,
            df,
            item_id=int(self.ins_item.get()),
            top_k=self._safe_topk(self.ins_topk.get()),
            simulate_behavior=3,
        )
        self.ins_summary.set(f"item={s.get('item_id')} users={s.get('users')} avgQ={s.get('avg_quality_score',0):.1f} avgRankImprove={s.get('avg_rank_improve',0):.1f}")
        self._fill_inspector(d)

    def _fill_inspector(self, df: pd.DataFrame):
        for r in self.ins_tree.get_children():
            self.ins_tree.delete(r)
        if df is None or len(df) == 0:
            return
        for _, x in df.iterrows():
            self.ins_tree.insert(
                "",
                "end",
                values=(
                    int(x.get("user_id", 0)),
                    int(x.get("item_id", self.ins_item.get())),
                    round(float(x.get("score_delta", 0) or 0), 4),
                    int(x.get("rank_improve", 0) or 0),
                    round(float(x.get("quality_score", 0) or 0), 2),
                    str(x.get("message", "")),
                ),
            )

    # ---------- data/papers ----------
    def on_refresh_data(self):
        for r in self.data_tree.get_children():
            self.data_tree.delete(r)
        files = [
            ("rec_system.db(source)", DEMO_DB_SRC),
            ("rec_system_runtime.db(writable)", DEMO_DB),
            ("artifacts/stgnn_artifact_v2.pt", DEMO_ARTIFACT),
            ("paper_alignment.json", PAPER_MAP),
            ("PAPER_PDF_LINKS.md", resolve_resource_path("PAPER_PDF_LINKS.md")),
            ("inspector_export_dir", EXPORT_DIR),
        ]
        for name, p in files:
            exists = os.path.exists(p)
            size_mb = round(os.path.getsize(p) / (1024 * 1024), 2) if exists and os.path.isfile(p) else "-"
            self.data_tree.insert("", "end", values=(name, exists, size_mb))
        demo_rows = len(self.demo_df) if isinstance(self.demo_df, pd.DataFrame) else 0
        custom_rows = len(self.custom_df) if isinstance(self.custom_df, pd.DataFrame) else 0
        beh = {}
        if isinstance(self.demo_df, pd.DataFrame) and len(self.demo_df) > 0:
            c = self.demo_df["b"].astype(int).value_counts().to_dict()
            beh = {"click": c.get(0, 0), "favorite": c.get(1, 0), "cart": c.get(2, 0), "buy": c.get(3, 0)}
        self.data_note.set(
            f"demo_rows={demo_rows}, custom_rows={custom_rows} | "
            f"demo_beh(click/fav/cart/buy)={beh.get('click',0)}/{beh.get('favorite',0)}/{beh.get('cart',0)}/{beh.get('buy',0)}"
        )

    def on_load_papers(self):
        for r in self.paper_tree.get_children():
            self.paper_tree.delete(r)
        if not os.path.exists(PAPER_MAP):
            return
        data = json.load(open(PAPER_MAP, "r", encoding="utf-8"))
        for x in data:
            self.paper_tree.insert("", "end", values=(x.get("paper", ""), x.get("url", ""), x.get("status", ""), x.get("aligned_idea", "")))

    def on_open_paper(self):
        sel = self.paper_tree.selection()
        if not sel:
            return
        vals = self.paper_tree.item(sel[0], "values")
        if len(vals) > 1 and vals[1]:
            webbrowser.open(vals[1])

    # ---------- UI ----------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Visual system
        style.configure("App.TFrame", background="#eef3f9")
        style.configure("Card.TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#eef3f9", foreground="#1f2937", font=("Segoe UI", 10))
        style.configure("TEntry", padding=5)
        style.configure("TCombobox", padding=4)
        style.configure("TCheckbutton", background="#ffffff", foreground="#1f2937")

        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        style.map("Accent.TButton", background=[("active", "#1959b3"), ("!disabled", "#1f6feb")], foreground=[("!disabled", "#ffffff")])
        style.configure("Muted.TButton", font=("Segoe UI", 10), padding=(10, 6))
        style.map("Muted.TButton", background=[("active", "#e5ecf6"), ("!disabled", "#f3f6fb")], foreground=[("!disabled", "#1f2937")])

        style.configure("TNotebook", background="#eef3f9", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 9), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff"), ("!selected", "#dfe7f2")], foreground=[("selected", "#0f172a"), ("!selected", "#334155")])

        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10), fieldbackground="#ffffff", background="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#e8eef7", foreground="#0f172a")
        style.map("Treeview", background=[("selected", "#cfe3ff")], foreground=[("selected", "#0f172a")])
        style.configure("Status.TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 9))

        self.root.configure(bg="#eef3f9")
        self.root.minsize(1180, 760)

        outer = ttk.Frame(self.root, style="App.TFrame", padding=12)
        outer.pack(fill="both", expand=True)

        acc = ttk.LabelFrame(outer, text="Account & Profile", style="Card.TLabelframe", padding=8)
        acc.pack(fill="x", pady=(0, 8))
        for idx in range(10):
            acc.columnconfigure(idx, weight=0)
        acc.columnconfigure(7, weight=1)

        ttk.Label(acc, text="Username").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        ttk.Entry(acc, textvariable=self.user_var, width=18).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(acc, text="Password").grid(row=0, column=2, padx=6, pady=6, sticky="w")
        ttk.Entry(acc, textvariable=self.pw_var, width=18, show="*").grid(row=0, column=3, padx=6, pady=6)
        ttk.Button(acc, text="Register", command=self.on_register, style="Muted.TButton").grid(row=0, column=4, padx=4, pady=6)
        ttk.Button(acc, text="Login", command=self.on_login, style="Accent.TButton").grid(row=0, column=5, padx=4, pady=6)
        ttk.Button(acc, text="Logout", command=self.on_logout, style="Muted.TButton").grid(row=0, column=6, padx=4, pady=6)
        ttk.Label(acc, textvariable=self.login_state).grid(row=0, column=7, padx=12, pady=6, sticky="w")
        ttk.Button(acc, text="Settings", command=self.on_open_settings, style="Muted.TButton").grid(row=0, column=8, padx=4, pady=6)
        ttk.Label(acc, textvariable=self.profile_text).grid(row=0, column=9, padx=10, pady=6, sticky="e")

        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True)

        tab_demo = ttk.Frame(nb)
        tab_custom = ttk.Frame(nb)
        tab_ins = ttk.Frame(nb)
        tab_user = ttk.Frame(nb)
        tab_data = ttk.Frame(nb)
        tab_paper = ttk.Frame(nb)
        nb.add(tab_demo, text="Demo Workspace")
        nb.add(tab_custom, text="Custom DB Workspace")
        nb.add(tab_ins, text="Inspector")
        nb.add(tab_user, text="User Visualization")
        nb.add(tab_data, text="Data")
        nb.add(tab_paper, text="Papers")

        # Demo tab
        dctl = ttk.LabelFrame(tab_demo, text="Demo (Built-in Thesis Data)", style="Card.TLabelframe", padding=8)
        dctl.pack(fill="x", padx=8, pady=8)
        ttk.Label(dctl, text="User").grid(row=0, column=0, padx=6, pady=6)
        ttk.Entry(dctl, textvariable=self.demo_uid, width=10).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(dctl, text="Top-K").grid(row=0, column=2, padx=6, pady=6)
        ttk.Entry(dctl, textvariable=self.demo_topk, width=8).grid(row=0, column=3, padx=6, pady=6)
        ttk.Checkbutton(dctl, text="Include seen", variable=self.demo_include).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(dctl, text="Recommend", command=self.on_demo_recommend, style="Accent.TButton").grid(row=0, column=5, padx=6, pady=6)
        ttk.Label(dctl, text="Item").grid(row=1, column=0, padx=6, pady=6)
        ttk.Entry(dctl, textvariable=self.demo_item, width=10).grid(row=1, column=1, padx=6, pady=6)
        ttk.Combobox(dctl, textvariable=self.demo_beh, values=["Click", "Favorite", "Cart", "Buy"], state="readonly", width=10).grid(row=1, column=2, padx=6, pady=6)
        ttk.Label(dctl, text="Max hist/user").grid(row=1, column=3, padx=6, pady=6)
        ttk.Entry(dctl, textvariable=self.demo_max_hist, width=8).grid(row=1, column=4, padx=6, pady=6)
        ttk.Button(dctl, text="Submit", command=self.on_demo_interact, style="Muted.TButton").grid(row=1, column=5, padx=6, pady=6)

        dtable = ttk.LabelFrame(tab_demo, text="Recommendations", style="Card.TLabelframe", padding=6)
        dtable.pack(fill="both", expand=True, padx=8, pady=8)
        self.demo_tree = ttk.Treeview(dtable, columns=("rank", "item_id", "score", "reason"), show="headings", height=18)
        for c, w in [("rank", 70), ("item_id", 120), ("score", 100), ("reason", 880)]:
            self.demo_tree.heading(c, text=c)
            self.demo_tree.column(c, width=w, anchor="w" if c == "reason" else "center")
        self.demo_tree.pack(fill="both", expand=True)

        # Custom tab
        req = ttk.LabelFrame(tab_custom, text="Custom DB Format Requirement", style="Card.TLabelframe", padding=8)
        req.pack(fill="x", padx=8, pady=8)
        ttk.Label(
            req,
            text=(
                "Table must contain columns: u,i,b,t. "
                "b in {0 click,1 favorite,2 cart,3 buy}. "
                "t must be unix timestamp. "
                "Eviction policy keeps latest N records per user."
            ),
            justify="left",
        ).pack(anchor="w", padx=6, pady=6)

        cctl = ttk.LabelFrame(tab_custom, text="Load & Use Custom DB", style="Card.TLabelframe", padding=8)
        cctl.pack(fill="x", padx=8, pady=8)
        ttk.Entry(cctl, textvariable=self.custom_db, width=80).grid(row=0, column=0, columnspan=4, padx=6, pady=6, sticky="w")
        ttk.Button(cctl, text="Browse", command=self.on_pick_custom_db, style="Muted.TButton").grid(row=0, column=4, padx=6, pady=6)
        ttk.Label(cctl, text="Table").grid(row=1, column=0, padx=6, pady=6)
        ttk.Entry(cctl, textvariable=self.custom_table, width=20).grid(row=1, column=1, padx=6, pady=6, sticky="w")
        ttk.Label(cctl, text="Epochs").grid(row=1, column=2, padx=6, pady=6)
        ttk.Entry(cctl, textvariable=self.custom_epochs, width=8).grid(row=1, column=3, padx=6, pady=6, sticky="w")
        ttk.Button(cctl, text="Validate", command=self.on_validate_custom, style="Muted.TButton").grid(row=1, column=4, padx=6, pady=6)
        ttk.Button(cctl, text="Load+Train", command=self.on_load_custom, style="Accent.TButton").grid(row=1, column=5, padx=6, pady=6)
        ttk.Label(cctl, textvariable=self.custom_note).grid(row=2, column=0, columnspan=6, padx=6, pady=6, sticky="w")

        cuse = ttk.LabelFrame(tab_custom, text="Recommend / Update", style="Card.TLabelframe", padding=8)
        cuse.pack(fill="x", padx=8, pady=8)
        ttk.Label(cuse, text="User").grid(row=0, column=0, padx=6, pady=6)
        ttk.Entry(cuse, textvariable=self.custom_uid, width=10).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(cuse, text="Top-K").grid(row=0, column=2, padx=6, pady=6)
        ttk.Entry(cuse, textvariable=self.custom_topk, width=8).grid(row=0, column=3, padx=6, pady=6)
        ttk.Checkbutton(cuse, text="Include seen", variable=self.custom_include).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(cuse, text="Recommend", command=self.on_custom_recommend, style="Accent.TButton").grid(row=0, column=5, padx=6, pady=6)
        ttk.Label(cuse, text="Item").grid(row=1, column=0, padx=6, pady=6)
        ttk.Entry(cuse, textvariable=self.custom_item, width=10).grid(row=1, column=1, padx=6, pady=6)
        ttk.Combobox(cuse, textvariable=self.custom_beh, values=["Click", "Favorite", "Cart", "Buy"], state="readonly", width=10).grid(row=1, column=2, padx=6, pady=6)
        ttk.Label(cuse, text="Max hist/user").grid(row=1, column=3, padx=6, pady=6)
        ttk.Entry(cuse, textvariable=self.custom_max_hist, width=8).grid(row=1, column=4, padx=6, pady=6)
        ttk.Button(cuse, text="Submit", command=self.on_custom_interact, style="Muted.TButton").grid(row=1, column=5, padx=6, pady=6)

        ctable = ttk.LabelFrame(tab_custom, text="Custom Recommendations", style="Card.TLabelframe", padding=6)
        ctable.pack(fill="both", expand=True, padx=8, pady=8)
        self.custom_tree = ttk.Treeview(ctable, columns=("rank", "item_id", "score", "reason"), show="headings", height=14)
        for c, w in [("rank", 70), ("item_id", 120), ("score", 100), ("reason", 880)]:
            self.custom_tree.heading(c, text=c)
            self.custom_tree.column(c, width=w, anchor="w" if c == "reason" else "center")
        self.custom_tree.pack(fill="both", expand=True)

        # Inspector
        ictl = ttk.LabelFrame(tab_ins, text="Inspector", style="Card.TLabelframe", padding=8)
        ictl.pack(fill="x", padx=8, pady=8)
        ttk.Combobox(ictl, textvariable=self.ins_ws, values=["Demo", "Custom"], state="readonly", width=10).grid(row=0, column=0, padx=6, pady=6)
        ttk.Entry(ictl, textvariable=self.ins_sample, width=8).grid(row=0, column=1, padx=6, pady=6)
        ttk.Entry(ictl, textvariable=self.ins_topk, width=8).grid(row=0, column=2, padx=6, pady=6)
        ttk.Button(ictl, text="Run Batch", command=self.on_run_batch, style="Accent.TButton").grid(row=0, column=3, padx=6, pady=6)
        ttk.Button(ictl, text="Export Batch", command=self.on_export_batch, style="Muted.TButton").grid(row=0, column=4, padx=6, pady=6)
        ttk.Entry(ictl, textvariable=self.ins_item, width=12).grid(row=1, column=0, padx=6, pady=6)
        ttk.Button(ictl, text="Diagnose Item", command=self.on_diag_item, style="Muted.TButton").grid(row=1, column=1, padx=6, pady=6)
        ttk.Label(ictl, textvariable=self.ins_summary).grid(row=2, column=0, columnspan=5, padx=6, pady=6, sticky="w")

        itable = ttk.LabelFrame(tab_ins, text="Inspector Output", style="Card.TLabelframe", padding=6)
        itable.pack(fill="both", expand=True, padx=8, pady=8)
        self.ins_tree = ttk.Treeview(itable, columns=("user_id", "item_id", "score_delta", "rank_improve", "quality_score", "message"), show="headings", height=16)
        for c, w in [("user_id", 90), ("item_id", 100), ("score_delta", 110), ("rank_improve", 110), ("quality_score", 110), ("message", 760)]:
            self.ins_tree.heading(c, text=c)
            self.ins_tree.column(c, width=w, anchor="w" if c == "message" else "center")
        self.ins_tree.pack(fill="both", expand=True)

        # User visualization
        uctl = ttk.LabelFrame(tab_user, text="User Behavior Visualization", style="Card.TLabelframe", padding=8)
        uctl.pack(fill="x", padx=8, pady=8)
        ttk.Combobox(uctl, textvariable=self.view_ws, values=["Demo", "Custom"], state="readonly", width=10).grid(row=0, column=0, padx=6, pady=6)
        ttk.Label(uctl, text="User").grid(row=0, column=1, padx=6, pady=6)
        ttk.Entry(uctl, textvariable=self.view_uid, width=10).grid(row=0, column=2, padx=6, pady=6)
        ttk.Label(uctl, text="Top Items").grid(row=0, column=3, padx=6, pady=6)
        ttk.Entry(uctl, textvariable=self.view_topn, width=8).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(uctl, text="Visualize", command=self.on_visualize_user, style="Accent.TButton").grid(row=0, column=5, padx=6, pady=6)
        ttk.Label(uctl, textvariable=self.view_summary).grid(row=1, column=0, columnspan=6, sticky="w", padx=6, pady=6)

        utable = ttk.LabelFrame(tab_user, text="Per-item Stats for Selected User", style="Card.TLabelframe", padding=6)
        utable.pack(fill="both", expand=True, padx=8, pady=8)
        self.user_tree = ttk.Treeview(utable, columns=("item_id", "total", "click", "favorite", "cart", "buy", "last_time"), show="headings", height=16)
        for c, w in [("item_id", 120), ("total", 100), ("click", 100), ("favorite", 100), ("cart", 100), ("buy", 100), ("last_time", 300)]:
            self.user_tree.heading(c, text=c)
            self.user_tree.column(c, width=w, anchor="center")
        self.user_tree.pack(fill="both", expand=True)

        # Data
        dctl2 = ttk.Frame(tab_data, style="App.TFrame")
        dctl2.pack(fill="x", padx=8, pady=8)
        ttk.Button(dctl2, text="Refresh Data", command=self.on_refresh_data, style="Accent.TButton").pack(side="left")
        self.data_note = StringVar(value="Click refresh")
        ttk.Label(tab_data, textvariable=self.data_note).pack(fill="x", padx=8)
        dtable2 = ttk.LabelFrame(tab_data, text="Files", style="Card.TLabelframe", padding=6)
        dtable2.pack(fill="both", expand=True, padx=8, pady=8)
        self.data_tree = ttk.Treeview(dtable2, columns=("file", "exists", "size_mb"), show="headings", height=16)
        for c, w in [("file", 560), ("exists", 100), ("size_mb", 120)]:
            self.data_tree.heading(c, text=c)
            self.data_tree.column(c, width=w, anchor="w" if c == "file" else "center")
        self.data_tree.pack(fill="both", expand=True)

        # Papers
        pctl = ttk.Frame(tab_paper, style="App.TFrame")
        pctl.pack(fill="x", padx=8, pady=8)
        ttk.Button(pctl, text="Load Papers", command=self.on_load_papers, style="Accent.TButton").pack(side="left")
        ttk.Button(pctl, text="Open URL", command=self.on_open_paper, style="Muted.TButton").pack(side="left", padx=6)
        ptable = ttk.LabelFrame(tab_paper, text="Paper Alignment", style="Card.TLabelframe", padding=6)
        ptable.pack(fill="both", expand=True, padx=8, pady=8)
        self.paper_tree = ttk.Treeview(ptable, columns=("paper", "url", "status", "aligned_idea"), show="headings", height=16)
        for c, w in [("paper", 320), ("url", 380), ("status", 120), ("aligned_idea", 460)]:
            self.paper_tree.heading(c, text=c)
            self.paper_tree.column(c, width=w, anchor="w")
        self.paper_tree.pack(fill="both", expand=True)

        status = ttk.Label(outer, textvariable=self.status, style="Status.TLabel", anchor="w", padding=(10, 6))
        status.pack(fill="x", pady=(8, 0))


def main():
    try:
        root = Tk()
        app = UnifiedApp(root)
        root.mainloop()
    except Exception:
        err = traceback.format_exc()
        safe_dir = USER_DATA_DIR if os.path.isdir(USER_DATA_DIR) else _fallback_user_dir()
        log_path = os.path.join(safe_dir, "app_error.log")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now().isoformat()}]\n{err}\n")
        except Exception:
            pass
        try:
            from tkinter import messagebox as _mb
            _mb.showerror("App Error", f"Application failed. Log saved to:\n{log_path}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
