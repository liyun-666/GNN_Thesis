import json
import os
import sqlite3
import sys
import time
import webbrowser
from tkinter import BooleanVar, IntVar, StringVar, Tk, messagebox
from tkinter import ttk

import pandas as pd

from qa_tool import diagnose_item_across_users, export_inspector_results, run_batch_diagnostics
from recommender_engine import STGNNPipeline, TrainConfig


def app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = app_base_dir()
DB_PATH = os.path.join(BASE_DIR, "rec_system.db")
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "stgnn_artifact_v2.pt")
INSPECT_EXPORT_DIR = os.path.join(BASE_DIR, "artifacts", "inspector_exports")
PAPER_ALIGNMENT_PATH = os.path.join(BASE_DIR, "paper_alignment.json")


class STGNNDesktopApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("ST-GNN Unified Desktop App")
        self.root.geometry("1200x760")

        self.user_id = IntVar(value=10009)
        self.top_k = IntVar(value=10)
        self.include_seen = BooleanVar(value=False)
        self.item_id = IntVar(value=460466)
        self.behavior_text = StringVar(value="Buy")
        self.status_text = StringVar(value="Ready")

        self.pipeline = None
        self.logs_df = None
        self.batch_df = None
        self.batch_summary = None
        self.diag_df = None
        self.diag_summary = None

        self._build_ui()
        self._load_assets_on_startup()

    def _load_assets_on_startup(self):
        try:
            self._ensure_local_assets(force_reload=True)
            self._set_status("Assets loaded. App is ready.")
        except Exception as e:
            self._set_status(f"Startup warning: {e}")
            messagebox.showwarning("Startup", f"Failed to fully initialize:\n{e}")

    def _ensure_local_assets(self, force_reload: bool = False):
        if force_reload or self.pipeline is None:
            if not os.path.exists(ARTIFACT_PATH):
                raise FileNotFoundError(f"Model artifact not found: {ARTIFACT_PATH}")
            pipe = STGNNPipeline(TrainConfig())
            pipe.load_artifact(ARTIFACT_PATH)
            self.pipeline = pipe

        if force_reload or self.logs_df is None:
            if not os.path.exists(DB_PATH):
                raise FileNotFoundError(f"DB not found: {DB_PATH}")
            conn = sqlite3.connect(DB_PATH)
            try:
                self.logs_df = pd.read_sql_query("select u,i,b,t from user_behavior_logs", conn)
            finally:
                conn.close()

    def _set_status(self, txt: str):
        self.status_text.set(txt)
        self.root.update_idletasks()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="Unified local app (no browser, no external API required)").pack(side="left")
        ttk.Button(top, text="Reload Assets", command=self.on_reload_assets).pack(side="right")

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        self.tab_main = ttk.Frame(notebook)
        self.tab_inspector = ttk.Frame(notebook)
        self.tab_data = ttk.Frame(notebook)
        self.tab_papers = ttk.Frame(notebook)
        notebook.add(self.tab_main, text="Main")
        notebook.add(self.tab_inspector, text="Inspector")
        notebook.add(self.tab_data, text="Data")
        notebook.add(self.tab_papers, text="Papers")

        self._build_main_tab()
        self._build_inspector_tab()
        self._build_data_tab()
        self._build_papers_tab()

        status = ttk.Label(outer, textvariable=self.status_text, relief="sunken", anchor="w")
        status.pack(fill="x", pady=(6, 0))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_main_tab(self):
        frm = self.tab_main

        row1 = ttk.LabelFrame(frm, text="Recommendation")
        row1.pack(fill="x", padx=6, pady=6)
        ttk.Label(row1, text="User ID").grid(row=0, column=0, padx=6, pady=6)
        ttk.Entry(row1, textvariable=self.user_id, width=12).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(row1, text="Top-K").grid(row=0, column=2, padx=6, pady=6)
        ttk.Entry(row1, textvariable=self.top_k, width=8).grid(row=0, column=3, padx=6, pady=6)
        ttk.Checkbutton(row1, text="Include seen items", variable=self.include_seen).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(row1, text="Get Recommendations", command=self.on_recommend).grid(row=0, column=5, padx=6, pady=6)

        row2 = ttk.LabelFrame(frm, text="Interaction")
        row2.pack(fill="x", padx=6, pady=6)
        ttk.Label(row2, text="Item ID").grid(row=0, column=0, padx=6, pady=6)
        ttk.Entry(row2, textvariable=self.item_id, width=12).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(row2, text="Behavior").grid(row=0, column=2, padx=6, pady=6)
        cb = ttk.Combobox(row2, textvariable=self.behavior_text, values=["Click", "Favorite", "Cart", "Buy"], state="readonly", width=12)
        cb.grid(row=0, column=3, padx=6, pady=6)
        cb.current(3)
        ttk.Button(row2, text="Submit Interaction", command=self.on_interact).grid(row=0, column=4, padx=6, pady=6)

        table = ttk.LabelFrame(frm, text="Top-K Result")
        table.pack(fill="both", expand=True, padx=6, pady=6)
        self.main_tree = ttk.Treeview(table, columns=("rank", "item_id", "score", "reason"), show="headings", height=18)
        for c, w in [("rank", 70), ("item_id", 120), ("score", 110), ("reason", 840)]:
            self.main_tree.heading(c, text=c)
            self.main_tree.column(c, width=w, anchor="w" if c == "reason" else "center")
        self.main_tree.pack(fill="both", expand=True)

    def _build_inspector_tab(self):
        frm = self.tab_inspector

        ctl = ttk.LabelFrame(frm, text="Inspector Controls")
        ctl.pack(fill="x", padx=6, pady=6)

        self.inspect_sample_size = IntVar(value=30)
        self.inspect_top_k = IntVar(value=10)
        self.inspect_item = IntVar(value=460466)

        ttk.Label(ctl, text="Sample Size").grid(row=0, column=0, padx=6, pady=6)
        ttk.Entry(ctl, textvariable=self.inspect_sample_size, width=8).grid(row=0, column=1, padx=6, pady=6)
        ttk.Label(ctl, text="Top-K").grid(row=0, column=2, padx=6, pady=6)
        ttk.Entry(ctl, textvariable=self.inspect_top_k, width=8).grid(row=0, column=3, padx=6, pady=6)
        ttk.Button(ctl, text="Run Batch Diagnostics", command=self.on_run_batch).grid(row=0, column=4, padx=6, pady=6)
        ttk.Button(ctl, text="Export Batch (CSV+PNG)", command=self.on_export_batch).grid(row=0, column=5, padx=6, pady=6)

        ttk.Label(ctl, text="Special Item").grid(row=1, column=0, padx=6, pady=6)
        ttk.Entry(ctl, textvariable=self.inspect_item, width=12).grid(row=1, column=1, padx=6, pady=6)
        ttk.Button(ctl, text="Diagnose Item", command=self.on_diag_item).grid(row=1, column=2, padx=6, pady=6)

        self.inspect_summary = StringVar(value="No diagnostics yet.")
        ttk.Label(ctl, textvariable=self.inspect_summary).grid(row=2, column=0, columnspan=6, padx=6, pady=4, sticky="w")

        table = ttk.LabelFrame(frm, text="Inspector Details")
        table.pack(fill="both", expand=True, padx=6, pady=6)
        cols = ("user_id", "item_id", "score_delta", "rank_improve", "quality_score", "message")
        self.inspect_tree = ttk.Treeview(table, columns=cols, show="headings", height=16)
        widths = {"user_id": 90, "item_id": 100, "score_delta": 110, "rank_improve": 110, "quality_score": 110, "message": 620}
        for c in cols:
            self.inspect_tree.heading(c, text=c)
            self.inspect_tree.column(c, width=widths[c], anchor="w" if c == "message" else "center")
        self.inspect_tree.pack(fill="both", expand=True)

    def _build_data_tab(self):
        frm = self.tab_data
        ctl = ttk.Frame(frm)
        ctl.pack(fill="x", padx=6, pady=6)
        ttk.Button(ctl, text="Refresh Data Stats", command=self.on_refresh_data).pack(side="left")

        self.data_summary = StringVar(value="Click Refresh Data Stats")
        ttk.Label(frm, textvariable=self.data_summary).pack(fill="x", padx=8, pady=4)

        table = ttk.LabelFrame(frm, text="File Stats")
        table.pack(fill="both", expand=True, padx=6, pady=6)
        cols = ("file", "exists", "size_mb")
        self.data_tree = ttk.Treeview(table, columns=cols, show="headings", height=16)
        for c, w in [("file", 420), ("exists", 100), ("size_mb", 120)]:
            self.data_tree.heading(c, text=c)
            self.data_tree.column(c, width=w, anchor="center" if c != "file" else "w")
        self.data_tree.pack(fill="both", expand=True)

    def _build_papers_tab(self):
        frm = self.tab_papers
        ctl = ttk.Frame(frm)
        ctl.pack(fill="x", padx=6, pady=6)
        ttk.Button(ctl, text="Load Paper Alignment", command=self.on_load_papers).pack(side="left")
        ttk.Button(ctl, text="Open Selected URL", command=self.on_open_paper_url).pack(side="left", padx=6)

        table = ttk.LabelFrame(frm, text="Paper Alignment")
        table.pack(fill="both", expand=True, padx=6, pady=6)
        cols = ("paper", "url", "status", "aligned_idea")
        self.paper_tree = ttk.Treeview(table, columns=cols, show="headings", height=16)
        widths = {"paper": 280, "url": 320, "status": 120, "aligned_idea": 430}
        for c in cols:
            self.paper_tree.heading(c, text=c)
            self.paper_tree.column(c, width=widths[c], anchor="w")
        self.paper_tree.pack(fill="both", expand=True)

    # actions
    def on_reload_assets(self):
        try:
            self._ensure_local_assets(force_reload=True)
            self._set_status("Assets reloaded.")
        except Exception as e:
            self._set_status(f"Reload failed: {e}")
            messagebox.showerror("Reload", str(e))

    def on_recommend(self):
        try:
            self._ensure_local_assets()
            rec = self.pipeline.recommend_for_raw_user(
                int(self.user_id.get()),
                top_k=int(self.top_k.get()),
                filter_seen=not bool(self.include_seen.get()),
            )
            for row in self.main_tree.get_children():
                self.main_tree.delete(row)
            for _, it in rec.iterrows():
                self.main_tree.insert(
                    "",
                    "end",
                    values=(
                        int(it.get("rank", 0)),
                        int(it.get("item_id", 0)),
                        f"{float(it.get('score', 0)):.4f}",
                        str(it.get("reason", "")),
                    ),
                )
            self._set_status(f"Loaded {len(rec)} recommendations.")
        except Exception as e:
            self._set_status(f"Recommend error: {e}")
            messagebox.showerror("Recommend", str(e))

    def on_interact(self):
        beh_map = {"Click": 0, "Favorite": 1, "Cart": 2, "Buy": 3}
        behavior = int(beh_map.get(self.behavior_text.get(), 3))
        uid = int(self.user_id.get())
        iid = int(self.item_id.get())
        ts = int(time.time())

        try:
            self._ensure_local_assets()
            conn = sqlite3.connect(DB_PATH)
            try:
                conn.execute("INSERT INTO user_behavior_logs (u, i, b, t) VALUES (?, ?, ?, ?)", (uid, iid, behavior, ts))
                conn.commit()
            finally:
                conn.close()

            self.pipeline.append_interaction(uid, iid, behavior, ts)
            self.logs_df = None
            self._set_status("Interaction accepted and recommendation updated.")
            self.on_recommend()
        except Exception as e:
            self._set_status(f"Interact error: {e}")
            messagebox.showerror("Interact", str(e))

    def on_run_batch(self):
        try:
            self._set_status("Running batch diagnostics...")
            self._ensure_local_assets()
            if self.logs_df is None:
                conn = sqlite3.connect(DB_PATH)
                try:
                    self.logs_df = pd.read_sql_query("select u,i,b,t from user_behavior_logs", conn)
                finally:
                    conn.close()

            df, summary = run_batch_diagnostics(
                pipeline=self.pipeline,
                logs_df=self.logs_df,
                sample_size=int(self.inspect_sample_size.get()),
                top_k=int(self.inspect_top_k.get()),
                random_seed=42,
            )
            self.batch_df = df
            self.batch_summary = summary
            self.inspect_summary.set(
                f"Batch: cases={summary.get('cases',0)}, pass_rate={summary.get('pass_rate',0)*100:.1f}%, avg_quality={summary.get('avg_quality_score',0):.1f}"
            )
            self._fill_inspector_table(df, include_item=True)
            self._set_status("Batch diagnostics completed.")
        except Exception as e:
            self._set_status(f"Inspector error: {e}")
            messagebox.showerror("Inspector", str(e))

    def on_export_batch(self):
        if self.batch_df is None or self.batch_summary is None or len(self.batch_df) == 0:
            messagebox.showwarning("Export", "Run batch diagnostics first.")
            return
        try:
            paths = export_inspector_results(self.batch_df, self.batch_summary, INSPECT_EXPORT_DIR, prefix="batch_diag")
            self._set_status(f"Exported: {paths.get('csv')}")
            messagebox.showinfo("Export Done", f"CSV: {paths.get('csv')}\nPNG: {paths.get('plot')}\nSummary: {paths.get('summary')}")
        except Exception as e:
            self._set_status(f"Export error: {e}")
            messagebox.showerror("Export", str(e))

    def on_diag_item(self):
        try:
            self._set_status("Running special item diagnosis...")
            self._ensure_local_assets()
            if self.logs_df is None:
                conn = sqlite3.connect(DB_PATH)
                try:
                    self.logs_df = pd.read_sql_query("select u,i,b,t from user_behavior_logs", conn)
                finally:
                    conn.close()

            df, summary = diagnose_item_across_users(
                pipeline=self.pipeline,
                logs_df=self.logs_df,
                item_id=int(self.inspect_item.get()),
                top_k=int(self.inspect_top_k.get()),
                simulate_behavior=3,
            )
            self.diag_df = df
            self.diag_summary = summary
            self.inspect_summary.set(
                f"Item {summary.get('item_id')} | users={summary.get('users')} | avg_quality={summary.get('avg_quality_score',0):.1f} | avg_rank_improve={summary.get('avg_rank_improve',0):.1f}"
            )
            self._fill_inspector_table(df, include_item=False)
            self._set_status("Special item diagnosis completed.")
        except Exception as e:
            self._set_status(f"Diagnosis error: {e}")
            messagebox.showerror("Diagnosis", str(e))

    def _fill_inspector_table(self, df: pd.DataFrame, include_item: bool):
        for row in self.inspect_tree.get_children():
            self.inspect_tree.delete(row)
        if df is None or len(df) == 0:
            return

        view = df.copy()
        if include_item and "item_id" not in view.columns:
            view["item_id"] = "-"
        if not include_item and "item_id" not in view.columns:
            view["item_id"] = int(self.inspect_item.get())

        for _, r in view.iterrows():
            self.inspect_tree.insert(
                "",
                "end",
                values=(
                    r.get("user_id", ""),
                    r.get("item_id", ""),
                    round(float(r.get("score_delta", 0) or 0), 4),
                    int(r.get("rank_improve", 0) or 0),
                    round(float(r.get("quality_score", 0) or 0), 2),
                    str(r.get("message", "")),
                ),
            )

    def on_refresh_data(self):
        try:
            self._ensure_local_assets()
            files = [
                "UserBehavior.csv",
                "final_real_data.csv",
                "final_real_data_clean.csv",
                "final_real_data_clean_strict.csv",
                "rec_system.db",
                "artifacts/stgnn_artifact_v2.pt",
            ]
            for row in self.data_tree.get_children():
                self.data_tree.delete(row)
            for name in files:
                path = os.path.join(BASE_DIR, name)
                exists = os.path.exists(path)
                size_mb = round(os.path.getsize(path) / (1024 * 1024), 2) if exists else "-"
                self.data_tree.insert("", "end", values=(name, exists, size_mb))

            conn = sqlite3.connect(DB_PATH)
            try:
                df = pd.read_sql_query("select u,i,b,t from user_behavior_logs", conn)
            finally:
                conn.close()

            model_rows = sum(len(v) for v in self.pipeline.user_hist.values())
            model_users = len(self.pipeline.idx2user)
            model_items = len(self.pipeline.idx2item)

            self.data_summary.set(
                f"DB rows/users/items = {len(df)}/{df['u'].nunique()}/{df['i'].nunique()} | "
                f"Model rows/users/items = {model_rows}/{model_users}/{model_items}"
            )
            if len(df) > model_rows or df["u"].nunique() > model_users or df["i"].nunique() > model_items:
                self._set_status("Model-DB drift detected. Retraining recommended.")
            else:
                self._set_status("Data stats refreshed.")
        except Exception as e:
            self._set_status(f"Data refresh error: {e}")
            messagebox.showerror("Data", str(e))

    def on_load_papers(self):
        try:
            for row in self.paper_tree.get_children():
                self.paper_tree.delete(row)
            if not os.path.exists(PAPER_ALIGNMENT_PATH):
                messagebox.showwarning("Papers", "paper_alignment.json not found.")
                return
            data = json.load(open(PAPER_ALIGNMENT_PATH, "r", encoding="utf-8"))
            for it in data:
                self.paper_tree.insert(
                    "",
                    "end",
                    values=(
                        it.get("paper", ""),
                        it.get("url", ""),
                        it.get("status", ""),
                        it.get("aligned_idea", ""),
                    ),
                )
            self._set_status(f"Loaded {len(data)} paper alignment entries.")
        except Exception as e:
            self._set_status(f"Load papers error: {e}")
            messagebox.showerror("Papers", str(e))

    def on_open_paper_url(self):
        sel = self.paper_tree.selection()
        if not sel:
            messagebox.showinfo("Open URL", "Select a paper row first.")
            return
        vals = self.paper_tree.item(sel[0], "values")
        url = vals[1] if len(vals) > 1 else ""
        if not url:
            messagebox.showwarning("Open URL", "No URL in selected row.")
            return
        webbrowser.open(url)

    def on_close(self):
        self.root.destroy()


def main():
    root = Tk()
    STGNNDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
