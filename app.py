import json
import os
import sqlite3
import time

import pandas as pd
import streamlit as st

from qa_tool import (
    diagnose_item_across_users,
    export_inspector_results,
    run_batch_diagnostics,
    run_single_interaction_check,
)
from recommender_engine import BEHAVIOR_NAME, STGNNPipeline, TrainConfig, load_behavior_df


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rec_system.db")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "stgnn_artifact_v2.pt")
DATA_REPORT_PATH = os.path.join(ARTIFACT_DIR, "data_quality_report.json")
EXP_DIR = os.path.join(ARTIFACT_DIR, "experiments")
PAPER_ALIGNMENT_PATH = os.path.join(BASE_DIR, "paper_alignment.json")
INSPECT_EXPORT_DIR = os.path.join(ARTIFACT_DIR, "inspector_exports")

st.set_page_config(page_title="ST-GNN Recommender", layout="wide")
st.title("Spatio-Temporal Multi-Behavior Recommender")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> STGNNPipeline:
    pipe = STGNNPipeline(TrainConfig())

    if os.path.exists(MODEL_ARTIFACT_PATH):
        try:
            pipe.load_artifact(MODEL_ARTIFACT_PATH)
            return pipe
        except Exception:
            pass

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    strict_csv = os.path.join(BASE_DIR, "final_real_data_clean_strict.csv")
    if os.path.exists(strict_csv):
        df = pd.read_csv(strict_csv)
    else:
        df = load_behavior_df(DB_PATH)

    pipe.config.epochs = 4
    pipe.config.embed_dim = 64
    pipe.config.batch_size = 256
    pipe.prepare_from_df(df)
    pipe.train(verbose=False)
    pipe.save_artifact(MODEL_ARTIFACT_PATH)
    return pipe


@st.cache_data(show_spinner=False, ttl=5)
def load_logs() -> pd.DataFrame:
    return load_behavior_df(DB_PATH).sort_values(["u", "t"]).reset_index(drop=True)


def append_to_db(u: int, i: int, b: int, t: int):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT INTO user_behavior_logs (u, i, b, t) VALUES (?, ?, ?, ?)", (int(u), int(i), int(b), int(t)))
        conn.commit()
    finally:
        conn.close()


def behavior_text(b: int) -> str:
    return BEHAVIOR_NAME.get(int(b), f"Unknown({b})")


def draw_interest_trend(user_df: pd.DataFrame):
    if user_df.empty:
        st.info("No interaction history for this user.")
        return

    x = user_df.copy()
    x["datetime"] = pd.to_datetime(x["t"], unit="s")
    x["day"] = x["datetime"].dt.date.astype(str)
    x["behavior_name"] = x["b"].apply(behavior_text)

    trend = x.groupby(["day", "behavior_name"]).size().reset_index(name="count").sort_values("day")
    st.line_chart(trend, x="day", y="count", color="behavior_name")


def draw_user_purchase_profile(user_df: pd.DataFrame):
    if user_df.empty:
        st.info("No records for this user.")
        return

    x = user_df.copy()
    x["datetime"] = pd.to_datetime(x["t"], unit="s")

    behavior_dist = x["b"].value_counts().sort_index().rename(index=BEHAVIOR_NAME)
    st.markdown("#### Behavior Distribution")
    st.bar_chart(behavior_dist)

    buy = x[x["b"] == 3].copy()
    if buy.empty:
        st.info("This user has no buy records yet.")
        return

    summary = (
        buy.groupby("i")
        .agg(
            buy_count=("i", "size"),
            first_buy=("datetime", "min"),
            last_buy=("datetime", "max"),
        )
        .reset_index()
        .sort_values(["buy_count", "last_buy"], ascending=[False, False])
    )

    st.markdown("#### Top Purchased Items")
    topn = summary.head(15).copy()
    chart_df = topn[["i", "buy_count"]].rename(columns={"i": "item_id"})
    st.bar_chart(chart_df.set_index("item_id"))

    view_df = topn.copy()
    view_df["first_buy"] = view_df["first_buy"].dt.strftime("%Y-%m-%d %H:%M:%S")
    view_df["last_buy"] = view_df["last_buy"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(view_df.rename(columns={"i": "item_id"}), use_container_width=True, hide_index=True)


def ensure_session_profile(selected_user: int, user_hist: pd.DataFrame):
    if st.session_state.get("active_user") != selected_user:
        st.session_state["active_user"] = selected_user
        st.session_state["session_history"] = user_hist[["u", "i", "b", "t"]].copy()


def merge_session_history(base_df: pd.DataFrame) -> pd.DataFrame:
    extra = st.session_state.get("session_history", pd.DataFrame(columns=["u", "i", "b", "t"]))
    if extra.empty:
        return base_df

    merged = pd.concat([base_df, extra], axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["u", "i", "b", "t"], keep="last")
    merged = merged.sort_values(["u", "t"]).reset_index(drop=True)
    return merged


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def show_paper_alignment():
    data = read_json(PAPER_ALIGNMENT_PATH)
    if not data:
        st.info("No paper alignment file found.")
        return
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def file_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def count_lines_fast(path: str, chunk_size: int = 1024 * 1024) -> int:
    if not os.path.exists(path):
        return 0
    cnt = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            cnt += data.count(b"\n")
    return cnt


def show_demo_page(pipeline: STGNNPipeline, logs_df: pd.DataFrame):
    user_ids = sorted(pipeline.idx2user)
    if not user_ids:
        st.error("No users found for demo.")
        return

    col_left, col_right = st.columns([1.1, 2.2])

    with col_left:
        st.subheader("User Interaction")
        selected_user = st.selectbox("User ID", options=user_ids, index=0)

        user_hist_df = logs_df[logs_df["u"] == selected_user].copy()
        ensure_session_profile(selected_user, user_hist_df)

        merged_logs = merge_session_history(logs_df)
        current_user_hist = merged_logs[merged_logs["u"] == selected_user].sort_values("t")

        st.metric("History Size", len(current_user_hist))
        if not current_user_hist.empty:
            show_hist = current_user_hist.tail(10).copy()
            show_hist["time"] = pd.to_datetime(show_hist["t"], unit="s")
            show_hist["behavior"] = show_hist["b"].apply(behavior_text)
            st.dataframe(show_hist[["i", "behavior", "time"]], use_container_width=True)

        st.markdown("### Simulate New Interaction")
        default_item = int(current_user_hist["i"].iloc[-1]) if len(current_user_hist) else int(pipeline.idx2item[0])
        item_id_input = st.number_input("Item ID", min_value=1, value=default_item, step=1)

        behavior_label_to_id = {
            "Click": 0,
            "Favorite": 1,
            "Cart": 2,
            "Buy": 3,
        }
        behavior_label = st.selectbox("Behavior", options=list(behavior_label_to_id.keys()), index=0)

        if st.button("Submit & Refresh Recs", type="primary", use_container_width=True):
            ts = int(time.time())
            b = behavior_label_to_id[behavior_label]

            append_to_db(selected_user, int(item_id_input), b, ts)

            session_hist = st.session_state.get("session_history", pd.DataFrame(columns=["u", "i", "b", "t"]))
            st.session_state["session_history"] = pd.concat(
                [session_hist, pd.DataFrame([{"u": selected_user, "i": int(item_id_input), "b": b, "t": ts}])],
                ignore_index=True,
            )

            if int(item_id_input) not in set(pipeline.idx2item):
                st.warning("Item is out of current model vocabulary. It is stored in DB and will be used after retraining.")
            pipeline.append_interaction(selected_user, int(item_id_input), b, ts)

            st.success("Interaction accepted, recommendation refreshed.")
            st.rerun()

    with col_right:
        st.subheader("Top-K Recommendations")
        include_seen = st.checkbox("Include seen items (for repurchase inspection)", value=False)
        st.caption("If disabled, items already interacted by the user are filtered out from recommendation list.")
        rec_df = pipeline.recommend_for_raw_user(int(selected_user), top_k=10, filter_seen=not include_seen)

        rec_show = rec_df.copy()
        rec_show["score"] = rec_show["score"].map(lambda x: f"{x:.4f}")
        st.dataframe(rec_show, use_container_width=True, hide_index=True)

        st.markdown("### Dynamic Interest Trend")
        merged_logs = merge_session_history(logs_df)
        current_user_hist = merged_logs[merged_logs["u"] == selected_user].sort_values("t")
        draw_interest_trend(current_user_hist)

        with st.expander("User Purchase Profile (DB View)", expanded=False):
            draw_user_purchase_profile(current_user_hist)


def show_inspector_page(pipeline: STGNNPipeline, logs_df: pd.DataFrame):
    st.subheader("Model Inspector")
    st.caption("Auto-check whether recommendation score/rank responds to new interactions.")

    user_ids = sorted(pipeline.idx2user)
    if not user_ids:
        st.warning("No users in model vocabulary.")
        return

    left, right = st.columns([1.1, 1.9])

    with left:
        user_id = st.selectbox("Inspector User", options=user_ids, index=0)
        user_logs = logs_df[logs_df["u"] == user_id]
        default_item = int(user_logs["i"].iloc[-1]) if len(user_logs) else int(pipeline.idx2item[0])
        item_id = st.number_input("Inspector Item", min_value=1, value=default_item, step=1)

        behavior_map = {"Click": 0, "Favorite": 1, "Cart": 2, "Buy": 3}
        behavior = behavior_map[st.selectbox("Inspector Behavior", list(behavior_map.keys()), index=3)]
        top_k = st.slider("Top-K for check", min_value=5, max_value=50, value=10, step=5)

        if st.button("Run Single Check", type="primary", use_container_width=True):
            result = run_single_interaction_check(
                pipeline=pipeline,
                user_id=int(user_id),
                item_id=int(item_id),
                behavior=int(behavior),
                top_k=int(top_k),
            )
            st.session_state["single_check_result"] = result

        sample_n = st.slider("Batch sample size", min_value=10, max_value=100, value=30, step=10)
        if st.button("Run Batch Diagnostics", use_container_width=True):
            detail_df, summary = run_batch_diagnostics(
                pipeline=pipeline,
                logs_df=logs_df,
                sample_size=int(sample_n),
                top_k=int(top_k),
                random_seed=42,
            )
            st.session_state["batch_diag_df"] = detail_df
            st.session_state["batch_diag_summary"] = summary

        if st.button("Export Batch Result (CSV + PNG)", use_container_width=True):
            detail_df = st.session_state.get("batch_diag_df")
            summary = st.session_state.get("batch_diag_summary")
            if detail_df is None or summary is None or len(detail_df) == 0:
                st.warning("Run batch diagnostics first.")
            else:
                paths = export_inspector_results(
                    detail_df=detail_df,
                    summary=summary,
                    out_dir=INSPECT_EXPORT_DIR,
                    prefix="batch_diag",
                )
                st.session_state["batch_export_paths"] = paths

        st.markdown("---")
        diag_item_id = st.number_input("Special Item ID", min_value=1, value=460466, step=1)
        if st.button("Run Special Item Diagnosis", use_container_width=True):
            diag_df, diag_summary = diagnose_item_across_users(
                pipeline=pipeline,
                logs_df=logs_df,
                item_id=int(diag_item_id),
                top_k=int(top_k),
                simulate_behavior=3,
            )
            st.session_state["diag_item_df"] = diag_df
            st.session_state["diag_item_summary"] = diag_summary

    with right:
        st.markdown("### Single Check Report")
        result = st.session_state.get("single_check_result")
        if result is None:
            st.info("No single-check result yet.")
        else:
            st.metric("Quality Score", f"{result.quality_score:.1f}/100")
            st.write(result.message)
            single_df = pd.DataFrame(
                [
                    {
                        "user_id": result.user_id,
                        "item_id": result.item_id,
                        "behavior": behavior_text(result.behavior),
                        "before_score": result.before_score,
                        "after_score": result.after_score,
                        "score_delta": result.score_delta,
                        "before_rank_unfiltered": result.before_rank_unfiltered,
                        "after_rank_unfiltered": result.after_rank_unfiltered,
                        "rank_improve": result.rank_improve,
                        "in_topk_before_unfiltered": result.in_topk_before_unfiltered,
                        "in_topk_after_unfiltered": result.in_topk_after_unfiltered,
                        "in_topk_before_filtered": result.in_topk_before_filtered,
                        "in_topk_after_filtered": result.in_topk_after_filtered,
                    }
                ]
            )
            st.dataframe(single_df, use_container_width=True, hide_index=True)

        st.markdown("### Batch Diagnostics")
        summary = st.session_state.get("batch_diag_summary")
        detail_df = st.session_state.get("batch_diag_df")
        if summary is None:
            st.info("No batch diagnostics yet.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Cases", summary["cases"])
            c2.metric("Pass Rate", f"{summary['pass_rate'] * 100:.1f}%")
            c3.metric("Avg Quality", f"{summary['avg_quality_score']:.1f}")

            st.caption("Pass condition: score increases OR rank improves after simulated interaction.")
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
            export_paths = st.session_state.get("batch_export_paths")
            if export_paths:
                st.success("Batch results exported.")
                st.write(export_paths)

        st.markdown("### Special Item Diagnosis")
        diag_summary = st.session_state.get("diag_item_summary")
        diag_df = st.session_state.get("diag_item_df")
        if diag_summary is None:
            st.info("No special-item diagnosis yet.")
        else:
            st.json(diag_summary)
            if diag_df is not None and len(diag_df):
                st.dataframe(diag_df, use_container_width=True, hide_index=True)


def show_data_page(logs_df: pd.DataFrame, pipeline: STGNNPipeline):
    st.subheader("Data Overview")

    files = [
        "UserBehavior.csv",
        "final_real_data.csv",
        "final_real_data_clean.csv",
        "final_real_data_clean_strict.csv",
        "rec_system.db",
    ]

    rows = []
    for name in files:
        path = os.path.join(BASE_DIR, name)
        exists = os.path.exists(path)
        rows.append(
            {
                "file": name,
                "exists": exists,
                "size_mb": round(file_size_mb(path), 2) if exists else None,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("DB rows(user_behavior_logs)", int(len(logs_df)))
    c2.metric("DB users", int(logs_df["u"].nunique()))
    c3.metric("DB items", int(logs_df["i"].nunique()))

    model_rows = sum(len(v) for v in pipeline.user_hist.values())
    model_users = len(pipeline.idx2user)
    model_items = len(pipeline.idx2item)
    d1, d2, d3 = st.columns(3)
    d1.metric("Model history rows", int(model_rows))
    d2.metric("Model users", int(model_users))
    d3.metric("Model items", int(model_items))

    if len(logs_df) > model_rows or logs_df["u"].nunique() > model_users or logs_df["i"].nunique() > model_items:
        st.warning("Model-DB drift detected: DB has newer or wider interactions than current model artifact. Retraining is recommended.")

    if st.button("Count lines in UserBehavior.csv (may take some time)"):
        path = os.path.join(BASE_DIR, "UserBehavior.csv")
        if not os.path.exists(path):
            st.error("UserBehavior.csv not found")
        else:
            with st.spinner("Counting lines..."):
                line_count = count_lines_fast(path)
            st.success(f"UserBehavior.csv lines: {line_count}")


def show_defense_page():
    st.subheader("Thesis Defense Board")

    st.markdown("#### Model Summary")
    st.markdown(
        "- Spatial branch: user-item graph + item transition graph\n"
        "- Temporal branch: behavior sequence encoder + time interval encoding\n"
        "- Multi-behavior branch: click/favorite/cart/buy semantic weighting"
    )

    st.markdown("#### Data Quality Report")
    report = read_json(DATA_REPORT_PATH)
    if report is None:
        st.warning("No report found. Run: python data_quality.py")
    else:
        before = report["before"]
        after = report["after"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Rows", before["rows"])
        c2.metric("Clean Rows", after["rows"])
        c3.metric("Raw Users", before["num_users"])
        st.json(report["recommendation"])

    st.markdown("#### Experiment Results")
    main_csv = os.path.join(EXP_DIR, "benchmark_main.csv")
    abl_csv = os.path.join(EXP_DIR, "ablation.csv")
    sens_csv = os.path.join(EXP_DIR, "sensitivity.csv")

    if os.path.exists(main_csv):
        st.markdown("Main Benchmark")
        st.dataframe(pd.read_csv(main_csv), use_container_width=True, hide_index=True)
        main_png = os.path.join(EXP_DIR, "benchmark_main_ndcg.png")
        if os.path.exists(main_png):
            st.image(main_png)

    if os.path.exists(abl_csv):
        st.markdown("Ablation")
        st.dataframe(pd.read_csv(abl_csv), use_container_width=True, hide_index=True)
        abl_png = os.path.join(EXP_DIR, "ablation_ndcg.png")
        if os.path.exists(abl_png):
            st.image(abl_png)

    if os.path.exists(sens_csv):
        st.markdown("Sensitivity")
        st.dataframe(pd.read_csv(sens_csv), use_container_width=True, hide_index=True)

    st.markdown("#### Paper-Aligned Landing")
    show_paper_alignment()


pipeline = load_pipeline()
logs_df = load_logs()

mode = st.sidebar.radio("Mode", ["Demo", "Inspector", "Data", "Defense"], index=0)
if mode == "Demo":
    show_demo_page(pipeline, logs_df)
elif mode == "Inspector":
    show_inspector_page(pipeline, logs_df)
elif mode == "Data":
    show_data_page(logs_df, pipeline)
else:
    show_defense_page()
