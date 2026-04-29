import os
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch

from recommender_engine import STGNNPipeline, TrainConfig


BASE_DIR = r"D:\GNN_Thesis"
DB_PATH = os.path.join(BASE_DIR, "rec_system.db")
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "stgnn_artifact_v2.pt")
EXP_DIR = os.path.join(BASE_DIR, "artifacts", "experiments")
OUT_DIR = BASE_DIR

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()
    return path


def load_df():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("select u,i,b,t from user_behavior_logs", conn)
    finally:
        conn.close()
    return df.dropna().astype({"u": int, "i": int, "b": int, "t": int})


def fig01_architecture():
    fig, ax = plt.subplots(figsize=(14, 8.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.4)
    ax.axis("off")

    layers = [
        (0.7, 6.3, 2.6, 1.2, "#e8f0fe", "输入层\n多行为序列 (u,i,b,t)"),
        (0.7, 4.8, 2.6, 1.1, "#e8f0fe", "行为语义映射\n点击/收藏/加购/购买"),
        (3.8, 5.4, 2.9, 1.5, "#e6f4ea", "时空异构图构建\n用户-商品-行为-时间"),
        (7.2, 6.4, 2.8, 1.0, "#fff4e5", "空间编码器\n异构关系传播"),
        (7.2, 5.0, 2.8, 1.0, "#f3e8fd", "时间编码器\n顺序+间隔建模"),
        (7.2, 3.6, 2.8, 1.0, "#fef3c7", "行为转移建模\nClick→Fav→Cart→Buy"),
        (10.5, 5.4, 2.8, 1.5, "#fde7e9", "时空融合门控\n动态兴趣状态 h_t"),
        (10.5, 3.7, 2.8, 1.0, "#e8f0fe", "预测层\nTop-K推荐打分"),
        (10.5, 2.2, 2.8, 1.0, "#e6f4ea", "在线更新\n新行为触发增量刷新"),
        (3.8, 2.2, 2.9, 1.0, "#fff4e5", "Inspector诊断\n评分/异常定位/专项分析"),
    ]

    for x, y, w, h, c, t in layers:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", ec="#333", fc=c, lw=1.3)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10)

    arrows = [
        ((3.3, 6.9), (3.8, 6.2)),
        ((3.3, 5.3), (3.8, 6.0)),
        ((6.7, 6.1), (7.2, 6.9)),
        ((6.7, 6.0), (7.2, 5.5)),
        ((6.7, 5.9), (7.2, 4.1)),
        ((10.0, 6.9), (10.5, 6.2)),
        ((10.0, 5.5), (10.5, 6.0)),
        ((10.0, 4.1), (10.5, 5.8)),
        ((11.9, 5.4), (11.9, 4.7)),
        ((11.9, 3.7), (11.9, 3.1)),
        ((10.5, 2.7), (6.7, 2.7)),
        ((3.8, 2.7), (2.0, 2.7)),
        ((2.0, 2.7), (2.0, 4.8)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#444"))

    ax.text(0.8, 1.1, "核心思想：统一学习“结构依赖（空间）+兴趣演化（时间）”，并支持用户新行为后的实时推荐更新。", fontsize=10, color="#444")
    return savefig("fig01_stgnn_architecture.png")


def fig02_system_flow():
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    nodes = [
        (0.7, 5.0, 2.8, 1.2, "演示工作区\n内置数据+预训练模型", "#e8f0fe"),
        (0.7, 2.9, 2.8, 1.2, "自定义工作区\n导入SQLite数据库", "#e8f0fe"),
        (4.1, 4.0, 2.9, 1.2, "数据校验与清洗\n字段校验(u,i,b,t)", "#e6f4ea"),
        (7.5, 4.0, 2.9, 1.2, "训练/加载引擎\nST-GNN Pipeline", "#fff4e5"),
        (10.9, 5.0, 2.5, 1.2, "桌面端功能层\n登录/推荐/可视化", "#f3e8fd"),
        (10.9, 2.9, 2.5, 1.2, "Inspector诊断层\n评分/导出/专项排查", "#fde7e9"),
    ]
    for x, y, w, h, t, c in nodes:
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", ec="#333", fc=c, lw=1.3)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10)

    arrows = [
        ((3.5, 5.6), (4.1, 4.6)),
        ((3.5, 3.5), (4.1, 4.4)),
        ((7.0, 4.6), (7.5, 4.6)),
        ((10.4, 4.6), (10.9, 5.6)),
        ((10.4, 4.6), (10.9, 3.5)),
        ((12.15, 5.0), (12.15, 4.1)),
        ((12.15, 2.9), (12.15, 2.1)),
        ((12.15, 2.1), (7.5, 2.1)),
        ((7.5, 2.1), (7.5, 4.0)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"))

    ax.text(7.4, 1.6, "在线反馈闭环：用户交互 → 追加行为 → 增量更新 → 推荐结果刷新", ha="center", fontsize=10, color="#555")
    ax.text(10.9, 6.5, "部署形态：源码运行 / EXE / 安装包", fontsize=10, color="#444")
    return savefig("fig02_system_pipeline.png")


def fig03_behavior_distribution(df):
    names = {0: "点击", 1: "收藏", 2: "加购", 3: "购买"}
    cnt = df["b"].value_counts().sort_index()
    labels = [names[i] for i in cnt.index]
    vals = cnt.values
    plt.figure(figsize=(8.8, 5.2))
    bars = plt.bar(labels, vals, color=["#6baed6", "#9ecae1", "#fdae6b", "#74c476"])
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{int(v):,}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("数量")
    return savefig("fig03_behavior_distribution.png")


def fig04_user_activity_hist(df):
    uc = df.groupby("u").size()
    plt.figure(figsize=(8.8, 5.2))
    plt.hist(uc.values, bins=45, color="#5dade2", edgecolor="white", alpha=0.9)
    plt.yscale("log")
    plt.xlabel("每个用户交互条数")
    plt.ylabel("用户频数（对数坐标）")
    return savefig("fig04_user_activity_distribution.png")


def fig05_item_popularity_hist(df):
    ic = df.groupby("i").size()
    plt.figure(figsize=(8.8, 5.2))
    plt.hist(ic.values, bins=55, color="#f5b041", edgecolor="white", alpha=0.9)
    plt.yscale("log")
    plt.xlabel("每个商品交互条数")
    plt.ylabel("商品频数（对数坐标）")
    return savefig("fig05_item_popularity_distribution.png")


def fig06_transition_heatmap(df):
    d = df.sort_values(["u", "t"]).copy()
    d["b_next"] = d.groupby("u")["b"].shift(-1)
    trans = d.dropna(subset=["b_next"]).copy()
    trans["b_next"] = trans["b_next"].astype(int)
    mat = np.zeros((4, 4), dtype=np.int64)
    for _, r in trans.iterrows():
        mat[int(r["b"]), int(r["b_next"])] += 1
    names = ["点击", "收藏", "加购", "购买"]
    plt.figure(figsize=(7.5, 6.2))
    plt.imshow(mat, cmap="YlOrRd")
    plt.colorbar(label="转移次数")
    plt.xticks(np.arange(4), names)
    plt.yticks(np.arange(4), names)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{mat[i,j]}", ha="center", va="center", fontsize=9, color="#222")
    plt.xlabel("下一行为")
    plt.ylabel("当前行为")
    return savefig("fig06_behavior_transition_heatmap.png")


def fig07_temporal_trend(df):
    dt = pd.to_datetime(df["t"], unit="s")
    tmp = pd.DataFrame({"dt": dt, "b": df["b"].values})
    tmp["month"] = tmp["dt"].dt.to_period("M").dt.to_timestamp()
    g = tmp.groupby(["month", "b"]).size().unstack(fill_value=0).sort_index()
    nmap = {0: "点击", 1: "收藏", 2: "加购", 3: "购买"}
    plt.figure(figsize=(10.5, 5.2))
    for b in [0, 1, 2, 3]:
        if b in g.columns:
            plt.plot(g.index, g[b], label=nmap[b], linewidth=1.8)
    plt.legend(ncol=4, fontsize=9)
    plt.xlabel("月份")
    plt.ylabel("交互次数")
    return savefig("fig07_monthly_behavior_trend.png")


def fig08_main_benchmark():
    df = pd.read_csv(os.path.join(EXP_DIR, "benchmark_main.csv"))
    use = df[df["model"].isin(["ST-GNN(ours)", "ItemCF", "BERT4Rec-lite", "LightGCN-lite", "MBA-lite", "PopRec"])].copy()
    use = use.sort_values("NDCG@10", ascending=False)
    plt.figure(figsize=(10.8, 5.4))
    x = np.arange(len(use))
    w = 0.24
    plt.bar(x - w, use["HR@10"], width=w, label="HR@10")
    plt.bar(x, use["NDCG@10"], width=w, label="NDCG@10")
    plt.bar(x + w, use["MRR@10"], width=w, label="MRR@10")
    plt.xticks(x, use["model"], rotation=15)
    plt.ylabel("指标值")
    plt.legend()
    return savefig("fig08_main_benchmark_metrics.png")


def fig09_ablation():
    df = pd.read_csv(os.path.join(EXP_DIR, "ablation.csv"))
    order = ["Full", "w/o Spatial", "w/o Temporal", "w/o Behavior"]
    df["setting"] = pd.Categorical(df["setting"], categories=order, ordered=True)
    df = df.sort_values("setting")
    plt.figure(figsize=(9.8, 5.2))
    x = np.arange(len(df))
    w = 0.24
    plt.bar(x - w, df["HR@10"], width=w, label="HR@10")
    plt.bar(x, df["NDCG@10"], width=w, label="NDCG@10")
    plt.bar(x + w, df["MRR@10"], width=w, label="MRR@10")
    plt.xticks(x, df["setting"])
    plt.ylabel("指标值")
    plt.legend()
    return savefig("fig09_ablation_metrics.png")


def fig10_item460466_diagnosis(df):
    pipe = STGNNPipeline(TrainConfig())
    pipe.load_artifact(ARTIFACT_PATH)
    uid, item, beh = 10009, 460466, 3
    base_ts = int(df[df["u"] == uid]["t"].max())
    u_idx, i_idx = pipe.user2idx[uid], pipe.item2idx[item]
    xs, rank_u, score_u, hidden_f = [], [], [], []
    for step in range(0, 8):
        seq_pack = pipe._make_seq_pack_for_user(u_idx)
        scores = pipe.model.score_user(u_idx, seq_pack)
        order = torch.argsort(scores, descending=True)
        pos = (order == i_idx).nonzero(as_tuple=False)
        rank = int(pos[0].item()) + 1 if len(pos) else scores.shape[0]
        seen = {x[0] for x in pipe.user_hist.get(u_idx, [])}
        xs.append(step)
        rank_u.append(rank)
        score_u.append(float(scores[i_idx].item()))
        hidden_f.append(int(i_idx in seen))
        pipe.append_interaction(uid, item, beh, base_ts + (step + 1) * 60)

    fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
    ax2 = ax1.twinx()
    l1 = ax1.plot(xs, rank_u, marker="o", color="#1f77b4", label="未过滤排名")
    l2 = ax2.plot(xs, score_u, marker="s", color="#d62728", label="未过滤得分")
    ax1.bar(xs, hidden_f, alpha=0.18, color="#2ca02c", width=0.5, label="过滤模式是否隐藏(0/1)")
    ax1.set_xlabel("连续追加购买行为次数")
    ax1.set_ylabel("排名（越小越好）")
    ax1.invert_yaxis()
    ax2.set_ylabel("模型打分")
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    lines.append(patches.Patch(color="#2ca02c", alpha=0.18, label="过滤模式是否隐藏(0/1)"))
    labels.append("过滤模式是否隐藏(0/1)")
    ax1.legend(lines, labels, loc="upper right", fontsize=9)
    return savefig("fig10_item460466_diagnosis.png")


def write_index(paths):
    md = os.path.join(OUT_DIR, "FIGURE_INDEX.md")
    lines = [
        "# 论文图包（10张）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "1. 图1 ST-GNN模型总体架构",
        "2. 图2 端到端落地流程",
        "3. 图3 行为类型分布",
        "4. 图4 用户活跃度分布",
        "5. 图5 商品流行度分布",
        "6. 图6 多行为转移矩阵",
        "7. 图7 月度行为趋势",
        "8. 图8 主实验基线对比",
        "9. 图9 消融实验对比",
        "10. 图10 商品460466专项诊断",
        "",
        "## 文件路径",
    ]
    for p in paths:
        lines.append(f"- {p}")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_out()
    df = load_df()
    paths = [
        fig01_architecture(),
        fig02_system_flow(),
        fig03_behavior_distribution(df),
        fig04_user_activity_hist(df),
        fig05_item_popularity_hist(df),
        fig06_transition_heatmap(df),
        fig07_temporal_trend(df),
        fig08_main_benchmark(),
        fig09_ablation(),
        fig10_item460466_diagnosis(df),
    ]
    write_index(paths)
    print("Generated figures:")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
