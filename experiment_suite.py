import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommender_engine import STGNNPipeline, TrainConfig


def leave_one_out_split(df: pd.DataFrame, min_user_inter: int = 5) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, set]]:
    df = df[["u", "i", "b", "t"]].copy().dropna()
    df = df.astype({"u": int, "i": int, "b": int, "t": int})
    df = df.sort_values(["u", "t"]).reset_index(drop=True)

    train_rows = []
    test_target = {}
    train_seen = {}

    for uid, g in df.groupby("u"):
        if len(g) < min_user_inter:
            continue
        g = g.sort_values("t")
        last = g.iloc[-1]
        train = g.iloc[:-1]
        if train.empty:
            continue

        test_target[int(uid)] = int(last["i"])
        train_seen[int(uid)] = set(train["i"].tolist())
        train_rows.append(train)

    if len(train_rows) == 0:
        raise RuntimeError("No users left after leave-one-out split.")

    train_df = pd.concat(train_rows, axis=0).reset_index(drop=True)
    return train_df, test_target, train_seen


def eval_topk(recommender, users: List[int], test_target: Dict[int, int], seen_map: Dict[int, set], k: int = 10) -> Dict[str, float]:
    hits, ndcgs, mrrs = [], [], []

    for u in users:
        gt = test_target[u]
        seen = seen_map[u]
        # Keep GT candidate even if it appeared in history (repeat-purchase case).
        # Otherwise evaluation can incorrectly mask the true target item.
        seen_for_filter = set(seen)
        if gt in seen_for_filter:
            seen_for_filter.remove(gt)
        rec = recommender.recommend(u, seen_for_filter, k)

        if gt in rec:
            rank = rec.index(gt) + 1
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))
            mrrs.append(1.0 / rank)
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
            mrrs.append(0.0)

    return {
        f"HR@{k}": float(np.mean(hits)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        f"MRR@{k}": float(np.mean(mrrs)),
        "users": int(len(users)),
    }


class PopularRec:
    name = "PopRec"

    def fit(self, train_df: pd.DataFrame):
        self.ranking = train_df.groupby("i").size().sort_values(ascending=False).index.tolist()

    def recommend(self, user_id: int, seen: set, k: int) -> List[int]:
        out = []
        for i in self.ranking:
            if i in seen:
                continue
            out.append(int(i))
            if len(out) >= k:
                break
        return out


class ItemCFRec:
    name = "ItemCF"

    def fit(self, train_df: pd.DataFrame):
        self.pop = train_df.groupby("i").size().sort_values(ascending=False)
        self.cooc = {}
        self.user_seq = {}
        train_df = train_df.sort_values(["u", "t"])

        for u, g in train_df.groupby("u"):
            items = g["i"].tolist()
            self.user_seq[int(u)] = items
            uniq = list(dict.fromkeys(items))
            for i in uniq:
                self.cooc.setdefault(i, {})
            for i in uniq:
                for j in uniq:
                    if i == j:
                        continue
                    self.cooc[i][j] = self.cooc[i].get(j, 0.0) + 1.0

    def recommend(self, user_id: int, seen: set, k: int) -> List[int]:
        hist = self.user_seq.get(int(user_id), [])
        score = {}
        for i in hist[-20:]:
            neigh = self.cooc.get(i, {})
            for j, w in neigh.items():
                if j in seen:
                    continue
                score[j] = score.get(j, 0.0) + w

        if not score:
            return [int(i) for i in self.pop.index.tolist() if i not in seen][:k]
        return [int(x[0]) for x in sorted(score.items(), key=lambda x: x[1], reverse=True)[:k]]


class MBALiteRec:
    name = "MBA-lite"

    def fit(self, train_df: pd.DataFrame):
        # behavior-aware transition score
        w = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        self.trans = {}
        self.user_seq = {}
        self.pop = train_df.groupby("i").size().sort_values(ascending=False)
        train_df = train_df.sort_values(["u", "t"])

        for u, g in train_df.groupby("u"):
            events = list(zip(g["i"].tolist(), g["b"].tolist(), g["t"].tolist()))
            self.user_seq[int(u)] = events
            for p in range(1, len(events)):
                i1, b1, t1 = events[p - 1]
                i2, b2, t2 = events[p]
                if i1 == i2:
                    continue
                dt = max(1, t2 - t1)
                val = (w.get(b1, 1.0) + w.get(b2, 1.0)) / (1.0 + np.log1p(dt))
                self.trans.setdefault(i1, {})
                self.trans[i1][i2] = self.trans[i1].get(i2, 0.0) + val

    def recommend(self, user_id: int, seen: set, k: int) -> List[int]:
        events = self.user_seq.get(int(user_id), [])
        score = {}
        for i, b, _ in events[-20:]:
            for j, w in self.trans.get(i, {}).items():
                if j in seen:
                    continue
                score[j] = score.get(j, 0.0) + (1.0 + 0.2 * b) * w

        if not score:
            return [int(i) for i in self.pop.index.tolist() if i not in seen][:k]
        return [int(x[0]) for x in sorted(score.items(), key=lambda x: x[1], reverse=True)[:k]]


class BERT4RecLiteRec:
    name = "BERT4Rec-lite"

    def fit(self, train_df: pd.DataFrame):
        # lightweight masked-sequence proxy via position-weighted transition
        self.trans = {}
        self.user_seq = {}
        self.pop = train_df.groupby("i").size().sort_values(ascending=False)

        train_df = train_df.sort_values(["u", "t"])
        for u, g in train_df.groupby("u"):
            items = g["i"].tolist()
            self.user_seq[int(u)] = items
            L = len(items)
            for p in range(L):
                for q in range(max(0, p - 5), p):
                    i_prev = items[q]
                    i_cur = items[p]
                    if i_prev == i_cur:
                        continue
                    dist = p - q
                    w = 1.0 / dist
                    self.trans.setdefault(i_prev, {})
                    self.trans[i_prev][i_cur] = self.trans[i_prev].get(i_cur, 0.0) + w

    def recommend(self, user_id: int, seen: set, k: int) -> List[int]:
        items = self.user_seq.get(int(user_id), [])
        score = {}
        for pos, i in enumerate(items[-10:]):
            alpha = (pos + 1) / max(1, len(items[-10:]))
            for j, w in self.trans.get(i, {}).items():
                if j in seen:
                    continue
                score[j] = score.get(j, 0.0) + alpha * w

        if not score:
            return [int(i) for i in self.pop.index.tolist() if i not in seen][:k]
        return [int(x[0]) for x in sorted(score.items(), key=lambda x: x[1], reverse=True)[:k]]


class LightGCNLiteRec:
    name = "LightGCN-lite"

    def __init__(self, embed_dim=48, epochs=3, lr=1e-2):
        self.embed_dim = embed_dim
        self.epochs = epochs
        self.lr = lr

    def fit(self, train_df: pd.DataFrame):
        train_df = train_df[["u", "i"]].copy()
        users = sorted(train_df["u"].unique().tolist())
        items = sorted(train_df["i"].unique().tolist())
        self.u2idx = {u: idx for idx, u in enumerate(users)}
        self.i2idx = {i: idx for idx, i in enumerate(items)}
        self.idx2item = items

        train_df["u_idx"] = train_df["u"].map(self.u2idx)
        train_df["i_idx"] = train_df["i"].map(self.i2idx)

        self.user_seen = {int(u): set(g["i"].tolist()) for u, g in train_df.groupby("u")}

        n_u, n_i = len(users), len(items)
        self.u_emb = nn.Embedding(n_u, self.embed_dim)
        self.i_emb = nn.Embedding(n_i, self.embed_dim)
        opt = torch.optim.Adam(list(self.u_emb.parameters()) + list(self.i_emb.parameters()), lr=self.lr)

        rows = train_df["u_idx"].values
        pos = train_df["i_idx"].values

        for _ in range(self.epochs):
            idx = np.random.permutation(len(rows))
            for s in range(0, len(idx), 512):
                b = idx[s:s+512]
                u = torch.tensor(rows[b], dtype=torch.long)
                p = torch.tensor(pos[b], dtype=torch.long)
                n = torch.randint(0, n_i, (len(b),), dtype=torch.long)

                ue = self.u_emb(u)
                pe = self.i_emb(p)
                ne = self.i_emb(n)

                loss = -F.logsigmoid((ue * pe).sum(-1) - (ue * ne).sum(-1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.pop = train_df.groupby("i").size().sort_values(ascending=False)

    @torch.no_grad()
    def recommend(self, user_id: int, seen: set, k: int) -> List[int]:
        if user_id not in self.u2idx:
            return [int(i) for i in self.pop.index.tolist() if i not in seen][:k]

        u_idx = self.u2idx[user_id]
        score = torch.matmul(self.i_emb.weight, self.u_emb.weight[u_idx])
        top = torch.topk(score, k=min(len(self.idx2item), k + len(seen) + 20)).indices.tolist()

        out = []
        for i_idx in top:
            item_id = int(self.idx2item[i_idx])
            if item_id in seen:
                continue
            out.append(item_id)
            if len(out) >= k:
                break

        if len(out) < k:
            for i in self.pop.index.tolist():
                if i in seen or i in out:
                    continue
                out.append(int(i))
                if len(out) >= k:
                    break
        return out


class STGNNRecWrapper:
    name = "ST-GNN(ours)"

    def __init__(self, epochs=3, embed_dim=64, max_seq_len=30, use_spatial=True, use_temporal=True, use_behavior=True, seed=42):
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal
        self.use_behavior = use_behavior
        self.seed = seed

    def fit(self, train_df: pd.DataFrame):
        cfg = TrainConfig(
            epochs=self.epochs,
            embed_dim=self.embed_dim,
            max_seq_len=self.max_seq_len,
            batch_size=256,
            seed=self.seed,
            use_spatial=self.use_spatial,
            use_temporal=self.use_temporal,
            use_behavior=self.use_behavior,
        )
        self.pipe = STGNNPipeline(cfg)
        self.pipe.prepare_from_df(train_df)
        self.pipe.train(verbose=False)

    def recommend(self, user_id: int, seen: set, k: int) -> List[int]:
        # Do not apply internal seen-filter here; enforce the external filter set
        # from eval to keep behavior consistent across all models.
        rec = self.pipe.recommend_for_raw_user(int(user_id), top_k=k + len(seen) + 100, filter_seen=False)
        out = []
        for item_id in rec["item_id"].astype(int).tolist():
            if item_id in seen:
                continue
            out.append(item_id)
            if len(out) >= k:
                break
        return out


def run_main_bench(
    train_df: pd.DataFrame,
    users: List[int],
    test_target: Dict[int, int],
    seen_map: Dict[int, set],
    topk: int,
    out_dir: Path,
    stgnn_epochs: int,
    lightgcn_epochs: int,
):
    models = [
        PopularRec(),
        ItemCFRec(),
        BERT4RecLiteRec(),
        LightGCNLiteRec(embed_dim=48, epochs=lightgcn_epochs),
        MBALiteRec(),
        STGNNRecWrapper(epochs=stgnn_epochs, embed_dim=64, max_seq_len=30),
    ]

    rows = []
    for model in models:
        print(f"[RUN] {model.name}")
        model.fit(train_df)
        metric = eval_topk(model, users, test_target, seen_map, k=topk)
        row = {"model": model.name, **metric}
        rows.append(row)
        print(f"[METRIC] {row}")

    df_res = pd.DataFrame(rows).sort_values(f"NDCG@{topk}", ascending=False)
    out = out_dir / "benchmark_main.csv"
    df_res.to_csv(out, index=False)
    return df_res


def run_ablation(
    train_df: pd.DataFrame,
    users: List[int],
    test_target: Dict[int, int],
    seen_map: Dict[int, set],
    topk: int,
    out_dir: Path,
    stgnn_epochs: int,
):
    settings = [
        ("Full", dict(use_spatial=True, use_temporal=True, use_behavior=True)),
        ("w/o Temporal", dict(use_spatial=True, use_temporal=False, use_behavior=True)),
        ("w/o Spatial", dict(use_spatial=False, use_temporal=True, use_behavior=True)),
        ("w/o Behavior", dict(use_spatial=True, use_temporal=True, use_behavior=False)),
    ]

    rows = []
    for name, cfg in settings:
        print(f"[ABL] {name}")
        m = STGNNRecWrapper(epochs=stgnn_epochs, embed_dim=64, max_seq_len=30, **cfg)
        m.fit(train_df)
        metric = eval_topk(m, users, test_target, seen_map, k=topk)
        rows.append({"setting": name, **metric})

    df_res = pd.DataFrame(rows).sort_values(f"NDCG@{topk}", ascending=False)
    out = out_dir / "ablation.csv"
    df_res.to_csv(out, index=False)
    return df_res


def run_sensitivity(
    train_df: pd.DataFrame,
    users: List[int],
    test_target: Dict[int, int],
    seen_map: Dict[int, set],
    topk: int,
    out_dir: Path,
    stgnn_epochs: int,
):
    rows = []
    for emb in [32, 64, 96]:
        m = STGNNRecWrapper(epochs=stgnn_epochs, embed_dim=emb, max_seq_len=30)
        m.fit(train_df)
        metric = eval_topk(m, users, test_target, seen_map, k=topk)
        rows.append({"param": "embed_dim", "value": emb, **metric})

    for L in [10, 20, 30, 50]:
        m = STGNNRecWrapper(epochs=stgnn_epochs, embed_dim=64, max_seq_len=L)
        m.fit(train_df)
        metric = eval_topk(m, users, test_target, seen_map, k=topk)
        rows.append({"param": "max_seq_len", "value": L, **metric})

    df_res = pd.DataFrame(rows)
    out = out_dir / "sensitivity.csv"
    df_res.to_csv(out, index=False)
    return df_res


def plot_results(out_dir: Path, topk: int):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available, skip plots")
        return

    main_csv = out_dir / "benchmark_main.csv"
    if main_csv.exists():
        d = pd.read_csv(main_csv)
        col = f"NDCG@{topk}"
        if col in d.columns:
            plt.figure(figsize=(10, 5))
            plt.bar(d["model"], d[col])
            plt.xticks(rotation=30, ha="right")
            plt.title(f"Main Benchmark NDCG@{topk}")
            plt.tight_layout()
            plt.savefig(out_dir / "benchmark_main_ndcg.png", dpi=150)
            plt.close()

    abl_csv = out_dir / "ablation.csv"
    if abl_csv.exists():
        d = pd.read_csv(abl_csv)
        col = f"NDCG@{topk}"
        if col in d.columns:
            plt.figure(figsize=(8, 5))
            plt.bar(d["setting"], d[col])
            plt.xticks(rotation=20, ha="right")
            plt.title(f"Ablation NDCG@{topk}")
            plt.tight_layout()
            plt.savefig(out_dir / "ablation_ndcg.png", dpi=150)
            plt.close()

    sens_csv = out_dir / "sensitivity.csv"
    if sens_csv.exists():
        d = pd.read_csv(sens_csv)
        col = f"NDCG@{topk}"
        if col not in d.columns:
            return
        for p in d["param"].unique():
            sub = d[d["param"] == p].sort_values("value")
            plt.figure(figsize=(6, 4))
            plt.plot(sub["value"], sub[col], marker="o")
            plt.title(f"Sensitivity: {p}")
            plt.xlabel(p)
            plt.ylabel(f"NDCG@{topk}")
            plt.tight_layout()
            plt.savefig(out_dir / f"sensitivity_{p}.png", dpi=150)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark/ablation/sensitivity for recommendation models")
    parser.add_argument("--input", type=str, default="final_real_data_clean.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts/experiments")
    parser.add_argument("--sample-users", type=int, default=300)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--stgnn-epochs", type=int, default=2)
    parser.add_argument("--lightgcn-epochs", type=int, default=2)
    parser.add_argument("--train-users", type=int, default=0, help="limit training to sampled users for fast demo")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.input).resolve())
    train_df, test_target, seen_map = leave_one_out_split(df, min_user_inter=5)

    users = sorted(test_target.keys())
    if args.sample_users > 0 and len(users) > args.sample_users:
        rng = np.random.default_rng(42)
        users = rng.choice(users, size=args.sample_users, replace=False).tolist()

    if args.train_users > 0:
        all_train_users = sorted(train_df["u"].unique().tolist())
        rng = np.random.default_rng(123)
        keep_users = set(users)
        extra_need = max(0, args.train_users - len(keep_users))
        if extra_need > 0:
            candidates = [u for u in all_train_users if u not in keep_users]
            if candidates:
                sampled = rng.choice(candidates, size=min(extra_need, len(candidates)), replace=False).tolist()
                keep_users.update(sampled)
        train_df = train_df[train_df["u"].isin(list(keep_users))].reset_index(drop=True)

    train_items = set(train_df["i"].unique().tolist())
    users = [u for u in users if test_target[u] in train_items]
    if not users:
        raise RuntimeError("No evaluable users left. Increase --train-users or disable aggressive filtering.")

    print(f"[INFO] train_rows={len(train_df)}, eval_users={len(users)}")

    main_res = run_main_bench(
        train_df,
        users,
        test_target,
        seen_map,
        args.topk,
        out_dir,
        stgnn_epochs=args.stgnn_epochs,
        lightgcn_epochs=args.lightgcn_epochs,
    )

    abl_res = pd.DataFrame()
    if not args.skip_ablation:
        abl_res = run_ablation(
            train_df,
            users,
            test_target,
            seen_map,
            args.topk,
            out_dir,
            stgnn_epochs=args.stgnn_epochs,
        )

    sens_res = pd.DataFrame()
    if not args.skip_sensitivity:
        sens_res = run_sensitivity(
            train_df,
            users,
            test_target,
            seen_map,
            args.topk,
            out_dir,
            stgnn_epochs=args.stgnn_epochs,
        )

    summary = {
        "topk": args.topk,
        "main_best": main_res.iloc[0].to_dict() if len(main_res) else {},
        "ablation_best": abl_res.iloc[0].to_dict() if len(abl_res) else {},
        "sensitivity_points": int(len(sens_res)),
        "num_eval_users": len(users),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_results(out_dir, args.topk)

    print("[OK] Experiment finished")
    print(f"[OK] outputs: {out_dir}")


if __name__ == "__main__":
    main()
