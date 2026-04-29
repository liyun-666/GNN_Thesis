from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


BEHAVIOR_NAME = {
    0: "click",
    1: "favorite",
    2: "cart",
    3: "buy",
}

DEFAULT_BEHAVIOR_WEIGHT = {
    0: 1.0,
    1: 1.25,
    2: 1.6,
    3: 2.1,
}


@dataclass
class TrainConfig:
    embed_dim: int = 64
    max_seq_len: int = 30
    num_layers: int = 2
    dropout: float = 0.15
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 6
    seed: int = 42

    # Ablation switches
    use_spatial: bool = True
    use_temporal: bool = True
    use_behavior: bool = True


class TimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        x = torch.log1p(delta_days.unsqueeze(-1))
        return self.fc(x)


class STGNNRec(nn.Module):
    """Spatio-Temporal Multi-Behavior Recommendation model."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embed_dim: int,
        item_adj: torch.Tensor,
        ui_norm: torch.Tensor,
        config: TrainConfig,
        num_behaviors: int = 4,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.config = config

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.behavior_embedding = nn.Embedding(num_behaviors, embed_dim)
        self.time_encoder = TimeEncoder(embed_dim)
        self.seq_encoder = nn.GRU(embed_dim, embed_dim, batch_first=True)

        self.gnn_item_linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(config.num_layers)])
        self.gnn_user_linears = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer("item_adj", item_adj)
        self.register_buffer("ui_norm", ui_norm)

    def spatial_propagation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.user_embedding.weight
        i = self.item_embedding.weight

        if not self.config.use_spatial:
            return u, i

        for l in range(self.config.num_layers):
            agg_u = torch.sparse.mm(self.ui_norm, i)
            agg_i_trans = torch.sparse.mm(self.item_adj, i)
            agg_i_ui = torch.sparse.mm(self.ui_norm.transpose(0, 1), u)

            u = F.relu(self.gnn_user_linears[l](u + agg_u))
            i = F.relu(self.gnn_item_linears[l](i + agg_i_trans + agg_i_ui))

            if l != self.config.num_layers - 1:
                u = self.dropout(u)
                i = self.dropout(i)

        return u, i

    def encode_sequence(
        self,
        seq_items: torch.Tensor,
        seq_behaviors: torch.Tensor,
        seq_delta_days: torch.Tensor,
        lengths: torch.Tensor,
        item_repr: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.use_temporal:
            return torch.zeros((seq_items.shape[0], self.embed_dim), dtype=torch.float32, device=seq_items.device)

        seq_item_emb = item_repr[seq_items]
        if self.config.use_behavior:
            # Moderate behavior injection to avoid overpowering item/time signals.
            seq_beh_emb = 0.35 * self.behavior_embedding(seq_behaviors)
        else:
            seq_beh_emb = torch.zeros_like(seq_item_emb)

        seq_time_emb = self.time_encoder(seq_delta_days)
        seq_x = self.layer_norm(seq_item_emb + seq_beh_emb + seq_time_emb)

        packed = nn.utils.rnn.pack_padded_sequence(
            seq_x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h = self.seq_encoder(packed)
        return h[-1]

    def forward_train(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        user_repr, item_repr = self.spatial_propagation()

        seq_repr = self.encode_sequence(
            seq_items=batch["seq_items"],
            seq_behaviors=batch["seq_behaviors"],
            seq_delta_days=batch["seq_delta_days"],
            lengths=batch["seq_len"],
            item_repr=item_repr,
        )

        u_base = user_repr[batch["user_idx"]]
        u_final = self.layer_norm(u_base + seq_repr)

        pos = item_repr[batch["pos_item_idx"]]
        neg = item_repr[batch["neg_item_idx"]]

        pos_score = (u_final * pos).sum(dim=-1)
        neg_score = (u_final * neg).sum(dim=-1)

        if self.config.use_behavior and "pos_behavior" in batch:
            # Emphasize high-intent targets (cart/buy) in pairwise ranking.
            w_map = torch.tensor([1.0, 1.25, 1.6, 2.1], dtype=torch.float32, device=pos_score.device)
            bw = w_map[batch["pos_behavior"].clamp(0, 3)]
            bpr_loss = (-F.logsigmoid(pos_score - neg_score) * bw).mean()
        else:
            bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()
        reg = (u_final.norm(dim=-1).mean() + pos.norm(dim=-1).mean() + neg.norm(dim=-1).mean()) * 1e-4
        loss = bpr_loss + reg

        with torch.no_grad():
            auc_approx = (pos_score > neg_score).float().mean().item()

        return loss, {"bpr_loss": float(bpr_loss.item()), "auc_pair": float(auc_approx)}

    @torch.no_grad()
    def score_user(self, user_idx: int, seq_pack: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        user_repr, item_repr = self.spatial_propagation()

        seq_repr = self.encode_sequence(
            seq_items=seq_pack["seq_items"],
            seq_behaviors=seq_pack["seq_behaviors"],
            seq_delta_days=seq_pack["seq_delta_days"],
            lengths=seq_pack["seq_len"],
            item_repr=item_repr,
        )

        u_final = self.layer_norm(user_repr[user_idx : user_idx + 1] + seq_repr)
        return torch.matmul(u_final, item_repr.t()).squeeze(0)


class STGNNPipeline:
    def __init__(self, config: TrainConfig = TrainConfig()):
        self.config = config
        self.device = torch.device("cpu")
        self.model: STGNNRec | None = None

        self.user2idx: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2user: List[int] = []
        self.idx2item: List[int] = []
        self.user_hist: Dict[int, List[Tuple[int, int, int]]] = {}

        self.item_pop: torch.Tensor | None = None

    @staticmethod
    def _set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def _normalize_sparse(indices: np.ndarray, values: np.ndarray, shape: Tuple[int, int]) -> torch.Tensor:
        mat = torch.sparse_coo_tensor(
            indices=torch.tensor(indices, dtype=torch.long),
            values=torch.tensor(values, dtype=torch.float32),
            size=shape,
        ).coalesce()
        row_sum = torch.sparse.sum(mat, dim=1).to_dense().clamp_min(1e-8)
        norm_values = mat.values() / row_sum[mat.indices()[0]]
        return torch.sparse_coo_tensor(mat.indices(), norm_values, size=shape).coalesce()

    def prepare_from_df(self, df: pd.DataFrame):
        df = df[["u", "i", "b", "t"]].copy()
        df = df.dropna().astype({"u": int, "i": int, "b": int, "t": int})
        df = df.sort_values(["u", "t"]).reset_index(drop=True)

        self.idx2user = sorted(df["u"].unique().tolist())
        self.idx2item = sorted(df["i"].unique().tolist())
        self.user2idx = {u: idx for idx, u in enumerate(self.idx2user)}
        self.item2idx = {i: idx for idx, i in enumerate(self.idx2item)}

        df["u_idx"] = df["u"].map(self.user2idx)
        df["i_idx"] = df["i"].map(self.item2idx)

        self.user_hist = {}
        for uid, g in df.groupby("u_idx"):
            self.user_hist[int(uid)] = list(zip(g["i_idx"].tolist(), g["b"].tolist(), g["t"].tolist()))

        behavior_weight = DEFAULT_BEHAVIOR_WEIGHT if self.config.use_behavior else {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

        ui_rows, ui_cols, ui_vals = [], [], []
        for _, r in df.iterrows():
            ui_rows.append(int(r["u_idx"]))
            ui_cols.append(int(r["i_idx"]))
            ui_vals.append(behavior_weight.get(int(r["b"]), 1.0))

        ui_indices = np.vstack([np.array(ui_rows, dtype=np.int64), np.array(ui_cols, dtype=np.int64)])
        ui_norm = self._normalize_sparse(ui_indices, np.array(ui_vals, dtype=np.float32), (len(self.idx2user), len(self.idx2item)))

        trans = {}
        for hist in self.user_hist.values():
            for p in range(1, len(hist)):
                prev_i, prev_b, prev_t = hist[p - 1]
                cur_i, cur_b, cur_t = hist[p]
                if prev_i == cur_i:
                    continue
                dt = max(1, cur_t - prev_t)
                w = (behavior_weight.get(cur_b, 1.0) + behavior_weight.get(prev_b, 1.0)) / (1.0 + np.log1p(dt))
                trans[(prev_i, cur_i)] = trans.get((prev_i, cur_i), 0.0) + w
                trans[(cur_i, prev_i)] = trans.get((cur_i, prev_i), 0.0) + 0.5 * w

        for i_idx in range(len(self.idx2item)):
            trans[(i_idx, i_idx)] = trans.get((i_idx, i_idx), 0.0) + 1.0

        ii_rows, ii_cols, ii_vals = zip(*[(r, c, v) for (r, c), v in trans.items()])
        ii_indices = np.vstack([np.array(ii_rows, dtype=np.int64), np.array(ii_cols, dtype=np.int64)])
        item_adj = self._normalize_sparse(ii_indices, np.array(ii_vals, dtype=np.float32), (len(self.idx2item), len(self.idx2item)))

        pop = df.groupby("i_idx").size().reindex(range(len(self.idx2item)), fill_value=0).values
        self.item_pop = torch.tensor(pop, dtype=torch.float32)

        self.model = STGNNRec(
            num_users=len(self.idx2user),
            num_items=len(self.idx2item),
            embed_dim=self.config.embed_dim,
            item_adj=item_adj,
            ui_norm=ui_norm,
            config=self.config,
        ).to(self.device)

    def _build_samples(self) -> List[Dict[str, int | List[int] | List[float]]]:
        samples = []
        L = self.config.max_seq_len

        for u_idx, hist in self.user_hist.items():
            if len(hist) < 2:
                continue
            user_seen = {x[0] for x in hist}

            for cut in range(1, len(hist)):
                target_i, _, target_t = hist[cut]
                prefix = hist[:cut][-L:]
                if len(prefix) == 0:
                    continue

                samples.append(
                    {
                        "user_idx": u_idx,
                        "seq_items": [x[0] for x in prefix],
                        "seq_behaviors": [x[1] for x in prefix],
                        "seq_delta_days": [max(0.0, (target_t - x[2]) / 86400.0) for x in prefix],
                        "seq_len": len(prefix),
                        "pos_item_idx": target_i,
                        "pos_behavior": int(hist[cut][1]),
                        "user_seen": user_seen,
                    }
                )
        return samples

    def _collate_batch(self, batch_samples: List[dict]) -> Dict[str, torch.Tensor]:
        B = len(batch_samples)
        max_len = max(int(x["seq_len"]) for x in batch_samples)

        seq_items = torch.zeros((B, max_len), dtype=torch.long)
        seq_beh = torch.zeros((B, max_len), dtype=torch.long)
        seq_dt = torch.zeros((B, max_len), dtype=torch.float32)
        seq_len = torch.zeros((B,), dtype=torch.long)

        user_idx = torch.zeros((B,), dtype=torch.long)
        pos_item_idx = torch.zeros((B,), dtype=torch.long)
        neg_item_idx = torch.zeros((B,), dtype=torch.long)
        pos_behavior = torch.zeros((B,), dtype=torch.long)

        n_items = len(self.idx2item)

        for bi, s in enumerate(batch_samples):
            l = int(s["seq_len"])
            seq_items[bi, :l] = torch.tensor(s["seq_items"], dtype=torch.long)
            seq_beh[bi, :l] = torch.tensor(s["seq_behaviors"], dtype=torch.long)
            seq_dt[bi, :l] = torch.tensor(s["seq_delta_days"], dtype=torch.float32)
            seq_len[bi] = l

            user_idx[bi] = int(s["user_idx"])
            pos_item_idx[bi] = int(s["pos_item_idx"])
            pos_behavior[bi] = int(s.get("pos_behavior", 0))

            seen = s["user_seen"]
            neg = np.random.randint(0, n_items)
            while neg in seen:
                neg = np.random.randint(0, n_items)
            neg_item_idx[bi] = int(neg)

        return {
            "seq_items": seq_items,
            "seq_behaviors": seq_beh,
            "seq_delta_days": seq_dt,
            "seq_len": seq_len,
            "user_idx": user_idx,
            "pos_item_idx": pos_item_idx,
            "pos_behavior": pos_behavior,
            "neg_item_idx": neg_item_idx,
        }

    def train(self, verbose: bool = True) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("Call prepare_from_df first.")

        self._set_seed(self.config.seed)
        samples = self._build_samples()
        if len(samples) == 0:
            raise RuntimeError("No training samples generated.")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        summary = {}
        for ep in range(1, self.config.epochs + 1):
            np.random.shuffle(samples)
            losses, aucs = [], []

            for s in range(0, len(samples), self.config.batch_size):
                batch = self._collate_batch(samples[s : s + self.config.batch_size])
                self.model.train()
                optimizer.zero_grad()
                loss, metric = self.model.forward_train(batch)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                aucs.append(metric["auc_pair"])

            summary = {
                "epoch": ep,
                "loss": float(np.mean(losses)),
                "auc_pair": float(np.mean(aucs)),
                "num_samples": int(len(samples)),
            }
            if verbose:
                print(f"Epoch {ep:02d} | loss={summary['loss']:.4f} | pair_auc={summary['auc_pair']:.4f} | samples={summary['num_samples']}")

        return summary

    def _make_seq_pack_for_user(self, u_idx: int) -> Dict[str, torch.Tensor]:
        L = self.config.max_seq_len
        hist = self.user_hist.get(u_idx, [])[-L:]

        if len(hist) == 0:
            return {
                "seq_items": torch.zeros((1, 1), dtype=torch.long),
                "seq_behaviors": torch.zeros((1, 1), dtype=torch.long),
                "seq_delta_days": torch.zeros((1, 1), dtype=torch.float32),
                "seq_len": torch.ones((1,), dtype=torch.long),
            }

        anchor_t = hist[-1][2]
        return {
            "seq_items": torch.tensor([[x[0] for x in hist]], dtype=torch.long),
            "seq_behaviors": torch.tensor([[x[1] for x in hist]], dtype=torch.long),
            "seq_delta_days": torch.tensor([[max(0.0, (anchor_t - x[2]) / 86400.0) for x in hist]], dtype=torch.float32),
            "seq_len": torch.tensor([len(hist)], dtype=torch.long),
        }

    @torch.no_grad()
    def recommend_for_raw_user(self, raw_user_id: int, top_k: int = 10, filter_seen: bool = True) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not ready.")

        if raw_user_id not in self.user2idx:
            top_idx = torch.argsort(self.item_pop, descending=True)[:top_k].cpu().numpy().tolist()
            return pd.DataFrame(
                {
                    "rank": list(range(1, len(top_idx) + 1)),
                    "item_id": [self.idx2item[x] for x in top_idx],
                    "score": [float(self.item_pop[x].item()) for x in top_idx],
                    "reason": ["cold-start: popular items" for _ in top_idx],
                }
            )

        u_idx = self.user2idx[raw_user_id]
        seq_pack = self._make_seq_pack_for_user(u_idx)
        scores = self.model.score_user(u_idx, seq_pack)

        # Behavior-aware online boost: strengthen recently high-intent interacted items
        # so that repeat-purchase preferences can be reflected in ranking.
        if self.config.use_behavior:
            hist = self.user_hist.get(u_idx, [])
            if hist:
                recent = hist[-20:]
                anchor_t = recent[-1][2]
                boost = torch.zeros_like(scores)
                for i_idx, b, t in recent:
                    recency = 1.0 / (1.0 + np.log1p(max(1, (anchor_t - t) // 86400 + 1)))
                    bw = DEFAULT_BEHAVIOR_WEIGHT.get(int(b), 1.0)
                    boost[i_idx] += float(0.35 * bw * recency)
                scores = scores + boost

        if filter_seen:
            seen = {x[0] for x in self.user_hist.get(u_idx, [])}
            if seen:
                scores[torch.tensor(sorted(list(seen)), dtype=torch.long)] = -1e9

        top_idx = torch.topk(scores, k=min(top_k, scores.shape[0])).indices.cpu().numpy().tolist()
        recent_beh = [BEHAVIOR_NAME.get(x[1], str(x[1])) for x in self.user_hist.get(u_idx, [])[-3:]]
        reasons = [
            f"fused recent behaviors ({','.join(recent_beh)}) with spatio-temporal neighbors"
            if recent_beh
            else "spatio-temporal preference signal"
            for _ in top_idx
        ]

        return pd.DataFrame(
            {
                "rank": list(range(1, len(top_idx) + 1)),
                "item_id": [self.idx2item[x] for x in top_idx],
                "score": [float(scores[x].item()) for x in top_idx],
                "reason": reasons,
            }
        )

    def append_interaction(self, raw_user_id: int, raw_item_id: int, behavior_id: int, ts: int):
        if raw_user_id not in self.user2idx:
            return
        if raw_item_id not in self.item2idx:
            return

        u_idx = self.user2idx[raw_user_id]
        i_idx = self.item2idx[raw_item_id]
        self.user_hist.setdefault(u_idx, []).append((i_idx, int(behavior_id), int(ts)))

    def save_artifact(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save.")
        payload = {
            "config": self.config.__dict__,
            "state_dict": self.model.state_dict(),
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "idx2user": self.idx2user,
            "idx2item": self.idx2item,
            "user_hist": self.user_hist,
            "item_pop": self.item_pop,
        }
        torch.save(payload, path)

    def load_artifact(self, path: str):
        payload = torch.load(path, map_location="cpu")
        self.config = TrainConfig(**payload["config"])

        self.user2idx = {int(k): int(v) for k, v in payload["user2idx"].items()}
        self.item2idx = {int(k): int(v) for k, v in payload["item2idx"].items()}
        self.idx2user = [int(x) for x in payload["idx2user"]]
        self.idx2item = [int(x) for x in payload["idx2item"]]
        self.user_hist = {int(k): [(int(a), int(b), int(c)) for a, b, c in v] for k, v in payload["user_hist"].items()}
        self.item_pop = payload["item_pop"].float()

        df = hist_to_df(self.user_hist, self.idx2user, self.idx2item)
        self.prepare_from_df(df)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()


def hist_to_df(user_hist: Dict[int, List[Tuple[int, int, int]]], idx2user: List[int], idx2item: List[int]) -> pd.DataFrame:
    rows = []
    for u_idx, events in user_hist.items():
        u_raw = idx2user[u_idx]
        for i_idx, b, t in events:
            rows.append((u_raw, idx2item[i_idx], b, t))
    return pd.DataFrame(rows, columns=["u", "i", "b", "t"])


def load_behavior_df(db_path: str) -> pd.DataFrame:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT u, i, b, t FROM user_behavior_logs", conn)
    finally:
        conn.close()
    return df
