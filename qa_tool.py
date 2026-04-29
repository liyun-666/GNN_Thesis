from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import time
import os

import numpy as np
import pandas as pd
import torch

from recommender_engine import STGNNPipeline


@dataclass
class SingleCheckResult:
    ok: bool
    user_id: int
    item_id: int
    behavior: int
    before_score: float | None
    after_score: float | None
    score_delta: float | None
    before_rank_unfiltered: int | None
    after_rank_unfiltered: int | None
    rank_improve: int | None
    in_topk_before_unfiltered: bool
    in_topk_after_unfiltered: bool
    in_topk_before_filtered: bool
    in_topk_after_filtered: bool
    quality_score: float
    message: str


def _user_scores(pipeline: STGNNPipeline, raw_user_id: int) -> torch.Tensor | None:
    if pipeline.model is None:
        return None
    if raw_user_id not in pipeline.user2idx:
        return None

    u_idx = pipeline.user2idx[raw_user_id]
    seq_pack = pipeline._make_seq_pack_for_user(u_idx)
    return pipeline.model.score_user(u_idx, seq_pack)


def _rank_of_item(scores: torch.Tensor, item_idx: int) -> int:
    # rank starts from 1
    order = torch.argsort(scores, descending=True)
    pos = (order == item_idx).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return int(scores.shape[0]) + 1
    return int(pos[0].item()) + 1


def run_single_interaction_check(
    pipeline: STGNNPipeline,
    user_id: int,
    item_id: int,
    behavior: int,
    top_k: int = 10,
    ts: int | None = None,
) -> SingleCheckResult:
    ts = ts or int(time.time())

    if user_id not in pipeline.user2idx:
        return SingleCheckResult(
            ok=False,
            user_id=user_id,
            item_id=item_id,
            behavior=behavior,
            before_score=None,
            after_score=None,
            score_delta=None,
            before_rank_unfiltered=None,
            after_rank_unfiltered=None,
            rank_improve=None,
            in_topk_before_unfiltered=False,
            in_topk_after_unfiltered=False,
            in_topk_before_filtered=False,
            in_topk_after_filtered=False,
            quality_score=0.0,
            message="User is out of model vocabulary. Retraining is required.",
        )

    if item_id not in pipeline.item2idx:
        return SingleCheckResult(
            ok=False,
            user_id=user_id,
            item_id=item_id,
            behavior=behavior,
            before_score=None,
            after_score=None,
            score_delta=None,
            before_rank_unfiltered=None,
            after_rank_unfiltered=None,
            rank_improve=None,
            in_topk_before_unfiltered=False,
            in_topk_after_unfiltered=False,
            in_topk_before_filtered=False,
            in_topk_after_filtered=False,
            quality_score=0.0,
            message="Item is out of model vocabulary. Retraining is required.",
        )

    u_idx = pipeline.user2idx[user_id]
    i_idx = pipeline.item2idx[item_id]

    # Snapshot and restore after simulation
    old_hist = list(pipeline.user_hist.get(u_idx, []))

    before_scores = _user_scores(pipeline, user_id)
    if before_scores is None:
        return SingleCheckResult(
            ok=False,
            user_id=user_id,
            item_id=item_id,
            behavior=behavior,
            before_score=None,
            after_score=None,
            score_delta=None,
            before_rank_unfiltered=None,
            after_rank_unfiltered=None,
            rank_improve=None,
            in_topk_before_unfiltered=False,
            in_topk_after_unfiltered=False,
            in_topk_before_filtered=False,
            in_topk_after_filtered=False,
            quality_score=0.0,
            message="Cannot compute score for this user.",
        )

    before_score = float(before_scores[i_idx].item())
    before_rank = _rank_of_item(before_scores, i_idx)

    before_unfiltered = pipeline.recommend_for_raw_user(user_id, top_k=top_k, filter_seen=False)["item_id"].tolist()
    before_filtered = pipeline.recommend_for_raw_user(user_id, top_k=top_k, filter_seen=True)["item_id"].tolist()

    pipeline.append_interaction(user_id, item_id, behavior, ts)
    after_scores = _user_scores(pipeline, user_id)
    after_score = float(after_scores[i_idx].item())
    after_rank = _rank_of_item(after_scores, i_idx)

    after_unfiltered = pipeline.recommend_for_raw_user(user_id, top_k=top_k, filter_seen=False)["item_id"].tolist()
    after_filtered = pipeline.recommend_for_raw_user(user_id, top_k=top_k, filter_seen=True)["item_id"].tolist()

    # Restore
    pipeline.user_hist[u_idx] = old_hist

    delta = after_score - before_score
    rank_improve = before_rank - after_rank

    in_topk_before_unfiltered = int(item_id in before_unfiltered) == 1
    in_topk_after_unfiltered = int(item_id in after_unfiltered) == 1
    in_topk_before_filtered = int(item_id in before_filtered) == 1
    in_topk_after_filtered = int(item_id in after_filtered) == 1

    quality = 0.0
    if delta > 0:
        quality += 45.0
    if rank_improve > 0:
        quality += 30.0
    if rank_improve < 0:
        quality -= 20.0
    if in_topk_after_unfiltered:
        quality += 15.0
    if in_topk_after_filtered or in_topk_before_filtered:
        quality += 10.0

    # Strong behavior expectation bonus
    if behavior in (2, 3) and delta > 0:
        quality += 5.0

    quality = float(min(100.0, max(0.0, quality)))

    msg = "OK"
    if not in_topk_after_filtered and in_topk_after_unfiltered:
        msg = "Item can be ranked high in unfiltered list, but is removed in filtered list because it is already seen."
    elif delta <= 0 and rank_improve <= 0:
        msg = "Warning: both score and rank do not improve after interaction. Retraining or parameter tuning is recommended."
    elif rank_improve < 0:
        msg = "Warning: item rank drops after interaction. Check temporal weighting and retrain with more recent data."

    return SingleCheckResult(
        ok=True,
        user_id=user_id,
        item_id=item_id,
        behavior=behavior,
        before_score=before_score,
        after_score=after_score,
        score_delta=delta,
        before_rank_unfiltered=before_rank,
        after_rank_unfiltered=after_rank,
        rank_improve=rank_improve,
        in_topk_before_unfiltered=in_topk_before_unfiltered,
        in_topk_after_unfiltered=in_topk_after_unfiltered,
        in_topk_before_filtered=in_topk_before_filtered,
        in_topk_after_filtered=in_topk_after_filtered,
        quality_score=quality,
        message=msg,
    )


def run_batch_diagnostics(
    pipeline: STGNNPipeline,
    logs_df: pd.DataFrame,
    sample_size: int = 30,
    top_k: int = 10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    random.seed(random_seed)

    x = logs_df[["u", "i", "b", "t"]].copy()
    x = x.dropna().astype({"u": int, "i": int, "b": int, "t": int})
    x = x[(x["u"].isin(pipeline.idx2user)) & (x["i"].isin(pipeline.idx2item))]

    if x.empty:
        return pd.DataFrame(), {
            "cases": 0,
            "pass_rate": 0.0,
            "avg_quality_score": 0.0,
        }

    if len(x) > sample_size:
        x = x.sample(n=sample_size, random_state=random_seed)

    rows = []
    for _, r in x.iterrows():
        res = run_single_interaction_check(
            pipeline=pipeline,
            user_id=int(r["u"]),
            item_id=int(r["i"]),
            behavior=int(r["b"]),
            top_k=top_k,
            ts=int(r["t"]) + 1,
        )
        rows.append(
            {
                "ok": res.ok,
                "user_id": res.user_id,
                "item_id": res.item_id,
                "behavior": res.behavior,
                "before_score": res.before_score,
                "after_score": res.after_score,
                "score_delta": res.score_delta,
                "before_rank_unfiltered": res.before_rank_unfiltered,
                "after_rank_unfiltered": res.after_rank_unfiltered,
                "rank_improve": res.rank_improve,
                "topk_before_unfiltered": res.in_topk_before_unfiltered,
                "topk_after_unfiltered": res.in_topk_after_unfiltered,
                "topk_before_filtered": res.in_topk_before_filtered,
                "topk_after_filtered": res.in_topk_after_filtered,
                "quality_score": res.quality_score,
                "message": res.message,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "cases": 0,
            "pass_rate": 0.0,
            "avg_quality_score": 0.0,
        }

    pass_cond = (df["score_delta"] > 0) | (df["rank_improve"] > 0)
    summary = {
        "cases": int(len(df)),
        "pass_rate": float(pass_cond.mean()),
        "avg_quality_score": float(df["quality_score"].mean()),
        "avg_score_delta": float(df["score_delta"].fillna(0.0).mean()),
        "avg_rank_improve": float(df["rank_improve"].fillna(0.0).mean()),
    }
    return df, summary


def diagnose_item_across_users(
    pipeline: STGNNPipeline,
    logs_df: pd.DataFrame,
    item_id: int,
    top_k: int = 10,
    simulate_behavior: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, float | int | str]]:
    x = logs_df[["u", "i", "b", "t"]].copy()
    x = x.dropna().astype({"u": int, "i": int, "b": int, "t": int})
    users = sorted(x.loc[x["i"] == int(item_id), "u"].unique().tolist())

    if item_id not in pipeline.item2idx:
        return pd.DataFrame(), {
            "item_id": int(item_id),
            "users": len(users),
            "in_vocab": False,
            "message": "Item is not in model vocabulary.",
        }

    rows = []
    for u in users:
        if u not in pipeline.user2idx:
            continue
        res = run_single_interaction_check(
            pipeline=pipeline,
            user_id=int(u),
            item_id=int(item_id),
            behavior=int(simulate_behavior),
            top_k=int(top_k),
        )
        rows.append(
            {
                "user_id": res.user_id,
                "before_score": res.before_score,
                "after_score": res.after_score,
                "score_delta": res.score_delta,
                "before_rank_unfiltered": res.before_rank_unfiltered,
                "after_rank_unfiltered": res.after_rank_unfiltered,
                "rank_improve": res.rank_improve,
                "in_topk_before_unfiltered": res.in_topk_before_unfiltered,
                "in_topk_after_unfiltered": res.in_topk_after_unfiltered,
                "in_topk_before_filtered": res.in_topk_before_filtered,
                "in_topk_after_filtered": res.in_topk_after_filtered,
                "quality_score": res.quality_score,
                "message": res.message,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, {
            "item_id": int(item_id),
            "users": len(users),
            "in_vocab": True,
            "message": "No users with this item are in model vocabulary.",
        }

    summary = {
        "item_id": int(item_id),
        "users": int(len(df)),
        "in_vocab": True,
        "avg_quality_score": float(df["quality_score"].mean()),
        "avg_score_delta": float(df["score_delta"].mean()),
        "avg_rank_improve": float(df["rank_improve"].mean()),
        "before_topk_unfiltered_rate": float(df["in_topk_before_unfiltered"].mean()),
        "after_topk_unfiltered_rate": float(df["in_topk_after_unfiltered"].mean()),
        "before_topk_filtered_rate": float(df["in_topk_before_filtered"].mean()),
        "after_topk_filtered_rate": float(df["in_topk_after_filtered"].mean()),
    }
    return df, summary


def export_inspector_results(
    detail_df: pd.DataFrame,
    summary: Dict[str, float],
    out_dir: str,
    prefix: str = "batch_diag",
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"{prefix}_{ts}.csv")
    detail_df.to_csv(csv_path, index=False)

    summary_path = os.path.join(out_dir, f"{prefix}_{ts}_summary.json")
    pd.Series(summary).to_json(summary_path, force_ascii=False, indent=2)

    img_path = ""
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        detail_df["quality_score"].hist(ax=ax1, bins=10)
        ax1.set_title("Quality Score Dist")
        ax2 = fig.add_subplot(1, 2, 2)
        detail_df["score_delta"].fillna(0).hist(ax=ax2, bins=10)
        ax2.set_title("Score Delta Dist")
        fig.tight_layout()
        img_path = os.path.join(out_dir, f"{prefix}_{ts}.png")
        fig.savefig(img_path, dpi=150)
        plt.close(fig)
    except Exception:
        img_path = ""

    return {
        "csv": csv_path,
        "summary": summary_path,
        "plot": img_path,
    }
