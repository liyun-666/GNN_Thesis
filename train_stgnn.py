import argparse
import os
import pandas as pd

from recommender_engine import TrainConfig, STGNNPipeline, load_behavior_df


def main():
    parser = argparse.ArgumentParser(description="Train ST-GNN multi-behavior recommender")
    parser.add_argument("--db", type=str, default="rec_system.db", help="path to sqlite db")
    parser.add_argument("--csv", type=str, default="", help="optional csv path with columns u,i,b,t")
    parser.add_argument("--artifact", type=str, default="artifacts/stgnn_artifact.pt", help="path to save artifact")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=30)
    parser.add_argument(
        "--recipe",
        type=str,
        default="default",
        choices=["default", "mba_like", "stc_hgat_like", "muse_like"],
        help="paper-aligned training recipe",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    artifact_path = os.path.abspath(args.artifact)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

    if args.csv:
        csv_path = os.path.abspath(args.csv)
        print(f"[INFO] Loading csv data from {csv_path}")
        df = pd.read_csv(csv_path)[["u", "i", "b", "t"]]
    else:
        print(f"[INFO] Loading db data from {db_path}")
        df = load_behavior_df(db_path)
    print(f"[INFO] Rows={len(df)}, Users={df['u'].nunique()}, Items={df['i'].nunique()}")

    recipe_cfg = {
        "default": dict(use_spatial=True, use_temporal=True, use_behavior=True),
        # MBA-style multi-behavior sequence-aware setting
        "mba_like": dict(use_spatial=True, use_temporal=True, use_behavior=True),
        # STC-HGAT style: emphasize spatial-temporal coupling
        "stc_hgat_like": dict(use_spatial=True, use_temporal=True, use_behavior=False),
        # MUSE style: long sequence preference
        "muse_like": dict(use_spatial=True, use_temporal=True, use_behavior=True),
    }

    max_seq_len = args.max_seq_len
    if args.recipe == "muse_like":
        max_seq_len = max(max_seq_len, 50)

    cfg = TrainConfig(
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=max_seq_len,
        **recipe_cfg[args.recipe],
    )
    pipe = STGNNPipeline(cfg)

    print(f"[INFO] Recipe={args.recipe}, Config={cfg}")
    print("[INFO] Building graph and sequences...")
    pipe.prepare_from_df(df)

    print("[INFO] Start training...")
    summary = pipe.train(verbose=True)
    print(f"[INFO] Final: {summary}")

    pipe.save_artifact(artifact_path)
    print(f"[OK] Artifact saved to {artifact_path}")


if __name__ == "__main__":
    main()
