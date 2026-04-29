import os
import sqlite3
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel

from recommender_engine import STGNNPipeline, TrainConfig, load_behavior_df


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rec_system.db")
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts", "stgnn_artifact_v2.pt")
WEB_DIR = os.path.join(BASE_DIR, "mobile_app_pwa")

app = FastAPI(title="ST-GNN Recommender API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve mobile web app from the same origin as API so external users can use one URL.
if os.path.isdir(WEB_DIR):
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")


class InteractionIn(BaseModel):
    user_id: int
    item_id: int
    behavior: int  # 0 click,1 fav,2 cart,3 buy
    timestamp: int | None = None


pipeline: STGNNPipeline | None = None


def init_pipeline() -> STGNNPipeline:
    pipe = STGNNPipeline(TrainConfig())
    if os.path.exists(ARTIFACT_PATH):
        try:
            pipe.load_artifact(ARTIFACT_PATH)
            return pipe
        except Exception:
            pass

    strict_csv = os.path.join(BASE_DIR, "final_real_data_clean_strict.csv")
    if os.path.exists(strict_csv):
        df = pd.read_csv(strict_csv)
    else:
        df = load_behavior_df(DB_PATH)
    pipe.config.epochs = 4
    pipe.prepare_from_df(df)
    pipe.train(verbose=False)
    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    pipe.save_artifact(ARTIFACT_PATH)
    return pipe


@app.on_event("startup")
def startup_event():
    global pipeline
    pipeline = init_pipeline()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return RedirectResponse(url="/web/index.html")


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = 10, include_seen: bool = False):
    rec = pipeline.recommend_for_raw_user(user_id, top_k=top_k, filter_seen=not include_seen)
    return {
        "user_id": user_id,
        "top_k": top_k,
        "include_seen": include_seen,
        "items": rec.to_dict(orient="records"),
    }


@app.post("/interact")
def interact(payload: InteractionIn):
    ts = payload.timestamp or int(time.time())

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO user_behavior_logs (u, i, b, t) VALUES (?, ?, ?, ?)",
            (int(payload.user_id), int(payload.item_id), int(payload.behavior), int(ts)),
        )
        conn.commit()
    finally:
        conn.close()

    pipeline.append_interaction(payload.user_id, payload.item_id, payload.behavior, ts)

    rec = pipeline.recommend_for_raw_user(payload.user_id, top_k=10, filter_seen=True)
    return {
        "message": "interaction accepted",
        "user_id": payload.user_id,
        "refresh_recommendation": rec.to_dict(orient="records"),
    }
