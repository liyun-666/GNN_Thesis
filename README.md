# ST-GNN Multi-Behavior Recommendation System

This is my ST-GNN based recommendation system project. It supports multi-behavior user sequence modeling, recommendation generation, interactive updates, desktop visualization, a FastAPI backend, and a mobile-style PWA demo.

## Features

- Multi-behavior recommendation for click, favorite, cart, and purchase events.
- ST-GNN pipeline with user-item graph, item transition graph, behavior embedding, and temporal sequence modeling.
- Desktop app with login, demo workspace, custom database workspace, recommendation view, diagnostics, and user behavior visualization.
- Streamlit demo for quick local inspection.
- FastAPI service with a mobile PWA frontend under `mobile_app_pwa/`.
- Data cleaning, model training, experiment, and QA utilities.

## Project Structure

```text
.
├── app.py                    # Streamlit demo
├── desktop_app_v2.py          # Desktop application
├── api_server.py              # FastAPI backend
├── recommender_engine.py      # Recommendation pipeline
├── st_gnn_model.py            # Model definition
├── train_stgnn.py             # Training script
├── experiment_suite.py        # Experiment utilities
├── qa_tool.py                 # Recommendation diagnostics
├── data_quality.py            # Data quality checks
├── mobile_app_pwa/            # Mobile-style web frontend
├── artifacts/                 # Lightweight model and experiment outputs
├── sample_data/               # Sample upload data
└── requirements.txt           # Python dependencies
```

## Requirements

- Python 3.10+
- Windows 10/11 recommended for the desktop startup scripts

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the Streamlit demo:

```bash
streamlit run app.py
```

Run the desktop app:

```bash
python desktop_app_v2.py
```

Or double-click on Windows:

```text
start_desktop_app.bat
```

Run the API and mobile PWA:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --app-dir .
```

Then open:

```text
http://127.0.0.1:8000/web/index.html
```

## Data Format

The system expects behavior data from SQLite or CSV with these fields:

| Field | Description |
| --- | --- |
| `u` | User ID |
| `i` | Item ID |
| `b` | Behavior type: `0=click`, `1=favorite`, `2=cart`, `3=buy` |
| `t` | Unix timestamp |

## Training

```bash
python train_stgnn.py
```

## Notes

Large local datasets, build outputs, installers, caches, and private runtime files are excluded by `.gitignore`. If large assets need to be shared later, use GitHub Releases, cloud storage, or Git LFS.
