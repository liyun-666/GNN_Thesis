# Project Function Specification

This project implements an ST-GNN based multi-behavior recommendation system. It supports offline training, recommendation inference, interactive behavior updates, desktop visualization, custom database loading, and diagnostic exports.

## Core Modules

- `recommender_engine.py`: recommendation pipeline, data preparation, training, artifact loading, and Top-K inference.
- `st_gnn_model.py`: ST-GNN model structure.
- `desktop_app_v2.py`: desktop application with login, demo workspace, custom database workspace, recommendation view, diagnostics, and user visualization.
- `api_server.py`: FastAPI backend for recommendation and behavior submission.
- `mobile_app_pwa/`: mobile-style web frontend served by the API.
- `qa_tool.py`: batch diagnostics and recommendation inspection utilities.
- `data_quality.py`: data validation and cleaning tools.
- `experiment_suite.py`: experiment runner and metric comparison utilities.

## Data Format

Custom SQLite or CSV data should contain:

- `u`: user ID
- `i`: item ID
- `b`: behavior type, where `0=click`, `1=favorite`, `2=cart`, `3=buy`
- `t`: Unix timestamp

## Highlights

- Combines graph-based and sequence-based recommendation features.
- Supports both built-in demo data and user-provided databases.
- Updates recommendations after new user interactions.
- Provides diagnostic exports for checking recommendation quality.
- Includes desktop, API, and mobile-style web demo entry points.

## Known Limits

- Account data is local-only.
- Large datasets and build artifacts are excluded from the Git repository.
- Custom databases should follow the expected field format for best compatibility.
