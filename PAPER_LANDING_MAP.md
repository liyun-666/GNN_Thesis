# Paper-Aligned Landing Map

This project is grounded against highly similar multi-behavior spatio-temporal recommendation papers and translated into reproducible engineering modules.

- Core paper map: `paper_alignment.json`
- Repro command (strict data):
```bash
python train_stgnn.py --csv final_real_data_clean_strict.csv --artifact artifacts/stgnn_artifact_v2.pt --recipe mba_like --epochs 6
python experiment_suite.py --input final_real_data_clean_strict.csv --output-dir artifacts/experiments --sample-users 80 --topk 10
```

## What is already landed
- Spatio-temporal joint encoder (graph + temporal sequence)
- Multi-behavior semantic embedding
- Online interaction feedback update in app
- Inspector for post-interaction recommendation correctness scoring
- Benchmark, ablation, sensitivity experiment pipeline
- Defense dashboard for thesis demonstration
