# Benchmark Results

Data source for this run: the **bundled synthetic flow generator** (`src/data/synthetic_flows.py`) — NOT the CIC-IDS2018 dataset required by the brief. Set `data.raw_flows_path` in `configs/default.yaml` to a real dataset CSV and re-run `python -m src.train` then this script to get the real benchmark.

Test set size: 120 sequences (0.0% positive infiltration rate)

## Sequence Composition (Attack Semantics)

- Pre-attack sequences: 0 (pure forecasting scenarios)
- During-attack sequences: 3 (detection scenarios)
- Benign sequences: 117 (negative examples)

## Detection Metrics (Overall)

| Model | F1 | Precision | Recall | FPR |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.000 | 0.000 | 0.000 | 0.000 |
| World Model (LSTM) | 0.000 | 0.000 | 0.000 | 0.000 |

AUC-ROC: World Model = nan, Logistic Regression = nan

## Proactive Forecasting Metrics (Pre-Attack Sequences Only)

These metrics evaluate the model's ability to warn about attacks *before* they start, not detect ongoing attacks.

*(No pre-attack sequences in test set — chronological ordering places all attacks in training)*
