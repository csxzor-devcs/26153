# Benchmark Results

Data source for this run: the **bundled synthetic flow generator** (`src/data/synthetic_flows.py`) — NOT the CIC-IDS2018 dataset required by the brief. Set `data.raw_flows_path` in `configs/default.yaml` to a real dataset CSV and re-run `python -m src.train` then this script to get the real benchmark.

Test set size: 390 sequences (15.4% positive infiltration rate)

| Model | F1 | Precision | Recall | FPR |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.722 | 0.946 | 0.583 | 0.006 |
| World Model (LSTM) | 0.614 | 0.964 | 0.450 | 0.003 |

AUC-ROC: World Model = 0.666, Logistic Regression = 0.815

**Caveat (read before citing these numbers):** on this bundled synthetic data the world model currently trails the logistic-regression baseline on F1 *and* AUC-ROC, not just threshold calibration — more training epochs narrows but does not close the gap. The likely cause is a property of the *synthetic generator*, not of temporal modeling generally: each attack phase here is a clean, rule-based, single-window signature (e.g. a PortScan window's SYN/ACK pattern is unambiguous on its own), so the single most-recent state already carries most of the separable signal and the LSTM's extra capacity buys little. Real CIC-IDS2018 traffic is noisier and attack signatures are smeared across multiple flows/windows, which is exactly the regime where trajectory memory should help — this needs to be re-run on real data before treating the world model's relative standing here as representative. Do not cite this comparison as evidence the temporal approach works; it is currently evidence only that the pipeline runs end to end.
