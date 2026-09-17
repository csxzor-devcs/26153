# Network World Model — Proactive Infiltration Forecasting

A world model that learns `P(S_{t+1} | S_t, ..., S_{t-T+1})` over network
traffic state and uses it to forecast, K windows ahead, the probability of
infiltration and the MITRE ATT&CK stage the network is trending toward —
with attention and SHAP explainability on every prediction.

See `../techsoln.md` for the full design and `../human.md` for a
plain-language walkthrough.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (fully offline, no dataset download)

Run everything from the repo root (`world-model-ids/`):

```bash
python -m tests.smoke_test
```

This trains the world model on **bundled synthetic flow data**
(`src/data/synthetic_flows.py` — a CICFlowMeter-schema generator with an
embedded recon → initial-access → lateral-movement → C2 → exfiltration-like
campaign), runs the benchmark against the logistic-regression baseline,
runs a K-step rollout, and runs SHAP + attention explainability — proving
the whole pipeline is wired correctly before you touch the real dataset.

Then launch the demo:

```bash
streamlit run app/streamlit_app.py
```

Pick an "Attacking session" or "Benign session" from the sidebar and watch
the infiltration-probability forecast, MITRE stage trajectory, attention
heatmap, and SHAP feature attribution update together.

## Using the real dataset (CIC-IDS2018/2017)

The bundled synthetic generator exists so the prototype runs with zero
setup — it is **not** a substitute for the benchmark the brief requires.
To run against real data:

1. Download CSE-CIC-IDS2018 ("Processed Traffic Data for ML Algorithms")
   from the official source referenced at nciipc.gov.in, or CIC-IDS2017 for
   `PortScan`-labelled reconnaissance traffic specifically.
2. Ensure the CSV has (or is renamed to) these CICFlowMeter columns:
   `Timestamp, Src IP, Dst IP, Src Port, Dst Port, Protocol, Flow Duration,
   Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts,
   Flow IAT Mean, Flow IAT Std, SYN Flag Cnt, ACK Flag Cnt, RST Flag Cnt,
   FIN Flag Cnt, Init Fwd Win Byts, Init Bwd Win Byts, Label`.
   (Optional packet-level columns `TTL`, `Retransmit Cnt` are picked up
   automatically if present; they degrade to zero otherwise.)
3. Place it at `data/raw/flows.csv` (or update `data.raw_flows_path` in
   `configs/default.yaml`).
4. Re-run:
   ```bash
   python -m src.train
   python -m evaluation.benchmark
   ```
   `evaluation/results.md` will report which data source was used —
   check this line before citing the numbers.

## Repository layout

```
configs/default.yaml     T, K, encoder choice, hidden size, loss weights
src/mitre_mapping.py     dataset-label -> MITRE stage map + exfil heuristic
src/data/synthetic_flows.py   bundled offline flow generator
src/features/extract.py       flow CSV -> per-window state vectors
src/features/windowing.py     state vectors -> (X, y_next, y_inf, y_stage)
src/models/world_model.py     LSTM/Transformer world model + multi-task loss
src/models/baseline_lr.py     logistic-regression baseline
src/train.py                  trains + saves all artifacts to weights/
src/rollout.py                K-step forward simulation (+ MC-dropout band)
src/explain.py                attention + SHAP explainability
evaluation/benchmark.py       world model vs baseline -> evaluation/results.md
app/streamlit_app.py          offline demo UI
tests/smoke_test.py           end-to-end pipeline check
```

## Known limitations

See `../techsoln.md` Section 12 — in particular, exfiltration-stage labels
are heuristic (no public dataset used here ships ground truth for that
stage), and the bundled synthetic data is for pipeline verification, not
for citing as a project result.

**Read before demoing:** on the bundled synthetic benchmark, the world
model currently trails the logistic-regression baseline on F1 and AUC-ROC
(see `evaluation/results.md`, generated fresh by every `python -m
evaluation.benchmark` run). This has been diagnosed, not hidden — see
techsoln.md sec 12 item 5 for why, and why it's expected to look different
on real CIC-IDS2018 data.
