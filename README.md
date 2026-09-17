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

The bundled synthetic generator proves the pipeline works end-to-end offline.
To benchmark against real network attack data:

### Option 1: Manual Preparation (Recommended)

1. Download CSE-CIC-IDS2018 ("Processed Traffic Data for ML Algorithms") from
   the official source referenced at nciipc.gov.in, or CIC-IDS2017 for
   PortScan reconnaissance traffic.

2. Prepare the dataset:
   ```bash
   python -m src.data.prepare \
     --input-dir /path/to/CIC-IDS2018/CSVs \
     --output data/raw/flows.csv \
     --source cic_ids2018
   ```
   
   This:
   - Handles multiple CSVs from a directory
   - Normalizes column names across dataset variants
   - Validates required CICFlowMeter columns
   - Removes malformed rows (NaN, Inf)
   - Logs data quality (row counts, class distribution, timestamp range, hosts)
   - Saves provenance (`flows_provenance.json`)

3. Train and evaluate:
   ```bash
   python -m src.train
   python -m evaluation.benchmark
   ```

### Option 2: Auto-Preparation

Point `configs/default.yaml` at the raw data directory:
```yaml
data:
  raw_input_dir: "/path/to/CIC-IDS2018/CSVs"
  dataset_source: "cic_ids2018"
```

Then run training — `src.train.py` will call `prepare.py` automatically.

### Column Format

Input CSV must have CICFlowMeter columns (or aliases):
- Core: `Timestamp, Src IP, Dst IP, Src Port, Dst Port, Protocol, Flow Duration,
  Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts,
  Flow IAT Mean, Flow IAT Std, SYN Flag Cnt, ACK Flag Cnt, RST Flag Cnt,
  FIN Flag Cnt, Init Fwd Win Byts, Init Bwd Win Byts, Label`
- Optional packet-level: `TTL, Retransmit Cnt` (degrade to 0.0 if absent)

### Evaluation Results

`evaluation/results.md` distinguishes:
- **Synthetic benchmark:** Bundled offline generator (pipeline verification only)
- **Real benchmark:** CIC-IDS2018/2017 (actual project result)

Do not cite synthetic results as project evidence.

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

## Forecasting vs. Detection

This system makes a specific claim: **"proactive infiltration forecasting"** —
predicting attacks *before* they complete, not just detecting them as they happen.

To validate this claim scientifically, we distinguish:

- **Pre-attack sequences:** History contains no attack traffic yet. Predicting
  infiltration here is genuine early-warning.
- **During-attack sequences:** History already contains attack traffic. Predicting
  infiltration here is detection during attack, not forecasting.

The attack semantics module (`src/data/attack_semantics.py`) automatically:
1. Detects when each attack session begins per host
2. Classifies sequences as pre-attack, during-attack, or benign
3. Measures lead-time (windows of warning before attack onset)
4. Reports early-warning rate separately from detection rate

Results distinguish these rigorously so judges can evaluate the forecasting claim.

## Known limitations

See `../techsoln.md` Section 12 — in particular, exfiltration-stage labels
are heuristic (no public dataset used here ships ground truth for that
stage), and the bundled synthetic data is for pipeline verification only.

**Synthetic benchmark caveat:** On bundled data, the world model currently
trails the logistic-regression baseline (see `evaluation/results.md`).
This is expected — synthetic attack signatures are single-window-separable.
Real CIC-IDS2018 traffic is noisier and should favor temporal modeling.
