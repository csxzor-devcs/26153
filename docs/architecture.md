# Network World Model Architecture

## System Overview

The Network World Model is an LSTM-based temporal forecasting system designed to predict network infiltration 3-30 minutes ahead of detection. It learns per-host network dynamics through a sequence-to-sequence architecture that:

1. **Encodes history** (T=20 windows) into a learned latent representation via multi-layer LSTM
2. **Forecasts ahead** (K=5 windows) via autoregressive rollout without teacher forcing
3. **Predicts three targets** simultaneously: next-state reconstruction, infiltration probability, and MITRE stage

The system processes raw network flow records (from CICFlowMeter or packet captures) and produces calibrated infiltration probabilities with uncertainty estimates via MC-dropout.

## Data Pipeline

### Flow-Level Input
Raw flows come from either real datasets (CIC-IDS2017) or the bundled synthetic generator. Each flow is a single bidirectional communication session with ~30 features:
- **Temporal**: Flow Duration, Flow IAT Mean/Std, packet inter-arrival times
- **Volume**: Tot Fwd Pkts, Tot Bwd Pkts, TotLen Fwd Pkts, TotLen Bwd Pkts
- **Flags**: SYN, ACK, RST, FIN flag counts (TCP handshake patterns)
- **Windows**: Init Fwd Win Byts, Init Bwd Win Byts
- **Packet-level** (optional): Pkt Len Mean, Pkt Len Std, Fwd Header Len, TTL, Retransmit Cnt

### Window Aggregation (60-second windows)
Flows are aggregated per (host, 60-second window) into **state vectors** (29 features):

**Baseline features** (per-window sum/mean/count):
- flow_count, mean_duration, std_duration
- bytes_in, bytes_out, pkts_in, pkts_out
- syn_count, ack_count, rst_count, fin_count
- iat_mean, iat_std, bidirectional_ratio, tcp_pct, win_size_mean
- ttl_mean, retransmission_count

**Graph-derived features** (host-level signatures):
- n_distinct_dst_ips: count of unique destination IPs this window
- n_distinct_dst_ports: count of unique destination ports
- fan_out_ratio: n_distinct_dst_ips / (total_flows + 1)
- new_dst_ratio: fraction of destination IPs not seen in previous window

**Packet-proxy features** (if present, else default 0.0):
- init_fwd_win_mean, init_bwd_win_mean (TCP window size dynamics)
- pkt_len_mean_agg, pkt_len_std_agg (packet size patterns)
- fwd_header_len_mean (protocol overhead patterns)

### Dataset Provenance & Column Normalization
The system handles two dataset variants:
- **CIC-IDS2017**: Contains "Source IP", "Destination IP" (plus legacy variants)
- **CIC-IDS2018 ML CSVs**: Lack Src/Dst IP, handled via flexible COLUMN_ALIASES

Preflight diagnostics detect missing columns with graceful degradation (default to 0.0).

## Temporal Correctness

### Canonical Sequence Indexing
All sequences follow strict temporal ordering to prevent information leakage:

```
History (T windows):    S_{t-19}, ..., S_t
Next state:            S_{t+1}
Infiltration (K=5):    any stage in {S_{t+1}, ..., S_{t+5}} > 0
Stage:                 stage(S_{t+1})
```

The model sees history up to window t, then predicts windows [t+1, t+5]. No future information leaks into the history.

### Per-Scenario Chronological Split
Unlike random splits that cluster attacks temporally:
- Group sequences by host (synthetic) or date (real data)
- Within each group, split chronologically 70/15/15
- Concatenate splits across groups

**Result**: Ensures attacks appear in all splits, preventing the model from memorizing "attacks happen at specific times."

### Temporal Leakage Verification
For each train/val/test split, compute effective intervals:
- History span: [t - (T-1)×60s, t]
- Target span: [t + 60s, t + K×60s]
- Total: [t - (T-1)×60s, t + K×60s]

Verify no target window from training overlaps history of validation/test (with optional purge gap).

## Model Architecture

### World Model (LSTM-based)
```
Input: X ∈ ℝ^(B×T×F)  [batch, T=20 windows, F=29 features]
       ↓
LSTM encoder:          [B×T×F] → [B×T×H] [H=64 hidden dim]
       ↓
Additive attention:    softmax([B×T×H] → [B×T×1]) → [B×H] context vector
       ↓ (feed context to three output heads)
Next state head:       [B×H] → [B×F] (MSE loss)
Infiltration head:     [B×H] → [B×1] logit → sigmoid → P(infiltration)
Stage head:            [B×H] → [B×7] logits → cross-entropy for MITRE stage
```

### Uncertainty via MC-Dropout
During forecast rollout, keep dropout active (0.2) and sample N=20 times:
- Collect N infiltration probability trajectories
- Compute mean and std per forecast window
- Std grows over K steps (autoregressive error compounds)

### Baseline Comparisons
1. **Logistic Regression**: Single-window features (no history) — shows importance of temporal context
2. **Static MLP**: Same architecture, takes only x[:, -1, :] — temporal ablation

Both use same threshold selection (sweep 0.1-0.9, max F1 on validation).

## Training Pipeline

1. **Data preparation** (`src/data.prepare`): Raw flows → state vectors → provenance JSON
2. **Sequence building** (`src/features.windowing`): State vectors → temporal sequences (T=20, K=5)
3. **Chronological split** (per-scenario): Ensures no temporal leakage
4. **Standardization**: Fit scaler on training, apply to all splits
5. **Multi-task training** (60 epochs, batch=32):
   - Next-state MSE (weight 1.0)
   - Infiltration BCE (weight 1.0)
   - Stage cross-entropy (weight 1.0)
   - Adam optimizer, LR=0.005, early stopping patience=5
6. **Threshold selection** (validation set): Sweep [0.1, 0.9] in 0.01 steps, select max F1
7. **Test evaluation** (frozen threshold): Report F1, precision, recall, FPR, AUC-ROC

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| T=20 windows | 20min history; more captures multi-phase attacks, less captures slow exfil |
| K=5 windows | 5min horizon; early warning before exploitation, before data exfiltration |
| Additive attention | Explainability: which windows mattered most? |
| Per-scenario split | Prevents temporal clustering from dominating model capacity |
| MC-dropout | Quantifies forecast confidence; wide bands → low certainty → operator override |
| Packet-proxy features | Second-order signatures (window sizes, packet lengths) capture subtle exploits |
| Graceful degradation | Works on flow-only data (most common); gains packet-level features when available |

## Evaluation Metrics

**Detection (at threshold t)**:
- F1 (primary): harmonic mean of precision and recall
- Precision: of predicted infiltrations, how many were real
- Recall: of actual infiltrations, how many were caught
- FPR: false-positive rate (false alarms per benign sequence)
- AUC-ROC: threshold-agnostic discrimination ability

**Forecasting (if attack semantics available)**:
- Lead time: windows between prediction and attack onset
- Proactive detection: % of attacks warned before the first suspicious flow

## Deployment Considerations

1. **Real-time inference**: Sequence takes ~1-2ms on CPU; stream windows through model
2. **Threshold tuning**: Adjust on new data via validation-set sweep; affects precision/recall tradeoff
3. **Drift handling**: Retraining recommended when FPR increases >5% or F1 decreases >10%
4. **Explainability**: Always show SHAP attribution and attention weights with predictions
5. **Uncertainty**: Only alert on high-confidence predictions (>70%); low-confidence allows human review
