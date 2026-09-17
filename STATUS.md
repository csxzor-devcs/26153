# SIH 2026 Network World Model — Project Status

**Date**: 2026-09-18  
**Overall Progress**: 2/8 Phases Complete (25%)  
**Next Phase**: Phase 3 (Graph-derived & Packet-proxy Features)

---

## ✅ Completed Phases

### Phase 1: CIC-IDS2017 Dataset Support
- ✓ Enhanced COLUMN_ALIASES for CIC-IDS2017 variants (Source IP, Destination IP, etc.)
- ✓ Added diagnose_dataset_type() function to detect dataset type and provide preflight diagnostics
- ✓ Updated validate_columns() with flexible normalization
- ✓ Changed default from cic_ids2018 to cic_ids2017 in src/data/prepare.py
- ✓ Smoke test: PASSED

**Files Modified**:
- `src/data/prepare.py` — Column normalization, validation, diagnostics

### Phase 2: Per-Scenario Chronological Split
- ✓ Modified campaign_start timing in synthetic_flows.py to scatter attacks across 25-75% timeline
- ✓ Implemented per_scenario_chronological_split() in windowing.py (per-host for synthetic, per-date for real)
- ✓ Updated train.py to use new split function with hosts_array parameter
- ✓ Made temporal leakage check dev-mode aware (tolerates overlapping intervals in per-scenario mode)
- ✓ Verified attack distribution across train/val/test splits
- ✓ Smoke test: PASSED

**Files Modified**:
- `src/data/synthetic_flows.py` — Campaign timing
- `src/features/windowing.py` — New per_scenario_chronological_split() function
- `src/train.py` — Split strategy call, leakage check tolerance
- `configs/dev_synthetic.yaml` — Reduced purge_gap to 0

---

## 📋 Remaining Phases (Ready to Execute)

### Phase 3: Graph-Derived & Packet-Proxy Features
**Location**: `src/features/extract.py`  
**Tasks**:
- Add graph-derived scalars per (host, 60s window):
  - `n_distinct_dst_ips` — count of unique destination IPs
  - `n_distinct_dst_ports` — count of unique destination ports
  - `fan_out_ratio` — n_distinct_dst_ips / (total flows + 1)
  - `new_dst_ratio` — fraction of DST IPs not in previous window (requires rolling set state)
  
- Add packet-proxy features from existing CICFlowMeter columns:
  - `init_fwd_win_mean` — mean of Init Fwd Win Byts
  - `init_bwd_win_mean` — mean of Init Bwd Win Byts
  - `pkt_len_mean_agg` — mean of Pkt Len Mean
  - `pkt_len_std_agg` — mean of Pkt Len Std
  - `fwd_header_len_mean` — mean of Fwd Header Len
  
- Update FEATURE_COLUMNS list
- Add graceful degradation (default to 0.0 if column absent, log warning)
- Update state vector dimension in configs/default.yaml
- **Verification**: `python -m tests.smoke_test`

### Phase 4: Validation-Set Threshold Selection
**Location**: `src/train.py`  
**Tasks**:
- Sweep thresholds 0.1-0.9 on validation set
- Select threshold with maximum F1 score
- Save selected threshold to `weights/threshold.json`
- Update model checkpoint saving
- **Verification**: `python -m tests.smoke_test`

### Phase 5: Static Neural Baseline
**Location**: `src/models/static_model.py` (new file)  
**Tasks**:
- Implement single-window MLP (no history, no attention)
- Compare vs Logistic Regression vs World Model
- Add to evaluation.benchmark
- Report results in evaluation/results.md
- **Verification**: `python -m tests.smoke_test`

### Phase 6: Rollout Correctness Tests
**Location**: `tests/test_rollout.py` (new file)  
**Tasks**:
- Test K-step autoregressive rollout without teacher forcing
- Verify prediction bounds (e.g., infiltration ∈ [0,1])
- Verify monotonic uncertainty growth with MC-dropout
- Verify rollout produces non-trivial predictions
- **Verification**: `python -m pytest tests/test_rollout.py -v`

### Phase 7: Update Streamlit Demo
**Location**: `app/streamlit_app.py`  
**Tasks**:
- Add scenario badges (host, date, MITRE stage)
- Display lead time (K windows ahead)
- Show test set composition (number of attacks, benign)
- Add uncertainty ribbon from MC-dropout
- **Verification**: Manual testing in browser

### Phase 8: Create Deliverable Documents
**Locations**: `docs/architecture.md`, `docs/presentation.md`  
**Tasks**:
- architecture.md (2 pages): System design, temporal correctness, feature extraction, model architecture
- presentation.md (5 slides in markdown): Overview, data pipeline, model, results, deployment
- **Verification**: `python -m tests.smoke_test`

---

## 🚀 Training on Separate Machine

Complete guide has been created: **TRAINING_SETUP.md**

### Quick Start
```bash
# On training machine
git clone <repo-url> world-model-ids
cd world-model-ids

# Environment setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Dataset preparation (one-time)
python -m src.data.prepare \
  --input-dir /data/cic_ids2017 \
  --output data/raw/flows.csv \
  --source cic_ids2017

# Training with progress monitoring
python train_monitor.py  # See TRAINING_SETUP.md for the script

# After training completes
python -m evaluation.benchmark
python -m src.rollout
python -m src.explain
```

### Output Artifacts
- `weights/world_model.pt` — Trained model
- `weights/scaler.pkl` — Feature scaler
- `weights/threshold.json` — Selected threshold
- `evaluation/results.md` — Test set performance

**Expected Training Time**: 20-40 minutes on RTX 4060  
**Expected Test F1**: 0.70-0.85 (depends on CIC-IDS2017 distribution)

---

## 📊 Constraints

**After EVERY phase**: Run smoke test to confirm nothing is broken
```bash
python -m tests.smoke_test
```

**Do not skip ahead**: Complete phases in order (3 → 4 → 5 → 6 → 7 → 8)

**Training**: Use separate laptop with TRAINING_SETUP.md guide

---

## 📁 Key Files for Next Phase

| File | Purpose |
|------|---------|
| `src/features/extract.py` | Feature extraction (Phase 3) |
| `src/train.py` | Training loop, thresholding (Phases 3-4) |
| `configs/default.yaml` | State vector dimension (Phase 3) |
| `tests/smoke_test.py` | Verification after each phase |
| `TRAINING_SETUP.md` | Separate machine training guide |

---

## ⚠️ Known Constraints

1. **Per-scenario split overlapping intervals**: Temporal leakage check is disabled for dev mode with purge_gap=0. This is expected for per-scenario split but does NOT violate chronological safety within scenarios.

2. **Synthetic data on single day**: dev_synthetic.yaml has purge_gap=0 because all synthetic data is on 2026-01-01. Per-scenario split handles temporal separation.

3. **CIC-IDS2017 column naming**: Some CSVs have "Source IP" instead of "Src IP". diagnose_dataset_type() handles this with COLUMN_ALIASES.

4. **Separate machine**: Phases 3-8 will be implemented on main dev machine. Training (via TRAINING_SETUP.md) runs on separate GPU machine.

---

## Next Command

Begin Phase 3 when ready:
```bash
# Implement graph-derived and packet-proxy features
# See Phase 3 specification above
```
