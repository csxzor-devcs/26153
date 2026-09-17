# AUDIT REPORT: Network World Model Implementation

**Date:** 2026-09-17  
**Status:** Pre-hardening audit (PHASE 0)  
**Purpose:** Document current implementation, identify gaps, and prioritize fixes for real-data evaluation

---

## EXECUTIVE SUMMARY

The prototype implements the core architecture described in `human.md` and `techsoln.md`:
- LSTM/Transformer world model with multi-task learning
- Attention + SHAP explainability
- K-step autoregressive rollout with MC-dropout uncertainty
- Logistic regression baseline
- Streamlit demo
- End-to-end smoke test

**However, the implementation is currently optimized for synthetic data verification, not real-data evaluation.**

**Critical gaps blocking the SIH project's core claim** ("forecast attacks before they complete"):
1. No real dataset ingestion (CIC-IDS2018/2017)
2. No PCAP support (designed in docs, not implemented)
3. No pre-attack evaluation (can't prove "early warning" vs "detection during attack")
4. No chronological split (only random host-level split)
5. No temporal leakage tests
6. No session/attack-boundary awareness
7. No lead-time metrics
8. Forecasting target semantics are under-specified

---

## A. IMPLEMENTATION vs. HUMAN.MD

| Aspect | Status | Notes |
|--------|--------|-------|
| Weather-model analogy clearly stated | ✅ | README.md references this |
| 60-second state snapshots (S_t) | ✅ | Implemented in `src/features/extract.py` |
| LSTM as default, Transformer as option | ✅ | Both implemented; LSTM is default |
| K-step rollout (controlled daydream) | ✅ | Implemented in `src/rollout.py` |
| Never-trust-black-box philosophy | ⚠️ | Attention + SHAP implemented, but scoped to current state only |
| Benchmark world model vs. LR baseline | ✅ | Implemented, but only on synthetic data |
| Explanation of why not full GNN | ✅ | Documented in `techsoln.md` with upgrade path |
| Explanation of why LSTM not Transformer-first | ✅ | Documented in `human.md` |
| Explanation of why LR not random forest | ✅ | Documented in `human.md` |
| MITRE mapping assumption documented | ✅ | Explicitly stated in `src/mitre_mapping.py` and `techsoln.md` |

**Assessment:** Design document is well-realized in code. ✅

---

## B. IMPLEMENTATION vs. IMPLEMENTATION_PLAN.MD

**Days 0-5 objectives (creation of artifacts):**
- Day 0: Smoke test ✅ (implemented, passes on synthetic data)
- Day 1: State extraction ✅
- Day 2: Windowing + split ✅
- Day 3: Model + loss ✅
- Day 4: Rollout + explain ✅
- Day 5: Streamlit demo ✅

**Day 6-7 objectives (deliverables for judges):**
- ≤2-page architecture doc ❌ (not yet created)
- ≤5-slide technical deck ❌ (not yet created)
- ≤2-minute demo video ❌ (not yet created)

**Assessment:** Core pipeline complete; media deliverables pending. ✅ (for now)

---

## C. IMPLEMENTATION vs. TECHSOLN.MD

| Section | Coverage |
|---------|----------|
| 1. Problem restatement | ✅ Implemented |
| 2. Data & assumptions | ⚠️ Partially implemented (only synthetic; real dataset path exists but untested) |
| 3. State representation S_t | ✅ Fully implemented (20 flow + 2 optional packet features) |
| 4. Sequence construction | ✅ Implemented (with minor caveat: forecasting target semantics not enforced) |
| 5. World model architecture | ✅ Both LSTM and Transformer implemented |
| 6. K-step rollout | ✅ Implemented with MC-dropout |
| 7. Explainability | ✅ Attention + SHAP implemented |
| 8. Baseline & split methodology | ⚠️ Host-level split implemented, but not chronological; no leakage tests |
| 9. System architecture | ✅ Matches diagram |
| 10. Repository layout | ✅ Matches structure |
| 11. Deliverables traceability | ⚠️ Core pipeline present; media artifacts missing |
| 12. Known limitations | ✅ Documented (exfil heuristic, synthetic data caveat, LSTM underperforms on synthetic) |

**Assessment:** Technical design 90% realized in code; real-data pathway and validation infrastructure missing.

---

## D. WHAT IS ACTUALLY IMPLEMENTED

### Core Pipeline (End-to-End)
1. **Flow ingestion** (`src/train.py::load_flows()`)
   - Loads CSV at `data.raw_flows_path` if exists
   - Falls back to bundled synthetic generator
   - No validation of required columns
   - No handling of malformed rows
   - No logging of data quality

2. **Feature extraction** (`src/features/extract.py`)
   - 20 flow-derived features aggregated per (host, 60s window)
   - 2 optional packet features (TTL, retransmit) default to 0.0
   - Applies exfiltration heuristic (z-score spike detection)
   - Feature names hardcoded in `FEATURE_COLUMNS`

3. **Windowing & split** (`src/features/windowing.py`)
   - Builds sequences: X = [S_{t-19}...S_t], y_next = S_{t+1}, y_inf = 1[any S_{t+1..t+5} is attack]
   - Host-level split (shuffles unique hosts, no chronological ordering)
   - Returns per-sequence metadata (host, timestamp)
   - **No attack-session awareness**

4. **Scaling** (`src/train.py`)
   - StandardScaler fit on training split only ✅
   - Applied to train/val/test correctly ✅
   - Saved to `weights/scaler.pkl` ✅

5. **Model training** (`src/train.py`, `src/models/world_model.py`)
   - LSTM encoder (2 layers, 64 hidden, 0.2 dropout)
   - Three heads: next_state (MSE), infiltration (BCE), stage (CE)
   - Multi-task loss with equal weights
   - Trained for 60 epochs on CPU or GPU

6. **Rollout** (`src/rollout.py`)
   - Genuinely autoregressive ✅ (feeds predicted state back in)
   - Returns probability trajectory, stage trajectory, uncertainty band
   - MC-dropout uncertainty (20 samples by default)
   - **Does NOT enforce pre-attack semantics**

7. **Explainability** (`src/explain.py`)
   - Attention weights across T=20 windows ✅
   - SHAP KernelExplainer on last state only ✅
   - Correctly scoped to current-state attribution ✅

8. **Baseline** (`src/models/baseline_lr.py`)
   - Logistic regression on S_t only (no history)
   - Same train/test split as world model ✅
   - Metrics: F1, precision, recall, FPR ✅

9. **Benchmark** (`evaluation/benchmark.py`)
   - Compares world model vs. LR on test split
   - Outputs F1, precision, recall, FPR, AUC-ROC
   - Marks synthetic data with explicit caveat ✅
   - **Does NOT report lead-time or early-warning rate**

10. **Demo** (`app/streamlit_app.py`)
    - Loads trained weights, renders forecast trajectory, MITRE stage, attention, SHAP
    - Correctly distinguishes attack/benign sessions
    - **Does NOT distinguish pre-attack from in-attack scenarios**

---

## E. WHAT IS ONLY DOCUMENTED (NOT IMPLEMENTED)

| Feature | Documented In | Implementation Status |
|---------|---|---|
| PCAP feature extraction | `techsoln.md` sec 3 | ❌ Not implemented (features default to 0) |
| Full GNN upgrade path | `techsoln.md` sec 9 | ❌ Not implemented (documented as Phase 2) |
| Transformer variant | `techsoln.md` sec 5.2 | ✅ Implemented, but not exercised |
| Dataset provenance tracking | `techsoln.md` | ❌ Not implemented |
| Chronological evaluation split | `techsoln.md` sec 8 | ❌ Not implemented (only random host split) |
| Pre-attack evaluation | `techsoln.md` sec 6 | ❌ Not implemented |
| Lead-time metrics | `techsoln.md` sec 8 | ❌ Not implemented |
| Session/scenario awareness | `human.md`, `techsoln.md` | ❌ Not implemented |

---

## F. SYNTHETIC vs. REAL DATA SUPPORT

| Aspect | Status |
|--------|--------|
| Synthetic data generator | ✅ Fully functional, embedded in `src/data/synthetic_flows.py` |
| Synthetic benchmark | ✅ Runs successfully, caveat documented |
| Real CIC-IDS2018 CSV ingest | ⚠️ **Path only** — code expects file at `data/raw/flows.csv`, no validation or preprocessing |
| Real CIC-IDS2017 support | ❌ Not documented how to merge multiple CSVs |
| Dataset preparation pipeline | ❌ No `python -m src.data.prepare` command |
| Missing-value handling | ⚠️ `fillna(0.0)` — too permissive for NaN (attack indicators?) |
| Malformed row handling | ❌ No validation, rows silently drop on NaN in numeric fields |
| Data quality logging | ❌ No row counts, no class distribution, no timestamp range logged |
| Dataset provenance in artifacts | ❌ `used_config.yaml` doesn't record whether synthetic or real was used |

**Assessment:** Real data path exists but is untested and lacks data-quality infrastructure.

---

## G. PCAP SUPPORT STATUS

**Current state:** Designed in `techsoln.md` sec 3, not implemented.

1. Optional packet-level features documented:
   - TTL mean/std/entropy
   - TCP window mean/std
   - Fragmentation ratio
   - Payload size distribution
   - Retransmission count

2. Current implementation:
   - If `TTL` column exists in input CSV, use it; else default to 0.0
   - If `Retransmit Cnt` column exists, use it; else default to 0.0
   - No PCAP parsing code exists
   - No mapping from PCAP flow timings to 60s window boundaries

3. **What would be needed for real PCAP support:**
   - PCAP parser (e.g., `dpkt`, `scapy`, or pre-extracted feature CSV from tools like Zeek)
   - Temporal alignment: map packet timestamps to 60s windows matching flow data
   - Aggregation: compute packet-level features per window per host
   - Join: merge with flow-level features
   - Tests: handle missing PCAP, misaligned timestamps, partial coverage

---

## H. LABEL / MITRE MAPPING CORRECTNESS

**Mapping (from `src/mitre_mapping.py`):**

| Dataset Label | MITRE Stage | Ground Truth? |
|---|---|---|
| `Benign` | Benign/None (0) | ✅ Yes |
| `PortScan` | Reconnaissance (1) | ✅ Yes (CIC-IDS2017) |
| `FTP-BruteForce`, `SSH-Bruteforce`, etc. | Initial Access (2) | ✅ Yes |
| `Infilteration` | Lateral Movement (3) | ✅ Yes |
| `Bot` | Command & Control (4) | ✅ Yes |
| Heuristic (z-score spike + Bot/Infilteration) | Exfiltration (5) | ⚠️ **Heuristic only** |
| `DoS`, `DDoS` | Impact/Noise (6) | ✅ Yes (out of MITRE 5-stage scope) |

**Issues:**
1. ✅ Exfiltration explicitly marked as heuristic
2. ✅ Mapping documented in source
3. ⚠️ No validation report generated (would help auditors)
4. ⚠️ Unknown labels silently map to Benign(0) — safe but can hide data problems
5. ❌ No per-dataset-source mapping report (which dataset → which stages used)

**Assessment:** Mapping is correct and well-documented; could benefit from data-quality report.

---

## I. DATA LEAKAGE RISKS

### Critical Issues

**1. Same-attack-session leakage across train/test**
- **Risk:** Host-level split is random, not chronological
- **Impact:** Attack sessions that span chronologically can have windows in both train and test
- **Example:** A Bot→C2→Exfil session spanning hours, split halfway through
- **Mitigation status:** ❌ Not enforced. No tests.
- **Fix required:** Chronological split + attack-session-aware masking

**2. Attack-onset timing leakage**
- **Risk:** Sequences constructed without knowing where attacks begin
- **Impact:** A sequence labeled "y_inf=1" might already contain attack traffic, making "forecast" into "detection during attack"
- **Semantics issue:** Does "y_inf=1 if any S_{t+1..t+K} is attack" mean "any future window" or "attack begins in that window"?
- **Mitigation status:** ❌ Not enforced. Ambiguous semantics.
- **Fix required:** Per-session attack-onset detection + pre-attack sequence filtering

**3. Scaler leakage** ✅ **NOT A RISK**
- StandardScaler fit on training split only
- Correctly applied to train/val/test
- Correctly saved and reloaded

**4. Label leakage in y_stage**
- **Risk:** y_stage = stage_label[t] (not t+1), so input X_t sees the stage it's predicting
- **Minor issue:** Technically y_stage should be stages[t:t+K] (horizon stage), not stages[t]
- **Current semantics:** "Predict current stage, given current history"
- **Intended semantics:** "Predict future stage, given past history"
- **Mitigation status:** ⚠️ Design choice not explicitly validated

---

## J. FORECASTING-VALIDITY RISKS

**Critical: The system cannot currently prove "early-warning"**

### Problem Statement

The brief asks for: **"forecast attacks before they complete"** — i.e., predict at time t that an attack will reach a certain stage by time t+K.

Current implementation can answer: **"given a sequence up to time t, is at least one future window an attack?"**

But it **cannot answer**: **"given a pre-attack sequence (no attack traffic yet), forecast that an attack is coming?"**

### Why This Matters

1. **Detection vs. Forecasting:**
   - Detection: "There is attack traffic right now" (y_inf=1 at window t)
   - Forecasting: "Attack traffic will appear in the future" (y_inf=1 predicted at t for windows t+1..t+K)

2. **Current implementation conflates them:**
   - Sequences are built from all windows in a session, regardless of attack onset
   - A sequence at window t=50 (deep in a C2 session) is labeled "forecast attack" when really it's "detect current attack"
   - Validation set contains mostly in-attack sequences, not pre-attack sequences

3. **Can't measure lead-time:**
   - No tracking of "first prediction > threshold before attack onset"
   - No distinction between "true early warning" and "detected during attack"

### What Would Fix This

1. **Per-session attack-onset detection**
   - For each attack session, identify t_onset = first window with stage > 0
   - Mark all windows t < t_onset as "pre-attack"
   - Mark windows t >= t_onset as "during-attack"

2. **Pre-attack sequence filtering**
   - Only use pre-attack sequences for training
   - Evaluate separately on pre-attack test set
   - Measure: "what % of attacks are warned before t_onset?"

3. **Lead-time metrics**
   - For each correctly-warned attack, compute: t_warning - t_onset
   - Report: median lead time, mean lead time, % of attacks warned > N windows early

---

## K. EVALUATION / SPLIT PROBLEMS

| Issue | Current | Correct | Risk |
|-------|---------|---------|------|
| **Split method** | Random host-level shuffle | Chronological by session | Can't prove temporal forecasting works |
| **Leakage test** | None | Automated test to verify no same-session in train/test | Silent contamination |
| **Attack-session boundary** | Not tracked | Detected and respected | Pre-attack claims invalid |
| **Pre-attack subset** | Not separated | Separate train/val/test for pre-attack | Can't measure early-warning |
| **Lead-time metric** | Not computed | Tracked per-attack | Can't prove "before completion" |
| **Chronological validation** | Not enforced | Strict time-ordering | Overstates generalization |

**Assessment:** Current evaluation is insufficient for "proactive forecasting" claim.

---

## L. EXPLAINABILITY CORRECTNESS

### Temporal Attention
- **Implementation:** ✅ Additive attention over T=20 windows
- **Semantics:** ✅ "Which past windows mattered?"
- **Correctness:** ✅ Computed from LSTM hidden states, softmax normalized
- **Caveat:** ⚠️ Explains current prediction, not future rollout predictions

### SHAP Attribution
- **Implementation:** ✅ KernelExplainer on state-swapping function
- **Semantics:** ⚠️ "Which current features drove current prediction?"
- **Scoping:** ✅ Correctly limited to S_t (last window), not full sequence
- **Caveat:** ❌ Does NOT explain why the model predicts S_{t+1} (next-state regression)
- **Caveat:** ❌ Does NOT explain which features matter for future rollout steps

**Assessment:** Explainability is correctly scoped but incomplete (current-state only).

---

## M. MODEL / ROLLOUT CORRECTNESS

### Next-State Prediction
- **Architecture:** ✅ Linear head from context vector
- **Loss:** ✅ MSE on scaled target
- **Target semantics:** ⚠️ y_next = S_{t+1} — but is this actually used to understand trajectory, or just auxiliary?
- **Rollout use:** ✅ Predicted next_state is fed back in at each step
- **Question:** Does the model actually learn predictive state transitions, or just class labels? (Can verify by checking if rollout diverges from real trajectories.)

### Infiltration Head
- **Loss:** ✅ BCE with y_inf = 1[any S_{t+1..t+K} is attack]
- **Semantics:** ⚠️ Predicts "will see attack" not "attack will START"
- **Threshold:** ✅ 0.5 is used in benchmark, but not optimized per-dataset
- **Concern:** Threshold should be selected using validation set, not hardcoded

### Stage Head
- **Loss:** ✅ Cross-entropy on stage labels
- **Semantics:** ⚠️ Predicts stage of S_{t+1}, which may be redundant with infiltration head if stages are known
- **Rollout:** ✅ argmax stages predicted at each step
- **Concern:** Is predicting next stage actually helpful, or should we predict stage trajectory over K windows?

### Autoregressive Rollout
- **Implementation:** ✅ Feeds predicted state back in correctly
- **Verification:** ⚠️ No test to confirm predicted state ≠ original history after K steps
- **Stability:** ⚠️ No check for divergence (does predicted trajectory grow unrealistic values?)

---

## N. BENCHMARK CORRECTNESS

**Current results (synthetic data):**
- World Model F1: 0.614, AUC: 0.666
- Logistic Regression F1: 0.722, AUC: 0.815
- **Caveat documented:** Synthetic signatures too single-window-separable; expected to flip on real data

**Issues:**
1. ❌ No real-data benchmark yet
2. ⚠️ Threshold (0.5) not optimized; should use validation set to select
3. ❌ No early-warning metrics
4. ❌ No lead-time reported
5. ❌ No ablation (static LSTM vs. LSTM with history)
6. ✅ Caveat is honest and documented

---

## O. OFFLINE STREAMLIT DEMO STATUS

**Implementation:** ✅ Fully functional
- Loads pre-trained weights
- Renders forecast trajectory + uncertainty band
- Shows MITRE stage trajectory
- Displays attention heatmap
- Displays SHAP attributions

**Issues:**
1. ⚠️ Hard-codes selection of "most confident attack" sample (good for demo, but hides false negatives)
2. ❌ Does not distinguish "pre-attack scenario" from "in-attack scenario"
3. ❌ Does not show lead-time information
4. ❌ Cannot demo early-warning (since no pre-attack sequences in test set)

---

## EXACT PRIORITIZED FIXES

Based on the above audit, here is the exact priority order for hardening the system:

### PHASE 1: DATASET ARCHITECTURE (Blocks everything else)
1. **Create `src/data/prepare.py`** — real dataset ingestion pipeline
   - Accept directory of CIC-IDS2018 CSVs
   - Normalize/validate columns
   - Handle malformed rows
   - Log data quality (row counts, class distribution, timestamp range, unique hosts)
   - Save processed dataset with provenance
   - Command: `python -m src.data.prepare`

### PHASE 2: REAL INGESTION (Can't evaluate without data)
2. **Extend `src/train.py`** — support real data mode
   - Call `src/data.prepare` if CSV not yet processed
   - Ingest real CIC-IDS2018
   - Add dataset provenance to config
   - Log which dataset was used

### PHASE 3: PCAP (Optional but important)
3. **Design PCAP feature extraction** (if data available)
   - Decide: pre-parsed feature CSV or raw PCAP?
   - Implement packet-level aggregation
   - Join with flow data
   - Make optional (fall back to zeros)

### PHASE 4: ATTACK SEMANTICS (Critical for forecasting claim)
4. **Implement attack-session detection**
   - Per-host: identify first window where stage > 0
   - Mark pre-attack, during-attack, post-attack windows
   - Add field `forecast_status` to sequences
   - Filter training to include pre-attack-only sequences

5. **Fix forecasting target semantics**
   - Change y_inf definition (if needed): should it be "attack *begins*" or "attack *present*"?
   - Ensure pre-attack sequences exist for evaluation

### PHASE 5: SPLIT STRATEGY (Can't prove temporal learning without this)
6. **Implement chronological split**
   - Add option for time-ordered train/val/test
   - Respect attack-session boundaries
   - Add automated leakage tests
   - Default: chronological; optional: host-disjoint

### PHASE 6: NORMALIZATION (Data integrity)
7. **Verify scaler train-only policy**
   - Add test that scaler is fit before any test data is touched
   - Document in README

### PHASE 7: EVALUATION (Measure what matters)
8. **Add early-warning metrics**
   - Lead-time (t_warning - t_onset)
   - % of attacks warned before onset
   - FPR on benign periods
   - Recall at fixed FPR

9. **Add static neural ablation**
   - Single-window LSTM/linear model (no temporal memory)
   - Compare vs. baseline (LR) vs. world model
   - Isolate temporal contribution

### PHASE 8: REAL BENCHMARK (Prove claims)
10. **Run on real CIC-IDS2018**
    - Generate results.md with real data
    - Separate from synthetic results
    - Report lead-time metrics
    - Report early-warning rate

### PHASE 9: DEMO (Show it works)
11. **Update Streamlit demo**
    - Distinguish pre-attack scenarios
    - Show lead-time information
    - Use real-data model if available

### PHASE 10: TESTING (Prevent regression)
12. **Add unit tests**
    - Feature extraction
    - Windowing / leakage
    - Scaler policy
    - Rollout autoregression
    - Label mapping
    - SHAP feature order

### PHASE 11: DOCS (Judge-defensible)
13. **Update README.md**
    - Real vs. synthetic pathways
    - PCAP optional support
    - Early-warning definition
    - How to reproduce results

---

## SUMMARY TABLE

| Component | Status | Confidence | Risk |
|-----------|--------|-----------|------|
| **Core Architecture** | ✅ Implemented | High | Low |
| **Synthetic Benchmark** | ✅ Working | High | Low |
| **Real Dataset Support** | ⚠️ Path only | Medium | High |
| **Attack Semantics** | ❌ Undefined | Low | **Critical** |
| **Pre-Attack Evaluation** | ❌ Missing | Low | **Critical** |
| **Forecasting Claim Validity** | ❌ Unproven | Low | **Critical** |
| **Chronological Split** | ❌ Missing | Low | High |
| **Temporal Leakage Tests** | ❌ Missing | Low | High |
| **Lead-Time Metrics** | ❌ Missing | Low | High |
| **PCAP Support** | ❌ Not implemented | Low | Medium |
| **Explainability Completeness** | ⚠️ Current-state only | Medium | Medium |
| **Demo Realism** | ⚠️ Hides false negatives | Medium | Medium |

---

## NEXT STEPS

Proceed to **PHASE 1** (Dataset Architecture) immediately. Do not attempt real-data evaluation until:
1. Dataset preparation pipeline exists
2. Attack-session awareness is implemented
3. Chronological split is in place
4. Early-warning metrics are defined

All fixes should preserve the existing synthetic-data smoke test as a regression guard.
