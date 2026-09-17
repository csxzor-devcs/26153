# Real-Data Hardening Progress

**Status:** Active implementation toward SIH project maturity  
**Target:** Scientifically-defensible network forecasting demonstrator  
**Timeline:** Ongoing (started from Phase 0 audit 2026-09-17)

---

## What Has Been Completed

### Phase 1: Dataset Architecture ✅
**Commit:** `db6e08b`  
**File:** `src/data/prepare.py` (285 lines)

Robust ingestion pipeline for CIC-IDS2018/2017:
- ✅ Accept single CSV or directory of CSVs
- ✅ Normalize column names across dataset variants
- ✅ Validate all required CICFlowMeter columns
- ✅ Safe removal of malformed rows (NaN/Inf handling)
- ✅ Comprehensive logging (rows, class distribution, timestamp range, hosts)
- ✅ Dataset provenance tracking (source, statistics saved to JSON)
- ✅ Optional packet-level features (TTL, retransmit)
- ✅ Tested and working on synthetic data

**Usage:**
```bash
python -m src.data.prepare \
  --input-dir /path/to/CIC-IDS2018/CSVs \
  --output data/raw/flows.csv \
  --source cic_ids2018
```

### Phase 2: Real Dataset Ingestion Integration ✅
**Commit:** `1301b80`  
**Files Modified:** `src/train.py`, `configs/default.yaml`

Extended training pipeline to support real data:
- ✅ Auto-detect pre-processed flows.csv with provenance
- ✅ Auto-invoke prepare.py if raw_input_dir is configured
- ✅ Fall back to synthetic for offline smoke testing
- ✅ Added config fields: `raw_input_dir`, `dataset_source`
- ✅ Backward compatible (smoke test still passes)

**Usage:** Set in `configs/default.yaml`:
```yaml
data:
  raw_input_dir: "/path/to/CIC-IDS2018/CSVs"
  dataset_source: "cic_ids2018"
```

Then run: `python -m src.train`

### Phase 4: Attack Semantics ✅
**Commit:** `9cf86f1`  
**File:** `src/data/attack_semantics.py` (251 lines)

Critical infrastructure for "proactive forecasting" claim:
- ✅ Detect attack onset per host (first window where stage > 0)
- ✅ Classify sequences: pre_attack, during_attack, benign
- ✅ Track lead-time (windows until attack onset)
- ✅ Separate evaluation subsets for distinct claims
- ✅ Early-warning metrics: count, false-alarm rate, median lead-time
- ✅ Tested on synthetic data (correctly identifies ~6% pre-attack, ~4% during, ~90% benign)

**Purpose:** Enable rigorous distinction between:
- **Early-warning:** "Predicted infiltration before attack began" (pre-attack test set)
- **Detection:** "Detected ongoing compromise" (during-attack test set)
- **False alarm rate:** "Wrong alerts on benign traffic"

---

## What Remains To Be Completed

### Phase 3: PCAP Support (Optional but Important) ⏳
**Priority:** Medium (optional enhancement)

- [ ] PCAP parser or feature-extraction tool integration (Zeek, dpkt)
- [ ] Temporal alignment: map packet timestamps to 60s windows
- [ ] Packet-level feature aggregation per window per host
- [ ] Join with flow-level features
- [ ] Tests: missing PCAP, timestamp misalignment, partial coverage

**Note:** Currently packet-level features (TTL, retransmit) degrade to 0.0 if absent. Full PCAP support would enhance these.

### Phase 5: Chronological Split & Leakage Tests ⏳
**Priority:** High (required for temporal forecasting claim)

- [ ] Implement time-ordered train/val/test split
- [ ] Respect attack-session boundaries during split
- [ ] Automated leakage tests:
  - [ ] Same-session windows don't span train/test
  - [ ] No temporal leakage (future in training)
  - [ ] Scaler fit on training only (already correct)
  - [ ] No label information from test during training
- [ ] Optional host-disjoint split (for population generalization)
- [ ] Default to chronological; option for random host split

**Files to modify:** `src/features/windowing.py`, new test suite

### Phase 6-10: Model & Evaluation Correctness ⏳
**Priority:** High

- [ ] Verify next-state regression actually learns trajectories (not just auxiliary)
- [ ] Verify rollout doesn't diverge unrealistically
- [ ] Add MC-dropout verification (check that uncertainty actually widens with K)
- [ ] Static neural ablation (single-window LSTM for comparison vs. temporal LSTM)
- [ ] Threshold selection using validation set (not hardcoded 0.5)
- [ ] Real-data benchmark with all three models (LR, Static, Temporal)

### Phase 11: Integration of Attack Semantics into Training ⏳
**Priority:** High

- [ ] Update `src/features/windowing.py` to call `build_sequences_with_status()`
- [ ] Filter sequences (optional): keep only pre-attack for training
- [ ] Add flags to config: `use_pre_attack_only`, `separate_pre_attack_test`
- [ ] Save sequence status to `test_split.npz` for benchmark
- [ ] Report both "overall metrics" and "pre-attack early-warning metrics"

**Files to create/modify:**
- New: `src/features/windowing_v2.py` (backward-compatible alternative)
- Modify: `src/train.py` (wire in new windowing if flag set)
- Modify: `evaluation/benchmark.py` (report early-warning metrics)

### Phase 12: Early-Warning Evaluation ⏳
**Priority:** Critical

- [ ] Benchmark script computes `compute_lead_time_metrics()`
- [ ] Report:
  - [ ] Standard metrics: F1, precision, recall, FPR, AUC-ROC (for detection)
  - [ ] Early-warning metrics: lead-time, % attacked warned, false-alarm rate (for forecasting)
  - [ ] Separate tables: pre-attack evaluation, during-attack evaluation
- [ ] Output to `evaluation/results.md` with clear captions
- [ ] Document which claim each metric supports

**Files to modify:** `evaluation/benchmark.py`

### Phase 13: Updated Streamlit Demo ⏳
**Priority:** Medium

- [ ] Show scenario metadata: "Pre-attack scenario" vs. "During-attack scenario"
- [ ] Display lead-time if known
- [ ] Show class distribution in test set
- [ ] Add mode selector: "Detection metrics" vs. "Forecasting metrics"
- [ ] Use real-data model if available (fallback to synthetic)

**Files to modify:** `app/streamlit_app.py`

### Phase 14: Test Suite Expansion ⏳
**Priority:** High (prevent regression)

Create `tests/` subdirectory with:
- [ ] `test_prepare.py` — dataset preparation
- [ ] `test_features.py` — feature extraction, NaN/Inf handling
- [ ] `test_windowing.py` — sequence construction, leakage detection
- [ ] `test_attack_semantics.py` — attack onset detection, status classification
- [ ] `test_scaler_leakage.py` — scaler is train-only
- [ ] `test_rollout.py` — autoregressive rollout divergence check
- [ ] `test_model_shapes.py` — input/output dimensions correct
- [ ] `test_baseline_equivalence.py` — LR baseline uses correct split

**Command:** `pytest tests/ -v`

### Phase 15: Documentation Updates ⏳
**Priority:** High

- [ ] Update README.md:
  - [ ] Real dataset pathways documented ✅ (done)
  - [ ] Forecasting vs. detection explained ✅ (done)
  - [ ] Early-warning definition and metrics
  - [ ] Prepare.py usage
  - [ ] Attack semantics explanation
- [ ] Update `techsoln.md` section 8 with exact split strategy once finalized
- [ ] Create architecture diagram showing data flow (state → sequences → split → train)
- [ ] Document known limitations (real data required, lead-time estimation uncertainty, etc.)

### Phase 16: Real-Data Benchmark Run ⏳
**Priority:** Critical (project deliverable)

Once all above phases complete:
1. [ ] Acquire CIC-IDS2018 CSV files
2. [ ] Run `python -m src.data.prepare`
3. [ ] Run `python -m src.train`
4. [ ] Run `python -m evaluation.benchmark`
5. [ ] Document results:
   - [ ] Dataset stats (rows, hosts, scenarios, duration)
   - [ ] Class distribution
   - [ ] Detection metrics (F1, precision, recall, FPR, AUC)
   - [ ] Forecasting metrics (lead-time, % warned, false-alarm rate)
   - [ ] Comparison: LR baseline, Static model, Temporal model

### Phase 17: Media Deliverables ⏳
**Priority:** Medium (for judges/demo)

- [ ] ≤2-page architecture document (architecture.pdf)
- [ ] ≤5-slide technical deck (presentation.pdf)
- [ ] ≤2-minute demo video (demo.mp4 or link)

---

## Current Test Status

### ✅ Synthetic Smoke Test (Passes)
```bash
python -m tests.smoke_test
```
- Generates synthetic dataset
- Trains world model + baseline
- Runs rollout, explain, benchmark
- Takes ~5-10 minutes on CPU

### ⏳ Real-Data Test (Pending)
- Requires CIC-IDS2018 CSV files
- Will validate entire hardened pipeline on real data

### ⏳ Unit Tests (Not Yet Implemented)
- See Phase 14 above

---

## Known Issues & Design Decisions

### 1. Synthetic vs. Real Performance Gap
**Issue:** On synthetic data, LSTM world model underperforms LR baseline (F1: 0.614 vs 0.722)  
**Diagnosis:** Synthetic attack signatures are too single-window-separable  
**Expected:** Real CIC-IDS2018 should show LSTM advantage (attacks are smeared across windows)  
**Mitigation:** Honest reporting; separate synthetic/real benchmarks

### 2. Exfiltration is Heuristic
**Issue:** No dataset ships ground-truth exfiltration labels  
**Solution:** Z-score heuristic (outbound byte spike after Bot/Infiltration)  
**Mitigation:** Explicitly mark as heuristic in all reports

### 3. Attack Onset Detection Limitations
**Issue:** Assumes first stage > 0 window is true attack start  
**Edge case:** False positives in flow labeling could misidentify onset  
**Mitigation:** Careful validation on real data; document assumption in reports

### 4. Lead-Time Measurement Uncertainty
**Issue:** Lead-time measured at window granularity (60s bins); true attack onset may be within a window  
**Mitigation:** Report as "windows of lead-time" not "exact seconds"

---

## Integration Checklist

Before using results for project submission:

- [ ] Real-data ingestion tested with actual CIC-IDS2018 CSVs
- [ ] Chronological split implemented and leakage tests passing
- [ ] Attack semantics integrated into training pipeline
- [ ] Early-warning metrics reported separately from detection metrics
- [ ] Synthetic and real benchmarks clearly separated in results
- [ ] All claims (forecasting, early-warning, MITRE mapping) explicitly scoped
- [ ] Documentation reflects actual implementation, not aspirational features
- [ ] Smoke test still passes (regression test)
- [ ] Unit test suite at >80% coverage

---

## Next Immediate Action

**Priority 1 (Blocking everything else):**
```bash
# Implement Phase 5: Chronological split
# File: src/features/windowing.py
# Add: chronological_time_split() function
# Test: test_windowing.py with leakage checks
```

**Priority 2 (Enables evaluation):**
```bash
# Implement Phase 11: Wire attack semantics into training
# File: src/train.py
# Add: use_attack_semantics flag to config
# Test: Full smoke test on synthetic with new windowing
```

**Priority 3 (Validates claims):**
```bash
# Implement Phase 12: Early-warning metrics
# File: evaluation/benchmark.py
# Add: compute_lead_time_metrics() call
# Test: Verify metrics make sense on synthetic data
```

---

## Contact & Status Updates

- **Audit Report:** `AUDIT_REPORT.md` (comprehensive gap analysis)
- **Technical Design:** `../techsoln.md` (design decisions, rationale)
- **Plain Language:** `../human.md` (high-level explanation)
- **Implementation Plan:** `../implementation_plan.md` (original vision)

Any questions about phasing, architecture, or requirements: refer to these documents first.
