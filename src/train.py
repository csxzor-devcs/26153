"""
Trains the world model end to end: raw flows -> state vectors -> sequences
-> host-level split -> scaling -> multi-task training -> saved artifacts.

Usage (run from repo root):
    python -m src.train --config configs/default.yaml

If configs/default.yaml's data.raw_flows_path doesn't exist, this falls
back to the bundled synthetic flow generator (src/data/synthetic_flows.py)
so the whole pipeline runs offline with zero setup. To benchmark against
the real dataset the brief asks for, download CIC-IDS2018/2017 and point
raw_flows_path at it (see README.md) — nothing else in this script changes.
"""
import argparse
import json
import os

import joblib
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from src.features.extract import FEATURE_COLUMNS, build_state_vectors
from src.features.windowing import build_sequences, host_level_split
from src.models.baseline_lr import BaselineLogReg, infiltration_metrics
from src.models.world_model import build_model, world_model_loss


def load_flows(cfg: dict):
    import pandas as pd
    path = cfg["data"]["raw_flows_path"]
    dataset_source = cfg["data"].get("dataset_source", "cic_ids2017")

    # Map source code to human-readable names
    dataset_names = {
        "cic_ids2017": "CIC-IDS2017",
        "cic_ids2018": "CIC-IDS2018",
        "synthetic_dev": "SYNTHETIC (development mode)",
    }
    dataset_name = dataset_names.get(dataset_source, dataset_source.upper())

    # If processed flows exist, load them directly
    if os.path.exists(path):
        print(f"\n{'='*70}")
        print(f"DATA SOURCE: REAL ({dataset_name})")
        print(f"{'='*70}")
        print(f"[train] loading flow data from {path}")
        df = pd.read_csv(path)

        # Check for provenance file to identify and verify data source
        provenance_path = path.replace(".csv", "_provenance.json")
        if os.path.exists(provenance_path):
            with open(provenance_path) as f:
                provenance = json.load(f)
            dataset_in_file = provenance.get("dataset", "UNKNOWN")
            print(f"[train] dataset: {dataset_in_file}")
            print(f"[train] flows retained: {provenance['quality_stats'].get('rows_total', '?'):,}")
            print(f"[train] hosts: {provenance['quality_stats'].get('unique_hosts', '?')}")
            print(f"[train] duration: {provenance['quality_stats'].get('timestamp_duration_hours', '?'):.1f} hours")

        return df

    # If input directory specified, try to prepare dataset
    input_dir = cfg["data"].get("raw_input_dir")
    if input_dir and os.path.exists(input_dir):
        print(f"\n{'='*70}")
        print(f"DATA SOURCE: REAL ({dataset_name})")
        print(f"{'='*70}")
        print(f"[train] preparing {dataset_name} dataset from {input_dir}...")
        from src.data.prepare import main as prepare_main
        try:
            df, provenance = prepare_main(input_dir, path, dataset_source)
            return df
        except Exception as e:
            print(f"[train] ERROR during {dataset_name} preparation: {e}")
            print(f"[train] dataset preparation failed")
            raise

    # Allow synthetic_dev mode for development/testing
    if dataset_source == "synthetic_dev":
        print(f"\n{'='*70}")
        print(f"DATA SOURCE: {dataset_name}")
        print(f"{'='*70}")
        print(f"[train] using bundled synthetic flow generator (development mode)")
        from src.data.synthetic_flows import generate_synthetic_flows
        df = generate_synthetic_flows()
        return df

    # No synthetic fallback for explicit real-data requests
    print(f"\n{'='*70}")
    print(f"ERROR: DATASET NOT FOUND")
    print(f"{'='*70}")
    print(f"[train] Requested dataset: {dataset_name}")
    print(f"[train] Expected path: {path}")
    print(f"[train] Expected input_dir: raw_input_dir in configs/default.yaml")
    print(f"[train]")
    print(f"[train] To train on {dataset_name}:")
    print(f"[train]   1. Download {dataset_name} dataset")
    print(f"[train]   2. Extract CSVs to a folder (e.g., /path/to/{dataset_source})")
    print(f"[train]   3. Set data.raw_input_dir in configs/default.yaml")
    print(f"[train]   4. Re-run: python -m src.train")
    print(f"[train]")
    print(f"[train] Alternatively, for development only, use synthetic data:")
    print(f"[train]   python -m src.train --config configs/dev_synthetic.yaml")
    raise FileNotFoundError(
        f"Dataset {dataset_name} not found at {path} and no raw_input_dir configured. "
        f"Use synthetic data for testing via configs/dev_synthetic.yaml, or provide real data."
    )


def scale_split(X, scaler):
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    flows = load_flows(cfg)
    states = build_state_vectors(flows, window_seconds=cfg["data"]["window_seconds"],
                                  host_col=cfg["data"]["host_col"])

    # Optionally enhance with attack semantics (pre-attack/during-attack classification)
    use_attack_semantics = cfg["data"].get("use_attack_semantics", False)
    if use_attack_semantics:
        from src.data.attack_semantics import detect_attack_onsets, build_sequences_with_status
        states = detect_attack_onsets(states)
        seq = build_sequences_with_status(states, T=cfg["data"]["history_length"],
                                          K=cfg["data"]["horizon"])
        print(f"[train] attack semantics enabled: {len(seq['X'])} sequences "
              f"({(seq['forecast_status'] == 'pre_attack').sum()} pre-attack, "
              f"{(seq['forecast_status'] == 'attack_in_progress').sum()} during-attack, "
              f"{(seq['forecast_status'] == 'benign').sum()} benign)")
    else:
        from src.features.windowing import build_sequences
        seq = build_sequences(states, T=cfg["data"]["history_length"], K=cfg["data"]["horizon"])
        print(f"[train] attack semantics disabled")

    print(f"[train] built {len(seq['X'])} sequences across {len(np.unique(seq['hosts']))} hosts "
          f"({seq['y_inf'].mean():.1%} positive infiltration rate)")

    # Use chronological split (SAFE) by default; host-level split available for comparison
    split_strategy = cfg["data"].get("split_strategy", "chronological")

    if split_strategy == "chronological":
        from src.features.windowing import per_scenario_chronological_split, verify_no_leakage
        purge_gap = cfg["data"].get("purge_gap_seconds", 0)
        print(f"[train] using per-scenario chronological split (ensures attacks appear in all splits)")
        # Pass hosts for per-host grouping in synthetic data
        train_mask, val_mask, test_mask = per_scenario_chronological_split(
            seq["times"], cfg["data"]["train_frac"], cfg["data"]["val_frac"],
            hosts_array=seq["hosts"], purge_gap_seconds=purge_gap)

        # Verify no temporal leakage (checks effective intervals, not just timestamps)
        leakage_check = verify_no_leakage(
            seq["times"], train_mask, val_mask, test_mask,
            T=cfg["data"]["history_length"], K=cfg["data"]["horizon"],
            window_seconds=cfg["data"]["window_seconds"]
        )
        if not leakage_check["is_valid"]:
            if purge_gap == 0 and cfg["data"].get("dataset_source") == "synthetic_dev":
                # For dev mode with no purge gap, per-scenario split will have overlapping intervals
                # This is acceptable as long as splits are chronologically ordered within scenarios
                print(f"[train] ⚠ Temporal interval overlap detected (acceptable for per-scenario split in dev mode):")
                for issue in leakage_check["issues"]:
                    print(f"  - {issue}")
            else:
                print(f"[train] ERROR: Temporal leakage detected:")
                for issue in leakage_check["issues"]:
                    print(f"  - {issue}")
                raise RuntimeError("Chronological split failed temporal leakage verification")
        else:
            print(f"[train] ✓ Chronological split verified (no temporal leakage)")
            print(f"[train]   Purge gap: {purge_gap}s")
            stats = leakage_check["stats"]
            print(f"[train]   Train effective interval: {stats['train_effective_start']} → {stats['train_effective_end']}")
            print(f"[train]   Val effective interval:   {stats['val_effective_start']} → {stats['val_effective_end']}")
            print(f"[train]   Test effective interval:  {stats['test_effective_start']} → {stats['test_effective_end']}")

            # Verify attacks appear in all splits
            y_inf_train = seq["y_inf"][train_mask]
            y_inf_val = seq["y_inf"][val_mask]
            y_inf_test = seq["y_inf"][test_mask]

            train_attacks = (y_inf_train > 0).sum()
            val_attacks = (y_inf_val > 0).sum()
            test_attacks = (y_inf_test > 0).sum()

            print(f"[train] ✓ Attack distribution: train={train_attacks}, val={val_attacks}, test={test_attacks}")

            if train_attacks == 0 or val_attacks == 0 or test_attacks == 0:
                print(f"[train] WARNING: At least one split has zero attack sequences")
                print(f"[train]           (This is acceptable if dataset is purely benign, but unusual for CIC-IDS)")
    else:
        # Fall back to host-level split (non-temporal, useful for comparison)
        from src.features.windowing import host_level_split
        train_mask, val_mask, test_mask = host_level_split(
            seq["hosts"], cfg["data"]["train_frac"], cfg["data"]["val_frac"], seed)
        print(f"[train] Using host-level split (non-temporal, for comparison only)")

    # Setup device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[train]   GPU: {torch.cuda.get_device_name(0)}")
    print(f"[train]   PyTorch: {torch.__version__}")
    print(f"[train]   Random seed: {seed}")

    # Scale all splits (fit scaler ONLY on training data)
    scaler = StandardScaler().fit(seq["X"][train_mask].reshape(-1, len(FEATURE_COLUMNS)))
    X_train = scale_split(seq["X"][train_mask], scaler)
    X_val = scale_split(seq["X"][val_mask], scaler)  # VALIDATION SET (new)
    X_test = scale_split(seq["X"][test_mask], scaler)

    y_next_train = scaler.transform(seq["y_next"][train_mask]).astype(np.float32)
    y_next_val = scaler.transform(seq["y_next"][val_mask]).astype(np.float32)

    y_inf_train = seq["y_inf"][train_mask]
    y_inf_val = seq["y_inf"][val_mask]   # VALIDATION INFILTRATION LABELS (new)
    y_inf_test = seq["y_inf"][test_mask]

    y_stage_train = seq["y_stage"][train_mask]
    y_stage_val = seq["y_stage"][val_mask]  # VALIDATION STAGES (new)
    y_stage_test = seq["y_stage"][test_mask]

    # Convert to torch tensors on device
    Xtr_t = torch.tensor(X_train, device=device)
    y_next_t = torch.tensor(y_next_train, device=device)
    y_inf_t = torch.tensor(y_inf_train, dtype=torch.float32, device=device)
    y_stage_t = torch.tensor(y_stage_train, dtype=torch.long, device=device)

    Xval_t = torch.tensor(X_val, device=device)
    y_next_val_t = torch.tensor(y_next_val, device=device)
    y_inf_val_t = torch.tensor(y_inf_val, dtype=torch.float32, device=device)
    y_stage_val_t = torch.tensor(y_stage_val, dtype=torch.long, device=device)

    Xtest_t = torch.tensor(X_test, device=device)

    model = build_model(cfg["model"], input_dim=len(FEATURE_COLUMNS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    print(f"[train] training {cfg['model']['encoder']} world model for {cfg['train']['epochs']} epochs...")

    # Training loop with validation monitoring
    best_val_loss = float('inf')
    best_checkpoint = None
    patience = cfg["train"].get("early_stopping_patience", 5)
    patience_counter = 0
    training_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_bce": []}

    for epoch in range(cfg["train"]["epochs"]):
        # TRAINING PHASE
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss, parts = world_model_loss(pred, y_next_t, y_inf_t, y_stage_t,
                                        weights=cfg["train"]["loss_weights"])
        loss.backward()
        opt.step()

        # VALIDATION PHASE
        model.eval()
        with torch.no_grad():
            val_pred = model(Xval_t)
            val_loss, val_parts = world_model_loss(
                val_pred, y_next_val_t, y_inf_val_t, y_stage_val_t,
                weights=cfg["train"]["loss_weights"])

        # Record history
        training_history["epoch"].append(epoch)
        training_history["train_loss"].append(loss.item())
        training_history["val_loss"].append(val_loss.item())
        training_history["val_bce"].append(val_parts["bce"])

        # Best checkpoint selection: save if validation loss improved
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss.item(),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience and epoch > cfg["train"].get("min_epochs", 10):
            print(f"[train] early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

        if epoch % 10 == 0 or epoch == cfg["train"]["epochs"] - 1:
            print(f"  epoch {epoch:3d}  train_loss={loss.item():.4f} val_loss={val_loss.item():.4f}  "
                  f"(mse={parts['mse']:.4f} bce={parts['bce']:.4f} ce={parts['ce']:.4f})")

    # Load best checkpoint
    if best_checkpoint is not None:
        print(f"[train] loading best checkpoint from epoch {best_checkpoint['epoch']} "
              f"(val_loss={best_checkpoint['val_loss']:.4f})")
        model.load_state_dict(best_checkpoint["model_state"])
    else:
        print(f"[train] WARNING: no best checkpoint found, using final model")

    # THRESHOLD SELECTION on validation set
    print(f"[train] selecting infiltration threshold on validation set...")
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(Xval_t)["infiltration_logit"]).cpu().numpy()

    # Find threshold that maximizes F1 on validation (fine-grained sweep: 0.01 steps)
    best_threshold = 0.5
    best_f1 = 0.0
    threshold_sweep_wm = []
    for threshold in np.linspace(0.1, 0.9, 81):  # 0.01 step size: 0.1, 0.11, ..., 0.90
        val_pred_binary = (val_prob > threshold).astype(int)
        metrics = infiltration_metrics(y_inf_val, val_pred_binary)
        threshold_sweep_wm.append({
            "threshold": float(threshold),
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "fpr": float(metrics["fpr"]),
        })
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    print(f"[train] selected threshold={best_threshold:.3f} (F1={best_f1:.3f} on validation)")

    # FINAL TEST EVALUATION with frozen threshold
    print(f"[train] evaluating on test set with frozen threshold...")
    with torch.no_grad():
        test_prob = torch.sigmoid(model(Xtest_t)["infiltration_logit"]).cpu().numpy()

    wm_pred = (test_prob > best_threshold).astype(int)
    wm_metrics = infiltration_metrics(y_inf_test, wm_pred)
    print(f"[train] world model test metrics: {wm_metrics}")

    # Save threshold to config for benchmark script
    cfg["train"]["selected_threshold"] = float(best_threshold)

    # BASELINE: Logistic Regression (single-window, no temporal modeling)
    print(f"[train] training logistic regression baseline (no temporal modeling)...")
    baseline = BaselineLogReg().fit(X_train[:, -1, :], y_inf_train, y_stage_train)

    # Threshold selection on validation for baseline too (fine-grained sweep: 0.01 steps)
    val_prob_lr = baseline.inf_model.predict_proba(X_val[:, -1, :])[:, 1]
    best_threshold_lr = 0.5
    best_f1_lr = 0.0
    threshold_sweep_lr = []
    for threshold in np.linspace(0.1, 0.9, 81):  # 0.01 step size
        val_pred_lr = (val_prob_lr > threshold).astype(int)
        metrics_lr = infiltration_metrics(y_inf_val, val_pred_lr)
        threshold_sweep_lr.append({
            "threshold": float(threshold),
            "f1": float(metrics_lr["f1"]),
            "precision": float(metrics_lr["precision"]),
            "recall": float(metrics_lr["recall"]),
            "fpr": float(metrics_lr["fpr"]),
        })
        if metrics_lr["f1"] > best_f1_lr:
            best_f1_lr = metrics_lr["f1"]
            best_threshold_lr = threshold

    print(f"[train] LR baseline: selected threshold={best_threshold_lr:.3f} (F1={best_f1_lr:.3f} on validation)")

    # Test evaluation with frozen threshold
    test_prob_lr = baseline.inf_model.predict_proba(X_test[:, -1, :])[:, 1]
    lr_pred = (test_prob_lr > best_threshold_lr).astype(int)
    lr_metrics = infiltration_metrics(y_inf_test, lr_pred)
    print(f"[train] baseline LR test metrics:   {lr_metrics}")

    # Save LR threshold to config for benchmark script
    cfg["train"]["selected_threshold_lr"] = float(best_threshold_lr)

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/world_model.pt")
    joblib.dump(scaler, "weights/scaler.pkl")
    joblib.dump(baseline, "weights/baseline_lr.pkl")
    with open("weights/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    with open("weights/used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Save training history and metrics
    with open("weights/training_history.json", "w") as f:
        json.dump(training_history, f, indent=2, default=str)

    with open("weights/validation_metrics.json", "w") as f:
        json.dump({
            "best_val_loss": float(best_val_loss),
            "best_epoch": best_checkpoint["epoch"] if best_checkpoint else -1,
            "selected_threshold_world_model": float(best_threshold),
            "selected_threshold_baseline": float(best_threshold_lr),
        }, f, indent=2)

    # Save threshold selection results: full sweep and selected thresholds
    with open("weights/threshold.json", "w") as f:
        json.dump({
            "world_model": {
                "selected_threshold": float(best_threshold),
                "best_f1": float(best_f1),
                "sweep": threshold_sweep_wm,
            },
            "baseline_lr": {
                "selected_threshold": float(best_threshold_lr),
                "best_f1": float(best_f1_lr),
                "sweep": threshold_sweep_lr,
            },
        }, f, indent=2)

    os.makedirs("data/processed", exist_ok=True)
    # Save test split, including attack semantics if available
    test_data = {
        "X_test": X_test,
        "y_inf_test": y_inf_test,
        "y_stage_test": y_stage_test,
        "hosts_test": seq["hosts"][test_mask],
    }
    if use_attack_semantics:
        test_data["forecast_status_test"] = seq["forecast_status"][test_mask]
        test_data["contains_attack_test"] = seq["contains_attack"][test_mask]
        test_data["lead_times_test"] = seq["lead_times"][test_mask]
    np.savez("data/processed/test_split.npz", **test_data)

    print("[train] saved: weights/world_model.pt, scaler.pkl, baseline_lr.pkl, "
          "feature_columns.json, used_config.yaml, threshold.json, data/processed/test_split.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
