"""
Train the world model end to end: raw flows -> state vectors -> sequences
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
from torch.utils.tensorboard import SummaryWriter

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
            print(
                f"[train] flows retained: "
                f"{provenance['quality_stats'].get('rows_total', '?'):,}"
            )
            print(
                f"[train] hosts: "
                f"{provenance['quality_stats'].get('unique_hosts', '?')}"
            )
            print(
                f"[train] duration: "
                f"{provenance['quality_stats'].get('timestamp_duration_hours', '?'):.1f} hours"
            )

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
        print(
            "[train] using bundled synthetic flow generator "
            "(development mode)"
        )

        from src.data.synthetic_flows import generate_synthetic_flows

        df = generate_synthetic_flows()
        return df

    # No synthetic fallback for explicit real-data requests
    print(f"\n{'='*70}")
    print(f"ERROR: DATASET NOT FOUND")
    print(f"{'='*70}")
    print(f"[train] Requested dataset: {dataset_name}")
    print(f"[train] Expected path: {path}")
    print(
        f"[train] Expected input_dir: "
        f"raw_input_dir in configs/default.yaml"
    )
    print(f"[train]")
    print(f"[train] To train on {dataset_name}:")
    print(f"[train]   1. Download {dataset_name} dataset")
    print(
        f"[train]   2. Extract CSVs to a folder "
        f"(e.g., /path/to/{dataset_source})"
    )
    print(
        f"[train]   3. Set data.raw_input_dir in "
        f"configs/default.yaml"
    )
    print(f"[train]   4. Re-run: python -m src.train")
    print(f"[train]")
    print(
        "[train] Alternatively, for development only, "
        "use synthetic data:"
    )
    print(
        "[train]   python -m src.train "
        "--config configs/dev_synthetic.yaml"
    )

    raise FileNotFoundError(
        f"Dataset {dataset_name} not found at {path} and no raw_input_dir "
        f"configured. Use synthetic data for testing via "
        f"configs/dev_synthetic.yaml, or provide real data."
    )


def scale_split(X, scaler):
    shape = X.shape
    return scaler.transform(
        X.reshape(-1, shape[-1])
    ).reshape(shape).astype(np.float32)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    flows = load_flows(cfg)

    states = build_state_vectors(
        flows,
        window_seconds=cfg["data"]["window_seconds"],
        host_col=cfg["data"]["host_col"],
    )

    # Optionally enhance with attack semantics
    # (pre-attack/during-attack classification)
    use_attack_semantics = cfg["data"].get(
        "use_attack_semantics",
        False,
    )

    if use_attack_semantics:
        from src.data.attack_semantics import (
            detect_attack_onsets,
            build_sequences_with_status,
        )

        states = detect_attack_onsets(states)

        seq = build_sequences_with_status(
            states,
            T=cfg["data"]["history_length"],
            K=cfg["data"]["horizon"],
        )

        print(
            f"[train] attack semantics enabled: {len(seq['X'])} sequences "
            f"({(seq['forecast_status'] == 'pre_attack').sum()} pre-attack, "
            f"{(seq['forecast_status'] == 'attack_in_progress').sum()} "
            f"during-attack, "
            f"{(seq['forecast_status'] == 'benign').sum()} benign)"
        )

    else:
        from src.features.windowing import build_sequences

        seq = build_sequences(
            states,
            T=cfg["data"]["history_length"],
            K=cfg["data"]["horizon"],
        )

        print("[train] attack semantics disabled")

    print(
        f"[train] built {len(seq['X'])} sequences across "
        f"{len(np.unique(seq['hosts']))} hosts "
        f"({seq['y_inf'].mean():.1%} positive infiltration rate)"
    )

    # Use chronological split (SAFE) by default;
    # host-level split available for comparison
    split_strategy = cfg["data"].get(
        "split_strategy",
        "chronological",
    )

    if split_strategy == "chronological":
        from src.features.windowing import (
            per_scenario_chronological_split,
            verify_no_leakage,
        )

        purge_gap = cfg["data"].get(
            "purge_gap_seconds",
            0,
        )

        print(
            "[train] using per-scenario chronological split "
            "(ensures attacks appear in all splits)"
        )

        # Pass hosts for per-host grouping in synthetic data
        train_mask, val_mask, test_mask = (
            per_scenario_chronological_split(
                seq["times"],
                cfg["data"]["train_frac"],
                cfg["data"]["val_frac"],
                hosts_array=seq["hosts"],
                purge_gap_seconds=purge_gap,
            )
        )

        # Verify no temporal leakage
        leakage_check = verify_no_leakage(
            seq["times"],
            train_mask,
            val_mask,
            test_mask,
            T=cfg["data"]["history_length"],
            K=cfg["data"]["horizon"],
            window_seconds=cfg["data"]["window_seconds"],
            hosts_array=seq["hosts"],
        )

        if not leakage_check["is_valid"]:
            if (
                purge_gap == 0
                and cfg["data"].get("dataset_source")
                == "synthetic_dev"
            ):
                print(
                    "[train] ⚠ Temporal interval overlap detected "
                    "(acceptable for per-scenario split in dev mode):"
                )

                for issue in leakage_check["issues"]:
                    print(f"  - {issue}")

            else:
                print("[train] ERROR: Temporal leakage detected:")

                for issue in leakage_check["issues"]:
                    print(f"  - {issue}")

                raise RuntimeError(
                    "Chronological split failed temporal leakage verification"
                )

        else:
            print(
                "[train] ✓ Chronological split verified "
                "(no temporal leakage)"
            )
            print(f"[train]   Purge gap: {purge_gap}s")

            stats = leakage_check["stats"]

            print(
                f"[train]   Train effective interval: "
                f"{stats['train_effective_start']} → "
                f"{stats['train_effective_end']}"
            )

            print(
                f"[train]   Val effective interval:   "
                f"{stats['val_effective_start']} → "
                f"{stats['val_effective_end']}"
            )

            print(
                f"[train]   Test effective interval:  "
                f"{stats['test_effective_start']} → "
                f"{stats['test_effective_end']}"
            )

            # Verify attacks appear in all splits
            y_inf_train = seq["y_inf"][train_mask]
            y_inf_val = seq["y_inf"][val_mask]
            y_inf_test = seq["y_inf"][test_mask]

            train_attacks = (y_inf_train > 0).sum()
            val_attacks = (y_inf_val > 0).sum()
            test_attacks = (y_inf_test > 0).sum()

            print(
                f"[train] ✓ Attack distribution: "
                f"train={train_attacks}, "
                f"val={val_attacks}, "
                f"test={test_attacks}"
            )

            if (
                train_attacks == 0
                or val_attacks == 0
                or test_attacks == 0
            ):
                print(
                    "[train] WARNING: At least one split has "
                    "zero attack sequences"
                )
                print(
                    "[train]           (This is acceptable if dataset "
                    "is purely benign, but unusual for CIC-IDS)"
                )

    else:
        # Fall back to host-level split
        # (non-temporal, useful for comparison)
        from src.features.windowing import host_level_split

        train_mask, val_mask, test_mask = host_level_split(
            seq["hosts"],
            cfg["data"]["train_frac"],
            cfg["data"]["val_frac"],
            seed,
        )

        print(
            "[train] Using host-level split "
            "(non-temporal, for comparison only)"
        )

    # Setup device (GPU if available, else CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[train] Using device: {device}")

    if torch.cuda.is_available():
        print(
            f"[train]   GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"[train]   PyTorch: {torch.__version__}")
    print(f"[train]   Random seed: {seed}")

    # Scale all splits
    # (fit scaler ONLY on training data)
    scaler = StandardScaler().fit(
        seq["X"][train_mask].reshape(
            -1,
            len(FEATURE_COLUMNS),
        )
    )

    X_train = scale_split(
        seq["X"][train_mask],
        scaler,
    )

    X_val = scale_split(
        seq["X"][val_mask],
        scaler,
    )

    X_test = scale_split(
        seq["X"][test_mask],
        scaler,
    )

    y_next_train = scaler.transform(
        seq["y_next"][train_mask]
    ).astype(np.float32)

    y_next_val = scaler.transform(
        seq["y_next"][val_mask]
    ).astype(np.float32)

    y_inf_train = seq["y_inf"][train_mask]
    y_inf_val = seq["y_inf"][val_mask]
    y_inf_test = seq["y_inf"][test_mask]

    y_stage_train = seq["y_stage"][train_mask]
    y_stage_val = seq["y_stage"][val_mask]
    y_stage_test = seq["y_stage"][test_mask]

    # Convert to torch tensors on device
    Xtr_t = torch.tensor(
        X_train,
        device=device,
    )

    y_next_t = torch.tensor(
        y_next_train,
        device=device,
    )

    y_inf_t = torch.tensor(
        y_inf_train,
        dtype=torch.float32,
        device=device,
    )

    y_stage_t = torch.tensor(
        y_stage_train,
        dtype=torch.long,
        device=device,
    )

    Xval_t = torch.tensor(
        X_val,
        device=device,
    )

    y_next_val_t = torch.tensor(
        y_next_val,
        device=device,
    )

    y_inf_val_t = torch.tensor(
        y_inf_val,
        dtype=torch.float32,
        device=device,
    )

    y_stage_val_t = torch.tensor(
        y_stage_val,
        dtype=torch.long,
        device=device,
    )

    Xtest_t = torch.tensor(
        X_test,
        device=device,
    )

    model = build_model(
        cfg["model"],
        input_dim=len(FEATURE_COLUMNS),
    ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
    )

    print(
        f"[train] training "
        f"{cfg['model']['encoder']} world model for "
        f"{cfg['train']['epochs']} epochs..."
    )

    # ============================================================
    # TENSORBOARD
    # ============================================================
    writer = SummaryWriter("runs/experiment")

    # Training loop with validation monitoring
    best_val_loss = float("inf")
    best_checkpoint = None
    patience = cfg["train"].get(
        "early_stopping_patience",
        5,
    )
    patience_counter = 0

    training_history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_bce": [],
    }

    # Mini-batch the training step: a full-batch forward+backward over every
    # training sequence at once needs the LSTM to retain per-timestep
    # activations for the whole training set, which is several GB for the
    # real CIC-IDS2017 split (~65k sequences) and doesn't fit an 8GB GPU.
    # Validation/test forward passes stay full-batch since torch.no_grad()
    # doesn't need to retain those activations for backprop.
    n_train = Xtr_t.shape[0]
    batch_size = cfg["train"].get("batch_size", 512)

    # Infiltration positives are a small fraction of sequences (real
    # CIC-IDS2017 attack windows are rare), so unweighted BCE lets the model
    # minimize loss by predicting negative for everything and never learn to
    # detect them. pos_weight = n_negative / n_positive rebalances the BCE
    # gradient so a missed positive costs as much as it would under a
    # balanced dataset.
    n_pos_train = float(y_inf_t.sum().item())
    n_neg_train = float(n_train - n_pos_train)
    pos_weight = torch.tensor(
        n_neg_train / n_pos_train if n_pos_train > 0 else 1.0,
        device=device,
    )

    for epoch in range(cfg["train"]["epochs"]):
        # TRAINING PHASE
        model.train()

        perm = torch.randperm(n_train, device=device)
        loss_sum = 0.0
        parts_sum = {"mse": 0.0, "bce": 0.0, "ce": 0.0}

        for start in range(0, n_train, batch_size):
            batch_idx = perm[start:start + batch_size]
            n_batch = len(batch_idx)

            opt.zero_grad()

            pred = model(Xtr_t[batch_idx])

            batch_loss, batch_parts = world_model_loss(
                pred,
                y_next_t[batch_idx],
                y_inf_t[batch_idx],
                y_stage_t[batch_idx],
                weights=cfg["train"]["loss_weights"],
                pos_weight=pos_weight,
            )

            batch_loss.backward()
            opt.step()

            loss_sum += batch_loss.item() * n_batch
            for k in parts_sum:
                parts_sum[k] += batch_parts[k] * n_batch

        train_loss_value = loss_sum / n_train
        parts = {k: v / n_train for k, v in parts_sum.items()}

        # VALIDATION PHASE
        model.eval()

        with torch.no_grad():
            val_pred = model(Xval_t)

            val_loss, val_parts = world_model_loss(
                val_pred,
                y_next_val_t,
                y_inf_val_t,
                y_stage_val_t,
                weights=cfg["train"]["loss_weights"],
                pos_weight=pos_weight,
            )

        # Record history
        training_history["epoch"].append(epoch)
        training_history["train_loss"].append(
            train_loss_value
        )
        training_history["val_loss"].append(
            val_loss.item()
        )
        training_history["val_bce"].append(
            val_parts["bce"]
        )

        # ========================================================
        # TENSORBOARD METRICS
        # ========================================================
        writer.add_scalar(
            "Loss/train",
            train_loss_value,
            epoch,
        )

        writer.add_scalar(
            "Loss/validation",
            val_loss.item(),
            epoch,
        )

        writer.add_scalar(
            "Loss/validation_bce",
            val_parts["bce"],
            epoch,
        )

        # Best checkpoint selection
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
        if (
            patience_counter >= patience
            and epoch > cfg["train"].get(
                "min_epochs",
                10,
            )
        ):
            print(
                f"[train] early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

        if (
            epoch % 10 == 0
            or epoch == cfg["train"]["epochs"] - 1
        ):
            print(
                f"  epoch {epoch:3d}  "
                f"train_loss={train_loss_value:.4f} "
                f"val_loss={val_loss.item():.4f}  "
                f"(mse={parts['mse']:.4f} "
                f"bce={parts['bce']:.4f} "
                f"ce={parts['ce']:.4f})"
            )

    # Load best checkpoint
    if best_checkpoint is not None:
        print(
            f"[train] loading best checkpoint from epoch "
            f"{best_checkpoint['epoch']} "
            f"(val_loss={best_checkpoint['val_loss']:.4f})"
        )

        model.load_state_dict(
            best_checkpoint["model_state"]
        )

    else:
        print(
            "[train] WARNING: no best checkpoint found, "
            "using final model"
        )

    # THRESHOLD SELECTION on validation set
    print(
        "[train] selecting infiltration threshold "
        "on validation set..."
    )

    model.eval()

    with torch.no_grad():
        val_prob = torch.sigmoid(
            model(Xval_t)["infiltration_logit"]
        ).cpu().numpy()

    # Find threshold that maximizes F1
    best_threshold = 0.5
    best_f1 = 0.0
    threshold_sweep_wm = []

    for threshold in np.linspace(0.1, 0.9, 81):
        val_pred_binary = (
            val_prob > threshold
        ).astype(int)

        metrics = infiltration_metrics(
            y_inf_val,
            val_pred_binary,
        )

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

    print(
        f"[train] selected threshold="
        f"{best_threshold:.3f} "
        f"(F1={best_f1:.3f} on validation)"
    )

    # FINAL TEST EVALUATION
    print(
        "[train] evaluating on test set "
        "with frozen threshold..."
    )

    with torch.no_grad():
        test_prob = torch.sigmoid(
            model(Xtest_t)["infiltration_logit"]
        ).cpu().numpy()

    wm_pred = (
        test_prob > best_threshold
    ).astype(int)

    wm_metrics = infiltration_metrics(
        y_inf_test,
        wm_pred,
    )

    print(
        f"[train] world model test metrics: "
        f"{wm_metrics}"
    )

    cfg["train"]["selected_threshold"] = float(
        best_threshold
    )

    # BASELINE: Logistic Regression
    print(
        "[train] training logistic regression baseline "
        "(no temporal modeling)..."
    )

    baseline = BaselineLogReg().fit(
        X_train[:, -1, :],
        y_inf_train,
        y_stage_train,
    )

    # Threshold selection for baseline
    val_prob_lr = (
        baseline.inf_model.predict_proba(
            X_val[:, -1, :]
        )[:, 1]
    )

    best_threshold_lr = 0.5
    best_f1_lr = 0.0
    threshold_sweep_lr = []

    for threshold in np.linspace(0.1, 0.9, 81):
        val_pred_lr = (
            val_prob_lr > threshold
        ).astype(int)

        metrics_lr = infiltration_metrics(
            y_inf_val,
            val_pred_lr,
        )

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

    print(
        f"[train] LR baseline: selected threshold="
        f"{best_threshold_lr:.3f} "
        f"(F1={best_f1_lr:.3f} on validation)"
    )

    test_prob_lr = (
        baseline.inf_model.predict_proba(
            X_test[:, -1, :]
        )[:, 1]
    )

    lr_pred = (
        test_prob_lr > best_threshold_lr
    ).astype(int)

    lr_metrics = infiltration_metrics(
        y_inf_test,
        lr_pred,
    )

    print(
        f"[train] baseline LR test metrics:   "
        f"{lr_metrics}"
    )

    cfg["train"]["selected_threshold_lr"] = float(
        best_threshold_lr
    )

    # STATIC MODEL
    print(
        "[train] training static MLP "
        "(no temporal modeling)..."
    )

    from src.models.static_model import build_static_model

    static_model = build_static_model(
        cfg["model"],
        input_dim=len(FEATURE_COLUMNS),
    ).to(device)

    static_optim = torch.optim.Adam(
        static_model.parameters(),
        lr=cfg["train"]["lr"],
    )

    static_best_val_loss = float("inf")
    static_best_checkpoint = None
    static_training_history = []

    for epoch in range(cfg["train"]["epochs"]):
        # Training phase
        static_model.train()

        train_loss_epoch = 0.0

        for i in range(0, len(X_train), 32):
            Xbatch = torch.from_numpy(
                X_train[i:i + 32]
            ).to(device)

            y_inf_batch = torch.from_numpy(
                y_inf_train[i:i + 32]
            ).to(device)

            y_stage_batch = torch.from_numpy(
                y_stage_train[i:i + 32]
            ).to(device)

            y_next_batch = torch.from_numpy(
                y_next_train[i:i + 32]
            ).to(device)

            out = static_model(Xbatch)

            mse_loss = torch.nn.functional.mse_loss(
                out["next_state"],
                y_next_batch,
            )

            bce_loss = (
                torch.nn.functional
                .binary_cross_entropy_with_logits(
                    out["infiltration_logit"],
                    y_inf_batch,
                    pos_weight=pos_weight,
                )
            )

            ce_loss = (
                torch.nn.functional.cross_entropy(
                    out["stage_logit"],
                    y_stage_batch,
                )
            )

            loss = (
                cfg["train"]["loss_weights"][0]
                * mse_loss
                + cfg["train"]["loss_weights"][1]
                * bce_loss
                + cfg["train"]["loss_weights"][2]
                * ce_loss
            )

            static_optim.zero_grad()
            loss.backward()
            static_optim.step()

            train_loss_epoch += loss.item()

        train_loss_epoch /= max(
            1,
            (len(X_train) + 31) // 32,
        )

        # Validation phase
        static_model.eval()

        with torch.no_grad():
            val_out = static_model(Xval_t)

            val_mse = (
                torch.nn.functional.mse_loss(
                    val_out["next_state"],
                    y_next_val_t,
                )
            )

            val_bce = (
                torch.nn.functional
                .binary_cross_entropy_with_logits(
                    val_out["infiltration_logit"],
                    y_inf_val_t,
                    pos_weight=pos_weight,
                )
            )

            val_ce = (
                torch.nn.functional.cross_entropy(
                    val_out["stage_logit"],
                    y_stage_val_t,
                )
            )

            val_loss_epoch = (
                cfg["train"]["loss_weights"][0]
                * val_mse
                + cfg["train"]["loss_weights"][1]
                * val_bce
                + cfg["train"]["loss_weights"][2]
                * val_ce
            )

            val_loss_epoch = val_loss_epoch.item()

        static_training_history.append({
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch,
        })

        if (
            epoch % 10 == 0
            or epoch == cfg["train"]["epochs"] - 1
        ):
            print(
                f"  epoch {epoch:3d}  "
                f"train_loss={train_loss_epoch:.4f} "
                f"val_loss={val_loss_epoch:.4f}"
            )

        # Early stopping
        if val_loss_epoch < static_best_val_loss:
            static_best_val_loss = val_loss_epoch

            static_best_checkpoint = {
                "model": static_model.state_dict(),
                "epoch": epoch,
            }

        elif (
            epoch >= cfg["train"]["min_epochs"]
            and epoch
            - static_best_checkpoint["epoch"]
            >= cfg["train"]["early_stopping_patience"]
        ):
            print(
                f"[train] static model: "
                f"early stopping at epoch {epoch}"
            )
            break

    # Load best checkpoint
    if static_best_checkpoint:
        static_model.load_state_dict(
            static_best_checkpoint["model"]
        )

    # Threshold selection for static model
    static_model.eval()

    with torch.no_grad():
        static_val_prob = torch.sigmoid(
            static_model(Xval_t)["infiltration_logit"]
        ).cpu().numpy()

    best_threshold_static = 0.5
    best_f1_static = 0.0
    threshold_sweep_static = []

    for threshold in np.linspace(0.1, 0.9, 81):
        val_pred_static = (
            static_val_prob > threshold
        ).astype(int)

        metrics_static = infiltration_metrics(
            y_inf_val,
            val_pred_static,
        )

        threshold_sweep_static.append({
            "threshold": float(threshold),
            "f1": float(metrics_static["f1"]),
            "precision": float(metrics_static["precision"]),
            "recall": float(metrics_static["recall"]),
            "fpr": float(metrics_static["fpr"]),
        })

        if metrics_static["f1"] > best_f1_static:
            best_f1_static = metrics_static["f1"]
            best_threshold_static = threshold

    print(
        f"[train] static model: selected threshold="
        f"{best_threshold_static:.3f} "
        f"(F1={best_f1_static:.3f} on validation)"
    )

    # Test evaluation
    with torch.no_grad():
        test_prob_static = torch.sigmoid(
            static_model(Xtest_t)["infiltration_logit"]
        ).cpu().numpy()

    static_pred = (
        test_prob_static > best_threshold_static
    ).astype(int)

    static_metrics = infiltration_metrics(
        y_inf_test,
        static_pred,
    )

    print(
        f"[train] static model test metrics:  "
        f"{static_metrics}"
    )

    cfg["train"]["selected_threshold_static"] = float(
        best_threshold_static
    )

    # Save artifacts
    os.makedirs("weights", exist_ok=True)

    torch.save(
        model.state_dict(),
        "weights/world_model.pt",
    )

    torch.save(
        static_model.state_dict(),
        "weights/static_model.pt",
    )

    joblib.dump(
        scaler,
        "weights/scaler.pkl",
    )

    joblib.dump(
        baseline,
        "weights/baseline_lr.pkl",
    )

    with open(
        "weights/feature_columns.json",
        "w",
    ) as f:
        json.dump(
            FEATURE_COLUMNS,
            f,
        )

    with open(
        "weights/used_config.yaml",
        "w",
    ) as f:
        yaml.safe_dump(
            cfg,
            f,
        )

    # Save training history and metrics
    with open(
        "weights/training_history.json",
        "w",
    ) as f:
        json.dump(
            training_history,
            f,
            indent=2,
            default=str,
        )

    with open(
        "weights/validation_metrics.json",
        "w",
    ) as f:
        json.dump(
            {
                "best_val_loss": float(best_val_loss),
                "best_epoch": (
                    best_checkpoint["epoch"]
                    if best_checkpoint
                    else -1
                ),
                "selected_threshold_world_model": float(
                    best_threshold
                ),
                "selected_threshold_baseline": float(
                    best_threshold_lr
                ),
            },
            f,
            indent=2,
        )

    # Save threshold selection results
    with open(
        "weights/threshold.json",
        "w",
    ) as f:
        json.dump(
            {
                "world_model": {
                    "selected_threshold": float(
                        best_threshold
                    ),
                    "best_f1": float(best_f1),
                    "sweep": threshold_sweep_wm,
                },
                "baseline_lr": {
                    "selected_threshold": float(
                        best_threshold_lr
                    ),
                    "best_f1": float(best_f1_lr),
                    "sweep": threshold_sweep_lr,
                },
                "static_model": {
                    "selected_threshold": float(
                        best_threshold_static
                    ),
                    "best_f1": float(best_f1_static),
                    "sweep": threshold_sweep_static,
                },
            },
            f,
            indent=2,
        )

    os.makedirs(
        "data/processed",
        exist_ok=True,
    )

    # Save test split
    test_data = {
        "X_test": X_test,
        "y_inf_test": y_inf_test,
        "y_stage_test": y_stage_test,
        "hosts_test": seq["hosts"][test_mask],
    }

    if use_attack_semantics:
        test_data["forecast_status_test"] = (
            seq["forecast_status"][test_mask]
        )
        test_data["contains_attack_test"] = (
            seq["contains_attack"][test_mask]
        )
        test_data["lead_times_test"] = (
            seq["lead_times"][test_mask]
        )

    np.savez(
        "data/processed/test_split.npz",
        **test_data,
    )

    print(
        "[train] saved: weights/world_model.pt, "
        "static_model.pt, scaler.pkl, baseline_lr.pkl, "
        "feature_columns.json, used_config.yaml, "
        "threshold.json, data/processed/test_split.npz"
    )

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
    )

    args = parser.parse_args()

    main(args.config)