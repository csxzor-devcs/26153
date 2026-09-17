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

    # If processed flows exist, load them directly
    if os.path.exists(path):
        print(f"[train] loading flow data from {path}")
        df = pd.read_csv(path)

        # Check for provenance file to identify data source
        provenance_path = path.replace(".csv", "_provenance.json")
        if os.path.exists(provenance_path):
            with open(provenance_path) as f:
                provenance = json.load(f)
            print(f"[train] data source: {provenance.get('source', 'unknown')}")
            print(f"[train] flows retained: {provenance['quality_stats'].get('rows_total', '?'):,}")

        return df

    # If input directory specified, try to prepare dataset
    input_dir = cfg["data"].get("raw_input_dir")
    if input_dir and os.path.exists(input_dir):
        print(f"[train] preparing dataset from {input_dir}...")
        from src.data.prepare import main as prepare_main
        try:
            source = cfg["data"].get("dataset_source", "cic_ids2018")
            df, provenance = prepare_main(input_dir, path, source)
            return df
        except Exception as e:
            print(f"[train] ERROR during preparation: {e}")
            print(f"[train] falling back to synthetic data")

    # Fall back to synthetic
    print(f"[train] WARNING: {path} not found and no input_dir configured.")
    print(f"[train] Using bundled synthetic flow generator for this run.")
    print(f"[train] To benchmark against real CIC-IDS2018:")
    print(f"[train]   1. Download dataset to /path/to/CSVs")
    print(f"[train]   2. Set data.raw_input_dir in configs/default.yaml")
    print(f"[train]   3. Re-run: python -m src.train")
    from src.data.synthetic_flows import generate_synthetic_flows
    return generate_synthetic_flows()


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
    seq = build_sequences(states, T=cfg["data"]["history_length"], K=cfg["data"]["horizon"])
    print(f"[train] built {len(seq['X'])} sequences across {len(np.unique(seq['hosts']))} hosts "
          f"({seq['y_inf'].mean():.1%} positive infiltration rate)")

    train_mask, val_mask, test_mask = host_level_split(
        seq["hosts"], cfg["data"]["train_frac"], cfg["data"]["val_frac"], seed)

    scaler = StandardScaler().fit(seq["X"][train_mask].reshape(-1, len(FEATURE_COLUMNS)))
    X_train = scale_split(seq["X"][train_mask], scaler)
    X_test = scale_split(seq["X"][test_mask], scaler)
    y_next_train = scaler.transform(seq["y_next"][train_mask]).astype(np.float32)
    y_inf_train, y_inf_test = seq["y_inf"][train_mask], seq["y_inf"][test_mask]
    y_stage_train, y_stage_test = seq["y_stage"][train_mask], seq["y_stage"][test_mask]

    model = build_model(cfg["model"], input_dim=len(FEATURE_COLUMNS))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    Xtr_t = torch.tensor(X_train)
    y_next_t = torch.tensor(y_next_train)
    y_inf_t = torch.tensor(y_inf_train, dtype=torch.float32)
    y_stage_t = torch.tensor(y_stage_train, dtype=torch.long)

    print(f"[train] training {cfg['model']['encoder']} world model for {cfg['train']['epochs']} epochs...")
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss, parts = world_model_loss(pred, y_next_t, y_inf_t, y_stage_t,
                                        weights=cfg["train"]["loss_weights"])
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == cfg["train"]["epochs"] - 1:
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  "
                  f"(mse={parts['mse']:.4f} bce={parts['bce']:.4f} ce={parts['ce']:.4f})")

    model.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(model(torch.tensor(X_test))["infiltration_logit"]).numpy()
    wm_metrics = infiltration_metrics(y_inf_test, (test_prob > 0.5).astype(int))
    print(f"[train] world model test metrics: {wm_metrics}")

    baseline = BaselineLogReg().fit(X_train[:, -1, :], y_inf_train, y_stage_train)
    lr_pred = baseline.predict_infiltration(X_test[:, -1, :])
    lr_metrics = infiltration_metrics(y_inf_test, lr_pred)
    print(f"[train] baseline LR test metrics:   {lr_metrics}")

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/world_model.pt")
    joblib.dump(scaler, "weights/scaler.pkl")
    joblib.dump(baseline, "weights/baseline_lr.pkl")
    with open("weights/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    with open("weights/used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/test_split.npz",
              X_test=X_test, y_inf_test=y_inf_test, y_stage_test=y_stage_test,
              hosts_test=seq["hosts"][test_mask])

    print("[train] saved: weights/world_model.pt, scaler.pkl, baseline_lr.pkl, "
          "feature_columns.json, used_config.yaml, data/processed/test_split.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
