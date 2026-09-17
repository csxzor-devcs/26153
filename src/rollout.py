"""
K-step forward simulation: the infiltration prediction engine
(techsoln.md sec 6). Feeds the model's own predicted next state back into
itself autoregressively, so this is genuine forward simulation of the
learned transition dynamics, not repeated one-step classification.

Usage (run from repo root, after `python -m src.train`):
    python -m src.rollout
"""
import json

import numpy as np
import torch

from src.mitre_mapping import STAGE_NAMES


@torch.no_grad()
def rollout(model, x_history: torch.Tensor, K: int = 5) -> dict:
    """Does NOT force model.eval()/train() — the caller decides the mode.
    Point-estimate callers should call model.eval() first; rollout_with_uncertainty
    needs model.train() active during its sampling loop for MC-dropout to have
    any effect. If this function forces eval() internally, every "stochastic"
    sample below becomes identical and prob_std collapses to zero silently."""
    seq = x_history.clone()
    probs, stages, imagined_states, attn_trace = [], [], [], []

    for _ in range(K):
        out = model(seq)
        probs.append(torch.sigmoid(out["infiltration_logit"]).item())
        stages.append(int(torch.argmax(out["stage_logits"], dim=-1).item()))
        next_state = out["next_state"].unsqueeze(1)
        imagined_states.append(next_state.squeeze().tolist())
        attn_trace.append(out["attention_weights"].squeeze().tolist())
        seq = torch.cat([seq[:, 1:, :], next_state], dim=1)

    return {
        "infiltration_prob_trajectory": probs,
        "predicted_stage_trajectory": stages,
        "predicted_stage_names": [STAGE_NAMES[s] for s in stages],
        "imagined_states": imagined_states,
        "attention_trace": attn_trace,
    }


def rollout_with_uncertainty(model, x_history: torch.Tensor, K: int = 5, n_samples: int = 20) -> dict:
    """
    Runs the rollout N times with dropout active (MC-dropout) to expose how
    uncertainty compounds over the autoregressive horizon — the demo shows
    this as a widening band rather than a single point estimate past step
    2-3 (techsoln.md sec 6 calibration note).
    """
    model.train()  # keep dropout active; LSTM/Transformer have no BatchNorm here
    all_probs = []
    for _ in range(n_samples):
        with torch.no_grad():
            r = rollout(model, x_history, K=K)
        all_probs.append(r["infiltration_prob_trajectory"])
    model.eval()

    all_probs = np.array(all_probs)  # (n_samples, K)
    point_estimate = rollout(model, x_history, K=K)
    return {
        **point_estimate,
        "prob_mean": all_probs.mean(axis=0).tolist(),
        "prob_std": all_probs.std(axis=0).tolist(),
    }


def _load_artifacts():
    import joblib
    from src.models.world_model import build_model
    import yaml

    with open("weights/used_config.yaml") as f:
        cfg = yaml.safe_load(f)
    feature_columns = json.load(open("weights/feature_columns.json"))
    model = build_model(cfg["model"], input_dim=len(feature_columns))
    model.load_state_dict(torch.load("weights/world_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    scaler = joblib.load("weights/scaler.pkl")
    return model, scaler, cfg


if __name__ == "__main__":
    model, scaler, cfg = _load_artifacts()
    data = np.load("data/processed/test_split.npz")  # first-party artifact from src/train.py, plain numeric/string arrays
    X_test, y_inf_test = data["X_test"], data["y_inf_test"]

    attack_idx = np.where(y_inf_test == 1)[0]
    if len(attack_idx) == 0:
        raise SystemExit("No positive-infiltration sequences in the test split to demo rollout on.")

    # Pick the attack sequence the model is most confident about, not just the
    # first one — attack_idx[0] can land on a false negative (model correctly
    # trained but wrong on that one sequence), which makes for a dead-looking
    # demo (saturated near-zero logit) rather than a wired-code problem.
    with torch.no_grad():
        attack_probs = torch.sigmoid(
            model(torch.tensor(X_test[attack_idx]))["infiltration_logit"]).squeeze(-1).numpy()
    best = attack_idx[np.argmax(attack_probs)]
    sample = torch.tensor(X_test[best]).unsqueeze(0)
    result = rollout_with_uncertainty(model, sample, K=cfg["data"]["horizon"],
                                       n_samples=cfg["explain"]["mc_dropout_samples"])
    print(json.dumps({
        "infiltration_prob_trajectory": [round(p, 3) for p in result["infiltration_prob_trajectory"]],
        "prob_mean": [round(p, 3) for p in result["prob_mean"]],
        "prob_std": [round(p, 3) for p in result["prob_std"]],
        "predicted_stage_names": result["predicted_stage_names"],
    }, indent=2))
