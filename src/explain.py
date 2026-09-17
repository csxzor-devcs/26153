"""
Explainability (mandatory deliverable, techsoln.md sec 7): attention
weights answer "when" (which past windows mattered), SHAP answers "what"
(which features drove the score). Both are always surfaced together.

Usage (run from repo root, after `python -m src.train`):
    python -m src.explain
"""
import json

import numpy as np
import torch


@torch.no_grad()
def extract_attention(model, x_history: torch.Tensor) -> list:
    model.eval()
    out = model(x_history)
    return out["attention_weights"].squeeze().tolist()


def explain_features(model, background_states: np.ndarray, query_history: np.ndarray,
                      feature_columns: list, nsamples: int = 100) -> dict:
    """
    SHAP KernelExplainer over the CURRENT (last) state slice of a fixed
    history — this explains "what in the latest window drove the score,"
    which is the "flags, ports, or flow patterns" attribution the brief
    asks for. Attention (above) is what explains the temporal "when";
    SHAP is deliberately scoped to the single most recent state, not the
    full rollout (techsoln.md sec 7.2).
    """
    import shap

    def predict_infiltration_from_last_state(last_states_np: np.ndarray) -> np.ndarray:
        batch = np.repeat(query_history[np.newaxis, :, :], len(last_states_np), axis=0)
        batch[:, -1, :] = last_states_np
        with torch.no_grad():
            out = model(torch.tensor(batch, dtype=torch.float32))
            return torch.sigmoid(out["infiltration_logit"]).numpy()

    explainer = shap.KernelExplainer(predict_infiltration_from_last_state, background_states)
    query_last_state = query_history[-1].reshape(1, -1)
    shap_values = explainer.shap_values(query_last_state, nsamples=nsamples)[0]

    attribution = dict(zip(feature_columns, shap_values.tolist()))
    return dict(sorted(attribution.items(), key=lambda kv: abs(kv[1]), reverse=True))


if __name__ == "__main__":
    import json as _json

    from src.rollout import _load_artifacts

    model, scaler, cfg = _load_artifacts()
    feature_columns = _json.load(open("weights/feature_columns.json"))
    data = np.load("data/processed/test_split.npz")
    X_test, y_inf_test = data["X_test"], data["y_inf_test"]

    attack_idx = np.where(y_inf_test == 1)[0]
    if len(attack_idx) == 0:
        raise SystemExit("No positive-infiltration sequences in the test split to explain.")

    # Pick the most confidently-predicted attack sequence, not just the
    # first — attack_idx[0] can be a false negative, whose saturated logit
    # makes SHAP attributions collapse to ~0 for reasons unrelated to
    # whether SHAP itself is wired correctly.
    with torch.no_grad():
        attack_probs = torch.sigmoid(
            model(torch.tensor(X_test[attack_idx]))["infiltration_logit"]).squeeze(-1).numpy()
    query_history = X_test[attack_idx[np.argmax(attack_probs)]]
    attn = extract_attention(model, torch.tensor(query_history).unsqueeze(0))
    background = X_test[:cfg["explain"]["shap_background_size"], -1, :]
    shap_attr = explain_features(model, background, query_history, feature_columns,
                                  nsamples=cfg["explain"]["shap_nsamples"])

    print("Attention over the last", len(attn), "windows (most recent last):")
    print([round(a, 3) for a in attn])
    print("\nTop-5 SHAP feature attributions for this window's infiltration score:")
    for name, value in list(shap_attr.items())[:5]:
        print(f"  {name:<22} {value:+.4f}")
