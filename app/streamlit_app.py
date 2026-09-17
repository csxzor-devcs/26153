"""
Offline demo interface (techsoln.md sec 9). Loads pre-trained artifacts
from weights/ and data/processed/test_split.npz — no network calls, no
external services. Run from repo root:

    streamlit run app/streamlit_app.py

Requires `python -m src.train` to have been run first so weights/ and
data/processed/test_split.npz exist.
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `from src...` imports

from src.explain import explain_features, extract_attention  # noqa: E402
from src.mitre_mapping import STAGE_NAMES  # noqa: E402
from src.rollout import _load_artifacts, rollout_with_uncertainty  # noqa: E402

st.set_page_config(page_title="Network World Model — Infiltration Forecast", layout="wide")
st.title("Proactive Infiltration Forecasting — Offline Demo")
st.caption("Runs fully offline against pre-trained weights. No network calls.")


@st.cache_resource
def load_everything():
    model, scaler, cfg = _load_artifacts()
    feature_columns = json.load(open("weights/feature_columns.json"))
    data = np.load("data/processed/test_split.npz")
    return model, scaler, cfg, feature_columns, data


try:
    model, scaler, cfg, feature_columns, data = load_everything()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.train` from the repo root first.")
    st.stop()

X_test, y_inf_test, hosts_test = data["X_test"], data["y_inf_test"], data["hosts_test"]
attack_idx = np.where(y_inf_test == 1)[0]
benign_idx = np.where(y_inf_test == 0)[0]

st.sidebar.header("Scenario & Configuration")

# Test set composition
st.sidebar.subheader("Test Set Composition")
st.sidebar.metric("Total sequences", len(X_test))
st.sidebar.metric("Attacking sessions", len(attack_idx))
st.sidebar.metric("Benign sessions", len(benign_idx))
st.sidebar.divider()

scenario_kind = st.sidebar.radio("Pick a sample from the held-out test set",
                                  ["Attacking session", "Benign session"])
pool = attack_idx if scenario_kind == "Attacking session" else benign_idx
if len(pool) == 0:
    st.warning(f"No '{scenario_kind}' sequences in the test split.")
    st.stop()

choice = st.sidebar.selectbox(
    "Session", list(range(min(len(pool), 20))),
    format_func=lambda i: f"Host {hosts_test[pool[i]]} — idx {pool[i]}")
sample_idx = pool[choice]
history = X_test[sample_idx]

# Display scenario badge
col_host, col_type = st.columns(2)
with col_host:
    st.metric("Host", hosts_test[sample_idx])
with col_type:
    scenario_badge = "🔴 Attacking" if y_inf_test[sample_idx] == 1 else "🟢 Benign"
    st.metric("Classification", scenario_badge)

st.divider()

K = st.sidebar.slider("Forecast horizon (windows)", 1, cfg["data"]["horizon"], cfg["data"]["horizon"])
st.sidebar.caption(f"Lead time: {K} windows × {cfg['data']['window_seconds']}s = ~{K * cfg['data']['window_seconds'] // 60}min")

history_t = torch.tensor(history).unsqueeze(0)
result = rollout_with_uncertainty(model, history_t, K=K, n_samples=cfg["explain"]["mc_dropout_samples"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Infiltration probability forecast")
    prob_mean = np.array(result["prob_mean"][:K])
    prob_std = np.array(result["prob_std"][:K])
    chart_data = {
        "window ahead": list(range(1, K + 1)),
        "P(infiltration)": prob_mean,
        "upper (±1σ)": np.clip(prob_mean + prob_std, 0, 1),
        "lower (±1σ)": np.clip(prob_mean - prob_std, 0, 1),
    }
    st.line_chart(chart_data, x="window ahead",
                   y=["P(infiltration)", "upper (±1σ)", "lower (±1σ)"])
    st.caption("Uncertainty widens with horizon — autoregressive rollout error "
               "compounds by construction (see techsoln.md sec 6).")

with col2:
    st.subheader("Predicted MITRE ATT&CK stage trajectory")
    st.table({
        "Step": list(range(1, K + 1)),
        "Predicted stage": [STAGE_NAMES[s] for s in result["predicted_stage_trajectory"][:K]],
        "P(infiltration)": [f"{p:.0%}" for p in prob_mean],
    })

peak_step = int(np.argmax(prob_mean))
if prob_mean[peak_step] > 0.5:
    predicted_stage = STAGE_NAMES[result['predicted_stage_trajectory'][peak_step]]
    col_alert_stage, col_alert_conf = st.columns([2, 1])
    with col_alert_stage:
        st.error(f"⚠ **{predicted_stage}** predicted within {peak_step + 1} window(s)")
    with col_alert_conf:
        st.metric("Confidence", f"{prob_mean[peak_step]:.0%}",
                 delta=f"±{prob_std[peak_step]:.0%}")
else:
    st.success("✓ No infiltration convergence predicted within the forecast horizon.")

st.divider()
st.subheader("Explainability")
c1, c2 = st.columns(2)
with c1:
    st.caption("Attention over recent history — **when** it mattered")
    attn = extract_attention(model, history_t)
    st.bar_chart({"attention weight": attn})
with c2:
    st.caption("SHAP feature attribution for the latest window — **what** drove the score")
    background = X_test[:cfg["explain"]["shap_background_size"], -1, :]
    with st.spinner("Computing SHAP values..."):
        shap_attr = explain_features(model, background, history, feature_columns,
                                       nsamples=cfg["explain"]["shap_nsamples"])
    top5 = dict(list(shap_attr.items())[:5])
    st.bar_chart(top5)

with st.expander("Raw session's last observed state vector"):
    st.json(dict(zip(feature_columns, [round(float(v), 3) for v in history[-1]])))
