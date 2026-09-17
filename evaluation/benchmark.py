"""
Benchmark: world model vs logistic-regression baseline on the held-out test
split saved by src/train.py (techsoln.md sec 8). Writes evaluation/results.md.

Usage (run from repo root, after `python -m src.train`):
    python -m evaluation.benchmark
"""
import json

import numpy as np
import torch

from src.models.baseline_lr import infiltration_metrics
from src.rollout import _load_artifacts


def main():
    model, scaler, cfg = _load_artifacts()
    import joblib
    baseline = joblib.load("weights/baseline_lr.pkl")  # first-party artifact from src/train.py

    data = np.load("data/processed/test_split.npz")
    X_test, y_inf_test = data["X_test"], data["y_inf_test"]
    hosts_test = data["hosts_test"]

    # Check if attack semantics data is available
    has_attack_semantics = "forecast_status_test" in data.files
    if has_attack_semantics:
        forecast_status = data["forecast_status_test"]
        lead_times = data["lead_times_test"]

    # Load frozen thresholds selected during validation
    selected_threshold_wm = cfg["train"].get("selected_threshold", 0.5)
    selected_threshold_lr = cfg["train"].get("selected_threshold_lr", 0.5)

    model.eval()
    with torch.no_grad():
        wm_prob = torch.sigmoid(model(torch.tensor(X_test))["infiltration_logit"]).numpy()
    wm_pred = (wm_prob > selected_threshold_wm).astype(int)  # Use frozen threshold
    wm_metrics = infiltration_metrics(y_inf_test, wm_pred)

    lr_prob = baseline.inf_model.predict_proba(X_test[:, -1, :])[:, 1]
    lr_pred = (lr_prob > selected_threshold_lr).astype(int)  # Use frozen threshold
    lr_metrics = infiltration_metrics(y_inf_test, lr_pred)

    from sklearn.metrics import roc_auc_score
    wm_auc = roc_auc_score(y_inf_test, wm_prob.ravel())
    lr_auc = roc_auc_score(y_inf_test, lr_prob)

    # Compute early-warning metrics if attack semantics available
    early_warning_metrics_wm = None
    early_warning_metrics_lr = None
    if has_attack_semantics:
        from src.data.attack_semantics import compute_lead_time_metrics
        early_warning_metrics_wm = compute_lead_time_metrics(
            y_inf_test, wm_prob, lead_times, hosts_test, forecast_status, selected_threshold_wm
        )
        early_warning_metrics_lr = compute_lead_time_metrics(
            y_inf_test, lr_prob, lead_times, hosts_test, forecast_status, selected_threshold_lr
        )

    using_real_data = __import__("os").path.exists(cfg["data"]["raw_flows_path"])
    data_note = ("real flow data at `{}`".format(cfg["data"]["raw_flows_path"]) if using_real_data
                 else "the **bundled synthetic flow generator** "
                      "(`src/data/synthetic_flows.py`) — NOT the CIC-IDS2018 dataset "
                      "required by the brief. Set `data.raw_flows_path` in "
                      "`configs/default.yaml` to a real dataset CSV and re-run "
                      "`python -m src.train` then this script to get the real benchmark.")

    lines = [
        "# Benchmark Results\n",
        f"Data source for this run: {data_note}\n",
        f"Test set size: {len(y_inf_test)} sequences "
        f"({y_inf_test.mean():.1%} positive infiltration rate)\n",
    ]

    if has_attack_semantics:
        pre_attack_count = (forecast_status == "pre_attack").sum()
        during_attack_count = (forecast_status == "attack_in_progress").sum()
        benign_count = (forecast_status == "benign").sum()
        lines += [
            "## Sequence Composition (Attack Semantics)\n",
            f"- Pre-attack sequences: {pre_attack_count} (pure forecasting scenarios)",
            f"- During-attack sequences: {during_attack_count} (detection scenarios)",
            f"- Benign sequences: {benign_count} (negative examples)",
            "",
        ]

    lines += [
        "## Detection Metrics (Overall)\n",
        "| Model | F1 | Precision | Recall | FPR |",
        "|---|---|---|---|---|",
        f"| Logistic Regression (baseline) | {lr_metrics['f1']:.3f} | {lr_metrics['precision']:.3f} "
        f"| {lr_metrics['recall']:.3f} | {lr_metrics['fpr']:.3f} |",
        f"| World Model ({cfg['model']['encoder'].upper()}) | {wm_metrics['f1']:.3f} "
        f"| {wm_metrics['precision']:.3f} | {wm_metrics['recall']:.3f} | {wm_metrics['fpr']:.3f} |",
        "",
        f"AUC-ROC: World Model = {wm_auc:.3f}, Logistic Regression = {lr_auc:.3f}",
        "",
    ]

    if has_attack_semantics and early_warning_metrics_wm is not None:
        lines += [
            "## Proactive Forecasting Metrics (Pre-Attack Sequences Only)\n",
            "These metrics evaluate the model's ability to warn about attacks *before* they start, not detect ongoing attacks.\n",
        ]

        if early_warning_metrics_wm.get("total_pre_attack_sequences", 0) > 0:
            lines += [
                "| Metric | Logistic Regression | World Model |",
                "|---|---|---|",
                f"| Early warnings (TP) | {early_warning_metrics_lr['early_warnings']} | {early_warning_metrics_wm['early_warnings']} |",
                f"| False alarms (FP) | {early_warning_metrics_lr['false_early_alarms']} | {early_warning_metrics_wm['false_early_alarms']} |",
                f"| Missed attacks (FN) | {early_warning_metrics_lr['missed_attacks']} | {early_warning_metrics_wm['missed_attacks']} |",
                f"| Attacks warned (%) | {early_warning_metrics_lr['pct_attacks_warned_before_onset']*100:.1f}% | {early_warning_metrics_wm['pct_attacks_warned_before_onset']*100:.1f}% |",
            ]

            if early_warning_metrics_wm['median_lead_time_windows'] is not None:
                lines += [
                    f"| Median lead-time | {early_warning_metrics_lr['median_lead_time_windows']:.1f} windows | {early_warning_metrics_wm['median_lead_time_windows']:.1f} windows |",
                ]
            lines += [""]
        else:
            lines += [
                "*(No pre-attack sequences in test set — chronological ordering places all attacks in training)*",
                "",
            ]

    if wm_auc < lr_auc:
        lines += [
            "**Caveat (read before citing these numbers):** on this bundled synthetic "
            "data the world model currently trails the logistic-regression baseline on "
            "F1 *and* AUC-ROC, not just threshold calibration — more training epochs "
            "narrows but does not close the gap. The likely cause is a property of the "
            "*synthetic generator*, not of temporal modeling generally: each attack "
            "phase here is a clean, rule-based, single-window signature (e.g. a "
            "PortScan window's SYN/ACK pattern is unambiguous on its own), so the "
            "single most-recent state already carries most of the separable signal "
            "and the LSTM's extra capacity buys little. Real CIC-IDS2018 traffic is "
            "noisier and attack signatures are smeared across multiple flows/windows, "
            "which is exactly the regime where trajectory memory should help — this "
            "needs to be re-run on real data before treating the world model's "
            "relative standing here as representative. Do not cite this comparison as "
            "evidence the temporal approach works; it is currently evidence only that "
            "the pipeline runs end to end.",
            "",
        ]

    with open("evaluation/results.md", "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("[benchmark] wrote evaluation/results.md")


if __name__ == "__main__":
    main()
