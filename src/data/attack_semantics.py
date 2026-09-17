"""
Attack session detection and pre-attack/during-attack/post-attack classification.

This module is critical for the "proactive forecasting" claim. Without it, the
system cannot distinguish "predicted before attack started" from "detected
during attack" — both will show positive y_inf, but only the former is true
early-warning.

Key definitions:
- Attack onset (t_onset): First window where stage_label > 0 for a host
- Pre-attack windows: t < t_onset (no attack traffic yet)
- Attack-in-progress: t >= t_onset (attack visible in state)
- Post-attack: Windows after attack ends (if detectable)

Forecasting claim: "Given pre-attack history (X_t with t < t_onset), predict
that infiltration will occur (y_inf=1) within next K windows, before t_onset+K."

Detection claim: "Given during-attack history, detect current compromise."

These are fundamentally different and require separate evaluation.
"""
import numpy as np
import pandas as pd


def detect_attack_onsets(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each host, identify the first window where stage_label > 0.

    Returns state_df with added columns:
    - attack_onset_window: first window index where stage > 0 (or None)
    - forecast_status: one of {pre_attack, attack_in_progress, post_attack, benign}
    - windows_until_onset: windows remaining until attack starts (inf if benign)
    """
    df = state_df.copy()
    df["attack_onset_window"] = None
    df["forecast_status"] = "benign"
    df["windows_until_onset"] = np.inf

    for host in df["host"].unique():
        host_mask = df["host"] == host
        host_df = df[host_mask].sort_values("window_start").reset_index(drop=True)

        # Find first attack window
        attack_mask = host_df["stage_label"] > 0
        if not attack_mask.any():
            # Benign host, all windows remain "benign" with onset=None
            df.loc[host_mask, "attack_onset_window"] = None
            df.loc[host_mask, "forecast_status"] = "benign"
            continue

        onset_idx = attack_mask.idxmax()
        onset_timestamp = host_df.loc[onset_idx, "window_start"]

        # Mark all windows of this host
        for idx in host_df.index:
            ts = host_df.loc[idx, "window_start"]
            if ts < onset_timestamp:
                df.loc[idx, "forecast_status"] = "pre_attack"
                df.loc[idx, "windows_until_onset"] = (onset_timestamp - ts).total_seconds() / 60  # minutes
            elif ts == onset_timestamp:
                df.loc[idx, "forecast_status"] = "attack_in_progress"
                df.loc[idx, "windows_until_onset"] = 0
            else:
                df.loc[idx, "forecast_status"] = "attack_in_progress"  # During or after onset
                df.loc[idx, "windows_until_onset"] = 0

            df.loc[idx, "attack_onset_window"] = onset_timestamp

    return df


def build_sequences_with_status(state_df: pd.DataFrame, T: int = 20, K: int = 5,
                                 feature_columns: list = None) -> dict:
    """
    Build sequences with attack status tracking.

    Extends the standard sequence-building to track:
    - forecast_status_at_t: pre_attack, during_attack, benign, etc. at window t
    - contains_attack_history: whether input history X_t contains any attack windows
    - y_attack_begins: 1 if any window in [t+1..t+K] is first attack of the session
    - lead_time_windows: windows between t and expected attack onset (if pre-attack)

    Returns dict with standard fields plus:
    - forecast_status: array of forecast status for each sequence
    - contains_attack: array of bool (does X_t contain attack windows?)
    - lead_times: array of windows until onset (inf if benign or during-attack)
    """
    from src.features.extract import FEATURE_COLUMNS
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    X, y_next, y_inf, y_stage = [], [], [], []
    hosts, times, forecast_status_list, contains_attack_list = [], [], [], []
    lead_times = []

    for host in state_df["host"].unique():
        host_df = state_df[state_df["host"] == host].sort_values("window_start").reset_index(drop=True)
        if len(host_df) < T + K:
            continue  # Skip hosts without enough history + horizon

        feats = host_df[feature_columns].to_numpy(dtype=np.float32)
        stages = host_df["stage_label"].to_numpy(dtype=np.int64)
        status = host_df["forecast_status"].to_numpy()
        onset_ts = host_df["attack_onset_window"].iloc[0]  # Same for all rows of host

        n = len(host_df)
        for i in range(T, n - K):
            # Input history [S_{i-T+1}, ..., S_i]
            # In Python 0-indexing: feats[i-T+1:i+1]
            X.append(feats[i - T + 1:i + 1])

            # Targets (canonical: future-only forecast, matching windowing.py)
            # y_next: S_{i+1}
            y_next.append(feats[i + 1])
            # y_inf: infiltration in NEXT K windows [S_{i+1}, ..., S_{i+K}]
            y_inf.append(1.0 if np.any(stages[i + 1:i + 1 + K] > 0) else 0.0)
            # y_stage: stage of S_{i+1}
            y_stage.append(stages[i + 1])

            # Status tracking
            hosts.append(host)
            times.append(host_df["window_start"].iloc[i])  # Timestamp at history endpoint
            forecast_status_list.append(status[i])
            # Whether history [S_{i-T+1}, ..., S_i] contains any attack windows
            contains_attack_list.append(1.0 if np.any(stages[i - T + 1:i + 1] > 0) else 0.0)
            lead_times.append(
                (onset_ts - host_df["window_start"].iloc[i]).total_seconds() / 60
                if onset_ts is not None and status[i] == "pre_attack"
                else np.inf
            )

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y_next": np.asarray(y_next, dtype=np.float32),
        "y_inf": np.asarray(y_inf, dtype=np.float32),
        "y_stage": np.asarray(y_stage, dtype=np.int64),
        "hosts": np.asarray(hosts),
        "times": np.asarray(times),
        "forecast_status": np.asarray(forecast_status_list),  # pre_attack, during, benign
        "contains_attack": np.asarray(contains_attack_list),  # bool: history has attack?
        "lead_times": np.asarray(lead_times),  # windows until onset (-inf if benign/during)
    }


def create_pre_attack_split(sequences: dict, seed: int = 42) -> tuple:
    """
    Create separate evaluation subsets for pre-attack and during-attack scenarios.

    Returns:
        (pre_attack_mask, during_attack_mask, benign_mask)

    This allows evaluation to distinguish:
    - Early-warning: "Warned before attack onset" (pre-attack test)
    - Detection: "Detected ongoing attack" (during-attack test)
    """
    status = sequences["forecast_status"]

    pre_attack_mask = status == "pre_attack"
    during_attack_mask = status == "attack_in_progress"
    benign_mask = status == "benign"

    return pre_attack_mask, during_attack_mask, benign_mask


def compute_lead_time_metrics(y_inf_true: np.ndarray, y_inf_pred: np.ndarray,
                               lead_times: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute early-warning specific metrics.

    Only counts predictions as "early warning" if:
    1. Predicted infiltration > threshold
    2. Prediction made before attack onset (lead_time > 0)
    3. True infiltration is 1 (attack actually occurred)

    Returns:
    - early_warnings: count of true early warnings
    - false_early_alarms: count of false positives on benign traffic
    - missed_attacks: count of true attacks that were not warned before onset
    - median_lead_time: median windows of warning before attack
    - pct_attacks_warned: percentage of attacks warned before onset
    """
    # Separate pre-attack sequences (lead_time < inf)
    pre_attack_mask = ~np.isinf(lead_times)

    if not pre_attack_mask.any():
        return {
            "early_warnings": 0,
            "false_early_alarms": 0,
            "missed_attacks": 0,
            "median_lead_time": None,
            "pct_attacks_warned": None,
            "note": "No pre-attack sequences in test set"
        }

    y_pred_binary = (y_inf_pred > threshold).astype(int)

    # Count early warnings: predicted+true+pre_attack
    early_warnings = np.sum(
        (y_pred_binary == 1) & (y_inf_true == 1) & pre_attack_mask
    )

    # False alarms: predicted+not_true+pre_attack
    false_early_alarms = np.sum(
        (y_pred_binary == 1) & (y_inf_true == 0) & pre_attack_mask
    )

    # Missed attacks: not_predicted+true+pre_attack
    missed_attacks = np.sum(
        (y_pred_binary == 0) & (y_inf_true == 1) & pre_attack_mask
    )

    # Lead time for successful early warnings
    successful_warnings = (y_pred_binary == 1) & (y_inf_true == 1) & pre_attack_mask
    if successful_warnings.any():
        lead_times_valid = lead_times[successful_warnings]
        lead_times_valid = lead_times_valid[~np.isinf(lead_times_valid)]
        median_lead_time = np.median(lead_times_valid) if len(lead_times_valid) > 0 else None
    else:
        median_lead_time = None

    # Percentage of attacks warned
    total_pre_attack_attacks = np.sum((y_inf_true == 1) & pre_attack_mask)
    pct_warned = early_warnings / total_pre_attack_attacks if total_pre_attack_attacks > 0 else None

    return {
        "early_warnings": int(early_warnings),
        "false_early_alarms": int(false_early_alarms),
        "missed_attacks": int(missed_attacks),
        "median_lead_time_windows": float(median_lead_time) if median_lead_time else None,
        "pct_attacks_warned_before_onset": float(pct_warned) if pct_warned else None,
        "total_pre_attack_sequences": int(pre_attack_mask.sum()),
        "total_pre_attack_attacks": int(total_pre_attack_attacks),
    }


if __name__ == "__main__":
    # Example usage
    from src.features.extract import build_state_vectors
    from src.data.synthetic_flows import generate_synthetic_flows

    flows = generate_synthetic_flows(n_hosts=10, session_minutes=60, seed=42)
    states = build_state_vectors(flows)
    states_with_status = detect_attack_onsets(states)

    print(f"States with attack detection:")
    print(f"  Total windows: {len(states_with_status)}")
    print(f"  Pre-attack windows: {(states_with_status['forecast_status'] == 'pre_attack').sum()}")
    print(f"  During-attack windows: {(states_with_status['forecast_status'] == 'attack_in_progress').sum()}")
    print(f"  Benign windows: {(states_with_status['forecast_status'] == 'benign').sum()}")
    print(f"\nSample sequence build with status:")
    from src.features.extract import FEATURE_COLUMNS
    sequences = build_sequences_with_status(states_with_status, T=5, K=2, feature_columns=FEATURE_COLUMNS)
    print(f"  Sequences built: {len(sequences['X'])}")
    print(f"  Pre-attack sequences: {(sequences['forecast_status'] == 'pre_attack').sum()}")
    print(f"  With attack history: {sequences['contains_attack'].sum()}")
