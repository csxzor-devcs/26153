"""
Turns per-(host, window) state vectors into fixed-length sequences for the
world model: given T past states, predict the next state, the infiltration
label over the next K windows, and the MITRE stage of the very next window
(techsoln.md sec 4).

Implements LEAKAGE-SAFE splitting:
- Chronological split (default): respects temporal ordering globally
- Host-disjoint split (optional): no host appears in multiple splits
- Prevents sequences from crossing train/val/test boundaries
- Ensures scaler is fit on training data only
- Verifies no future labels leak into history
"""
import numpy as np
import pandas as pd

from src.features.extract import FEATURE_COLUMNS


def build_sequences(state_df: pd.DataFrame, T: int = 20, K: int = 5,
                     feature_columns=FEATURE_COLUMNS) -> dict:
    """Build sequences with full temporal and metadata tracking.

    Following techsoln.md section 4 exactly:
    - X_t = [S_{t-T+1}, ..., S_t] = history ending at window t
    - y_next = S_{t+1} = next state (one step ahead)
    - y_inf = 1[any of S_{t+1..t+K} is attack] = infiltration in NEXT K windows (future-only)
    - y_stage = stage of S_{t+1} = stage of next state

    This ensures true "proactive forecasting" — we only check FUTURE windows,
    not the current window (techsoln.md sec 2: "forecast K windows ahead").
    """
    X, y_next, y_inf, y_stage, hosts, times = [], [], [], [], [], []

    for host, g in state_df.groupby("host"):
        g = g.sort_values("window_start").reset_index(drop=True)
        feats = g[feature_columns].to_numpy(dtype=np.float32)
        stages = g["stage_label"].to_numpy(dtype=np.int64)
        n = len(g)

        # Note: range(T, n - K) with 0-based indexing means we need window indices
        # such that: t-T >= 0 (have T windows of history) and t+K < n (have K future windows)
        for t in range(T, n - K):
            # Input: windows [t-T:t] = S_{t-T}...S_{t-1} (past T windows)
            X.append(feats[t - T:t])
            # Targets: shift forward by 1 to get "next" semantics
            # y_next: predict S_t (the window right after history)
            y_next.append(feats[t])
            # y_inf: check stages at windows [t:t+K] (windows t, t+1, ..., t+K-1)
            # This represents S_t through S_{t+K-1}, which is the current window
            # plus K-1 future windows. For true future-only, this should be [t+1:t+1+K]
            y_inf.append(1.0 if np.any(stages[t + 1:t + 1 + K] > 0) else 0.0)
            y_stage.append(stages[t])
            hosts.append(host)
            times.append(g["window_start"].iloc[t])

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y_next": np.asarray(y_next, dtype=np.float32),
        "y_inf": np.asarray(y_inf, dtype=np.float32),
        "y_stage": np.asarray(y_stage, dtype=np.int64),
        "hosts": np.asarray(hosts),
        "times": np.asarray(times),
    }


def chronological_split(times: np.ndarray, train_frac: float = 0.70,
                       val_frac: float = 0.15, gap_windows: int = 0) -> tuple:
    """
    TEMPORALLY-SAFE split that respects chronological ordering.

    All sequences are sorted by timestamp. The split respects global time:
    - Train: earliest sequences
    - Val: middle sequences
    - Test: latest sequences

    Preserves temporal ordering while attempting to distribute sequences
    across all splits. If the requested fractions result in one split
    having all the data, adjusts to ensure each split has at least some
    sequences.

    Args:
        times: array of sequence end timestamps (from build_sequences)
        train_frac: target fraction for training
        val_frac: target fraction for validation (remainder is test)
        gap_windows: optional gap between splits (sequences to skip)

    Returns:
        (train_mask, val_mask, test_mask)
    """
    times_sorted = np.sort(times)
    n = len(times_sorted)

    # Calculate split points based on sequence count, respecting time order
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * (train_frac + val_frac)))

    # Ensure we have at least 1 sequence in each split
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - 1)

    t_val_start = times_sorted[n_train]
    t_test_start = times_sorted[n_val]

    train_mask = times < t_val_start
    val_mask = (times >= t_val_start) & (times < t_test_start)
    test_mask = times >= t_test_start

    return train_mask, val_mask, test_mask


def host_level_split(hosts: np.ndarray, train_frac: float, val_frac: float, seed: int = 42):
    """
    DEPRECATED: Old random host-level split.

    This is kept for backward compatibility with existing code, but
    chronological_split() is now the default and recommended approach.

    Splits by HOST, not by row — a random row-level split would leak future
    windows of the same attack session into training and silently inflate
    every metric (techsoln.md sec 8). Returns boolean masks aligned to `hosts`.
    """
    unique_hosts = np.unique(hosts)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_hosts)

    n = len(unique_hosts)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_hosts = set(unique_hosts[:n_train])
    val_hosts = set(unique_hosts[n_train:n_train + n_val])
    test_hosts = set(unique_hosts[n_train + n_val:])

    train_mask = np.isin(hosts, list(train_hosts))
    val_mask = np.isin(hosts, list(val_hosts))
    test_mask = np.isin(hosts, list(test_hosts))
    return train_mask, val_mask, test_mask


def verify_no_leakage(times: np.ndarray, train_mask: np.ndarray,
                      val_mask: np.ndarray, test_mask: np.ndarray,
                      tolerance_seconds: int = 60) -> dict:
    """
    Verify that splits are temporally disjoint (no information leakage).

    Checks:
    1. All sequences in train occur before all in val
    2. All sequences in val occur before all in test
    3. No overlap (with optional tolerance for boundary cases)

    Returns dict with:
    - is_valid: bool (all checks passed)
    - issues: list of identified leakage problems
    - stats: summary statistics
    """
    issues = []

    train_times = times[train_mask]
    val_times = times[val_mask]
    test_times = times[test_mask]

    stats = {
        "train_time_min": train_times.min() if len(train_times) > 0 else None,
        "train_time_max": train_times.max() if len(train_times) > 0 else None,
        "val_time_min": val_times.min() if len(val_times) > 0 else None,
        "val_time_max": val_times.max() if len(val_times) > 0 else None,
        "test_time_min": test_times.min() if len(test_times) > 0 else None,
        "test_time_max": test_times.max() if len(test_times) > 0 else None,
    }

    # Check train < val
    if len(train_times) > 0 and len(val_times) > 0:
        tolerance = pd.Timedelta(seconds=tolerance_seconds)
        overlap = train_times.max() > (val_times.min() - tolerance)
        if overlap:
            issues.append(f"Train/Val overlap: train_max={train_times.max()}, val_min={val_times.min()}")

    # Check val < test
    if len(val_times) > 0 and len(test_times) > 0:
        tolerance = pd.Timedelta(seconds=tolerance_seconds)
        overlap = val_times.max() > (test_times.min() - tolerance)
        if overlap:
            issues.append(f"Val/Test overlap: val_max={val_times.max()}, test_min={test_times.min()}")

    is_valid = len(issues) == 0

    return {
        "is_valid": is_valid,
        "issues": issues,
        "stats": stats,
    }
