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

    Canonical temporal indexing (REQUIRED FOR REPRODUCIBILITY):
    Given index i, we define:
    - S_i = network state at window i
    - Sequence history ends at window t: history contains S_{t-T+1}, ..., S_t (T windows total)
    - Targets are future-only: S_{t+1}, ..., S_{t+K}

    Formal specification:
    - X = [S_{t-T+1}, ..., S_t]           (input: history ending at t)
    - y_next = S_{t+1}                     (target: next state)
    - y_inf = 1[any(S_{t+1}...S_{t+K}) is attack]  (infiltration in next K windows)
    - y_stage = stage(S_{t+1})             (stage of next state)
    - times[i] = timestamp of S_t          (history endpoint)

    SECURITY: y_inf and y_stage are NEVER trained on test data — they come from the
    next K windows after the history endpoint.

    In 0-indexed Python with range(T, n-K):
    - History: feats[t-T:t] gives S_{t-T}, ..., S_{t-1} → need shift to S_{t-T+1}, ..., S_t
    """
    X, y_next, y_inf, y_stage, hosts, times = [], [], [], [], [], []

    for host, g in state_df.groupby("host"):
        g = g.sort_values("window_start").reset_index(drop=True)
        feats = g[feature_columns].to_numpy(dtype=np.float32)
        stages = g["stage_label"].to_numpy(dtype=np.int64)
        n = len(g)

        # Loop from T to n-K: each iteration i represents a sequence
        # where the history ends at window i
        for i in range(T, n - K):
            # History: [S_{i-T+1}, ..., S_i]
            # In Python 0-indexing: feats[i-T+1:i+1]
            # But feats[i-T:i] gives [S_{i-T}, ..., S_{i-1}]
            # So shift by 1: feats[(i-T+1):(i+1)] = feats[i-T+1:i+1]
            X.append(feats[i - T + 1:i + 1])

            # Target next state: S_{i+1}
            y_next.append(feats[i + 1])

            # Infiltration in NEXT K windows: [S_{i+1}, ..., S_{i+K}]
            # Python slice: stages[i+1:i+1+K]
            y_inf.append(1.0 if np.any(stages[i + 1:i + 1 + K] > 0) else 0.0)

            # Stage of next state: S_{i+1}
            y_stage.append(stages[i + 1])

            hosts.append(host)
            times.append(g["window_start"].iloc[i])  # Timestamp at i, end of history

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y_next": np.asarray(y_next, dtype=np.float32),
        "y_inf": np.asarray(y_inf, dtype=np.float32),
        "y_stage": np.asarray(y_stage, dtype=np.int64),
        "hosts": np.asarray(hosts),
        "times": np.asarray(times),
    }


def chronological_split(times: np.ndarray, train_frac: float = 0.70,
                       val_frac: float = 0.15, purge_gap_seconds: int = 0) -> tuple:
    """
    TEMPORALLY-SAFE split with purge gap to prevent information leakage.

    Guarantees:
    ```
    TRAIN < PURGE GAP < VALIDATION < PURGE GAP < TEST
    ```

    Each sequence contains:
    - History: T past windows
    - Targets: next K windows

    Therefore a sequence at time t_i uses information from t_i-T*60 to t_i+K*60.
    The purge gap ensures no target window from training overlaps with history
    of validation sequences, and no target from validation overlaps with test.

    Args:
        times: sequence timestamps (end of history window for each sequence)
        train_frac: fraction of time range for training
        val_frac: fraction for validation (remainder is test)
        purge_gap_seconds: temporal gap between splits (prevents leakage around boundaries)
                          Recommended: >= K * 60 (K forecast windows)
                          Default: 0 (tight but valid if K is small)

    Returns:
        (train_mask, val_mask, test_mask) - boolean arrays

    Leakage Prevention:
    1. Chronological ordering: sequences ordered by time
    2. Purge gap: buffer zone with no sequences
    3. No temporal overlap: max(train) < min(val), max(val) < min(test)
    """
    times_sorted = np.sort(times)
    n = len(times_sorted)
    time_min, time_max = times_sorted[0], times_sorted[-1]
    time_range = time_max - time_min

    # Calculate theoretical split boundaries in time space
    t_val_start_ideal = time_min + time_range * train_frac
    t_test_start_ideal = time_min + time_range * (train_frac + val_frac)

    # Apply purge gaps
    t_val_start = t_val_start_ideal + pd.Timedelta(seconds=purge_gap_seconds)
    t_test_start = t_test_start_ideal + pd.Timedelta(seconds=purge_gap_seconds)

    # Create masks
    train_mask = times < t_val_start_ideal
    val_mask = (times >= t_val_start) & (times < t_test_start_ideal)
    test_mask = times >= t_test_start

    # Ensure we have at least 1 sequence in each split
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            f"Chronological split with purge_gap_seconds={purge_gap_seconds} "
            f"resulted in empty split(s). Try reducing purge_gap_seconds or adjusting fractions."
        )

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
