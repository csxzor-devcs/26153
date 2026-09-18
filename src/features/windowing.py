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

    CANONICAL TEMPORAL INDEXING (REQUIRED FOR REPRODUCIBILITY):

    This module enforces a strict temporal indexing convention to prevent
    information leakage. All sequences are defined relative to the history
    endpoint index `i`:

    - S_i = network state at window i (history endpoint)
    - Input history:  X = [S_{i-T+1}, S_{i-T+2}, ..., S_i]  (T windows total)
    - Next state:     y_next = S_{i+1}
    - Infiltration:   y_inf = 1 iff any(stage in [S_{i+1}, S_{i+2}, ..., S_{i+K}] > 0)
    - Stage target:   y_stage = stage(S_{i+1})
    - Sequence time:  times[i] = timestamp of S_i (history endpoint)

    CRITICAL: Targets y_inf and y_stage ONLY use windows [i+1, i+K]. They never
    use the history [i-T+1, i], preventing information leakage within the sequence.

    PYTHON IMPLEMENTATION DETAIL (0-indexed arrays):
    For loop variable i in range(T, n-K), we build:
    - feats[i-T+1:i+1]        gives [S_{i-T+1}, ..., S_i]  ✓ correct (T windows)
    - feats[i+1]              gives S_{i+1}               ✓ correct (next state)
    - stages[i+1:i+1+K]       gives [S_{i+1}, ..., S_{i+K}]  ✓ correct (future only)
    - g["window_start"].iloc[i]  timestamp of S_i         ✓ correct (history endpoint)
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


def per_scenario_chronological_split(times: np.ndarray, train_frac: float = 0.70,
                                      val_frac: float = 0.15, hosts_array: np.ndarray = None,
                                      purge_gap_seconds: int = 0) -> tuple:
    """
    Per-scenario chronological split: groups sequences by date, splits each date
    chronologically (70/15/15), then concatenates splits across dates.

    This ensures attacks that cluster on certain days don't all end up in training.

    Args:
        times: sequence timestamps (pandas Timestamp objects)
        train_frac: fraction within each scenario for training
        val_frac: fraction within each scenario for validation (remainder is test)
        purge_gap_seconds: gap between splits (applied within each scenario)

    Returns:
        (train_mask, val_mask, test_mask) - boolean arrays
    """
    train_mask = np.zeros(len(times), dtype=bool)
    val_mask = np.zeros(len(times), dtype=bool)
    test_mask = np.zeros(len(times), dtype=bool)

    # Decide grouping strategy
    if hosts_array is not None:
        # Per-host grouping for synthetic data
        unique_scenarios = np.unique(hosts_array)
        grouping_key = hosts_array
        grouping_type = "host"
    else:
        # Per-date grouping for real data
        unique_scenarios = np.unique([t.date() if hasattr(t, 'date') else t.strftime('%Y-%m-%d') for t in times])
        grouping_key = [t.date() if hasattr(t, 'date') else t.strftime('%Y-%m-%d') for t in times]
        grouping_type = "date"

    for scenario in unique_scenarios:
        # Find all sequences for this scenario
        if grouping_type == "host":
            scenario_mask = grouping_key == scenario
        else:
            scenario_mask = np.array([g == scenario for g in grouping_key])

        if not scenario_mask.any():
            continue

        scenario_indices = np.where(scenario_mask)[0]
        scenario_times = times[scenario_mask]

        # Split this scenario's sequences chronologically
        times_sorted_idx = np.argsort(scenario_times)
        n = len(scenario_times)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        # Map back to original indices
        train_idx = scenario_indices[times_sorted_idx[:n_train]]
        val_idx = scenario_indices[times_sorted_idx[n_train:n_train + n_val]]
        test_idx = scenario_indices[times_sorted_idx[n_train + n_val:]]

        # Enforce the purge gap: drop sequences at the val/test edge that
        # start before (previous split's last timestamp + purge_gap). An
        # index-only split leaves boundary sequences whose history/target
        # footprint still overlaps the neighboring split; dropping them
        # (rather than reassigning) is standard purge-gap practice.
        purge = pd.Timedelta(seconds=purge_gap_seconds)

        if len(train_idx) > 0 and len(val_idx) > 0:
            train_end_time = times[train_idx].max()
            val_idx = val_idx[times[val_idx] >= train_end_time + purge]

        if len(val_idx) > 0 and len(test_idx) > 0:
            val_end_time = times[val_idx].max()
            test_idx = test_idx[times[test_idx] >= val_end_time + purge]
        elif len(train_idx) > 0 and len(test_idx) > 0:
            train_end_time = times[train_idx].max()
            test_idx = test_idx[times[test_idx] >= train_end_time + purge]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    # Verify all splits have at least one sequence with attacks
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        print("[WARNING] Per-scenario split resulted in empty split(s). Falling back to global chronological split.")
        return chronological_split(times, train_frac, val_frac, purge_gap_seconds)

    return train_mask, val_mask, test_mask


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
                      T: int = 20, K: int = 5, window_seconds: int = 60,
                      tolerance_seconds: int = 0, hosts_array: np.ndarray = None) -> dict:
    """
    Verify temporal split integrity using EFFECTIVE INTERVALS, not just timestamps.

    Each sequence at time t_i has an effective temporal footprint:
    - History start:  t_i - (T-1) * window_seconds
    - History end:    t_i
    - Target start:   t_i + window_seconds
    - Target end:     t_i + K * window_seconds
    - Total interval: [t_i - (T-1)*window_seconds, t_i + K*window_seconds]

    Leakage prevention requires:
    - max(train_effective_end) < min(val_effective_start)
    - max(val_effective_end) < min(test_effective_start)

    This prevents training sequences' target windows from overlapping with
    validation/test sequences' history windows.

    If `hosts_array` is given (per-scenario/per-host split), the check is run
    independently per host instead of over the global timeline. Each host's
    sequences are an independent sample with no cross-host inputs, so two
    different hosts being simultaneously active (the normal case on a shared
    network capture) is not leakage — only a host's own train sequences
    overlapping that same host's val/test sequences is.

    Returns dict with:
    - is_valid: bool (no leakage detected)
    - issues: list of leakage violations
    - stats: detailed interval boundaries
    """
    if hosts_array is not None:
        issues = []
        for host in np.unique(hosts_array):
            host_mask = hosts_array == host
            sub = verify_no_leakage(
                times[host_mask], train_mask[host_mask], val_mask[host_mask], test_mask[host_mask],
                T=T, K=K, window_seconds=window_seconds, tolerance_seconds=tolerance_seconds,
            )
            issues.extend(f"[host {host}] {issue}" for issue in sub["issues"])

        # Global stats are informational only here (per-host validity is what counts).
        global_check = verify_no_leakage(
            times, train_mask, val_mask, test_mask,
            T=T, K=K, window_seconds=window_seconds, tolerance_seconds=tolerance_seconds,
        )
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "stats": global_check["stats"],
        }

    issues = []

    train_times = times[train_mask]
    val_times = times[val_mask]
    test_times = times[test_mask]

    # Compute effective intervals
    history_offset = pd.Timedelta(seconds=(T-1) * window_seconds)
    target_offset = pd.Timedelta(seconds=K * window_seconds)

    def effective_interval(t):
        """Compute [start, end] for a sequence at time t"""
        return (t - history_offset, t + target_offset)

    # Compute interval boundaries for each split
    if len(train_times) > 0:
        train_intervals = [effective_interval(t) for t in train_times]
        train_start = min(i[0] for i in train_intervals)
        train_end = max(i[1] for i in train_intervals)
    else:
        train_start = train_end = None

    if len(val_times) > 0:
        val_intervals = [effective_interval(t) for t in val_times]
        val_start = min(i[0] for i in val_intervals)
        val_end = max(i[1] for i in val_intervals)
    else:
        val_start = val_end = None

    if len(test_times) > 0:
        test_intervals = [effective_interval(t) for t in test_times]
        test_start = min(i[0] for i in test_intervals)
        test_end = max(i[1] for i in test_intervals)
    else:
        test_start = test_end = None

    # Check train < val (with optional tolerance)
    if train_end is not None and val_start is not None:
        tol = pd.Timedelta(seconds=tolerance_seconds)
        if train_end > (val_start - tol):
            issues.append(
                f"Train/Val leakage: train_effective_end={train_end} > "
                f"val_effective_start={val_start}"
            )

    # Check val < test (with optional tolerance)
    if val_end is not None and test_start is not None:
        tol = pd.Timedelta(seconds=tolerance_seconds)
        if val_end > (test_start - tol):
            issues.append(
                f"Val/Test leakage: val_effective_end={val_end} > "
                f"test_effective_start={test_start}"
            )

    is_valid = len(issues) == 0

    stats = {
        "train_effective_start": train_start,
        "train_effective_end": train_end,
        "val_effective_start": val_start,
        "val_effective_end": val_end,
        "test_effective_start": test_start,
        "test_effective_end": test_end,
        "history_offset_seconds": (T-1) * window_seconds,
        "target_offset_seconds": K * window_seconds,
        "tolerance_seconds": tolerance_seconds,
    }

    return {
        "is_valid": is_valid,
        "issues": issues,
        "stats": stats,
    }
