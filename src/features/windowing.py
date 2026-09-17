"""
Turns per-(host, window) state vectors into fixed-length sequences for the
world model: given T past states, predict the next state, the infiltration
label over the next K windows, and the MITRE stage of the very next window
(techsoln.md sec 4).
"""
import numpy as np
import pandas as pd

from src.features.extract import FEATURE_COLUMNS


def build_sequences(state_df: pd.DataFrame, T: int = 20, K: int = 5,
                     feature_columns=FEATURE_COLUMNS) -> dict:
    X, y_next, y_inf, y_stage, hosts, times = [], [], [], [], [], []

    for host, g in state_df.groupby("host"):
        g = g.sort_values("window_start").reset_index(drop=True)
        feats = g[feature_columns].to_numpy(dtype=np.float32)
        stages = g["stage_label"].to_numpy(dtype=np.int64)
        n = len(g)

        for t in range(T, n - K):
            X.append(feats[t - T:t])
            y_next.append(feats[t])
            y_inf.append(1.0 if np.any(stages[t:t + K] > 0) else 0.0)
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


def host_level_split(hosts: np.ndarray, train_frac: float, val_frac: float, seed: int):
    """
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
