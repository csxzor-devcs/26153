"""
MITRE ATT&CK mapping and exfiltration heuristics.

This module provides:
- Mapping of CIC-IDS2017 attack labels to attack stages.
- A lightweight exfiltration heuristic used during feature extraction.
"""

from __future__ import annotations

import pandas as pd


LABEL_TO_STAGE = {
    "BENIGN": 0,

    # Reconnaissance
    "PortScan": 1,

    # Initial Access / Credential Access
    "FTP-Patator": 2,
    "SSH-Patator": 2,

    # Discovery / Execution / Impact
    "DoS Hulk": 3,
    "DoS GoldenEye": 3,
    "DoS slowloris": 3,
    "DoS Slowhttptest": 3,
    "DDoS": 3,

    # Credential / Application attacks
    # Note: the CIC-IDS2017 CSVs encode these labels with a literal 0x96
    # (en dash) byte between "Web Attack" and the sub-type, which survives
    # as U+0096 after latin-1 decoding — not a double space.
    "Web Attack \x96 Brute Force": 4,
    "Web Attack \x96 XSS": 4,
    "Web Attack \x96 Sql Injection": 4,

    # Command and Control / Persistence
    "Bot": 5,
    "Infiltration": 5,

    # Exfiltration / other high-impact activity
    "Heartbleed": 6,
}


def label_to_stage(label: str) -> int:
    """
    Convert a CIC-IDS2017 label into an attack stage.

    An unknown or misspelled dataset label never crashes the pipeline
    silently-wrong — it just contributes no attack signal, which is the
    safe failure mode.
    """
    if pd.isna(label):
        return 0

    return LABEL_TO_STAGE.get(str(label).strip(), 0)


def apply_exfiltration_heuristic(
    state_df: pd.DataFrame,
    z_thresh: float = 2.0,
) -> pd.DataFrame:
    """
    Apply a lightweight per-host exfiltration heuristic.

    Escalates a window to the exfiltration/high-impact stage (6) only when
    BOTH hold:
    - the host is already flagged Bot/Infiltration (stage 5) — i.e. a
      foothold is already established, matching this heuristic's original
      intent of catching "a large outbound transfer right after a
      foothold/beacon", not just any bursty traffic day; and
    - that window's outbound bytes are a statistical outlier (z_thresh)
      relative to that host's own PAST behavior only (expanding mean/std,
      shifted by one window) — using the host's full-history mean/std
      (including future windows) would leak future information into a
      past window's label.

    The implementation intentionally avoids DataFrameGroupBy.apply()
    so that the `host` grouping column is preserved across pandas
    versions.
    """
    df = state_df.copy()

    def _flag(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("window_start").copy()

        if "bytes_out" not in g.columns or "stage_label" not in g.columns:
            return g

        values = pd.to_numeric(g["bytes_out"], errors="coerce").fillna(0.0)

        roll_mean = values.expanding().mean().shift(1)
        roll_std = values.expanding().std().shift(1).replace(0, pd.NA)

        z = (values - roll_mean) / roll_std
        already_compromised = g["stage_label"] >= 5
        upgrade = (z > z_thresh).fillna(False) & already_compromised

        g.loc[upgrade, "stage_label"] = 6

        return g

    groups = []

    for _, group in df.groupby("host", sort=False):
        groups.append(_flag(group))

    if not groups:
        return df

    result = pd.concat(groups, ignore_index=True)

    return result