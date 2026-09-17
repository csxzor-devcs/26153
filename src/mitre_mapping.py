"""
Maps dataset attack labels to MITRE ATT&CK stages (see techsoln.md sec 2).

This mapping is an explicit, documented assumption: public flow datasets
(CIC-IDS2017/2018) were not built with MITRE stages in mind. Exfiltration in
particular has no ground-truth label in these datasets, so it is never
assigned directly from a raw label — it is only ever produced by
`apply_exfiltration_heuristic` below, and every consumer of stage 5 must
treat it as heuristic, not ground truth.
"""
import numpy as np
import pandas as pd

STAGE_NAMES = [
    "Benign/None",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration (heuristic)",
    "Impact/Noise (DoS-DDoS, out of MITRE-5-stage scope)",
]

LABEL_TO_STAGE = {
    "Benign": 0,
    "PortScan": 1,
    "FTP-BruteForce": 2,
    "SSH-Bruteforce": 2,
    "Brute Force -Web": 2,
    "Brute Force -XSS": 2,
    "SQL Injection": 2,
    "Infilteration": 3,
    "Bot": 4,
    "DoS attacks-GoldenEye": 6,
    "DoS attacks-Slowloris": 6,
    "DoS attacks-SlowHTTPTest": 6,
    "DoS attacks-Hulk": 6,
    "DDoS attack-HOIC": 6,
    "DDoS attacks-LOIC-HTTP": 6,
    "DDOS attack-LOIC-UDP": 6,
}


def label_to_stage(label: str) -> int:
    """Unknown labels default to Benign(0) rather than raising, so a stray
    or misspelled dataset label never crashes the pipeline silently-wrong —
    it just contributes no attack signal, which is the safe failure mode."""
    return LABEL_TO_STAGE.get(label, 0)


def apply_exfiltration_heuristic(state_df: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Upgrades a window's stage_label to Exfiltration(5) when outbound bytes
    spike (z-score > z_thresh vs that host's own history) while the host is
    already in Lateral Movement(3) or Command & Control(4) — i.e. "large
    outbound transfer right after a foothold/beacon is established."
    This is a heuristic label, not dataset ground truth (see module docstring).
    """
    df = state_df.copy()

    def _flag(g: pd.DataFrame) -> pd.DataFrame:
        roll_mean = g["bytes_out"].expanding().mean().shift(1)
        roll_std = g["bytes_out"].expanding().std().shift(1).replace(0, np.nan)
        z = (g["bytes_out"] - roll_mean) / roll_std
        upgrade = (z > z_thresh) & g["stage_label"].isin([3, 4])
        g.loc[upgrade, "stage_label"] = 5
        return g

    return df.groupby("host", group_keys=False).apply(_flag)
