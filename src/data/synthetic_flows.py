"""
Bundled offline data source: generates flow-level records in the same
column schema CICFlowMeter emits for CIC-IDS2017/2018, with an embedded
kill-chain campaign (recon -> initial access -> lateral movement -> C2,
with an exfiltration-like outbound burst for the heuristic in
mitre_mapping.py to catch). This exists so the prototype runs fully offline
without requiring the (large, application-gated) real dataset download.

To use the real dataset instead: set data.raw_flows_path in
configs/default.yaml to a real CIC-IDS2018/2017 CSV (or a directory of
them) — src/train.py only calls this generator when that path is missing.
"""
import numpy as np
import pandas as pd


FLOW_COLUMNS = [
    "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol",
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std",
    "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Label",
]


def _benign_flow(rng, t, src_ip):
    dst_port = rng.choice([80, 443, 53, 22, 3389])
    return {
        "Timestamp": t, "Src IP": src_ip,
        "Dst IP": f"10.0.0.{rng.integers(2, 250)}",
        "Src Port": rng.integers(1024, 65535), "Dst Port": dst_port,
        "Protocol": 6 if dst_port != 53 else 17,
        "Flow Duration": max(rng.normal(500_000, 150_000), 1000),
        "Tot Fwd Pkts": rng.integers(2, 15), "Tot Bwd Pkts": rng.integers(2, 15),
        "TotLen Fwd Pkts": max(rng.normal(800, 200), 40),
        "TotLen Bwd Pkts": max(rng.normal(800, 200), 40),
        "Flow IAT Mean": max(rng.normal(20_000, 5_000), 100),
        "Flow IAT Std": max(rng.normal(5_000, 1_000), 50),
        "SYN Flag Cnt": 1, "ACK Flag Cnt": 1, "RST Flag Cnt": 0, "FIN Flag Cnt": 1,
        "Init Fwd Win Byts": rng.integers(8192, 65535), "Init Bwd Win Byts": rng.integers(8192, 65535),
        "Label": "Benign",
    }


def _recon_flow(rng, t, src_ip, target_ip):
    """PortScan: many distinct dst ports on one target, SYN-only, no reply."""
    return {
        "Timestamp": t, "Src IP": src_ip, "Dst IP": target_ip,
        "Src Port": rng.integers(1024, 65535), "Dst Port": rng.integers(1, 1024),
        "Protocol": 6,
        "Flow Duration": max(rng.normal(2_000, 500), 100),
        "Tot Fwd Pkts": 1, "Tot Bwd Pkts": 0,
        "TotLen Fwd Pkts": 40, "TotLen Bwd Pkts": 0,
        "Flow IAT Mean": max(rng.normal(500, 100), 10),
        "Flow IAT Std": max(rng.normal(100, 20), 5),
        "SYN Flag Cnt": 1, "ACK Flag Cnt": 0, "RST Flag Cnt": 0, "FIN Flag Cnt": 0,
        "Init Fwd Win Byts": rng.integers(1024, 8192), "Init Bwd Win Byts": 0,
        "Label": "PortScan",
    }


def _bruteforce_flow(rng, t, src_ip, target_ip, success=False):
    """SSH-Bruteforce: repeated attempts on port 22, mostly RST (failed)."""
    return {
        "Timestamp": t, "Src IP": src_ip, "Dst IP": target_ip,
        "Src Port": rng.integers(1024, 65535), "Dst Port": 22,
        "Protocol": 6,
        "Flow Duration": max(rng.normal(3_000, 800), 200),
        "Tot Fwd Pkts": 3, "Tot Bwd Pkts": 1 if success else 0,
        "TotLen Fwd Pkts": max(rng.normal(120, 20), 40),
        "TotLen Bwd Pkts": 60 if success else 0,
        "Flow IAT Mean": max(rng.normal(300, 50), 10),
        "Flow IAT Std": max(rng.normal(60, 10), 5),
        "SYN Flag Cnt": 1, "ACK Flag Cnt": 1 if success else 0,
        "RST Flag Cnt": 0 if success else 1, "FIN Flag Cnt": 1 if success else 0,
        "Init Fwd Win Byts": rng.integers(1024, 8192), "Init Bwd Win Byts": rng.integers(0, 4096),
        "Label": "SSH-Bruteforce",
    }


def _lateral_flow(rng, t, src_ip, target_ip):
    """Infilteration: larger, bidirectional, longer-lived internal flows."""
    return {
        "Timestamp": t, "Src IP": src_ip, "Dst IP": target_ip,
        "Src Port": rng.integers(1024, 65535), "Dst Port": rng.choice([445, 3389, 135, 5985]),
        "Protocol": 6,
        "Flow Duration": max(rng.normal(1_200_000, 300_000), 50_000),
        "Tot Fwd Pkts": rng.integers(20, 80), "Tot Bwd Pkts": rng.integers(20, 80),
        "TotLen Fwd Pkts": max(rng.normal(4_000, 1_000), 200),
        "TotLen Bwd Pkts": max(rng.normal(4_000, 1_000), 200),
        "Flow IAT Mean": max(rng.normal(15_000, 3_000), 100),
        "Flow IAT Std": max(rng.normal(4_000, 800), 50),
        "SYN Flag Cnt": 1, "ACK Flag Cnt": 1, "RST Flag Cnt": 0, "FIN Flag Cnt": 1,
        "Init Fwd Win Byts": rng.integers(8192, 65535), "Init Bwd Win Byts": rng.integers(8192, 65535),
        "Label": "Infilteration",
    }


def _c2_flow(rng, t, src_ip, c2_ip, exfil_burst=False):
    """Bot: small periodic beacons to a fixed external IP; occasional large
    outbound burst (still labelled 'Bot' at flow level — the exfiltration
    heuristic in mitre_mapping.py is what promotes this to stage 5, from
    the aggregate byte spike, exactly as documented in techsoln.md)."""
    bytes_out = max(rng.normal(50_000, 8_000), 10_000) if exfil_burst else max(rng.normal(300, 60), 40)
    return {
        "Timestamp": t, "Src IP": src_ip, "Dst IP": c2_ip,
        "Src Port": rng.integers(1024, 65535), "Dst Port": 443,
        "Protocol": 6,
        "Flow Duration": max(rng.normal(400_000, 50_000), 10_000),
        "Tot Fwd Pkts": rng.integers(1, 5), "Tot Bwd Pkts": rng.integers(1, 5),
        "TotLen Fwd Pkts": bytes_out, "TotLen Bwd Pkts": max(rng.normal(200, 40), 40),
        "Flow IAT Mean": max(rng.normal(400_000, 20_000), 1_000),  # regular beacon cadence
        "Flow IAT Std": max(rng.normal(5_000, 1_000), 50),
        "SYN Flag Cnt": 1, "ACK Flag Cnt": 1, "RST Flag Cnt": 0, "FIN Flag Cnt": 1,
        "Init Fwd Win Byts": rng.integers(8192, 65535), "Init Bwd Win Byts": rng.integers(8192, 65535),
        "Label": "Bot",
    }


def generate_synthetic_flows(n_hosts: int = 40, attack_fraction: float = 0.5,
                              session_minutes: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Builds one session per host: `session_minutes` of benign baseline, and
    for `attack_fraction` of hosts, a kill-chain campaign (recon ->
    initial access -> lateral movement -> C2 -> exfil-like burst) injected
    partway through, at realistically higher flow rates than the benign
    baseline (matching how these phases actually look in CIC-IDS traffic).
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2026-01-01 00:00:00")

    for h in range(n_hosts):
        src_ip = f"192.168.1.{h + 2}"
        is_attack = rng.random() < attack_fraction
        session_seconds = session_minutes * 60
        t = 0.0

        # benign baseline for the whole session (background traffic never stops)
        while t < session_seconds:
            rows.append(_benign_flow(rng, base_time + pd.Timedelta(seconds=t), src_ip))
            t += rng.exponential(3.0)

        if not is_attack:
            continue

        target_ip = f"10.0.0.{rng.integers(2, 250)}"
        c2_ip = f"203.0.113.{rng.integers(2, 250)}"
        # Campaign must start well past the model's history length (T windows,
        # default 20 * 60s) or every attack window gets trimmed away as pure
        # "history" before a single (X, y) sequence is ever built from it —
        # starting at 55-70% of the session leaves >=20 benign windows before
        # the campaign AND >=20 windows of runway after it for K-step targets.
        campaign_start = rng.uniform(session_seconds * 0.55, session_seconds * 0.70)
        t = campaign_start

        for _ in range(rng.integers(80, 150)):     # Reconnaissance: dense port-scan burst
            rows.append(_recon_flow(rng, base_time + pd.Timedelta(seconds=t), src_ip, target_ip))
            t += rng.exponential(1.0)

        for i in range(rng.integers(40, 80)):       # Initial Access: brute force, succeeds near the end
            success = i > 35
            rows.append(_bruteforce_flow(rng, base_time + pd.Timedelta(seconds=t), src_ip, target_ip, success))
            t += rng.exponential(1.5)

        for _ in range(rng.integers(30, 60)):        # Lateral Movement
            rows.append(_lateral_flow(rng, base_time + pd.Timedelta(seconds=t), src_ip, target_ip))
            t += rng.exponential(4.0)

        n_beacons = rng.integers(20, 40)              # Command & Control: regular beacons, then an exfil burst
        for i in range(n_beacons):
            exfil_burst = i >= n_beacons - 4
            rows.append(_c2_flow(rng, base_time + pd.Timedelta(seconds=t), src_ip, c2_ip, exfil_burst))
            t += rng.exponential(3.0) + 1.0

    df = pd.DataFrame(rows, columns=FLOW_COLUMNS)
    return df.sort_values(["Src IP", "Timestamp"]).reset_index(drop=True)


if __name__ == "__main__":
    flows = generate_synthetic_flows()
    print(flows["Label"].value_counts())
    print(f"\n{len(flows)} flows across {flows['Src IP'].nunique()} hosts")
