"""
Feature extraction pipeline: flow-level rows (CICFlowMeter schema, whether
real CIC-IDS2018/2017 CSVs or the bundled synthetic generator) -> a
normalised per-window, per-host state vector S_t (techsoln.md sec 3).

Packet-level features (TTL, TCP window size, fragmentation, retransmission)
are a second block that degrades gracefully to zero when PCAP-derived
columns aren't present in the input — flow-only CSVs (the common case for
enterprise NetFlow/IPFIX collectors) still produce a valid state vector.
"""
import numpy as np
import pandas as pd

from src.mitre_mapping import label_to_stage, apply_exfiltration_heuristic

FEATURE_COLUMNS = [
    "flow_count", "mean_duration", "std_duration",
    "bytes_in", "bytes_out", "pkts_in", "pkts_out",
    "unique_dst_ip", "unique_dst_port",
    "syn_count", "ack_count", "rst_count", "fin_count",
    "iat_mean", "iat_std", "bidirectional_ratio", "tcp_pct", "win_size_mean",
    "ttl_mean", "retransmission_count",           # packet-level block (zero if unavailable)
    "n_distinct_dst_ips", "n_distinct_dst_ports", "fan_out_ratio", "new_dst_ratio",  # graph-derived
    "init_fwd_win_mean", "init_bwd_win_mean", "pkt_len_mean_agg", "pkt_len_std_agg", "fwd_header_len_mean",  # packet-proxy
]


def build_state_vectors(flows_df: pd.DataFrame, window_seconds: int = 60,
                         host_col: str = "Src IP") -> pd.DataFrame:
    """
    Aggregates flow rows into fixed-size state vectors per (host, window).
    Returns a DataFrame with columns: host, window_start, FEATURE_COLUMNS...,
    stage_label (int, via mitre_mapping — see module docstring on exfil).

    Includes graph-derived features (distinct IPs/ports, fan_out, new_dst) and
    packet-proxy features (window sizes, packet lengths). Missing columns default
    to 0.0 with debug warnings.
    """
    df = flows_df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["window_start"] = df["Timestamp"].dt.floor(f"{window_seconds}s")

    has_ttl = "TTL" in df.columns
    has_retransmit = "Retransmit Cnt" in df.columns
    has_pkt_len_mean = "Packet Length Mean" in df.columns
    has_pkt_len_std = "Packet Length Std" in df.columns
    has_fwd_header_len = "Fwd Header Length" in df.columns
    has_init_bwd_win = "Init Bwd Win Byts" in df.columns

    if not has_pkt_len_mean:
        print(f"[extract] WARNING: 'Pkt Len Mean' column not found (synthetic data?), defaulting to 0.0")
    if not has_pkt_len_std:
        print(f"[extract] WARNING: 'Pkt Len Std' column not found, defaulting to 0.0")
    if not has_fwd_header_len:
        print(f"[extract] WARNING: 'Fwd Header Len' column not found, defaulting to 0.0")
    if not has_init_bwd_win:
        print(f"[extract] WARNING: 'Init Bwd Win Byts' column not found, defaulting to 0.0")

    rows = []
    prev_dst_ips = {}  # Track per-host previous window's destination IPs

    # Single pass grouped by (host, window) instead of re-scanning the full
    # frame per host: with ~17k hosts, filtering the whole DataFrame once per
    # host (the old approach) is O(hosts * rows) and takes hours on the real
    # CIC-IDS2017 data. groupby does it in one pass over the data.
    df = df.sort_values([host_col, "window_start"])
    for (host, wstart), g in df.groupby([host_col, "window_start"], sort=False):
        n_distinct_dst_ips = g["Dst IP"].nunique()
        n_distinct_dst_ports = g["Dst Port"].nunique()
        n_flows = len(g)
        fan_out_ratio = n_distinct_dst_ips / (n_flows + 1)

        # new_dst_ratio: fraction of IPs not seen in previous window
        current_dst_ips = set(g["Dst IP"].unique())
        if host in prev_dst_ips and len(prev_dst_ips[host]) > 0:
            n_new_dst_ips = len(current_dst_ips - prev_dst_ips[host])
            new_dst_ratio = n_new_dst_ips / max(len(current_dst_ips), 1)
        else:
            new_dst_ratio = 1.0 if len(current_dst_ips) > 0 else 0.0
        prev_dst_ips[host] = current_dst_ips

        row = {
            "host": host,
            "window_start": wstart,
            "flow_count": n_flows,
            "mean_duration": g["Flow Duration"].mean(),
            "std_duration": g["Flow Duration"].std(ddof=0),
            "bytes_in": g["TotLen Bwd Pkts"].sum(),
            "bytes_out": g["TotLen Fwd Pkts"].sum(),
            "pkts_in": g["Tot Bwd Pkts"].sum(),
            "pkts_out": g["Tot Fwd Pkts"].sum(),
            "unique_dst_ip": g["Dst IP"].nunique(),
            "unique_dst_port": g["Dst Port"].nunique(),
            "syn_count": g["SYN Flag Cnt"].sum(),
            "ack_count": g["ACK Flag Cnt"].sum(),
            "rst_count": g["RST Flag Cnt"].sum(),
            "fin_count": g["FIN Flag Cnt"].sum(),
            "iat_mean": g["Flow IAT Mean"].mean(),
            "iat_std": g["Flow IAT Std"].mean(),
            "bidirectional_ratio": (g["Tot Bwd Pkts"] > 0).mean(),
            "tcp_pct": (g["Protocol"] == 6).mean(),
            "win_size_mean": g["Init Fwd Win Byts"].mean(),
            "ttl_mean": g["TTL"].mean() if has_ttl else 0.0,
            "retransmission_count": g["Retransmit Cnt"].sum() if has_retransmit else 0.0,
            "n_distinct_dst_ips": n_distinct_dst_ips,
            "n_distinct_dst_ports": n_distinct_dst_ports,
            "fan_out_ratio": fan_out_ratio,
            "new_dst_ratio": new_dst_ratio,
            "init_fwd_win_mean": g["Init Fwd Win Byts"].mean(),
            "init_bwd_win_mean": g["Init Bwd Win Byts"].mean() if has_init_bwd_win else 0.0,
            "pkt_len_mean_agg": g["Packet Length Mean"].mean() if has_pkt_len_mean else 0.0,
            "pkt_len_std_agg": g["Packet Length Std"].mean() if has_pkt_len_std else 0.0,
            "fwd_header_len_mean": g["Fwd Header Length"].mean() if has_fwd_header_len else 0.0,
            "stage_label": int(g["Label"].map(label_to_stage).max()),
        }
        rows.append(row)

    state_df = pd.DataFrame(rows).sort_values(["host", "window_start"]).reset_index(drop=True)
    state_df[FEATURE_COLUMNS] = state_df[FEATURE_COLUMNS].fillna(0.0)
    state_df = apply_exfiltration_heuristic(state_df)
    return state_df


if __name__ == "__main__":
    from src.data.synthetic_flows import generate_synthetic_flows

    flows = generate_synthetic_flows()
    states = build_state_vectors(flows)
    print(states[["host", "window_start", "flow_count", "stage_label"]].head(10))
    print(f"\n{len(states)} state windows across {states['host'].nunique()} hosts")
    print("Stage label distribution:\n", states["stage_label"].value_counts().sort_index())
