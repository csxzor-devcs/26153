"""
Real dataset preparation: CSE-CIC-IDS2018 / CIC-IDS2017 ingestion pipeline.

This module provides a robust ingestion path for public network intrusion
detection datasets. It handles:
- Multiple CSV files from a directory
- Column name normalization (dataset variants use different names)
- Required column validation
- Malformed row detection and safe removal
- Infinity and NaN handling
- Data quality logging (row counts, class distribution, timestamp range, hosts)
- Dataset provenance tracking

Usage (from repo root):
    python -m src.data.prepare \\
        --input-dir /path/to/CIC-IDS2018/CSVs \\
        --output data/raw/flows.csv \\
        --source cic_ids2018
"""
import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[prepare] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# CICFlowMeter column names and common aliases
REQUIRED_COLUMNS = [
    "Timestamp", "Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol",
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow IAT Mean", "Flow IAT Std",
    "SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt", "FIN Flag Cnt",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Label"
]

# CIC-IDS datasets sometimes use different column names
COLUMN_ALIASES = {
    # Timestamp variations
    "Timestamp": ["Timestamp", "Dst Port "],  # Some datasets have trailing space
    "Src IP": ["Src IP", "Source IP"],
    "Dst IP": ["Dst IP", "Destination IP"],
    # Some datasets use shortened names
    "Flow Duration": ["Flow Duration", "Fwd IAT Total"],
    "Protocol": ["Protocol", "Protocol Code"],
}

# Optional packet-level features (degrade to 0 if absent)
OPTIONAL_COLUMNS = ["TTL", "Retransmit Cnt", "Retransmission Count"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard CICFlowMeter names."""
    df = df.copy()

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()

    # Try to match standard columns with aliases
    col_mapping = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and standard_name not in df.columns:
                col_mapping[alias] = standard_name
                break

    if col_mapping:
        df = df.rename(columns=col_mapping)
        logger.info(f"Normalized columns: {col_mapping}")

    return df


def validate_columns(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate that all required columns are present.
    Returns (is_valid, missing_columns).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def clean_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove malformed rows and handle special values (Inf, NaN).
    Returns (cleaned_df, removal_stats).
    """
    stats = {
        "rows_before": len(df),
        "rows_removed_na": 0,
        "rows_removed_inf": 0,
        "rows_removed_non_numeric": 0,
    }

    df = df.copy()

    # Identify numeric columns (all except Timestamp, IPs, Label)
    numeric_cols = [col for col in df.columns if col not in
                    ["Timestamp", "Src IP", "Dst IP", "Label"]]

    # Convert to numeric, coercing errors to NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag rows with NaN in critical columns
    critical_cols = [col for col in numeric_cols if col not in OPTIONAL_COLUMNS]
    rows_with_na = df[critical_cols].isna().any(axis=1)
    stats["rows_removed_na"] = rows_with_na.sum()

    # Flag rows with Inf
    rows_with_inf = np.isinf(df[numeric_cols]).any(axis=1)
    stats["rows_removed_inf"] = rows_with_inf.sum()

    # Remove flagged rows
    df = df[~(rows_with_na | rows_with_inf)].reset_index(drop=True)

    # Replace Inf with NaN, then forward-fill per host
    for col in numeric_cols:
        df.loc[df[col] == np.inf, col] = np.nan
        df.loc[df[col] == -np.inf, col] = np.nan

    # Fill NaN in optional columns with 0
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Validate Timestamp and IPs are present
    if df["Timestamp"].isna().any():
        stats["rows_removed_na"] += df["Timestamp"].isna().sum()
        df = df[~df["Timestamp"].isna()].reset_index(drop=True)

    if df["Label"].isna().any():
        stats["rows_removed_na"] += df["Label"].isna().sum()
        df = df[~df["Label"].isna()].reset_index(drop=True)

    stats["rows_after"] = len(df)
    stats["rows_removed_total"] = stats["rows_before"] - stats["rows_after"]

    return df, stats


def load_csvs(input_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a directory.
    Handles multiple scenarios (one file or many).
    """
    input_path = Path(input_dir)

    if input_path.is_file():
        # Single CSV file
        logger.info(f"Loading single CSV: {input_path}")
        return pd.read_csv(input_path)

    if input_path.is_dir():
        # Directory of CSVs
        csv_files = sorted(glob.glob(str(input_path / "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")

        logger.info(f"Found {len(csv_files)} CSV files in {input_dir}")
        dfs = []
        for csv_file in csv_files:
            logger.info(f"  Loading: {Path(csv_file).name}")
            dfs.append(pd.read_csv(csv_file))

        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Concatenated {len(csv_files)} files: {len(df)} total rows")
        return df

    raise FileNotFoundError(f"Input path not found: {input_dir}")


def log_data_quality(df: pd.DataFrame, source: str) -> dict:
    """
    Compute and log data quality metrics.
    Returns metrics dict for provenance recording.
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    stats = {
        "source": source,
        "rows_total": len(df),
        "unique_hosts": int(df["Src IP"].nunique()),
        "timestamp_min": df["Timestamp"].min().isoformat(),
        "timestamp_max": df["Timestamp"].max().isoformat(),
        "timestamp_duration_hours": (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 3600,
    }

    # Class distribution
    if "Label" in df.columns:
        label_dist = df["Label"].value_counts().to_dict()
        stats["label_distribution"] = label_dist
        logger.info(f"Label distribution: {label_dist}")

    logger.info(f"Data quality summary:")
    logger.info(f"  Total rows: {stats['rows_total']:,}")
    logger.info(f"  Unique hosts: {stats['unique_hosts']}")
    logger.info(f"  Timestamp range: {stats['timestamp_min']} to {stats['timestamp_max']}")
    logger.info(f"  Duration: {stats['timestamp_duration_hours']:.1f} hours")

    return stats


def main(input_dir: str, output_file: str, source: str = "cic_ids2018"):
    """
    End-to-end dataset preparation.

    Args:
        input_dir: Path to input CSV or directory of CSVs
        output_file: Path to save processed flows.csv
        source: Dataset source identifier (cic_ids2018, cic_ids2017, etc.)
    """
    logger.info(f"Starting dataset preparation: {source}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_file}")

    # Load
    df = load_csvs(input_dir)
    logger.info(f"Loaded {len(df)} raw rows")

    # Normalize columns
    df = normalize_columns(df)

    # Validate
    is_valid, missing = validate_columns(df)
    if not is_valid:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("✓ All required columns present")

    # Clean
    df, clean_stats = clean_rows(df)
    logger.info(f"Cleaning stats:")
    for key, val in clean_stats.items():
        logger.info(f"  {key}: {val}")

    # Ensure optional columns exist
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    # Log quality
    quality_stats = log_data_quality(df, source)

    # Save
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved processed flows: {output_file}")

    # Save provenance with detailed dataset identification
    provenance_file = output_file.replace(".csv", "_provenance.json")

    # Map source to human-readable dataset name
    dataset_names = {
        "cic_ids2017": "CIC-IDS2017",
        "cic_ids2018": "CIC-IDS2018",
    }
    dataset_name = dataset_names.get(source, source.upper())

    provenance = {
        "dataset": dataset_name,
        "dataset_source_code": source,
        "input_path": input_dir,
        "rows_loaded": clean_stats["rows_before"],
        "rows_retained": clean_stats["rows_after"],
        "rows_removed": clean_stats["rows_removed_total"],
        "quality_stats": quality_stats,
        "timestamp_generated": pd.Timestamp.now().isoformat(),
    }
    with open(provenance_file, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    logger.info(f"✓ Saved provenance: {provenance_file}")
    logger.info(f"✓ Dataset: {dataset_name}")

    logger.info("Dataset preparation complete")
    return df, provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare real network datasets for world model training")
    parser.add_argument("--input-dir", required=True, help="Input CSV file or directory of CSVs")
    parser.add_argument("--output", default="data/raw/flows.csv", help="Output CSV path")
    parser.add_argument("--source", default="cic_ids2018", help="Dataset source identifier")
    args = parser.parse_args()

    df, provenance = main(args.input_dir, args.output, args.source)
    print(f"\nDataset ready: {len(df)} flows from {provenance['quality_stats']['unique_hosts']} hosts")
