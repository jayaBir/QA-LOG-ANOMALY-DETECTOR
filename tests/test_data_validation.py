import pandas as pd
from pathlib import Path

def test_clean_logs_file_exists():
    path = Path("data/processed/clean_logs.csv")
    assert path.exists(), "Parsed log CSV does not exist."

def test_no_missing_critical_fields():
    df = pd.read_csv("data/processed/clean_logs.csv")
    
    # Critical fields: host, timestamp, method, url, status, bytes
    critical_cols = ["host", "timestamp", "method", "url", "status", "size"]
    missing = df[critical_cols].isnull().sum()
    for col, count in missing.items():
        assert count == 0, f"Column {col} has {count} missing values."

def test_bytes_column_non_negative():
    df = pd.read_csv("data/processed/clean_logs.csv")
    assert (df["size"] >= 0).all(), "Bytes column has negative values."