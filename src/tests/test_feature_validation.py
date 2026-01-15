import pandas as pd

DATA_PATH = "data/processed/clean_logs.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def test_status_code_valid_range():
    df = load_data()
    assert df["status"].between(100, 599).all(), "Invalid HTTP status codes detected"


def test_http_method_allowed_values():
    df = load_data()
    allowed_methods = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}
    assert df["method"].isin(allowed_methods).all(), "Unexpected HTTP methods found"


def test_size_non_negative():
    df = load_data()
    assert (df["size"] >= 0).all(), "Negative response sizes detected"


def test_status_has_variance():
    df = load_data()
    assert df["status"].nunique() > 1, "Status column has no variance"


def test_size_has_variance():
    df = load_data()
    assert df["size"].std() > 0, "Size column has zero variance"


def test_no_empty_urls():
    df = load_data()
    assert df["url"].notnull().all(), "Null URLs found"
    assert (df["url"] != "-").all(), "Placeholder '-' URLs found"
    assert (df["url"].str.len() >= 1).all(), "Empty or invalid URLs found"