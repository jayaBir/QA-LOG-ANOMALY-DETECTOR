import pandas as pd

from src.monitoring.drift_detection import detect_drift


def write_features(path, rpm_values, error_values=None, size_values=None):
    row_count = len(rpm_values)
    error_values = error_values or [0.01] * row_count
    size_values = size_values or [1000] * row_count

    df = pd.DataFrame({
        "host": [f"host-{idx}" for idx in range(row_count)],
        "requests_per_minute": rpm_values,
        "error_rate": error_values,
        "avg_response_size": size_values
    })
    df.to_csv(path, index=False)


def test_drift_detection_creates_baseline(tmp_path):
    current_path = tmp_path / "features.csv"
    baseline_path = tmp_path / "features_baseline.csv"
    report_path = tmp_path / "drift_report.json"

    write_features(current_path, [10, 11, 12, 13, 14])

    report = detect_drift(current_path, baseline_path, report_path)

    assert report["status"] == "baseline_created"
    assert report["drift_detected"] is False
    assert baseline_path.exists()
    assert report_path.exists()


def test_drift_detection_flags_shifted_features(tmp_path):
    current_path = tmp_path / "features.csv"
    baseline_path = tmp_path / "features_baseline.csv"
    report_path = tmp_path / "drift_report.json"

    write_features(
        baseline_path,
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        error_values=[0.01] * 10,
        size_values=[1000] * 10
    )
    write_features(
        current_path,
        [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
        error_values=[0.75] * 10,
        size_values=[100] * 10
    )

    report = detect_drift(current_path, baseline_path, report_path)

    assert report["status"] == "checked"
    assert report["drift_detected"] is True
    assert "requests_per_minute" in report["drifted_features"]
    assert report["features"]["requests_per_minute"]["drift_detected"] is True
