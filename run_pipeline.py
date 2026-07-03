from src.parser.log_parser import parse_log_file
from src.features.feature_builder import build_features
from src.models.train import train_model
from src.models.predict import predict
from src.models.explain_anomalies import explain
from src.monitoring.drift_detection import detect_drift
from src.utils.s3_utils import download_file, upload_file
import os


USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

S3_BUCKET = os.getenv("S3_BUCKET", "qa-log-anomaly-detector-jaya-752575507593-ap-southeast-2-an")
S3_INPUT_KEY = os.getenv("S3_INPUT_KEY", "raw/NASA_Jul95")

S3_CLEAN_KEY = "processed/clean_logs.csv"
S3_FEATURES_KEY = "processed/features.csv"
S3_OUTPUT_KEY = "processed/anomalies.csv"
S3_EXPLANATION_KEY = "processed/anomalies_explained.csv"
S3_DRIFT_REPORT_KEY = "processed/drift_report.json"


RAW_FILE = "data/raw/sample_logs.txt"
CLEAN_LOG = "data/processed/clean_logs.csv"
FEATURES_FILE = "data/processed/features.csv"
ANOMALIES_FILE = "data/processed/anomalies.csv"
EXPLAINED_FILE = "data/processed/anomalies_explained.csv"
DRIFT_REPORT_FILE = "data/processed/drift_report.json"


def run_pipeline():
    print("\nStarting QA Log Anomaly Detection Pipeline\n")

    # Step 0: Download input from S3 (if enabled)
    input_file = RAW_FILE
    if USE_S3:
        input_file = "data/raw/input_from_s3.txt"
        print(f"Downloading input file from s3://{S3_BUCKET}/{S3_INPUT_KEY}...")
        download_file(S3_BUCKET, S3_INPUT_KEY, input_file)

    # Step 1: Parse logs
    print("Step 1: Parsing logs...")
    parse_log_file(input_file, CLEAN_LOG)

    # Step 2: Feature engineering
    print("Step 2: Building features...")
    build_features()

    # Step 3: Check feature drift
    print("Step 3: Checking feature drift...")
    drift_report = detect_drift()
    if drift_report["drift_detected"]:
        print(f"Feature drift detected: {drift_report['drifted_features']}")
    else:
        print("No feature drift detected.")

    # Step 4: Train model
    print("Step 4: Training model...")
    train_model()

    # Step 5: Detect anomalies
    print("Step 5: Detecting anomalies...")
    predict()

    # Step 6: Explain anomalies
    print("Step 6: Explaining anomalies...")
    explain()

   
    if USE_S3:
        print("Uploading results to S3...")

        upload_file(CLEAN_LOG, S3_BUCKET, S3_CLEAN_KEY)
        upload_file(FEATURES_FILE, S3_BUCKET, S3_FEATURES_KEY)
        upload_file(ANOMALIES_FILE, S3_BUCKET, S3_OUTPUT_KEY)
        upload_file(EXPLAINED_FILE, S3_BUCKET, S3_EXPLANATION_KEY)
        upload_file(DRIFT_REPORT_FILE, S3_BUCKET, S3_DRIFT_REPORT_KEY)

    print("\nPipeline completed successfully\n")


if __name__ == "__main__":
    run_pipeline()
