from src.parser.log_parser import parse_log_file
from src.features.feature_builder import build_features
from src.models.train import train_model
from src.models.predict import predict
from src.models.explain_anomalies import explain
from src.utils.s3_utils import download_file, upload_file
import os


USE_S3 = os.getenv("USE_S3", "false").lower() == "true"

S3_BUCKET = "qa-log-anomaly-detector-jaya-752575507593-ap-southeast-2-an"
S3_INPUT_KEY = "raw/sample_logs.txt"

S3_CLEAN_KEY = "processed/clean_logs.csv"
S3_FEATURES_KEY = "processed/features.csv"
S3_OUTPUT_KEY = "processed/anomalies.csv"
S3_EXPLANATION_KEY = "processed/anomalies_explained.csv"


RAW_FILE = "data/raw/sample_logs.txt"
CLEAN_LOG = "data/processed/clean_logs.csv"
FEATURES_FILE = "data/processed/features.csv"
ANOMALIES_FILE = "data/processed/anomalies.csv"
EXPLAINED_FILE = "data/processed/anomalies_explained.csv"


def run_pipeline():
    print("\nStarting QA Log Anomaly Detection Pipeline\n")

    # Step 0: Download input from S3 (if enabled)
    input_file = RAW_FILE
    if USE_S3:
        input_file = "data/raw/input_from_s3.txt"
        print("Downloading input file from S3...")
        download_file(S3_BUCKET, S3_INPUT_KEY, input_file)

    # Step 1: Parse logs
    print("Step 1: Parsing logs...")
    parse_log_file(input_file, CLEAN_LOG)

    # Step 2: Feature engineering
    print("Step 2: Building features...")
    build_features()

    # Step 3: Train model
    print("Step 3: Training model...")
    train_model()

    # Step 4: Detect anomalies
    print("Step 4: Detecting anomalies...")
    predict()

    # Step 5: Explain anomalies
    print("Step 5: Explaining anomalies...")
    explain()

   
    if USE_S3:
        print("Uploading results to S3...")

        upload_file(CLEAN_LOG, S3_BUCKET, S3_CLEAN_KEY)
        upload_file(FEATURES_FILE, S3_BUCKET, S3_FEATURES_KEY)
        upload_file(ANOMALIES_FILE, S3_BUCKET, S3_OUTPUT_KEY)
        upload_file(EXPLAINED_FILE, S3_BUCKET, S3_EXPLANATION_KEY)

    print("\nPipeline completed successfully\n")


if __name__ == "__main__":
    run_pipeline()