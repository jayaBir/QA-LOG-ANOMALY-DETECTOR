from src.parser.log_parser import parse_log_file
from src.features.feature_builder import build_features
from src.models.train import train_model
from src.models.predict import predict
from src.models.explain_anomalies import explain


RAW_FILE = "data/raw/sample_logs.txt"
CLEAN_LOG = "data/processed/clean_logs.csv"


def run_pipeline():

    print("\nStarting QA Log Anomaly Detection Pipeline\n")

    # Step 1: parse logs
    print("Step 1: Parsing logs...")
    parse_log_file(RAW_FILE, CLEAN_LOG)

    # Step 2: feature engineering
    print("Step 2: Building features...")
    build_features()

    # Step 3: train model
    print("Step 3: Training model...")
    train_model()

    # Step 4: detect anomalies
    print("Step 4: Detecting anomalies...")
    predict()

    # Step 5: explain anomalies
    print("Step 5: Explaining anomalies...")
    explain()

    print("\nPipeline completed successfully\n")


if __name__ == "__main__":
    run_pipeline()