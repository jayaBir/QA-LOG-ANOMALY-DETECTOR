**QA Log Anomaly Detection Service**

Real-time + batch anomaly detection for web server logs using Isolation Forest. Built with MLOps best practices: automated training, experiment tracking, model registry, containerized serving, and CI/CD.

Stack: Python, FastAPI, scikit-learn, MLflow, Docker, GitHub Actions, AWS S3


**Architecture**
graph TD
    %% CI/CD Entry
    A[Git Push / PR] --> B[GitHub Actions: ci.yml]
    B --> B1[Setup Python 3.10]
    B1 --> B2[pip install -r requirements.txt]
    B2 --> B3[docker build -f docker/pipeline.Dockerfile]
    B3 --> B4[docker run qa-pipeline]
    B4 --> B5[python run_pipeline.py]
    B5 --> B6[pytest]
    B6 --> B7[docker build -f docker/Dockerfile.api]

    %% Pipeline Container Details
    B4 --> C[qa-pipeline Container]
    subgraph C [Training: docker/pipeline.Dockerfile]
        C1{USE_S3 env?} -->|false| C2[data/raw/sample_logs.txt]
        C1 -->|true| C3[S3: raw/sample_logs.txt<br/>boto3.download_file]
        C2 --> C4[src/parser/log_parser.py<br/>→ clean_logs.csv]
        C3 --> C4
        C4 --> C5[src/features/feature_builder.py<br/>→ features.csv]
        C5 --> C6[src/models/train.py<br/>IsolationForest]
        C6 --> C7[MLflow: sqlite:///mlflow.db<br/>log_model + register]
        C6 --> C8[joblib.dump → src/models/model.pkl]
        C8 --> C9[src/models/predict.py<br/>→ anomalies.csv]
        C9 --> C10[src/models/explain_anomalies.py<br/>→ anomalies_explained.csv]
        C10 --> C11{USE_S3?}
        C11 -->|true| C12[boto3.upload_file<br/>processed/*.csv to S3]
    end

    %% MLflow Details 
    C7 --> D[MLflow Registry]
    subgraph D [mlruns/340127053401822666]
        D1[Experiment: qa-log-anomaly-detector]
        D1 --> D2[Run: params=model_type, features]
        D1 --> D3[Metrics: training_rows]
        D1 --> D4[Artifacts: model.pkl]
        D1 --> D5[Registered Model Versions]
    end

    %% API Container Details
    B7 --> E[qa-api Container]
    subgraph E [Serving: docker/Dockerfile.api]
        E1[uvicorn src.service.app:app] --> E2[joblib.load src/models/model.pkl]
        E1 --> E3[GET /health]
        E1 --> E4[POST /predict_anomaly<br/>LogRequest: rpm, error_rate, avg_size]
        E1 --> E5[POST /detect<br/>reads anomalies_explained.csv]
        E5 --> E6[Return: total_anomalies + sample]
    end

    %% Docker Compose Runtime
    F[docker-compose.yml] --> G[services.pipeline<br/>depends_on]
    G --> H[services.api<br/>port 8000:8000]
    G -.volumes.-> I[./data:/app/data<br/>./mlruns:/app/mlruns<br/>./src/models:/app/src/models]
    H -.volumes.-> I

    %% Data Flow Files
    subgraph I [Shared Volumes]
        I1[data/raw/NASA_Jul95.gz]
        I2[data/processed/clean_logs.csv]
        I3[data/processed/features.csv]
        I4[data/processed/anomalies.csv]
        I5[data/processed/anomalies_explained.csv]
        I6[src/models/model.pkl]
        I7[mlruns/]
    end

    %% Tests
    B6 --> J[tests/]
    subgraph J [pytest]
        J1[test_api.py → TestClient]
        J2[test_data_validation.py]
        J3[test_feature_validation.py]
        J4[test_isolation_forest.py]
        J5[test_explanation_logic.py]
    end

    %% Current Limitations 
    C7 -.uses.-> K[sqlite:///mlflow.db<br/>Not Postgres]
    E2 -.loads local.-> L[src/models/model.pkl<br/>Not from MLflow Registry]

    %% Styling
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style D fill:#ffe1e1,stroke:#cc0000,stroke-width:2px
    style E fill:#e1ffe1,stroke:#00aa00,stroke-width:2px
    style K fill:#ffcccc,stroke:#990000,stroke-dasharray: 5 5
    style L fill:#ffcccc,stroke:#990000,stroke-dasharray: 5 5


**Key Features**:
•	Automated Pipeline: run_pipeline.py orchestrates end-to-end flow
•	Dual Data Mode: sample_logs.txt for CI, S3 for production training via USE_S3=true
•	Experiment Tracking: All runs logged to MLflow with params, metrics, artifacts
•	Model Registry: Models versioned in MLflow as qa-log-anomaly-detector
•	Serving: FastAPI with /health, /predict_anomaly real-time, /detect batch endpoints
•	Explainability: Z-score based feature attribution for each anomaly
•	CI/CD: GitHub Actions runs tests + builds Docker images on every PR


**Project Structure**
├── .github/workflows/ci.yml     # CI: test, train on sample, build images
├── data/
│   ├── raw/sample_logs.txt      # Small dataset for CI
│   └── processed/               # Clean logs, features, anomalies
├── docker/
│   ├── Dockerfile.api           # FastAPI serving container
│   └── pipeline.Dockerfile      # Training pipeline container  
├── src/
│   ├── parser/log_parser.py     # NASA log format → CSV
│   ├── features/feature_builder.py  # rpm, error_rate, avg_response_size
│   ├── models/
│   │   ├── train.py             # IsolationForest + MLflow logging
│   │   ├── predict.py           # Batch inference
│   │   └── explain_anomalies.py # Z-score explainability
│   ├── service/app.py           # FastAPI app
│   └── utils/s3_utils.py        # S3 upload/download
├── tests/                       # pytest: API, data, model, feature tests
├── mlruns/                      # MLflow local tracking
├── run_pipeline.py              # Main orchestrator
└── docker-compose.yml           # Local: pipeline + API

Quick Start
1. Local Development
 **Install deps**
pip install -r requirements.txt

 **Run full pipeline on sample data**
python run_pipeline.py

 **Start API**

uvicorn src.service.app:app --reload

 Visit http://localhost:8000/docs

2. Docker Compose

docker-compose up --build
# Pipeline runs first, then API starts on :8000

3. Production Training with S3

export USE_S3=true
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=xxx
python run_pipeline.py
API Endpoints
Endpoint	Method	Description
/health	GET	Liveness check. Returns {"status": "ok"}
/predict_anomaly	POST	Real-time scoring. Body: {"requests_per_minute": 120, "error_rate": 0.05, "avg_response_size": 5000}
/detect?limit=10	POST	Returns latest anomalies from batch run with explanations

Example:
curl -X POST "http://localhost:8000/predict_anomaly" -H "Content-Type: application/json" -d '{"requests_per_minute": 500, "error_rate": 0.8, "avg_response_size": 100}'
MLflow Tracking
All training runs are logged. To view UI:
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
Registered Model: qa-log-anomaly-detector

Stages: None → Staging → Production. Promote via MLflow UI or API.
CI/CD Flow

.github/workflows/ci.yml triggers on main push + PR:
•	pytest - Runs all tests in tests/
•	python run_pipeline.py - Trains on data/raw/sample_logs.txt
•	docker build - Builds qa-pipeline + qa-api images

Note: CI uses sample data to keep runs < 5 min. Full S3 training is manual or scheduled.

**Data Flow**
1. Raw: NASA_Jul95 web logs or sample_logs.txt
2. Parsed: clean_logs.csv - host, timestamp, method, url, status, size
3. Features: features.csv - aggregated per host: requests_per_minute, error_rate, avg_response_size
4. Anomalies: anomalies.csv - added is_anomaly, anomaly_score
5. Explained: anomalies_explained.csv - added explanation with z-score reasons
   
**Model Details**
Algorithm: sklearn.ensemble.IsolationForest
Contamination: 0.05 = expects 5% anomalies
Features: 3 engineered metrics per host per minute
Explainability: Flags any feature with |z-score| >= 3.0 as abnormal_feature

**Testing**
pytest -v
Coverage:
•	test_api.py - Endpoint status + schema
•	test_data_validation.py - Schema, nulls, value ranges
•	test_feature_validation.py - HTTP method/status validation
•	test_isolation_forest.py - Anomaly rate + score sanity checks
•	test_explanation_logic.py - Z-score explanations are statistically valid

**Deployment**
Current: EC2 running Docker containers built by CI.
**Recommended next steps:**
•	Point MLflow to remote Postgres instead of sqlite:///mlflow.db
•	Load model in FastAPI via mlflow.pyfunc.load_model("models:/qa-log-anomaly-detector/Production")
•	Add Prometheus metrics + Grafana dashboard
•	Split CI: fast pr.yml for tests only, scheduled train.yml for S3 training
**Environment Variables**
Variable	Default	Description
USE_S3	false	If true, downloads raw logs from S3 and uploads results
AWS_ACCESS_KEY_ID	-	Required if USE_S3=true
AWS_SECRET_ACCESS_KEY	-	Required if USE_S3=true
MLFLOW_TRACKING_URI	sqlite:///mlflow.db	Set to remote server for prod
**Limitations & Future Work**
•	Drift Detection: No automated monitoring of input distribution shifts yet
•	Rollback: Manual MLflow stage transition. No automated canary
•	Scale: Single EC2. For >1k RPS, move to ECS + auto-scaling
•	Feature Store: Features recomputed each run. Consider Feast for reuse
