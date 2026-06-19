
# End-to-End Movie Review Sentiment Analysis with MLOps

A production-ready, end-to-end MLOps pipeline for movie review sentiment classification. The project fine-tunes a **DistilBERT** model on movie reviews, tracks experiments with **MLflow**, serves predictions via a **FastAPI** REST interface, and provides a polished **Streamlit** frontend. The entire stack is containerized with **Docker** and automated using **GitHub Actions CI/CD**.

---

## System Architecture

```mermaid
graph TD
    A[IMDB Dataset / HF Datasets] -->|Preprocess & Tokenize| B[DistilBERT Fine-Tuning]
    B -->|Log Params, Loss, Metrics| C[(MLflow Server & SQLite)]
    B -->|Save Best Weights| D[notebooks/distilbert_imdb_tf_model/]
    C -->|Register Model| E[MLflow Model Registry]
    
    D -->|Load Weights| F[FastAPI Service on Hugging Face]
    F -->|Inference Route| G[Streamlit UI Webapp on Streamlit Cloud]
    
    subgraph CI/CD & Deployment
    H[GitHub Repo] -->|Push/PR Trigger| I[GitHub Actions]
    I -->|Run Pytest & Build Check| J[Docker Build]
    J -->|Container Image| K[Hugging Face Spaces]
    G -->|Query over HTTPS| F
    end
```

---

## Folder Structure

```
movie-sentiment-mlops/
│
├── .github/
│   └── workflows/
│       └── ci_cd.yml          # GitHub Actions CI/CD automation workflow
│
├── api/
│   ├── main.py                # FastAPI main entrypoint and middleware logging
│   ├── inference.py           # Thread-safe model predictor singleton wrapper
│   └── schemas.py             # Pydantic request/response validation schemas
│
├── app/
│   └── app.py                 # Streamlit UI web interface
│
├── data/                      # Local data cache folder (created dynamically)
│
├── notebooks/
│   ├── DistilBert_Transformer.ipynb # Jupyter notebook showcasing pipelines
│   └── distilbert_imdb_tf_model/    # Local save path for best fine-tuned model checkpoint
│
├── src/
│   ├── config.py              # Central hyperparameters and file path configurations
│   ├── preprocess.py          # Data ingestion, splitting, and tokenization loaders
│   ├── train.py               # PyTorch training, validation, and MLflow logging
│   ├── evaluate.py            # Evaluation metrics report and confusion matrix plotting
│   ├── predict.py             # Predictor wrapper class with CPU/GPU support
│   └── utils.py               # Structured logging configurations & seed initializers
│
├── tests/
│   ├── test_api.py            # FastAPI integration client tests
│   ├── test_inference.py      # Predictor module unit tests
│   └── test_preprocess.py     # Preprocessing and dataset splitting unit tests
│
├── Dockerfile                 # Multi-stage optimized Docker setup
├── docker-compose.yml         # Container coordinator (FastAPI, Streamlit, MLflow)
├── requirements.txt           # Python packages pin list
├── setup.py                   # Python package setup for editable installations
└── README.md                  # System documentation
```

---

## Local Quickstart Setup

### Prerequisites
* Python 3.11
* Docker & Docker Compose (Optional)

### 1. Set Up Virtual Environment & Packages
First, clone this repository and create a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies and editable package
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Run Training Pipeline & MLflow Tracking
We use a local SQLite backend to track runs and support the MLflow Model Registry.

1. **Start the MLflow Server** (in a separate terminal):
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
   ```
2. **Execute Training**:
   By default, training will run for `3` epochs on a subset of `1000` samples to ensure fast, CPU-friendly verification. To run on the full IMDB dataset, set `MAX_SAMPLES=-1` in your environment variables.
   ```bash
   # Run training
   python src/train.py
   ```
   *This logs training metrics, final test metrics, registers the best model under the name `DistilBERTSentimentModel`, and saves checkpoints under `notebooks/distilbert_imdb_tf_model/`.*

3. **Verify Evaluation**:
   Run evaluation independently to compute the classification report and save the confusion matrix plot:
   ```bash
   python src/evaluate.py
   ```
   *The confusion matrix plot is saved to `logs/confusion_matrix.png` and logged to MLflow.*

---

## API Service Documentation (FastAPI)

Start the FastAPI application locally:
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Access the interactive API documentation at: `http://127.0.0.1:8000/docs`

### 1. API Healthcheck
* **Endpoint**: `GET /`
* **Response**:
  ```json
  {
    "status": "running",
    "model": "DistilBERT Sentiment Classifier"
  }
  ```

### 2. Model Prediction
* **Endpoint**: `POST /predict`
* **Request Payload**:
  ```json
  {
    "text": "This movie was absolutely fantastic. The acting was superb!"
  }
  ```
* **Response Payload**:
  ```json
  {
    "sentiment": "Positive",
    "confidence": 0.9934
  }
  ```
* **cURL command**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/predict" \
       -H "Content-Type: application/json" \
       -d "{\"text\": \"This movie was a terrible waste of time.\"}"
  ```

---

## Streamlit Web Interface

To launch the interactive dashboard, run:
```bash
streamlit run app/app.py
```
This launches a browser tab at `http://localhost:8501`. Features include:
* **Interactive Text Input**: Write or copy movie reviews.
* **Graphical Confidence Meters**: Visual progress bars mapping sentiment probability.
* **Recent Predictions**: Lists previous evaluations in a session-state tracker.
* **Dynamic API Health Checks**: Live status indicators monitoring the FastAPI backend.

---

## Docker Deployment

You can build and deploy the entire multi-service container system in one click.

### Build and Run with Docker Compose
```bash
# Build and run API, Streamlit and MLflow tracking servers
docker-compose up --build
```
Once started, the services map to the following local ports:
* **FastAPI Service**: `http://localhost:8000`
* **Streamlit UI Frontend**: `http://localhost:8501`
* **MLflow experiment server**: `http://localhost:5000`

### Run API standalone via Dockerfile
```bash
# Build standalone image
docker build -t movie-sentiment-mlops .

# Run container exposing port 8000
docker run -p 8000:8000 movie-sentiment-mlops
```

---

## Testing Suite

We use `pytest` for unit and integration testing. Run:
```bash
pytest tests/ -v
```
The test suite validates:
* **Preprocessing**: Splits integrity, tokenization length mappings, and PyTorch dataloader dimensions.
* **Inference**: Predictor boundary scores, empty text handling, and list batch prediction methods.
* **API Endpoints**: Correct GET `/` responses, POST `/predict` status checks, and Pydantic validation error code mappings (422/400).

---

## Production Cloud Architecture

For cloud deployments, follow this MLOps topology:
1. **GitHub Repository**: Holds the source code and configuration.
2. **GitHub Actions**: Runs code linting, executes pytest (pulling weights via Git LFS), and verifies Docker builds.
3. **MLflow database store**: Deployed on a managed cloud database (e.g., PostgreSQL) with artifacts stored on an S3/GCS bucket (Optional).
4. **FastAPI Web Service**: Deployed on **Hugging Face Spaces** using the `Dockerfile`, pulling the tracked LFS model weights.
5. **Streamlit UI**: Deployed on **Streamlit Community Cloud**, querying the public Hugging Face Spaces FastAPI endpoint over HTTPS.
