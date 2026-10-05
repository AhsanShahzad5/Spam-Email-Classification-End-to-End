# SMS Spam Detection API

A lightweight machine-learning inference service that classifies SMS messages as **Spam** or **Not Spam**. It takes a model developed in a Jupyter notebook and serves it through a typed FastAPI API with startup-time artifact loading, request validation, and automated API tests.

## Highlights

- FastAPI service with generated OpenAPI documentation
- TF-IDF features and a Multinomial Naive Bayes classifier
- Deterministic preprocessing with no runtime corpus downloads
- Model artifacts loaded once during application startup
- Typed and validated request/response schemas
- Reproducible dependencies managed with `uv`
- Separate application, artifact, data, notebook, and test directories

## Architecture

```text
HTTP request
    |
    v
FastAPI and Pydantic validation
    |
    v
Lowercase -> tokenize -> filter -> stem
    |
    v
TF-IDF vectorizer -> Multinomial Naive Bayes
    |
    v
JSON prediction response
```

The trained vectorizer and classifier are loaded once through FastAPI's lifespan handler and reused across requests.

## Project structure

```text
.
|-- app/
|   |-- __init__.py
|   |-- constants.py       # Local English stop-word set
|   |-- main.py            # FastAPI application and endpoints
|   |-- predictor.py       # Preprocessing and model inference
|   `-- schemas.py         # Request and response models
|-- artifacts/
|   |-- model.pkl          # Trained MultinomialNB classifier
|   `-- vectorizer.pkl     # Fitted TF-IDF vectorizer
|-- data/
|   `-- spam.csv           # Training dataset
|-- notebooks/
|   |-- README.md          # Experiment results and limitations
|   `-- sms_spam_classifier_notebook.ipynb
|-- tests/
|   `-- test_api.py
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

## Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/)

## Local setup

Install the locked dependencies from the repository root:

```bash
uv sync
```

Start the API:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For development with automatic reload:

```bash
uv run uvicorn app.main:app --reload
```

The service is then available at:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

Run these commands from the project root so the `app` package and model artifacts resolve correctly.

## API reference

### Health check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

```bash
curl http://127.0.0.1:8000/health
```

### Classify a message

```http
POST /predict
Content-Type: application/json
```

Request:

```json
{
  "message": "Congratulations! You won a free prize. Call now!"
}
```

Response:

```json
{
  "label": "Spam"
}
```

A non-spam prediction is returned as `{"label":"Not Spam"}`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"Congratulations! You won a free prize. Call now!"}'
```

The `message` field must contain non-whitespace text and cannot exceed 5,000 characters. Invalid requests receive FastAPI's standard `422 Unprocessable Entity` response.

## Model and preprocessing

The inference pipeline mirrors the notebook's training pipeline:

1. Convert the message to lowercase.
2. Tokenize it with NLTK's `TreebankWordTokenizer`.
3. Keep alphanumeric tokens only.
4. Remove English stop words.
5. Apply Porter stemming.
6. Transform the cleaned text with a TF-IDF vectorizer limited to 3,000 features.
7. Classify the vector with Multinomial Naive Bayes.

The committed model reported approximately **97.10% accuracy** and **100% precision** on the notebook's single test split, with confusion matrix `[[896, 0], [30, 108]]`. These are experimental results, not guarantees for real-world traffic. See [the notebook notes](notebooks/README.md) for classifier comparisons and evaluation caveats.

## Testing

```bash
uv run pytest
```

The tests cover health, spam and non-spam predictions, and blank-message validation.

## Production readiness

This is a strong **production-style foundation**, but it is not yet production-ready. It already demonstrates separation of concerns, startup-time initialization, input validation, pinned model compatibility, a health endpoint, and API-level tests.

Before deploying it as a public or business-critical service, add:

- A CI pipeline that runs the test suite for every change
- Container packaging and production deployment configuration
- Structured logging, request correlation IDs, metrics, and tracing
- A readiness check that verifies the model is loaded and usable
- Authentication and rate limiting if the service is not private
- Model/data versioning, drift monitoring, and a documented retraining process
- Cross-validation and leakage-free evaluation (split before fitting TF-IDF)
- Dependency and container vulnerability scanning
- Strict artifact provenance and integrity checks, or a safer artifact format
- Load testing, worker sizing, timeouts, and resource limits

## Security note

Python pickle files can execute arbitrary code during deserialization. Only start this service with trusted artifacts. Never accept model or vectorizer files from users or unverified sources.

## Known limitations

- Training data is imbalanced: roughly 87% ham and 13% spam.
- Reported metrics come from one train/test split rather than cross-validation.
- The notebook fitted TF-IDF before splitting the data, introducing vocabulary leakage that may make the metrics optimistic.
- Language, slang, obfuscation patterns, and spam campaigns change over time.
- The endpoint returns a label only; it does not expose a confidence score or model version.
- The model targets English SMS-like text and should not be assumed to generalize to other languages or domains.

## Retraining

The original exploration and training workflow is in [`notebooks/sms_spam_classifier_notebook.ipynb`](notebooks/sms_spam_classifier_notebook.ipynb). Run it from the repository root so `data/spam.csv` resolves correctly.

Notebook-only experimentation packages are intentionally excluded from the lean API dependencies. The notebook's export cell writes `model.pkl` and `vectorizer.pkl` to the repository root; move reviewed, trusted outputs into `artifacts/` before starting the service.

When retraining, preserve the preprocessing contract or version the API and artifacts together. Fit preprocessing only on the training split and evaluate precision, recall, F1, the confusion matrix, and performance on a genuinely held-out set.

## License

No license file is currently included. Add one before distributing the project or accepting external contributions.
