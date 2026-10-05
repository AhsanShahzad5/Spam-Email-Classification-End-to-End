from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.predictor import SpamPredictor
from app.schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.predictor = SpamPredictor()
    try:
        yield
    finally:
        del app.state.predictor


app = FastAPI(
    title="SMS Spam Detection API",
    version="1.0.0",
    description="Classify an SMS message as spam or ham.",
    lifespan=lifespan,
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor: SpamPredictor = request.app.state.predictor
    is_spam = predictor.predict(payload.message)
    return PredictionResponse(
        label="Spam" if is_spam else "Not Spam",
    )


# uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
