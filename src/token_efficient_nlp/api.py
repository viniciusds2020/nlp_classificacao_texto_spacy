"""FastAPI application for local-first classification with Groq fallback."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from token_efficient_nlp.model import LocalTextClassifier
from token_efficient_nlp.providers import GroqProvider
from token_efficient_nlp.router import HybridClassifier


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=1_000)


@lru_cache
def router() -> HybridClassifier:
    model = LocalTextClassifier.load(os.getenv("MODEL_PATH", "artifacts/classifier.joblib"))
    use_groq = os.getenv("GROQ_ENABLED", "false").lower() in {"1", "true", "yes"}
    provider = (
        GroqProvider(model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
        if use_groq
        else None
    )

    return HybridClassifier(
        model,
        provider=provider,
        confidence_threshold=float(os.getenv("LOCAL_CONFIDENCE_THRESHOLD", "0.80")),
        max_prompt_chars=int(os.getenv("MAX_PROMPT_CHARS", "2000")),
    )


app = FastAPI(
    title="Token-Efficient NLP Router",
    version="1.1.0",
    description="Classificação local com fallback seletivo para Groq/Llama.",
)


@app.get("/health")
def health() -> dict:
    active_router = router()
    return {
        "status": "ok",
        "labels": active_router.local_model.labels,
        "fallback": getattr(active_router.provider, "name", None),
        "fallback_model": getattr(active_router.provider, "model", None),
    }


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict:
    return router().classify(request.text).__dict__


@app.post("/classify/batch")
def classify_batch(request: BatchRequest) -> list[dict]:
    return [item.__dict__ for item in router().classify_batch(request.texts)]


@app.get("/metrics")
def metrics() -> dict:
    return router().metrics_snapshot()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "token_efficient_nlp.api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
