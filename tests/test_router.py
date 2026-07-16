from dataclasses import dataclass

from token_efficient_nlp.model import Prediction
from token_efficient_nlp.providers import LLMResult
from token_efficient_nlp.router import HybridClassifier


class FakeLocalModel:
    labels = ["queda", "incendio"]

    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def predict_one(self, text: str) -> Prediction:
        return Prediction(
            label="queda",
            confidence=self.confidence,
            alternatives=[{"label": "queda", "confidence": self.confidence}],
        )

    def normalize_for_prompt(self, text: str, *, max_chars: int) -> str:
        return text.lower()[:max_chars]


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, text: str, labels: list[str]) -> LLMResult:
        self.calls += 1
        return LLMResult(
            label="incendio",
            confidence=0.91,
            input_tokens=20,
            output_tokens=5,
            provider=self.name,
            model=self.model,
        )


def test_high_confidence_is_resolved_locally_without_tokens():
    provider = FakeProvider()
    router = HybridClassifier(
        FakeLocalModel(0.95),
        provider=provider,
        confidence_threshold=0.80,
    )

    result = router.classify("risco de queda")

    assert result.source == "local"
    assert result.input_tokens == 0
    assert provider.calls == 0
    assert router.metrics_snapshot()["local_resolution_rate"] == 1.0


def test_low_confidence_escalates_and_second_request_uses_cache():
    provider = FakeProvider()
    router = HybridClassifier(
        FakeLocalModel(0.40),
        provider=provider,
        confidence_threshold=0.80,
    )

    first = router.classify("fumaça no equipamento")
    second = router.classify("fumaça no equipamento")

    assert first.source == "llm"
    assert first.label == "incendio"
    assert second.cached
    assert provider.calls == 1
    assert router.metrics_snapshot()["cache_hits"] == 1
