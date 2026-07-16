"""Confidence-based router that minimizes LLM usage."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

from token_efficient_nlp.metrics import MetricsStore
from token_efficient_nlp.model import LocalTextClassifier, Prediction
from token_efficient_nlp.providers import ClassificationProvider


@dataclass(frozen=True)
class RoutingDecision:
    label: str
    confidence: float | None
    source: str
    reason: str
    local_prediction: Prediction
    provider: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    estimated_tokens_avoided: int
    cached: bool


class TokenEstimator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model
        self._encoding = None
        self._encoding = None

    def estimate(self, text: str) -> int:
        # Provider usage supplies exact tokens after a Groq call. Before the
        # call, use a conservative language-agnostic character heuristic.
        return max(1, len(text.encode("utf-8")) // 4)


class HybridClassifier:
    """Resolve confident requests locally and escalate uncertain ones."""

    def __init__(
        self,
        local_model: LocalTextClassifier,
        *,
        provider: ClassificationProvider | None = None,
        confidence_threshold: float = 0.80,
        max_prompt_chars: int = 2_000,
        max_cache_entries: int = 10_000,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        self.local_model = local_model
        self.provider = provider
        self.confidence_threshold = confidence_threshold
        self.max_prompt_chars = max_prompt_chars
        self.max_cache_entries = max_cache_entries
        self.metrics = MetricsStore()
        self.estimator = TokenEstimator(getattr(provider, "model", None))
        self._cache: OrderedDict[str, RoutingDecision] = OrderedDict()

    def classify(self, text: str) -> RoutingDecision:
        text = str(text or "")
        local = self.local_model.predict_one(text)
        baseline_tokens = self.estimator.estimate(text) + 80

        if local.confidence >= self.confidence_threshold:
            decision = RoutingDecision(
                label=local.label,
                confidence=local.confidence,
                source="local",
                reason="confidence_threshold",
                local_prediction=local,
                provider=None,
                model=None,
                input_tokens=0,
                output_tokens=0,
                estimated_tokens_avoided=baseline_tokens,
                cached=False,
            )
            self.metrics.record(local=True, cached=False, baseline_tokens=baseline_tokens)
            return decision

        if self.provider is None:
            decision = RoutingDecision(
                label=local.label,
                confidence=local.confidence,
                source="local",
                reason="provider_unavailable",
                local_prediction=local,
                provider=None,
                model=None,
                input_tokens=0,
                output_tokens=0,
                estimated_tokens_avoided=baseline_tokens,
                cached=False,
            )
            self.metrics.record(local=True, cached=False, baseline_tokens=baseline_tokens)
            return decision

        compact_text = self.local_model.normalize_for_prompt(
            text,
            max_chars=self.max_prompt_chars,
        )
        key = self._cache_key(compact_text)
        if key in self._cache:
            cached = self._cache.pop(key)
            decision = RoutingDecision(
                **{
                    **cached.__dict__,
                    "cached": True,
                    "reason": "cache_hit",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "estimated_tokens_avoided": baseline_tokens,
                }
            )
            self._cache[key] = decision
            self.metrics.record(local=False, cached=True, baseline_tokens=baseline_tokens)
            return decision

        result = self.provider.classify(compact_text, self.local_model.labels)
        if result.label not in self.local_model.labels:
            raise ValueError(f"Provider returned invalid label: {result.label}")

        actual_tokens = result.input_tokens + result.output_tokens
        decision = RoutingDecision(
            label=result.label,
            confidence=result.confidence,
            source="llm",
            reason="low_local_confidence",
            local_prediction=local,
            provider=result.provider,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_tokens_avoided=max(baseline_tokens - actual_tokens, 0),
            cached=False,
        )
        self._cache[key] = decision
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        self.metrics.record(
            local=False,
            cached=False,
            baseline_tokens=baseline_tokens,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return decision

    def classify_batch(self, texts: list[str]) -> list[RoutingDecision]:
        return [self.classify(text) for text in texts]

    def metrics_snapshot(self) -> dict[str, int | float]:
        return self.metrics.snapshot()

    def _cache_key(self, compact_text: str) -> str:
        provider = getattr(self.provider, "name", "none")
        model = getattr(self.provider, "model", "none")
        labels = "|".join(self.local_model.labels)
        payload = f"{provider}|{model}|{labels}|{compact_text}".encode()
        return hashlib.sha256(payload).hexdigest()
