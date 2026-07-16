"""Thread-safe token economy metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Lock


@dataclass
class TokenEconomy:
    requests: int = 0
    local_decisions: int = 0
    llm_escalations: int = 0
    cache_hits: int = 0
    estimated_llm_only_tokens: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0

    @property
    def tokens_avoided(self) -> int:
        actual = self.actual_input_tokens + self.actual_output_tokens
        return max(self.estimated_llm_only_tokens - actual, 0)

    @property
    def local_resolution_rate(self) -> float:
        return self.local_decisions / self.requests if self.requests else 0.0

    def snapshot(self) -> dict[str, int | float]:
        return {
            **asdict(self),
            "tokens_avoided": self.tokens_avoided,
            "local_resolution_rate": self.local_resolution_rate,
        }


class MetricsStore:
    def __init__(self) -> None:
        self._metrics = TokenEconomy()
        self._lock = Lock()

    def record(
        self,
        *,
        local: bool,
        cached: bool,
        baseline_tokens: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._metrics.requests += 1
            self._metrics.estimated_llm_only_tokens += baseline_tokens
            self._metrics.actual_input_tokens += input_tokens
            self._metrics.actual_output_tokens += output_tokens
            self._metrics.local_decisions += int(local)
            self._metrics.llm_escalations += int(not local and not cached)
            self._metrics.cache_hits += int(cached)

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return self._metrics.snapshot()
