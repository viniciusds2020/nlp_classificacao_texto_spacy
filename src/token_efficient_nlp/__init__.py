"""Local-first, token-efficient text classification."""

from token_efficient_nlp.model import LocalTextClassifier, Prediction
from token_efficient_nlp.router import HybridClassifier, RoutingDecision

__version__ = "1.0.0"

__all__ = [
    "LocalTextClassifier",
    "Prediction",
    "HybridClassifier",
    "RoutingDecision",
]
