"""Trainable local text classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from token_efficient_nlp.preprocessing import SpacyPreprocessor


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    alternatives: list[dict[str, float | str]]


class LocalTextClassifier:
    """TF-IDF + Logistic Regression classifier with spaCy normalization."""

    def __init__(
        self,
        *,
        spacy_model: str = "pt_core_news_sm",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int | None = 50_000,
        random_state: int = 42,
    ) -> None:
        self.spacy_model = spacy_model
        self.preprocessor = SpacyPreprocessor(spacy_model)
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.classifier = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=random_state,
        )
        self.metadata: dict[str, Any] = {}
        self._fitted = False

    @property
    def labels(self) -> list[str]:
        if not self._fitted:
            return []
        return [str(label) for label in self.classifier.classes_]

    def fit(self, texts: list[str], labels: list[str]) -> "LocalTextClassifier":
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")
        if len(set(labels)) < 2:
            raise ValueError("At least two classes are required.")

        normalized = self.preprocessor.transform(texts)
        matrix = self.vectorizer.fit_transform(normalized)
        self.classifier.fit(matrix, labels)
        self._fitted = True
        self.metadata = {
            "trained_at": datetime.now(UTC).isoformat(),
            "samples": len(texts),
            "labels": self.labels,
            "spacy_model": self.spacy_model,
            "vocabulary_size": len(self.vectorizer.vocabulary_),
        }
        return self

    def predict(self, texts: list[str], *, top_k: int = 3) -> list[Prediction]:
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted.")
        normalized = self.preprocessor.transform(texts)
        matrix = self.vectorizer.transform(normalized)
        probabilities = self.classifier.predict_proba(matrix)
        classes = np.asarray(self.classifier.classes_)

        predictions: list[Prediction] = []
        for row in probabilities:
            order = np.argsort(row)[::-1][:top_k]
            alternatives = [
                {"label": str(classes[index]), "confidence": float(row[index])}
                for index in order
            ]
            predictions.append(
                Prediction(
                    label=str(classes[order[0]]),
                    confidence=float(row[order[0]]),
                    alternatives=alternatives,
                )
            )
        return predictions

    def predict_one(self, text: str, *, top_k: int = 3) -> Prediction:
        return self.predict([text], top_k=top_k)[0]

    def normalize_for_prompt(self, text: str, *, max_chars: int = 2_000) -> str:
        normalized = self.preprocessor.transform_one(text) or str(text)
        return normalized[:max_chars]

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "spacy_model": self.spacy_model,
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "metadata": self.metadata,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LocalTextClassifier":
        artifact = joblib.load(path)
        instance = cls(spacy_model=artifact["spacy_model"])
        instance.vectorizer = artifact["vectorizer"]
        instance.classifier = artifact["classifier"]
        instance.metadata = artifact.get("metadata", {})
        instance._fitted = True
        return instance

    def model_card(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "labels": self.labels,
            "classifier": self.classifier.__class__.__name__,
            "vectorizer": self.vectorizer.__class__.__name__,
        }
