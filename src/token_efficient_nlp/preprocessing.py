"""spaCy-based normalization optimized for classical NLP models."""

from __future__ import annotations

from collections.abc import Iterable

import spacy
from spacy.language import Language


class SpacyPreprocessor:
    def __init__(self, model_name: str = "pt_core_news_sm") -> None:
        self.model_name = model_name
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            try:
                self._nlp = spacy.load(
                    self.model_name,
                    disable=["ner", "parser", "textcat"],
                )
            except OSError:
                self._nlp = spacy.blank("pt")
        return self._nlp

    @staticmethod
    def _normalize_doc(doc) -> str:
        tokens: list[str] = []
        for token in doc:
            if token.is_space or token.is_punct or token.is_stop:
                continue
            value = token.lemma_.strip() if token.lemma_ else token.text.strip()
            if value:
                tokens.append(value.lower())
        return " ".join(tokens)

    def transform(self, texts: Iterable[str], *, batch_size: int = 128) -> list[str]:
        safe_texts = [str(text or "") for text in texts]
        docs = self.nlp.pipe(safe_texts, batch_size=batch_size)
        return [self._normalize_doc(doc) for doc in docs]

    def transform_one(self, text: str) -> str:
        return self.transform([text], batch_size=1)[0]
