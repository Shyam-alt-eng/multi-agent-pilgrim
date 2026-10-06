"""Dependency-free BM25 indexing for the processed chunk collection."""

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from langchain_core.documents import Document

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for indexing and querying."""

    return TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class BM25Result:
    document: Document
    score: float


class BM25Index:
    """In-memory Okapi BM25 index over LangChain documents."""

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(doc.page_content) for doc in self.documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            self._document_frequency.update(set(tokens))
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        predicate: Callable[[Document], bool] | None = None,
    ) -> list[BM25Result]:
        """Return highest-scoring documents, optionally after metadata filtering."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = tokenize(query)
        if not query_terms:
            return []

        candidate_count = sum(
            1
            for document in self.documents
            if predicate is None or predicate(document)
        )
        if not candidate_count:
            return []

        scores: list[BM25Result] = []
        for index, document in enumerate(self.documents):
            if predicate is not None and not predicate(document):
                continue
            document_length = len(self._tokens[index])
            score = 0.0
            for term in query_terms:
                frequency = self._term_frequencies[index].get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency.get(term, 0)
                idf = math.log(
                    1
                    + (candidate_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                normalization = (
                    1
                    - self.b
                    + self.b * document_length / self._average_length
                    if self._average_length
                    else 1.0
                )
                score += idf * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * normalization
                )
            scores.append(BM25Result(document=document, score=score))

        scores.sort(
            key=lambda result: (
                result.score,
                str(result.document.metadata.get("chunk_id", "")),
            ),
            reverse=True,
        )
        return scores[:top_k]
