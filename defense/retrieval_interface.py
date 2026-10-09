from abc import ABC, abstractmethod
from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional, Set
from defense.types import Evidence


class EvidenceRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        pass

    @abstractmethod
    def add_evidence(self, evidence: Evidence) -> None:
        pass


class InMemoryRetriever(EvidenceRetriever):
    def __init__(self, trusted_sources: Optional[Set[str]] = None) -> None:
        self.documents: List[Evidence] = []
        self.trusted_sources = trusted_sources or {
            "official_docs",
            "verified_database",
            "trusted_api",
            "internal_kb",
        }

    def add_evidence(self, evidence: Evidence) -> None:
        self.documents.append(evidence)

    def _tokenize(self, text: str) -> Set[str]:
        words = re.findall(r"\b\w+\b", text.lower())
        return set(words)

    def _calculate_similarity(self, query_tokens: Set[str], doc_tokens: Set[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        intersection = query_tokens.intersection(doc_tokens)
        union = query_tokens.union(doc_tokens)
        return len(intersection) / len(union)

    def retrieve(self, query: str, top_k: int = 5) -> List[Evidence]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored_docs: List[tuple[float, Evidence]] = []
        for doc in self.documents:
            doc_tokens = self._tokenize(doc.content)
            sim = self._calculate_similarity(query_tokens, doc_tokens)
            if sim > 0.05:
                source_multiplier = 1.2 if doc.source_id in self.trusted_sources else 0.8
                final_score = sim * doc.quality_score * source_multiplier
                scored_docs.append((final_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]


class StructuredEvidenceStore:
    def __init__(self, retriever: Optional[EvidenceRetriever] = None) -> None:
        self.retriever = retriever or InMemoryRetriever()
        self.evidence_by_id: Dict[str, Evidence] = {}

    def register_evidence(
        self,
        source_id: str,
        content: str,
        timestamp: Optional[str] = None,
        quality_score: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        is_stale: bool = False,
    ) -> Evidence:
        ev = Evidence(
            source_id=source_id,
            content=content,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            quality_score=quality_score,
            metadata=metadata or {},
            is_stale=is_stale,
        )
        self.evidence_by_id[source_id] = ev
        self.retriever.add_evidence(ev)
        return ev

    def query_evidence(self, query: str, top_k: int = 5) -> List[Evidence]:
        return self.retriever.retrieve(query, top_k=top_k)
