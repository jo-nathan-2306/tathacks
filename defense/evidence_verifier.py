import re
from typing import Dict, List, Optional, Set, Tuple
from defense.config import HallucinationConfig
from defense.retrieval_interface import EvidenceRetriever
from defense.types import Assertion, ClaimStatus, ClaimType, Evidence


class EvidenceVerifier:
    NEGATION_WORDS = {"not", "never", "no", "cannot", "neither", "nor", "hardly", "scarcely", "without"}

    def __init__(
        self,
        retriever: EvidenceRetriever,
        config: Optional[HallucinationConfig] = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or HallucinationConfig()

    def _extract_numbers(self, text: str) -> Set[str]:
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        return set(nums)

    def _extract_keywords(self, text: str) -> Set[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "that", "this", "with", "from", "for", "have",
            "has", "had", "are", "were", "was", "is", "am", "be", "been",
            "being", "will", "would", "should", "could", "do", "does", "did",
            "about", "which", "their", "there", "what", "when", "where", "who",
            "why", "how", "whose", "whom", "its", "they", "them", "then",
            "than", "into", "onto", "upon", "your", "you", "our", "out", "not",
            "but", "can", "just", "also", "all", "any", "each", "some", "such",
            "other", "more", "most", "only", "own", "same", "very", "too", "per",
            "a", "an", "of", "to", "on", "in", "at", "by", "as", "or", "if",
        }
        return set(w for w in words if w not in stopwords)

    def _has_negation(self, text: str) -> bool:
        words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
        return bool(words.intersection(self.NEGATION_WORDS))

    def verify_single_claim(self, assertion: Assertion) -> Tuple[ClaimStatus, float, List[str], Optional[str], Optional[str]]:
        evidences = self.retriever.retrieve(assertion.text, top_k=5)
        if not evidences:
            return ClaimStatus.UNVERIFIABLE, 0.0, [], None, "No relevant evidence retrieved."

        claim_keywords = self._extract_keywords(assertion.text)
        claim_numbers = self._extract_numbers(assertion.text)
        claim_negated = self._has_negation(assertion.text)

        matched_evidence_ids: List[str] = []
        best_support_score = 0.0
        contradiction_found = False
        contradiction_source: Optional[str] = None
        suggested_correction: Optional[str] = None
        has_stale_evidence = False

        for ev in evidences:
            if ev.is_stale:
                has_stale_evidence = True
                continue

            ev_keywords = self._extract_keywords(ev.content)
            ev_numbers = self._extract_numbers(ev.content)

            overlap = claim_keywords.intersection(ev_keywords)
            if not overlap:
                continue

            overlap_ratio = len(overlap) / max(1, len(claim_keywords))
            if overlap_ratio < 0.25:
                continue

            matched_evidence_ids.append(ev.source_id)

            if claim_numbers and ev_numbers:
                number_match = bool(claim_numbers.intersection(ev_numbers))
                if not number_match and overlap_ratio >= 0.35:
                    contradiction_found = True
                    contradiction_source = ev.source_id
                    suggested_correction = (
                        f"Evidence from {ev.source_id} states: '{ev.content.strip()}' "
                        f"(contains {', '.join(sorted(list(ev_numbers)))} instead of {', '.join(sorted(list(claim_numbers)))})."
                    )
                    break

            # Check sentence-level polarity for the best matching sentence
            ev_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev.content) if s.strip()]
            best_s_overlap = 0.0
            best_s_negated = self._has_negation(ev.content)
            best_s_text = ev.content
            for s in ev_sentences:
                s_kw = self._extract_keywords(s)
                if s_kw and claim_keywords:
                    s_overlap = len(claim_keywords.intersection(s_kw)) / max(1, len(claim_keywords))
                    if s_overlap > best_s_overlap:
                        best_s_overlap = s_overlap
                        best_s_negated = self._has_negation(s)
                        best_s_text = s

            if overlap_ratio >= 0.5:
                if (
                    claim_negated != best_s_negated
                    and (best_s_overlap >= 0.45 or len(ev_sentences) == 1)
                    and not (claim_numbers and ev_numbers and bool(claim_numbers.intersection(ev_numbers)))
                ):
                    contradiction_found = True
                    contradiction_source = ev.source_id
                    suggested_correction = f"Evidence from {ev.source_id} contradicts polarity: '{best_s_text.strip()}'."
                    break
                else:
                    weighted_score = overlap_ratio * ev.quality_score
                    if weighted_score > best_support_score:
                        best_support_score = weighted_score

        if contradiction_found:
            return ClaimStatus.CONTRADICTED, 0.9, matched_evidence_ids, suggested_correction, f"Contradicted by {contradiction_source}."

        if not matched_evidence_ids:
            if has_stale_evidence:
                return ClaimStatus.UNVERIFIABLE, 0.0, [], None, "Only stale evidence was found."
            return ClaimStatus.UNVERIFIABLE, 0.0, [], None, "Evidence overlap insufficient."

        if best_support_score >= self.config.min_claim_confidence:
            return ClaimStatus.SUPPORTED, round(best_support_score, 3), matched_evidence_ids, None, None
        elif best_support_score > 0.2:
            return ClaimStatus.INSUFFICIENT, round(best_support_score, 3), matched_evidence_ids, None, "Partial evidence match below threshold."
        else:
            return ClaimStatus.UNVERIFIABLE, round(best_support_score, 3), matched_evidence_ids, None, "Insufficient evidence coverage."

    def verify_assertions(self, assertions: List[Assertion]) -> List[Assertion]:
        for assertion in assertions:
            status, conf, ev_ids, correction, conflict_notes = self.verify_single_claim(assertion)
            assertion.status = status
            assertion.confidence = conf
            assertion.evidence_ids = ev_ids
            assertion.correction = correction
            assertion.conflict_notes = conflict_notes
        return assertions
