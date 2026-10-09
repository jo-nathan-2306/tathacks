import re
import uuid
from typing import List, Optional, Tuple
from defense.config import HallucinationConfig
from defense.types import Assertion, ClaimStatus, ClaimType


class ClaimExtractor:
    OPINION_MARKERS = [
        r"\bi\s+think\b",
        r"\bi\s+believe\b",
        r"\bin\s+my\s+opinion\b",
        r"\bi\s+feel\b",
        r"\bit\s+seems\s+to\s+me\b",
        r"\barguably\b",
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bpossibly\b",
        r"\bfrom\s+my\s+perspective\b",
    ]

    CREATIVE_MARKERS = [
        r"\bonce\s+upon\s+a\s+time\b",
        r"\bimagine\s+a\b",
        r"\blet's\s+create\b",
        r"\bfictional\b",
        r"\bmetaphorically\b",
        r"\bin\s+this\s+story\b",
    ]

    DECISION_CRITICAL_MARKERS = [
        r"\bmust\s+be\s+administered\b",
        r"\bdosage\b",
        r"\bcontraindicated\b",
        r"\blegal\s+requirement\b",
        r"\bcompliance\b",
        r"\bcritical\s+threshold\b",
        r"\bmaximum\s+safe\s+limit\b",
        r"\binvestment\s+advice\b",
        r"\bsecurity\s+vulnerability\b",
        r"\bpassword\b",
        r"\bkey\b",
    ]

    NUMERICAL_REGEX = re.compile(
        r"\b\d+(\.\d+)?(%|\$|€|£|¥|ms|s|kg|g|mg|km|m|cm|mm|GB|MB|KB|TB|users|dollars|percent|years|days|hours)?\b",
        re.IGNORECASE,
    )

    TEMPORAL_REGEX = re.compile(
        r"\b(in\s+\d{4}|on\s+[A-Z][a-z]+\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4}|during\s+\d{4}|between\s+\d{4}\s+and\s+\d{4}|since\s+\d{4}|until\s+\d{4}|in\s+(January|February|March|April|May|June|July|August|September|October|November|December))\b",
        re.IGNORECASE,
    )

    def __init__(self, config: Optional[HallucinationConfig] = None) -> None:
        self.config = config or HallucinationConfig()
        self._compiled_opinions = [re.compile(p, re.IGNORECASE) for p in self.OPINION_MARKERS]
        self._compiled_creatives = [re.compile(p, re.IGNORECASE) for p in self.CREATIVE_MARKERS]
        self._compiled_criticals = [re.compile(p, re.IGNORECASE) for p in self.DECISION_CRITICAL_MARKERS]

    def split_into_sentences(self, text: str) -> List[str]:
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
        sentences: List[str] = []
        for s in raw_sentences:
            cleaned = s.strip()
            if len(cleaned) > 5:
                sentences.append(cleaned)
        return sentences

    def classify_sentence(self, sentence: str) -> Tuple[ClaimType, float]:
        for op in self._compiled_opinions:
            if op.search(sentence):
                return ClaimType.OPINION, 0.2

        for cr in self._compiled_creatives:
            if cr.search(sentence):
                return ClaimType.CREATIVE, 0.1

        for crit in self._compiled_criticals:
            if crit.search(sentence):
                return ClaimType.DECISION_CRITICAL, 0.95

        has_num = bool(self.NUMERICAL_REGEX.search(sentence))
        has_temp = bool(self.TEMPORAL_REGEX.search(sentence))

        if has_num and has_temp:
            return ClaimType.NUMERICAL, 0.85
        elif has_num:
            return ClaimType.NUMERICAL, 0.8
        elif has_temp:
            return ClaimType.TEMPORAL, 0.75

        return ClaimType.GENERAL_FACT, 0.5

    def extract_claims(self, text: str) -> List[Assertion]:
        if not self.config.enabled:
            return []

        sentences = self.split_into_sentences(text)
        assertions: List[Assertion] = []

        for idx, sentence in enumerate(sentences):
            claim_type, impact = self.classify_sentence(sentence)
            if claim_type in (ClaimType.OPINION, ClaimType.CREATIVE):
                continue

            assertion = Assertion(
                id=f"claim_{idx+1}_{uuid.uuid4().hex[:6]}",
                text=sentence,
                claim_type=claim_type,
                impact_score=impact,
                status=ClaimStatus.UNVERIFIABLE,
                confidence=0.0,
            )
            assertions.append(assertion)

        if self.config.prioritize_decision_critical or self.config.prioritize_numerical:
            assertions.sort(key=lambda a: a.impact_score, reverse=True)

        return assertions[: self.config.max_assertions_per_response]
