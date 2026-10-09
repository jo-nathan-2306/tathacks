from typing import Any, Dict, List, Optional, Tuple
from defense.claim_extractor import ClaimExtractor
from defense.config import HallucinationConfig
from defense.evidence_verifier import EvidenceVerifier
from defense.reasoning_validator import ReasoningValidator
from defense.retrieval_interface import EvidenceRetriever
from defense.types import (
    ActionType,
    Assertion,
    ClaimStatus,
    ClaimType,
    RiskLevel,
    VerificationStatus,
)


class HallucinationMitigationLayer:
    def __init__(
        self,
        retriever: EvidenceRetriever,
        config: Optional[HallucinationConfig] = None,
    ) -> None:
        self.config = config or HallucinationConfig()
        self.retriever = retriever
        self.extractor = ClaimExtractor(config=self.config)
        self.verifier = EvidenceVerifier(retriever=self.retriever, config=self.config)
        self.reasoning_validator = ReasoningValidator()

    def process(self, generated_text: str) -> Dict[str, Any]:
        if not self.config.enabled:
            return {
                "status": VerificationStatus.VERIFIED,
                "action": ActionType.PASS,
                "final_text": generated_text,
                "assertions": [],
                "evidence_coverage": 1.0,
                "uncertainty_score": 0.0,
                "reasoning_report": {},
            }

        assertions = self.extractor.extract_claims(generated_text)
        if not assertions:
            return {
                "status": VerificationStatus.VERIFIED,
                "action": ActionType.PASS,
                "final_text": generated_text,
                "assertions": [],
                "evidence_coverage": 1.0,
                "uncertainty_score": 0.0,
                "reasoning_report": {},
            }

        verified_assertions = self.verifier.verify_assertions(assertions)

        supported_count = sum(1 for a in verified_assertions if a.status == ClaimStatus.SUPPORTED)
        contradicted_count = sum(1 for a in verified_assertions if a.status == ClaimStatus.CONTRADICTED)
        insufficient_count = sum(1 for a in verified_assertions if a.status == ClaimStatus.INSUFFICIENT)
        unverifiable_count = sum(1 for a in verified_assertions if a.status == ClaimStatus.UNVERIFIABLE)
        total = len(verified_assertions)

        evidence_coverage = round(supported_count / total, 3) if total > 0 else 1.0
        reasoning_report = self.reasoning_validator.validate(
            generated_text, verified_assertions, evidence_coverage
        )

        final_text = generated_text
        action = ActionType.PASS
        status = VerificationStatus.VERIFIED

        if contradicted_count > 0:
            status = VerificationStatus.CONTRADICTED
            if self.config.auto_correct_contradictions:
                modified_sentences = []
                for a in verified_assertions:
                    if a.status == ClaimStatus.CONTRADICTED and a.correction:
                        modified_sentences.append(f"[Correction: {a.correction}]")
                    else:
                        modified_sentences.append(a.text)
                final_text = " ".join(modified_sentences)
                action = ActionType.CORRECT
            else:
                action = ActionType.WITHHOLD
        elif evidence_coverage >= self.config.min_evidence_coverage and (insufficient_count > 0 or unverifiable_count > 0):
            status = VerificationStatus.PARTIALLY_VERIFIED
            if self.config.qualify_uncertain_claims:
                qualified_sentences = []
                for a in verified_assertions:
                    if a.status in (ClaimStatus.UNVERIFIABLE, ClaimStatus.INSUFFICIENT):
                        if a.impact_score >= 0.8:
                            qualified_sentences.append(f"[Unverified Claim: {a.text}]")
                        else:
                            qualified_sentences.append(a.text)
                    else:
                        qualified_sentences.append(a.text)
                final_text = " ".join(qualified_sentences)
                action = ActionType.QUALIFY
            else:
                action = ActionType.PASS
        elif evidence_coverage < self.config.min_evidence_coverage:
            status = VerificationStatus.UNVERIFIED
            if reasoning_report.get("has_reasoning_errors", False):
                action = ActionType.WITHHOLD
            else:
                action = ActionType.QUALIFY

        return {
            "status": status,
            "action": action,
            "final_text": final_text,
            "assertions": verified_assertions,
            "evidence_coverage": evidence_coverage,
            "uncertainty_score": reasoning_report.get("uncertainty_score", 0.0),
            "reasoning_report": reasoning_report,
        }
