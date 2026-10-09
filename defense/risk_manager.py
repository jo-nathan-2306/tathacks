from typing import Any, Dict, List, Optional
from defense.config import PipelineConfig
from defense.types import (
    ActionType,
    AdversarialDetectionResult,
    Assertion,
    ClaimStatus,
    RiskLevel,
    VerificationStatus,
)


class RiskManager:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

    def assess_adversarial_risk(
        self, adv_result: Optional[AdversarialDetectionResult]
    ) -> RiskLevel:
        if not adv_result:
            return RiskLevel.LOW
        return adv_result.risk_level

    def assess_factual_risk(
        self, assertions: List[Assertion], evidence_coverage: float
    ) -> RiskLevel:
        if not assertions:
            return RiskLevel.LOW

        contradicted = any(a.status == ClaimStatus.CONTRADICTED for a in assertions)
        if contradicted:
            return RiskLevel.HIGH

        if evidence_coverage < 0.4:
            return RiskLevel.HIGH
        elif evidence_coverage < 0.75:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def assess_evidence_quality(
        self, assertions: List[Assertion]
    ) -> RiskLevel:
        if not assertions:
            return RiskLevel.LOW

        scores = [a.confidence for a in assertions if a.confidence > 0]
        if not scores:
            return RiskLevel.HIGH

        avg_score = sum(scores) / len(scores)
        if avg_score >= 0.7:
            return RiskLevel.LOW
        elif avg_score >= 0.4:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def assess_potential_impact(self, assertions: List[Assertion]) -> RiskLevel:
        if not assertions:
            return RiskLevel.LOW

        max_impact = max((a.impact_score for a in assertions), default=0.0)
        if max_impact >= 0.8:
            return RiskLevel.HIGH
        elif max_impact >= 0.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def evaluate_decision(
        self,
        adv_risk: RiskLevel,
        factual_risk: RiskLevel,
        potential_impact: RiskLevel,
        evidence_coverage: float,
        status: VerificationStatus,
        retry_count: int,
        max_retries: int,
    ) -> ActionType:
        if adv_risk == RiskLevel.HIGH:
            return ActionType.BLOCK

        if status == VerificationStatus.CONTRADICTED:
            if retry_count < max_retries:
                return ActionType.REGENERATE
            return ActionType.CORRECT

        if status == VerificationStatus.UNVERIFIED:
            if potential_impact == RiskLevel.HIGH:
                return ActionType.WITHHOLD
            return ActionType.QUALIFY

        if status == VerificationStatus.PARTIALLY_VERIFIED:
            return ActionType.QUALIFY

        return ActionType.PASS
