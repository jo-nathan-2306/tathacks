import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
from defense.adversarial_detector import AdversarialDetector
from defense.claim_extractor import ClaimExtractor
from defense.config import PipelineConfig
from defense.evidence_verifier import EvidenceVerifier
from defense.hallucination_layer import HallucinationMitigationLayer
from defense.memory_guard import MemoryGuard
from defense.privacy_sanitizer import PrivacySanitizer
from defense.reasoning_validator import ReasoningValidator
from defense.retrieval_interface import EvidenceRetriever, InMemoryRetriever
from defense.risk_manager import RiskManager
from defense.tool_validator import ToolValidator
from defense.trust_boundary import TrustBoundaryManager
from defense.types import (
    ActionType,
    AdversarialDetectionResult,
    Assertion,
    ClaimStatus,
    RiskLevel,
    ToolCallRequest,
    ToolValidationResult,
    TrustBoundaryType,
    ValidationResult,
    VerificationStatus,
)
from models.base import CallableModelAdapter, ModelAdapter


class DefensivePipeline:
    def __init__(
        self,
        model: Union[ModelAdapter, Callable[..., str]],
        retriever: Optional[EvidenceRetriever] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        if isinstance(model, ModelAdapter):
            self.model = model
        else:
            self.model = CallableModelAdapter(model)

        self.retriever = retriever or InMemoryRetriever(
            trusted_sources=self.config.trusted_sources
        )
        self.trust_boundary = TrustBoundaryManager()
        self.adversarial_detector = AdversarialDetector(config=self.config.adversarial)
        self.hallucination_layer = HallucinationMitigationLayer(
            retriever=self.retriever, config=self.config.hallucination
        )
        self.tool_validator = ToolValidator(config=self.config.tools)
        self.memory_guard = MemoryGuard()
        self.privacy_sanitizer = PrivacySanitizer(config=self.config.privacy)
        self.risk_manager = RiskManager(config=self.config)

    def validate_tool_call(self, request: ToolCallRequest) -> ToolValidationResult:
        if not self.config.modules.tool_validation:
            return ToolValidationResult(
                is_valid=True,
                sanitized_arguments=request.arguments,
                reasons=[],
                requires_user_confirmation=False,
            )
        return self.tool_validator.validate_tool_call(request)

    def sanitize_memories(self, memories: List[str]) -> List[str]:
        if not self.config.modules.memory_guard:
            return memories
        return self.memory_guard.sanitize_memories(memories)

    def run(
        self,
        user_input: str,
        system_instruction: str = "",
        retrieved_context: Optional[List[str]] = None,
        memory_context: Optional[List[str]] = None,
        tool_results: Optional[List[str]] = None,
    ) -> ValidationResult:
        start_time = time.perf_counter()
        errors: List[str] = []
        metrics: Dict[str, Any] = {}

        adv_result: Optional[AdversarialDetectionResult] = None
        if self.config.modules.adversarial_detection:
            adv_result = self.adversarial_detector.detect(user_input)
            if adv_result.is_adversarial and adv_result.risk_level == RiskLevel.HIGH:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                return ValidationResult(
                    status=VerificationStatus.UNVERIFIED,
                    action=ActionType.BLOCK,
                    final_output="Request blocked due to security policy violation.",
                    original_output="",
                    adversarial_result=adv_result,
                    assertions=[],
                    evidence_coverage=0.0,
                    adversarial_risk=RiskLevel.HIGH,
                    factual_risk=RiskLevel.LOW,
                    evidence_quality=RiskLevel.LOW,
                    potential_impact=RiskLevel.HIGH,
                    uncertainty_score=1.0,
                    retry_count=0,
                    latency_ms=latency_ms,
                    metrics={"blocked_by": "adversarial_detector"},
                    errors=["Adversarial pattern detected."],
                )

        if self.config.modules.memory_guard and memory_context:
            memory_context = self.memory_guard.sanitize_memories(memory_context)

        prompt = user_input
        if self.config.modules.input_validation:
            prompt = self.trust_boundary.format_prompt_with_boundaries(
                system_instruction=system_instruction,
                user_input=user_input,
                retrieved_context=retrieved_context,
                memory_context=memory_context,
                tool_results=tool_results,
            )

        raw_output = ""
        final_output = ""
        current_assertions: List[Assertion] = []
        coverage = 1.0
        uncertainty = 0.0
        status = VerificationStatus.VERIFIED
        action = ActionType.PASS
        retry_count = 0
        max_retries = (
            self.config.retries.max_retries
            if self.config.retries.retry_on_contradiction
            else 0
        )

        for attempt in range(max_retries + 1):
            retry_count = attempt
            try:
                raw_output = self.model.generate(prompt)
            except Exception as e:
                errors.append(f"Model generation error: {str(e)}")
                raw_output = self.config.fallback_response
                status = VerificationStatus.UNVERIFIED
                action = ActionType.WITHHOLD
                break

            if self.config.modules.evidence_verification or self.config.modules.claim_extraction:
                hal_report = self.hallucination_layer.process(raw_output)
                status = hal_report["status"]
                action = hal_report["action"]
                final_output = hal_report["final_text"]
                current_assertions = hal_report["assertions"]
                coverage = hal_report["evidence_coverage"]
                uncertainty = hal_report["uncertainty_score"]

                if status == VerificationStatus.CONTRADICTED and attempt < max_retries:
                    prompt = (
                        f"{prompt}\n\n[SYSTEM NOTE: Previous response contained factual contradictions. "
                        f"Please generate a strictly factual response adhering to verified sources.]"
                    )
                    continue
                else:
                    break
            else:
                final_output = raw_output
                break

        adv_risk = self.risk_manager.assess_adversarial_risk(adv_result)
        factual_risk = self.risk_manager.assess_factual_risk(current_assertions, coverage)
        ev_quality = self.risk_manager.assess_evidence_quality(current_assertions)
        impact = self.risk_manager.assess_potential_impact(current_assertions)

        if action == ActionType.WITHHOLD and not final_output:
            final_output = self.config.fallback_response

        if self.config.modules.privacy_sanitization:
            final_output = self.privacy_sanitizer.mask_text(final_output)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        audit_payload = {
            "status": status.value,
            "action": action.value,
            "assertion_count": len(current_assertions),
            "evidence_coverage": coverage,
            "uncertainty_score": uncertainty,
            "adversarial_score": adv_result.score if adv_result else 0.0,
            "retry_count": retry_count,
            "latency_ms": latency_ms,
        }
        sanitized_metrics = self.privacy_sanitizer.sanitize_log_payload(audit_payload)

        return ValidationResult(
            status=status,
            action=action,
            final_output=final_output,
            original_output=raw_output,
            adversarial_result=adv_result,
            assertions=current_assertions,
            evidence_coverage=coverage,
            adversarial_risk=adv_risk,
            factual_risk=factual_risk,
            evidence_quality=ev_quality,
            potential_impact=impact,
            uncertainty_score=uncertainty,
            retry_count=retry_count,
            latency_ms=latency_ms,
            metrics=sanitized_metrics,
            errors=errors,
        )
