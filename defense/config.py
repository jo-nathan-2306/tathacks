from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from defense.types import RiskLevel


@dataclass
class ModuleToggles:
    input_validation: bool = True
    adversarial_detection: bool = True
    claim_extraction: bool = True
    evidence_verification: bool = True
    reasoning_validation: bool = True
    tool_validation: bool = True
    memory_guard: bool = True
    privacy_sanitization: bool = True
    output_qualification: bool = True


@dataclass
class AdversarialConfig:
    enabled: bool = True
    strict_mode: bool = False
    lexicon_threshold: float = 0.65
    obfuscation_detection: bool = True
    distinguish_benign_research: bool = True
    trusted_roles: Set[str] = field(default_factory=lambda: {"system", "developer"})
    block_threshold: RiskLevel = RiskLevel.HIGH


@dataclass
class HallucinationConfig:
    enabled: bool = True
    min_claim_confidence: float = 0.5
    min_evidence_coverage: float = 0.4
    staleness_threshold_days: int = 365
    max_assertions_per_response: int = 20
    prioritize_numerical: bool = True
    prioritize_temporal: bool = True
    prioritize_decision_critical: bool = True
    auto_correct_contradictions: bool = True
    qualify_uncertain_claims: bool = True


@dataclass
class RetryConfig:
    max_retries: int = 2
    retry_on_contradiction: bool = True
    retry_on_high_adversarial_risk: bool = False
    backoff_factor: float = 0.5


@dataclass
class PrivacyConfig:
    sanitize_pii: bool = True
    sanitize_secrets: bool = True
    mask_credentials: bool = True
    store_full_prompts: bool = False
    store_full_responses: bool = False
    custom_secret_patterns: List[str] = field(default_factory=list)


@dataclass
class ToolConfig:
    enabled: bool = True
    sensitive_tools: Set[str] = field(
        default_factory=lambda: {
            "execute_code",
            "shell_command",
            "write_file",
            "delete_file",
            "transfer_funds",
            "send_email",
            "update_permissions",
        }
    )
    enforce_schema: bool = True
    sanitize_arguments: bool = True


@dataclass
class PipelineConfig:
    modules: ModuleToggles = field(default_factory=ModuleToggles)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    retries: RetryConfig = field(default_factory=RetryConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    trusted_sources: Set[str] = field(
        default_factory=lambda: {
            "official_docs",
            "verified_database",
            "trusted_api",
            "internal_kb",
        }
    )
    fallback_response: str = (
        "I am unable to verify the accuracy of this claim with high confidence, "
        "and have withheld potentially unreliable assertions."
    )
