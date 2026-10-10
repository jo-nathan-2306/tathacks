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
    output_safety: bool = True
    rate_limiting: bool = True
    audit_logging: bool = True


@dataclass
class AdversarialConfig:
    enabled: bool = True
    strict_mode: bool = False
    lexicon_threshold: float = 0.65
    obfuscation_detection: bool = True
    distinguish_benign_research: bool = True
    trusted_roles: Set[str] = field(default_factory=lambda: {"system", "developer"})
    block_threshold: RiskLevel = RiskLevel.HIGH

    def __post_init__(self) -> None:
        if not 0.0 <= self.lexicon_threshold <= 1.0:
            raise ValueError(f"lexicon_threshold must be in [0.0, 1.0], got {self.lexicon_threshold}")


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

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_claim_confidence <= 1.0:
            raise ValueError(f"min_claim_confidence must be in [0.0, 1.0], got {self.min_claim_confidence}")
        if not 0.0 <= self.min_evidence_coverage <= 1.0:
            raise ValueError(f"min_evidence_coverage must be in [0.0, 1.0], got {self.min_evidence_coverage}")
        if self.max_assertions_per_response <= 0:
            raise ValueError(f"max_assertions_per_response must be > 0, got {self.max_assertions_per_response}")
        if self.staleness_threshold_days <= 0:
            raise ValueError(f"staleness_threshold_days must be > 0, got {self.staleness_threshold_days}")


@dataclass
class RetryConfig:
    max_retries: int = 2
    retry_on_contradiction: bool = True
    retry_on_high_adversarial_risk: bool = False
    backoff_factor: float = 0.5

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.backoff_factor <= 0:
            raise ValueError(f"backoff_factor must be > 0, got {self.backoff_factor}")


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
    allow_unregistered_tools: bool = False
    max_argument_depth: int = 8
    max_string_length: int = 16_384
    max_collection_items: int = 256

    def __post_init__(self) -> None:
        if self.max_argument_depth <= 0:
            raise ValueError("max_argument_depth must be > 0")
        if self.max_string_length <= 0:
            raise ValueError("max_string_length must be > 0")
        if self.max_collection_items <= 0:
            raise ValueError("max_collection_items must be > 0")


@dataclass
class AuditConfig:
    enabled: bool = True
    log_dir: str = "audit_logs"
    max_file_size_mb: int = 50
    log_prompts: bool = False
    log_responses: bool = False

    def __post_init__(self) -> None:
        if self.max_file_size_mb <= 0:
            raise ValueError(f"max_file_size_mb must be > 0, got {self.max_file_size_mb}")


@dataclass
class OutputSafetyConfig:
    enabled: bool = True
    block_threshold: float = 0.8
    warn_threshold: float = 0.5
    blocked_categories: Set[str] = field(default_factory=lambda: {
        "TOXICITY", "VIOLENCE", "SELF_HARM", "ILLEGAL_ACTIVITY",
        "SEXUAL_CONTENT", "DANGEROUS_INSTRUCTIONS",
    })

    def __post_init__(self) -> None:
        if not 0.0 <= self.block_threshold <= 1.0:
            raise ValueError(f"block_threshold must be in [0.0, 1.0], got {self.block_threshold}")
        if not 0.0 <= self.warn_threshold <= 1.0:
            raise ValueError(f"warn_threshold must be in [0.0, 1.0], got {self.warn_threshold}")
        if self.block_threshold < self.warn_threshold:
            raise ValueError(f"block_threshold ({self.block_threshold}) must be >= warn_threshold ({self.warn_threshold})")


@dataclass
class RateLimitConfig:
    enabled: bool = True
    requests_per_window: int = 60
    window_seconds: int = 60
    burst_limit: int = 10
    burst_window_seconds: int = 5

    def __post_init__(self) -> None:
        if self.requests_per_window <= 0:
            raise ValueError(f"requests_per_window must be > 0, got {self.requests_per_window}")
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {self.window_seconds}")
        if self.burst_limit <= 0:
            raise ValueError(f"burst_limit must be > 0, got {self.burst_limit}")
        if self.burst_window_seconds <= 0:
            raise ValueError(f"burst_window_seconds must be > 0, got {self.burst_window_seconds}")
        if self.burst_limit > self.requests_per_window:
            raise ValueError(f"burst_limit ({self.burst_limit}) must be <= requests_per_window ({self.requests_per_window})")


@dataclass
class PipelineConfig:
    modules: ModuleToggles = field(default_factory=ModuleToggles)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    hallucination: HallucinationConfig = field(default_factory=HallucinationConfig)
    retries: RetryConfig = field(default_factory=RetryConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    output_safety: OutputSafetyConfig = field(default_factory=OutputSafetyConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    trusted_sources: Set[str] = field(
        default_factory=lambda: {
            "official_docs",
            "verified_database",
            "trusted_api",
            "internal_kb",
        }
    )
    fallback_response: str = (
        "I couldn't generate a response because the configured model did not return a usable answer. "
        "Check the provider status and try again."
    )
