from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class VerificationStatus(Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"


class ClaimStatus(Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNVERIFIABLE = "UNVERIFIABLE"


class ClaimType(Enum):
    NUMERICAL = "NUMERICAL"
    TEMPORAL = "TEMPORAL"
    DECISION_CRITICAL = "DECISION_CRITICAL"
    GENERAL_FACT = "GENERAL_FACT"
    OPINION = "OPINION"
    CREATIVE = "CREATIVE"


class TrustBoundaryType(Enum):
    SYSTEM = "SYSTEM"
    DEVELOPER = "DEVELOPER"
    USER = "USER"
    RETRIEVED_DOC = "RETRIEVED_DOC"
    MEMORY = "MEMORY"
    TOOL_OUTPUT = "TOOL_OUTPUT"


class ActionType(Enum):
    PASS = "PASS"
    QUALIFY = "QUALIFY"
    CORRECT = "CORRECT"
    REGENERATE = "REGENERATE"
    WITHHOLD = "WITHHOLD"
    BLOCK = "BLOCK"


@dataclass
class Evidence:
    source_id: str
    content: str
    timestamp: Optional[str] = None
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_stale: bool = False


@dataclass
class Assertion:
    id: str
    text: str
    claim_type: ClaimType
    impact_score: float = 0.5
    status: ClaimStatus = ClaimStatus.UNVERIFIABLE
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    correction: Optional[str] = None
    conflict_notes: Optional[str] = None


@dataclass
class AdversarialDetectionResult:
    is_adversarial: bool
    risk_level: RiskLevel
    score: float
    detected_patterns: List[str] = field(default_factory=list)
    attack_category: Optional[str] = None
    is_benign_context: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRequest:
    tool_name: str
    arguments: Dict[str, Any]
    caller_boundary: TrustBoundaryType = TrustBoundaryType.USER
    is_sensitive: bool = False
    requires_confirmation: bool = False


@dataclass
class ToolValidationResult:
    is_valid: bool
    sanitized_arguments: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    requires_user_confirmation: bool = False


@dataclass
class ValidationResult:
    status: VerificationStatus
    action: ActionType
    final_output: str
    original_output: str
    adversarial_result: Optional[AdversarialDetectionResult] = None
    assertions: List[Assertion] = field(default_factory=list)
    evidence_coverage: float = 0.0
    adversarial_risk: RiskLevel = RiskLevel.LOW
    factual_risk: RiskLevel = RiskLevel.LOW
    evidence_quality: RiskLevel = RiskLevel.LOW
    potential_impact: RiskLevel = RiskLevel.LOW
    uncertainty_score: float = 0.0
    retry_count: int = 0
    latency_ms: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
