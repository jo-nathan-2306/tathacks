"""Output content safety filter — detects unsafe generated content categories."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SafetyCategory(Enum):
    TOXICITY = "TOXICITY"
    VIOLENCE = "VIOLENCE"
    SELF_HARM = "SELF_HARM"
    ILLEGAL_ACTIVITY = "ILLEGAL_ACTIVITY"
    SEXUAL_CONTENT = "SEXUAL_CONTENT"
    DANGEROUS_INSTRUCTIONS = "DANGEROUS_INSTRUCTIONS"


class SafetyAction(Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class SafetyViolation:
    category: SafetyCategory
    matched_pattern: str
    severity: float  # 0.0 – 1.0
    action: SafetyAction


@dataclass
class SafetyFilterResult:
    is_safe: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    filtered_text: str = ""
    action: SafetyAction = SafetyAction.PASS


class OutputSafetyFilter:
    """Classifies generated text against unsafe content categories and optionally blocks/warns."""

    CATEGORY_PATTERNS: Dict[SafetyCategory, List[Tuple[re.Pattern, float]]] = {
        SafetyCategory.TOXICITY: [
            (re.compile(r"\b(kill\s+yourself|die\s+in\s+a\s+fire|hope\s+you\s+die)\b", re.I), 0.9),
            (re.compile(r"\b(racial\s+slur|ethnic\s+cleansing)\b", re.I), 0.85),
            (re.compile(r"\b(you\s+are\s+worthless|subhuman|degenerate)\b", re.I), 0.7),
        ],
        SafetyCategory.VIOLENCE: [
            (re.compile(r"\b((how\s+to|instructions?\s+to|guide\s+to|steps?\s+to)\s+(murder|kill|assassinate))\b", re.I), 0.95),
            (re.compile(r"\b(step[- ]by[- ]step\s+.*?(torture|mutilat|dismember))", re.I), 0.9),
            (re.compile(r"\b(commit\s+a\s+mass\s+shooting|plan\s+an?\s+attack\s+on)\b", re.I), 0.95),
        ],
        SafetyCategory.SELF_HARM: [
            (re.compile(r"\b((how\s+to|instructions?\s+to|guide\s+to|steps?\s+to)\s+commit\s+suicide|methods?\s+of\s+self[- ]harm)\b", re.I), 0.95),
            (re.compile(r"\b(best\s+way\s+to\s+end\s+(your|my)\s+life)\b", re.I), 0.95),
            (re.compile(r"\b(cut\s+yourself|overdose\s+instructions?)\b", re.I), 0.85),
        ],
        SafetyCategory.ILLEGAL_ACTIVITY: [
            (re.compile(r"\b((how\s+to|instructions?\s+to|guide\s+to|steps?\s+to)\s+(synthesize|cook|manufacture)\s+(meth|fentanyl|heroin|cocaine|drugs?))\b", re.I), 0.95),
            (re.compile(r"\b(launder\s+money|evade\s+law\s+enforcement)\b", re.I), 0.7),
            (re.compile(r"\b(forge\s+(documents?|identity|passport))\b", re.I), 0.75),
            (re.compile(r"\b(steal\s+credit\s+card|card\s+skimming\s+tutorial)\b", re.I), 0.85),
        ],
        SafetyCategory.SEXUAL_CONTENT: [
            (re.compile(r"\b(explicit\s+sexual\s+content\s+involving\s+minor)", re.I), 0.99),
            (re.compile(r"\b(child\s+(pornography|exploitation|sexual\s+abuse))\b", re.I), 0.99),
        ],
        SafetyCategory.DANGEROUS_INSTRUCTIONS: [
            (re.compile(r"\b((how\s+to|instructions?\s+to|guide\s+to|steps?\s+to)\s+(make|build|construct|manufacture)\s+(a\s+)?(bomb|explosive|IED|weapon))\b", re.I), 0.95),
            (re.compile(r"\b(synthesiz(e|ing)\s+(nerve\s+agent|chemical\s+weapon|bioweapon))\b", re.I), 0.95),
            (re.compile(r"\b(3[dD][- ]?print(ed|ing)?\s+(gun|firearm|weapon))\b", re.I), 0.85),
            (re.compile(r"\b(step[- ]by[- ]step\s+.*?hack(ing)?\s+(into|a\s+bank|government))\b", re.I), 0.8),
        ],
    }

    def __init__(
        self,
        block_threshold: float = 0.8,
        warn_threshold: float = 0.5,
        enabled: bool = True,
        blocked_categories: Optional[set] = None,
    ) -> None:
        self.block_threshold = block_threshold
        self.warn_threshold = warn_threshold
        self.enabled = enabled
        self.blocked_categories = blocked_categories or {c for c in SafetyCategory}

    def _check_category(
        self, text: str, category: SafetyCategory
    ) -> List[SafetyViolation]:
        """Run pattern checks for a single safety category."""
        violations: List[SafetyViolation] = []
        if category not in self.blocked_categories:
            return violations
        patterns = self.CATEGORY_PATTERNS.get(category, [])
        for pattern, severity in patterns:
            match = pattern.search(text)
            if match:
                if severity >= self.block_threshold:
                    action = SafetyAction.BLOCK
                elif severity >= self.warn_threshold:
                    action = SafetyAction.WARN
                else:
                    action = SafetyAction.PASS
                violations.append(
                    SafetyViolation(
                        category=category,
                        matched_pattern=match.group(0),
                        severity=severity,
                        action=action,
                    )
                )
        return violations

    def check(self, text: str) -> SafetyFilterResult:
        """Run all safety checks on generated text and return aggregate result."""
        if not self.enabled:
            return SafetyFilterResult(is_safe=True, filtered_text=text, action=SafetyAction.PASS)

        all_violations: List[SafetyViolation] = []
        for category in SafetyCategory:
            all_violations.extend(self._check_category(text, category))

        if not all_violations:
            return SafetyFilterResult(is_safe=True, filtered_text=text, action=SafetyAction.PASS)

        # Determine overall action from worst violation
        max_severity = max(v.severity for v in all_violations)
        if max_severity >= self.block_threshold:
            overall_action = SafetyAction.BLOCK
        elif max_severity >= self.warn_threshold:
            overall_action = SafetyAction.WARN
        else:
            overall_action = SafetyAction.PASS

        # Build filtered text — replace blocked content spans
        filtered = text
        if overall_action == SafetyAction.BLOCK:
            for v in all_violations:
                if v.action == SafetyAction.BLOCK:
                    filtered = filtered.replace(
                        v.matched_pattern,
                        f"[Content removed: {v.category.value}]",
                    )

        return SafetyFilterResult(
            is_safe=(overall_action == SafetyAction.PASS),
            violations=all_violations,
            filtered_text=filtered,
            action=overall_action,
        )
