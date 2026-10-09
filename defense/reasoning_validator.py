import math
import re
from typing import Any, Dict, List, Optional, Tuple
from defense.types import Assertion, ClaimStatus, ClaimType, RiskLevel


class ReasoningValidator:
    ARITHMETIC_PATTERNS = [
        re.compile(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"),
        re.compile(r"(\d+(?:\.\d+)?)\s+plus\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s+minus\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s+times\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    ]

    TEMPORAL_SEQUENCE_PATTERN = re.compile(
        r"(before|after)\s+(?:the\s+)?(?:year\s+)?(\d{4})[^\.\,]*?(?:in|happened\s+in)\s+(\d{4})",
        re.IGNORECASE,
    )

    CAUSAL_MARKERS = [
        r"\btherefore\b",
        r"\bas\s+a\s+direct\s+result\b",
        r"\bconsequently\b",
        r"\bproves\s+that\b",
        r"\bcausing\b",
    ]

    def __init__(self) -> None:
        self._compiled_causals = [re.compile(p, re.IGNORECASE) for p in self.CAUSAL_MARKERS]

    def validate_arithmetic(self, text: str) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []

        for pattern in self.ARITHMETIC_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 4:
                    num1 = float(groups[0])
                    op = groups[1]
                    num2 = float(groups[2])
                    claimed_res = float(groups[3])

                    expected_res = 0.0
                    if op == "+":
                        expected_res = num1 + num2
                    elif op == "-":
                        expected_res = num1 - num2
                    elif op == "*":
                        expected_res = num1 * num2
                    elif op == "/":
                        if num2 == 0:
                            continue
                        expected_res = num1 / num2

                    if not math.isclose(expected_res, claimed_res, rel_tol=1e-3, abs_tol=1e-3):
                        errors.append({
                            "type": "CALCULATION_ERROR",
                            "expression": match.group(0),
                            "expected": expected_res,
                            "claimed": claimed_res,
                        })
                elif len(groups) == 3:
                    num1 = float(groups[0])
                    num2 = float(groups[1])
                    claimed_res = float(groups[2])
                    matched_str = match.group(0).lower()

                    if "plus" in matched_str:
                        expected_res = num1 + num2
                    elif "minus" in matched_str:
                        expected_res = num1 - num2
                    elif "times" in matched_str:
                        expected_res = num1 * num2
                    else:
                        continue

                    if not math.isclose(expected_res, claimed_res, rel_tol=1e-3, abs_tol=1e-3):
                        errors.append({
                            "type": "CALCULATION_ERROR",
                            "expression": match.group(0),
                            "expected": expected_res,
                            "claimed": claimed_res,
                        })

        return errors

    def validate_temporal_order(self, text: str) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for match in self.TEMPORAL_SEQUENCE_PATTERN.finditer(text):
            relation = match.group(1).lower()
            ref_year = int(match.group(2))
            event_year = int(match.group(3))

            if relation == "before" and event_year > ref_year:
                errors.append({
                    "type": "TEMPORAL_ERROR",
                    "text": match.group(0),
                    "issue": f"Claimed before {ref_year} but occurred in {event_year}",
                })
            elif relation == "after" and event_year < ref_year:
                errors.append({
                    "type": "TEMPORAL_ERROR",
                    "text": match.group(0),
                    "issue": f"Claimed after {ref_year} but occurred in {event_year}",
                })
        return errors

    def check_internal_contradictions(self, assertions: List[Assertion]) -> List[Dict[str, Any]]:
        contradictions: List[Dict[str, Any]] = []
        n = len(assertions)
        for i in range(n):
            for j in range(i + 1, n):
                a1 = assertions[i]
                a2 = assertions[j]
                
                stems1 = set(w[:4] for w in re.findall(r"\b[a-zA-Z]{4,}\b", a1.text.lower()))
                stems2 = set(w[:4] for w in re.findall(r"\b[a-zA-Z]{4,}\b", a2.text.lower()))
                common = stems1.intersection(stems2)
                
                if len(common) >= 2:
                    neg1 = any(w in a1.text.lower().split() for w in ["not", "never", "no", "cannot"])
                    neg2 = any(w in a2.text.lower().split() for w in ["not", "never", "no", "cannot"])
                    if neg1 != neg2:
                        contradictions.append({
                            "type": "INTERNAL_CONTRADICTION",
                            "claim_1": a1.text,
                            "claim_2": a2.text,
                        })
        return contradictions

    def assess_uncertainty(
        self,
        assertions: List[Assertion],
        evidence_coverage: float,
        reasoning_errors: List[Dict[str, Any]],
    ) -> float:
        if not assertions:
            return 0.0

        supported_count = sum(1 for a in assertions if a.status == ClaimStatus.SUPPORTED)
        contradicted_count = sum(1 for a in assertions if a.status == ClaimStatus.CONTRADICTED)
        unverifiable_count = sum(1 for a in assertions if a.status in (ClaimStatus.UNVERIFIABLE, ClaimStatus.INSUFFICIENT))
        total = len(assertions)

        unverifiable_ratio = unverifiable_count / total
        contradiction_ratio = contradicted_count / total
        error_penalty = min(0.4, len(reasoning_errors) * 0.2)
        coverage_gap = max(0.0, 1.0 - evidence_coverage)

        uncertainty = (
            unverifiable_ratio * 0.35
            + contradiction_ratio * 0.4
            + coverage_gap * 0.15
            + error_penalty
        )
        return round(min(1.0, max(0.0, uncertainty)), 3)

    def validate(self, text: str, assertions: List[Assertion], evidence_coverage: float) -> Dict[str, Any]:
        arith_errors = self.validate_arithmetic(text)
        temp_errors = self.validate_temporal_order(text)
        contra_errors = self.check_internal_contradictions(assertions)
        all_errors = arith_errors + temp_errors + contra_errors

        uncertainty = self.assess_uncertainty(assertions, evidence_coverage, all_errors)

        return {
            "has_reasoning_errors": len(all_errors) > 0,
            "errors": all_errors,
            "uncertainty_score": uncertainty,
        }
