"""Detect and discard model-internal reasoning before it reaches users."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningFilterResult:
    text: str
    detected: bool


class InternalReasoningFilter:
    """Allow final answers while withholding explicit analysis/thinking output."""

    _THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
    _THINK_OPEN = re.compile(r"<think\b", re.IGNORECASE)
    _REASONING_PREFIX = re.compile(
        r"^\s*(?:"
        r"here(?:'s| is)\s+(?:my\s+)?(?:thinking process|chain of thought|reasoning)\s*:"
        r"|(?:private\s+)?(?:chain of thought|internal reasoning|reasoning process)\s*:"
        r"|(?:analysis|reasoning|thinking|thought)\s*:"
        r"|let(?:'s| us)\s+think\s+step\s+by\s+step\s*:"
        r")",
        re.IGNORECASE,
    )
    _FINAL_HEADING = re.compile(r"^\s*final answer\s*:\s*", re.IGNORECASE | re.MULTILINE)

    def filter(self, text: str) -> ReasoningFilterResult:
        value = str(text or "").strip()
        if not value:
            return ReasoningFilterResult("", False)

        had_think_block = bool(self._THINK_OPEN.search(value))
        value = self._THINK_BLOCK.sub("", value).strip()
        if had_think_block:
            # An unclosed reasoning tag makes the remainder ambiguous.
            if self._THINK_OPEN.search(value):
                value = self._THINK_OPEN.sub("", value).strip()
                return ReasoningFilterResult("", True)
            return ReasoningFilterResult(value, True)

        if self._REASONING_PREFIX.match(value):
            final_heading = self._FINAL_HEADING.search(value)
            if not final_heading:
                return ReasoningFilterResult("", True)
            return ReasoningFilterResult(value[final_heading.end():].strip(), True)

        return ReasoningFilterResult(value, False)
