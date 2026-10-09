import re
from typing import Any, Dict, List, Optional
from defense.config import PrivacyConfig


class PrivacySanitizer:
    PATTERNS = {
        "API_KEY": re.compile(r"(?:api[_-]?key|secret|token|password|auth)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]{12,})['\"]?", re.IGNORECASE),
        "BEARER_TOKEN": re.compile(r"bearer\s+([a-zA-Z0-9_\-\.]{20,})", re.IGNORECASE),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    def __init__(self, config: Optional[PrivacyConfig] = None) -> None:
        self.config = config or PrivacyConfig()
        self._custom_patterns = [
            re.compile(pat, re.IGNORECASE) for pat in self.config.custom_secret_patterns
        ]

    def mask_text(self, text: str) -> str:
        if not self.config.sanitize_pii and not self.config.sanitize_secrets:
            return text

        masked = text

        if self.config.sanitize_secrets:
            for pat in self._custom_patterns:
                masked = pat.sub("[REDACTED_CUSTOM_SECRET]", masked)

            masked = self.PATTERNS["API_KEY"].sub(r"api_key=[REDACTED_KEY]", masked)
            masked = self.PATTERNS["BEARER_TOKEN"].sub(r"Bearer [REDACTED_TOKEN]", masked)

        if self.config.sanitize_pii:
            masked = self.PATTERNS["EMAIL"].sub("[REDACTED_EMAIL]", masked)
            masked = self.PATTERNS["CREDIT_CARD"].sub("[REDACTED_CARD]", masked)
            masked = self.PATTERNS["PHONE_NUMBER"].sub("[REDACTED_PHONE]", masked)
            masked = self.PATTERNS["SSN"].sub("[REDACTED_SSN]", masked)

        return masked

    def sanitize_log_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for k, v in payload.items():
            if k in ("prompt", "user_input") and not self.config.store_full_prompts:
                sanitized[k] = f"[MASKED_LEN_{len(str(v))}]"
            elif k in ("response", "raw_output", "final_output") and not self.config.store_full_responses:
                sanitized[k] = f"[MASKED_LEN_{len(str(v))}]"
            elif isinstance(v, str):
                sanitized[k] = self.mask_text(v)
            elif isinstance(v, dict):
                sanitized[k] = self.sanitize_log_payload(v)
            elif isinstance(v, list):
                sanitized[k] = [
                    self.mask_text(item) if isinstance(item, str) else item
                    for item in v
                ]
            else:
                sanitized[k] = v
        return sanitized
