import re
from typing import Any, Dict, List, Optional
from defense.config import PrivacyConfig


class PrivacySanitizer:
    PATTERNS = {
        "API_KEY": re.compile(r"(?:api[_-]?key|secret|token|password|auth)[\s:=]+['\"]?([a-zA-Z0-9_\-\.]{12,})['\"]?", re.IGNORECASE),
        "BEARER_TOKEN": re.compile(r"bearer\s+([a-zA-Z0-9_\-\.]{20,})", re.IGNORECASE),
        "JWT_TOKEN": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "AWS_KEY": re.compile(r"\b(AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b"),
        "OPENAI_KEY": re.compile(r"\bsk-(?:proj-|org-|ant-|tinyfish-)?[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
        "GOOGLE_API_KEY": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
        "URL_CREDENTIALS": re.compile(r"\b(https?://)[^/\s:@]+:[^@/\s]+@", re.IGNORECASE),
        "GITHUB_TOKEN": re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}\b"),
        "PRIVATE_KEY": re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]+?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----", re.IGNORECASE),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
        "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "IPV4_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
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

            masked = self.PATTERNS["PRIVATE_KEY"].sub("[REDACTED_PRIVATE_KEY]", masked)
            masked = self.PATTERNS["OPENAI_KEY"].sub("[REDACTED_OPENAI_KEY]", masked)
            masked = self.PATTERNS["GOOGLE_API_KEY"].sub("[REDACTED_API_KEY]", masked)
            masked = self.PATTERNS["URL_CREDENTIALS"].sub(r"\1[REDACTED_CREDENTIALS]@", masked)
            masked = self.PATTERNS["GITHUB_TOKEN"].sub("[REDACTED_GITHUB_TOKEN]", masked)
            masked = self.PATTERNS["AWS_KEY"].sub("[REDACTED_AWS_KEY]", masked)
            masked = self.PATTERNS["JWT_TOKEN"].sub("[REDACTED_JWT_TOKEN]", masked)
            masked = self.PATTERNS["API_KEY"].sub(r"api_key=[REDACTED_KEY]", masked)
            masked = self.PATTERNS["BEARER_TOKEN"].sub(r"Bearer [REDACTED_TOKEN]", masked)

        if self.config.sanitize_pii:
            masked = self.PATTERNS["EMAIL"].sub("[REDACTED_EMAIL]", masked)
            masked = self.PATTERNS["CREDIT_CARD"].sub("[REDACTED_CARD]", masked)
            masked = self.PATTERNS["PHONE_NUMBER"].sub("[REDACTED_PHONE]", masked)
            masked = self.PATTERNS["SSN"].sub("[REDACTED_SSN]", masked)
            masked = self.PATTERNS["IPV4_ADDRESS"].sub("[REDACTED_IP]", masked)

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
