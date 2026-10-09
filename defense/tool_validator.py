import re
from typing import Any, Dict, List, Optional, Set, Tuple
from defense.config import ToolConfig
from defense.types import ToolCallRequest, ToolValidationResult, TrustBoundaryType


class ToolValidator:
    DANGEROUS_PATTERNS = [
        re.compile(r"(\||;|&&|\n|\`|\$\()", re.IGNORECASE),
        re.compile(r"\.\./", re.IGNORECASE),
        re.compile(r"rm\s+-rf", re.IGNORECASE),
        re.compile(r"drop\s+table", re.IGNORECASE),
        re.compile(r"/etc/passwd", re.IGNORECASE),
    ]

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        self.config = config or ToolConfig()

    def sanitize_argument(self, key: str, value: Any) -> Tuple[Any, List[str]]:
        reasons: List[str] = []
        if isinstance(value, str):
            sanitized = value
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern.search(sanitized):
                    reasons.append(f"Dangerous pattern detected in parameter '{key}'.")
                    sanitized = pattern.sub("_SANITIZED_", sanitized)
            return sanitized, reasons
        elif isinstance(value, dict):
            sanitized_dict = {}
            for k, v in value.items():
                san_v, sub_reasons = self.sanitize_argument(k, v)
                sanitized_dict[k] = san_v
                reasons.extend(sub_reasons)
            return sanitized_dict, reasons
        elif isinstance(value, list):
            sanitized_list = []
            for item in value:
                san_item, sub_reasons = self.sanitize_argument(key, item)
                sanitized_list.append(san_item)
                reasons.extend(sub_reasons)
            return sanitized_list, reasons
        return value, reasons

    def validate_tool_call(self, request: ToolCallRequest) -> ToolValidationResult:
        if not self.config.enabled:
            return ToolValidationResult(
                is_valid=True,
                sanitized_arguments=request.arguments,
                reasons=[],
                requires_user_confirmation=False,
            )

        is_sensitive = (
            request.is_sensitive or request.tool_name in self.config.sensitive_tools
        )
        requires_confirmation = is_sensitive or request.requires_confirmation

        sanitized_args: Dict[str, Any] = {}
        all_reasons: List[str] = []

        if self.config.sanitize_arguments:
            for k, v in request.arguments.items():
                san_v, reasons = self.sanitize_argument(k, v)
                sanitized_args[k] = san_v
                all_reasons.extend(reasons)
        else:
            sanitized_args = request.arguments

        is_valid = True
        if request.caller_boundary == TrustBoundaryType.RETRIEVED_DOC:
            is_valid = False
            all_reasons.append(
                "Direct tool execution from retrieved documents is blocked by trust boundary."
            )

        return ToolValidationResult(
            is_valid=is_valid,
            sanitized_arguments=sanitized_args,
            reasons=all_reasons,
            requires_user_confirmation=requires_confirmation,
        )
