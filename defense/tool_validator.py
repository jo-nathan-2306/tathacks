import re
import math
from copy import deepcopy
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

    # Default schemas for known sensitive tools — validates required params and types
    DEFAULT_SCHEMAS: Dict[str, Dict[str, Any]] = {
        "execute_code": {
            "required": ["code", "language"],
            "properties": {"code": {"type": "str"}, "language": {"type": "str"}},
        },
        "shell_command": {
            "required": ["command"],
            "properties": {"command": {"type": "str"}},
        },
        "write_file": {
            "required": ["path", "content"],
            "properties": {"path": {"type": "str"}, "content": {"type": "str"}},
        },
        "delete_file": {
            "required": ["path"],
            "properties": {"path": {"type": "str"}},
        },
        "send_email": {
            "required": ["to", "subject", "body"],
            "properties": {"to": {"type": "str"}, "subject": {"type": "str"}, "body": {"type": "str"}},
        },
        "transfer_funds": {
            "required": ["amount", "destination"],
            "properties": {"amount": {"type": "float"}, "destination": {"type": "str"}},
        },
    }

    TYPE_MAP: Dict[str, type] = {
        "str": str,
        "int": int,
        "float": (int, float),  # type: ignore[assignment]
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    def __init__(self, config: Optional[ToolConfig] = None) -> None:
        self.config = config or ToolConfig()
        self._tool_schemas: Dict[str, Dict[str, Any]] = dict(self.DEFAULT_SCHEMAS)

    def register_schema(self, tool_name: str, schema: Dict[str, Any]) -> None:
        """Register a constrained schema for an application-owned tool."""
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if not isinstance(schema, dict) or not isinstance(schema.get("properties", {}), dict):
            raise ValueError("schema must be a mapping with a properties mapping")
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            raise ValueError("schema required must be a list of parameter names")
        if any(key not in properties for key in required):
            raise ValueError("every required parameter must have a property schema")
        for key, definition in properties.items():
            if not isinstance(key, str) or not isinstance(definition, dict):
                raise ValueError("schema properties must map string names to mappings")
            if definition.get("type") not in self.TYPE_MAP:
                raise ValueError(f"unsupported type for schema property '{key}'")
        self._tool_schemas[tool_name.strip()] = deepcopy(schema)

    def _validate_schema(self, request: ToolCallRequest) -> List[str]:
        """Validate tool arguments against the registered schema."""
        schema = self._tool_schemas.get(request.tool_name)
        if not schema:
            return []

        violations: List[str] = []
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required parameters
        for param in required:
            if param not in request.arguments:
                violations.append(f"Missing required parameter '{param}' for tool '{request.tool_name}'.")

        # Check types of provided parameters
        for param, value in request.arguments.items():
            if param not in properties:
                violations.append(f"Unknown parameter '{param}' for tool '{request.tool_name}'.")
                continue
            expected_type_str = properties[param].get("type", "")
            expected_type = self.TYPE_MAP.get(expected_type_str)
            # bool subclasses int in Python; accepting True as an amount/count
            # creates surprising authorization and schema bypasses.
            wrong_type = expected_type and not isinstance(value, expected_type)
            if expected_type_str in ("int", "float") and isinstance(value, bool):
                wrong_type = True
            if wrong_type:
                violations.append(
                    f"Parameter '{param}' for tool '{request.tool_name}' "
                    f"expected type '{expected_type_str}', got '{type(value).__name__}'."
                )

        return violations

    def sanitize_argument(self, key: str, value: Any, depth: int = 0) -> Tuple[Any, List[str]]:
        reasons: List[str] = []
        if depth > self.config.max_argument_depth:
            return value, [f"Maximum nesting depth exceeded for parameter '{key}'."]
        if isinstance(value, str):
            if len(value) > self.config.max_string_length:
                return value[:self.config.max_string_length], [f"Maximum length exceeded for parameter '{key}'."]
            sanitized = value
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern.search(sanitized):
                    reasons.append(f"Dangerous pattern detected in parameter '{key}'.")
                    sanitized = pattern.sub("_SANITIZED_", sanitized)
            return sanitized, reasons
        elif isinstance(value, dict):
            sanitized_dict = {}
            if len(value) > self.config.max_collection_items:
                reasons.append(f"Maximum collection size exceeded for parameter '{key}'.")
            for index, (k, v) in enumerate(value.items()):
                if index >= self.config.max_collection_items:
                    break
                if not isinstance(k, str):
                    reasons.append(f"Non-string object key found in parameter '{key}'.")
                    continue
                san_v, sub_reasons = self.sanitize_argument(k, v, depth + 1)
                sanitized_dict[k] = san_v
                reasons.extend(sub_reasons)
            return sanitized_dict, reasons
        elif isinstance(value, list):
            if len(value) > self.config.max_collection_items:
                reasons.append(f"Maximum collection size exceeded for parameter '{key}'.")
            sanitized_list = []
            for item in value[:self.config.max_collection_items]:
                san_item, sub_reasons = self.sanitize_argument(key, item, depth + 1)
                sanitized_list.append(san_item)
                reasons.extend(sub_reasons)
            return sanitized_list, reasons
        elif isinstance(value, float) and not math.isfinite(value):
            return value, [f"Non-finite numeric value found in parameter '{key}'."]
        elif value is None or isinstance(value, (bool, int, float)):
            return value, reasons
        else:
            return value, [f"Unsupported argument value type for parameter '{key}'."]

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

        schema = self._tool_schemas.get(request.tool_name)
        if schema is None and not self.config.allow_unregistered_tools:
            all_reasons.append(f"Tool '{request.tool_name}' is not registered and is denied by default.")

        # Schema validation
        if self.config.enforce_schema:
            schema_violations = self._validate_schema(request)
            all_reasons.extend(schema_violations)

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

        # Schema violations also invalidate the call
        # Rewriting an argument is not proof that it is safe to execute.  The
        # caller must review and submit a clean request instead.
        if self.config.sanitize_arguments and any("Dangerous pattern" in r for r in all_reasons):
            all_reasons.append("Tool call blocked because an argument contains unsafe syntax.")

        if self.config.enforce_schema and any(
            marker in r for r in all_reasons
            for marker in ("Missing required", "expected type", "Unknown parameter")
        ):
            is_valid = False

        if schema is None and not self.config.allow_unregistered_tools:
            is_valid = False
        if any(
            marker in r for r in all_reasons
            for marker in ("Dangerous pattern", "Maximum nesting", "Maximum length", "Maximum collection", "Non-string object key", "Non-finite numeric", "Unsupported argument")
        ):
            is_valid = False

        return ToolValidationResult(
            is_valid=is_valid,
            sanitized_arguments=sanitized_args,
            reasons=all_reasons,
            requires_user_confirmation=requires_confirmation,
        )
