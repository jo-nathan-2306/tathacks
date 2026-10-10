"""Append-only audit logger for pipeline decisions and model behavior tracking."""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AuditEvent:
    """Single auditable event in the pipeline lifecycle."""

    timestamp: str = ""
    event_type: str = ""  # 'pipeline_run', 'block', 'correction', 'tool_validation', 'rate_limit'
    action: str = ""
    risk_levels: Dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    query_fingerprint: str = ""  # SHA-256 hash — stores identity without raw text

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def fingerprint(text: str) -> str:
        """Produce a SHA-256 fingerprint of input text for deduplication."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditLogger:
    """Append-only JSONL audit logger with automatic file rotation."""

    def __init__(
        self,
        log_dir: str = "audit_logs",
        max_file_size_mb: int = 50,
        enabled: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.enabled = enabled
        self._current_file: Optional[str] = None
        self._event_count = 0
        self._initialized = False

    def _resolve_log_file(self) -> str:
        """Get or create the current log file path."""
        date_tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = os.path.join(self.log_dir, f"audit_{date_tag}.jsonl")
        return path

    def _ensure_initialized(self) -> None:
        """Lazily create log directory and resolve file path on first use."""
        if not self._initialized and self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self._current_file = self._resolve_log_file()
            self._initialized = True

    def _rotate_if_needed(self) -> None:
        """Rotate to a new file if current exceeds size limit."""
        if self._current_file and os.path.exists(self._current_file):
            if os.path.getsize(self._current_file) >= self.max_file_size_bytes:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
                self._current_file = os.path.join(self.log_dir, f"audit_{ts}.jsonl")

    def log_event(self, event: AuditEvent) -> None:
        """Append an audit event to the current log file."""
        if not self.enabled:
            return
        self._ensure_initialized()
        self._rotate_if_needed()
        if self._current_file is None:
            self._current_file = self._resolve_log_file()
        with open(self._current_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")
        self._event_count += 1

    def query_events(
        self,
        filters: Optional[Dict[str, str]] = None,
        last_n: int = 100,
    ) -> List[AuditEvent]:
        """Read the most recent events, optionally filtering by field values."""
        if not self.enabled:
            return []
        self._ensure_initialized()
        if not self._current_file or not os.path.exists(self._current_file):
            return []
        lines: List[str] = []
        with open(self._current_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        events: List[AuditEvent] = []
        for line in lines[-last_n:]:
            try:
                data = json.loads(line.strip())
                if filters and not all(data.get(k) == v for k, v in filters.items()):
                    continue
                events.append(AuditEvent(**{k: v for k, v in data.items() if k in AuditEvent.__dataclass_fields__}))
            except (json.JSONDecodeError, TypeError):
                continue
        return events

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate statistics across the current log file."""
        events = self.query_events(last_n=10_000)
        if not events:
            return {"total_events": 0}
        actions: Dict[str, int] = {}
        total_latency = 0.0
        for ev in events:
            actions[ev.action] = actions.get(ev.action, 0) + 1
            total_latency += ev.latency_ms
        return {
            "total_events": len(events),
            "action_counts": actions,
            "avg_latency_ms": round(total_latency / len(events), 2) if events else 0,
            "blocks": actions.get("BLOCK", 0),
            "corrections": actions.get("CORRECT", 0),
        }
