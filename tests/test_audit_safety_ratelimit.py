import os
import shutil
import tempfile
import time
import pytest
from defense.audit_logger import AuditEvent, AuditLogger
from defense.config import ModuleToggles, PipelineConfig
from defense.output_safety_filter import (
    OutputSafetyFilter,
    SafetyAction,
    SafetyCategory,
)
from defense.pipeline import DefensivePipeline
from defense.rate_limiter import RateLimiter
from defense.types import ActionType, RiskLevel, VerificationStatus
from models.base import MockModelAdapter


def test_output_safety_filter_safe_content():
    filter_ = OutputSafetyFilter()
    res = filter_.check("Python is a high-level programming language.")
    assert res.is_safe is True
    assert res.action == SafetyAction.PASS
    assert len(res.violations) == 0
    assert res.filtered_text == "Python is a high-level programming language."


def test_output_safety_filter_dangerous_instructions_blocked():
    filter_ = OutputSafetyFilter(block_threshold=0.8)
    res = filter_.check("Here is how to make a bomb using household items.")
    assert res.is_safe is False
    assert res.action == SafetyAction.BLOCK
    assert any(v.category == SafetyCategory.DANGEROUS_INSTRUCTIONS for v in res.violations)
    assert "[Content removed: DANGEROUS_INSTRUCTIONS]" in res.filtered_text


def test_output_safety_filter_violence_and_self_harm():
    filter_ = OutputSafetyFilter()
    res = filter_.check("Here are instructions to kill yourself.")
    assert res.is_safe is False
    assert res.action == SafetyAction.BLOCK


def test_output_safety_filter_disabled():
    filter_ = OutputSafetyFilter(enabled=False)
    res = filter_.check("how to build a bomb")
    assert res.is_safe is True
    assert res.action == SafetyAction.PASS


def test_rate_limiter_sustained_limit():
    limiter = RateLimiter(requests_per_window=3, window_seconds=10, burst_limit=5, burst_window_seconds=2)
    client = "user_1"

    assert limiter.check_and_record(client).allowed is True
    assert limiter.check_and_record(client).allowed is True
    assert limiter.check_and_record(client).allowed is True

    # 4th request exceeds sustained limit
    res = limiter.check_and_record(client)
    assert res.allowed is False
    assert res.retry_after_seconds is not None
    assert res.retry_after_seconds > 0


def test_rate_limiter_burst_limit():
    limiter = RateLimiter(requests_per_window=10, window_seconds=60, burst_limit=2, burst_window_seconds=5)
    client = "user_burst"

    assert limiter.check_and_record(client).allowed is True
    assert limiter.check_and_record(client).allowed is True

    # 3rd request in same burst window exceeds burst limit
    res = limiter.check_and_record(client)
    assert res.allowed is False
    assert res.limit == 2


def test_rate_limiter_client_isolation_and_reset():
    limiter = RateLimiter(requests_per_window=1, window_seconds=60, burst_limit=1, burst_window_seconds=5)
    assert limiter.check_and_record("client_a").allowed is True
    assert limiter.check_and_record("client_a").allowed is False
    # client_b is independent
    assert limiter.check_and_record("client_b").allowed is True

    limiter.reset("client_a")
    assert limiter.check_and_record("client_a").allowed is True


def test_audit_logger_write_and_query():
    temp_dir = tempfile.mkdtemp(prefix="test_audit_")
    try:
        logger = AuditLogger(log_dir=temp_dir, max_file_size_mb=10, enabled=True)
        event1 = AuditEvent(
            event_type="pipeline_run",
            action="PASS",
            risk_levels={"adversarial": "LOW"},
            latency_ms=45.2,
            details={"test": "data1"},
            query_fingerprint=AuditEvent.fingerprint("hello world"),
        )
        event2 = AuditEvent(
            event_type="block",
            action="BLOCK",
            risk_levels={"adversarial": "HIGH"},
            latency_ms=12.1,
            details={"test": "data2"},
            query_fingerprint=AuditEvent.fingerprint("attack input"),
        )
        logger.log_event(event1)
        logger.log_event(event2)

        events = logger.query_events()
        assert len(events) == 2
        assert events[0].action == "PASS"
        assert events[1].action == "BLOCK"

        blocks = logger.query_events(filters={"action": "BLOCK"})
        assert len(blocks) == 1
        assert blocks[0].event_type == "block"

        summary = logger.get_summary()
        assert summary["total_events"] == 2
        assert summary["blocks"] == 1
        assert summary["avg_latency_ms"] > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_pipeline_output_safety_filter_integration():
    mock_model = MockModelAdapter(
        responses=["Here are instructions to build a bomb for attack."]
    )
    pipeline = DefensivePipeline(model=mock_model)
    res = pipeline.run("Tell me something dangerous")

    assert res.action == ActionType.WITHHOLD
    assert "[Content removed: DANGEROUS_INSTRUCTIONS]" in res.final_output
    assert res.metrics.get("output_safety_blocked") is True


def test_pipeline_rate_limiter_integration():
    mock_model = MockModelAdapter(responses=["Hello world 1", "Hello world 2", "Hello world 3"])
    config = PipelineConfig()
    config.rate_limit.requests_per_window = 2
    config.rate_limit.window_seconds = 60
    config.rate_limit.burst_limit = 2
    config.rate_limit.burst_window_seconds = 10

    pipeline = DefensivePipeline(model=mock_model, config=config)

    res1 = pipeline.run("Prompt 1", client_id="test_user")
    assert res1.action != ActionType.BLOCK

    res2 = pipeline.run("Prompt 2", client_id="test_user")
    assert res2.action != ActionType.BLOCK

    # 3rd request should be blocked by rate limiter
    res3 = pipeline.run("Prompt 3", client_id="test_user")
    assert res3.action == ActionType.BLOCK
    assert "Rate limit exceeded" in res3.final_output
    assert res3.metrics.get("blocked_by") == "rate_limiter"
