"""Sliding-window rate limiter for abuse prevention and DoS protection."""

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    current_count: int = 0
    limit: int = 0
    window_seconds: int = 0
    retry_after_seconds: Optional[float] = None


class RateLimiter:
    """Thread-safe sliding-window rate limiter with sustained and burst limits."""

    def __init__(
        self,
        requests_per_window: int = 60,
        window_seconds: int = 60,
        burst_limit: int = 10,
        burst_window_seconds: int = 5,
        enabled: bool = True,
    ) -> None:
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.burst_limit = burst_limit
        self.burst_window_seconds = burst_window_seconds
        self.enabled = enabled
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _evict_expired(self, client_id: str, now: float) -> None:
        """Remove timestamps outside both windows."""
        bucket = self._buckets[client_id]
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def check(self, client_id: str = "default") -> RateLimitResult:
        """Check whether a request is allowed under both sustained and burst limits."""
        if not self.enabled:
            return RateLimitResult(allowed=True)

        with self._lock:
            now = time.monotonic()
            self._evict_expired(client_id, now)
            bucket = self._buckets[client_id]

            # Sustained window check
            sustained_count = len(bucket)
            if sustained_count >= self.requests_per_window:
                oldest = bucket[0]
                retry_after = oldest + self.window_seconds - now
                return RateLimitResult(
                    allowed=False,
                    current_count=sustained_count,
                    limit=self.requests_per_window,
                    window_seconds=self.window_seconds,
                    retry_after_seconds=max(0.0, retry_after),
                )

            # Burst window check
            burst_cutoff = now - self.burst_window_seconds
            burst_count = sum(1 for ts in bucket if ts >= burst_cutoff)
            if burst_count >= self.burst_limit:
                retry_after = self.burst_window_seconds - (now - min(
                    (ts for ts in bucket if ts >= burst_cutoff), default=now
                ))
                return RateLimitResult(
                    allowed=False,
                    current_count=burst_count,
                    limit=self.burst_limit,
                    window_seconds=self.burst_window_seconds,
                    retry_after_seconds=max(0.0, retry_after),
                )

            return RateLimitResult(
                allowed=True,
                current_count=sustained_count,
                limit=self.requests_per_window,
                window_seconds=self.window_seconds,
            )

    def record(self, client_id: str = "default") -> None:
        """Record a request timestamp."""
        if not self.enabled:
            return
        with self._lock:
            self._buckets[client_id].append(time.monotonic())

    def check_and_record(self, client_id: str = "default") -> RateLimitResult:
        """Atomic check-then-record. Only records if allowed."""
        if not self.enabled:
            return RateLimitResult(allowed=True)
        with self._lock:
            now = time.monotonic()
            self._evict_expired(client_id, now)
            bucket = self._buckets[client_id]

            sustained_count = len(bucket)
            if sustained_count >= self.requests_per_window:
                oldest = bucket[0]
                return RateLimitResult(
                    allowed=False,
                    current_count=sustained_count,
                    limit=self.requests_per_window,
                    window_seconds=self.window_seconds,
                    retry_after_seconds=max(0.0, oldest + self.window_seconds - now),
                )

            burst_cutoff = now - self.burst_window_seconds
            burst_count = sum(1 for ts in bucket if ts >= burst_cutoff)
            if burst_count >= self.burst_limit:
                return RateLimitResult(
                    allowed=False,
                    current_count=burst_count,
                    limit=self.burst_limit,
                    window_seconds=self.burst_window_seconds,
                    retry_after_seconds=max(0.1, self.burst_window_seconds - (now - min(
                        (ts for ts in bucket if ts >= burst_cutoff), default=now
                    ))),
                )

            bucket.append(now)
            return RateLimitResult(
                allowed=True,
                current_count=sustained_count + 1,
                limit=self.requests_per_window,
                window_seconds=self.window_seconds,
            )

    def reset(self, client_id: Optional[str] = None) -> None:
        """Reset rate limit state for one or all clients."""
        with self._lock:
            if client_id:
                self._buckets.pop(client_id, None)
            else:
                self._buckets.clear()
