"""Simple in-memory rate limiting for API endpoints."""

import time
from collections import defaultdict
from typing import Dict, Tuple
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Simple sliding window rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per client
            burst_size: Maximum burst of requests allowed
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.window_size = 60.0  # 1 minute
        # client_id -> list of (timestamp,) for requests in current window
        self._requests: Dict[str, list] = defaultdict(list)

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if behind proxy, otherwise use client host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, client_id: str, now: float):
        """Remove requests outside the current window."""
        cutoff = now - self.window_size
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]

    def check(self, request: Request) -> Tuple[bool, int]:
        """
        Check if request should be allowed.

        Returns:
            Tuple of (allowed, remaining_requests)
        """
        client_id = self._get_client_id(request)
        now = time.time()

        self._cleanup_old_requests(client_id, now)

        current_count = len(self._requests[client_id])
        remaining = max(0, self.requests_per_minute - current_count)

        if current_count >= self.requests_per_minute:
            return False, 0

        # Check burst (requests in last second)
        recent_cutoff = now - 1.0
        recent_count = sum(1 for ts in self._requests[client_id] if ts > recent_cutoff)
        if recent_count >= self.burst_size:
            return False, remaining

        self._requests[client_id].append(now)
        return True, remaining - 1

    def get_retry_after(self, request: Request) -> int:
        """Get seconds until rate limit resets."""
        client_id = self._get_client_id(request)
        if not self._requests[client_id]:
            return 0
        oldest = min(self._requests[client_id])
        return max(1, int(self.window_size - (time.time() - oldest)))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app,
        limiter: RateLimiter,
        exempt_paths: list = None,
    ):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            limiter: RateLimiter instance
            exempt_paths: Paths exempt from rate limiting (e.g., ["/health"])
        """
        super().__init__(app)
        self.limiter = limiter
        self.exempt_paths = exempt_paths or ["/health", "/", "/docs", "/redoc", "/openapi.json"]

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        allowed, remaining = self.limiter.check(request)

        if not allowed:
            retry_after = self.limiter.get_retry_after(request)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# Default rate limiter instance
default_limiter = RateLimiter(requests_per_minute=60, burst_size=10)
