"""Rate limiter for LLM API calls with token bucket algorithm.

Provides reliable rate limiting to prevent 429 errors and ensure API quota compliance.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from collections import deque


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.
    
    Attributes:
        requests_per_minute: Maximum requests per minute (RPM)
        requests_per_second: Maximum requests per second (RPS), optional
        tokens_per_minute: Maximum tokens per minute (TPM), optional
        max_concurrent: Maximum concurrent requests
        burst_multiplier: Allow burst up to this multiplier of base rate
    """
    requests_per_minute: int = 60
    requests_per_second: Optional[int] = None
    tokens_per_minute: Optional[int] = None
    max_concurrent: int = 10
    burst_multiplier: float = 1.5


class TokenBucket:
    """Token bucket rate limiter implementation.
    
    Allows bursting up to bucket capacity while maintaining average rate.
    """
    
    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: float,  # maximum tokens in bucket
    ):
        """Initialize token bucket.
        
        Args:
            rate: Token refill rate per second
            capacity: Maximum bucket capacity (allows bursting)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity  # Start full
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait indefinitely)
            
        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.monotonic()
        
        async with self._lock:
            while True:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed + wait_time > timeout:
                        return False
                    wait_time = min(wait_time, timeout - elapsed)
                
                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate)."""
        elapsed = time.monotonic() - self.last_update
        return min(self.capacity, self.tokens + elapsed * self.rate)


class SlidingWindowCounter:
    """Sliding window rate limiter for precise rate limiting.
    
    More accurate than token bucket for strict per-minute/per-second limits.
    """
    
    def __init__(self, limit: int, window_seconds: float):
        """Initialize sliding window counter.
        
        Args:
            limit: Maximum requests in the window
            window_seconds: Window size in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Try to acquire a request slot.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if slot acquired, False if timeout
        """
        start_time = time.monotonic()
        
        async with self._lock:
            while True:
                self._cleanup()
                
                if len(self.requests) < self.limit:
                    self.requests.append(time.monotonic())
                    return True
                
                # Calculate wait time until oldest request expires
                if self.requests:
                    oldest = self.requests[0]
                    wait_time = (oldest + self.window_seconds) - time.monotonic()
                    wait_time = max(0.01, wait_time)  # Minimum wait
                else:
                    wait_time = 0.01
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        return False
                    wait_time = min(wait_time, timeout - elapsed)
                
                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()
    
    def _cleanup(self):
        """Remove expired requests from the window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    @property
    def current_count(self) -> int:
        """Get current request count in window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        return sum(1 for t in self.requests if t >= cutoff)


class RateLimiter:
    """Composite rate limiter with multiple constraints.
    
    Combines RPM, RPS, TPM, and concurrency limits.
    """
    
    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with config.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        
        # RPM limiter (sliding window for accuracy)
        self.rpm_limiter = SlidingWindowCounter(
            limit=config.requests_per_minute,
            window_seconds=60.0
        )
        
        # RPS limiter (optional, token bucket for smooth distribution)
        self.rps_limiter: Optional[TokenBucket] = None
        if config.requests_per_second:
            self.rps_limiter = TokenBucket(
                rate=config.requests_per_second,
                capacity=config.requests_per_second * config.burst_multiplier
            )
        
        # TPM limiter (optional)
        self.tpm_limiter: Optional[SlidingWindowCounter] = None
        if config.tokens_per_minute:
            # We'll track tokens, not requests
            self._token_window: deque = deque()
            self._token_lock = asyncio.Lock()
        
        # Concurrency semaphore
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        
        # Metrics
        self._total_requests = 0
        self._total_wait_time = 0.0
        self._throttled_requests = 0
    
    async def acquire(
        self,
        estimated_tokens: int = 0,
        timeout: Optional[float] = 30.0
    ) -> 'RateLimitContext':
        """Acquire permission to make a request.
        
        Args:
            estimated_tokens: Estimated total tokens for this request
            timeout: Maximum time to wait for rate limit
            
        Returns:
            Context manager that releases the slot on exit
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        start_time = time.monotonic()
        
        # Acquire concurrency slot
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._throttled_requests += 1
            raise
        
        try:
            remaining_timeout = timeout - (time.monotonic() - start_time) if timeout else None
            
            # Acquire RPM slot
            if not await self.rpm_limiter.acquire(timeout=remaining_timeout):
                self._semaphore.release()
                self._throttled_requests += 1
                raise asyncio.TimeoutError("RPM rate limit timeout")
            
            # Acquire RPS slot (if configured)
            if self.rps_limiter:
                remaining_timeout = timeout - (time.monotonic() - start_time) if timeout else None
                if not await self.rps_limiter.acquire(timeout=remaining_timeout):
                    self._semaphore.release()
                    self._throttled_requests += 1
                    raise asyncio.TimeoutError("RPS rate limit timeout")
            
            # Check TPM (if configured)
            if self.config.tokens_per_minute and estimated_tokens > 0:
                remaining_timeout = timeout - (time.monotonic() - start_time) if timeout else None
                await self._acquire_tokens(estimated_tokens, timeout=remaining_timeout)
            
            wait_time = time.monotonic() - start_time
            self._total_wait_time += wait_time
            self._total_requests += 1
            
            return RateLimitContext(self, estimated_tokens)
            
        except Exception:
            self._semaphore.release()
            raise
    
    async def _acquire_tokens(self, tokens: int, timeout: Optional[float] = None):
        """Acquire token quota from TPM limiter."""
        start_time = time.monotonic()
        
        async with self._token_lock:
            while True:
                self._cleanup_tokens()
                
                current_tokens = sum(t for _, t in self._token_window)
                if current_tokens + tokens <= self.config.tokens_per_minute:
                    self._token_window.append((time.monotonic(), tokens))
                    return
                
                # Wait for tokens to expire
                if self._token_window:
                    oldest_time, _ = self._token_window[0]
                    wait_time = (oldest_time + 60.0) - time.monotonic()
                    wait_time = max(0.01, wait_time)
                else:
                    wait_time = 0.01
                
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout:
                        raise asyncio.TimeoutError("TPM rate limit timeout")
                    wait_time = min(wait_time, timeout - elapsed)
                
                self._token_lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._token_lock.acquire()
    
    def _cleanup_tokens(self):
        """Remove expired token records."""
        if not hasattr(self, '_token_window'):
            return
        now = time.monotonic()
        cutoff = now - 60.0
        while self._token_window and self._token_window[0][0] < cutoff:
            self._token_window.popleft()
    
    def release(self):
        """Release concurrency slot."""
        self._semaphore.release()
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self._total_requests,
            "throttled_requests": self._throttled_requests,
            "avg_wait_time_ms": (self._total_wait_time / max(1, self._total_requests)) * 1000,
            "current_rpm": self.rpm_limiter.current_count,
            "current_concurrent": self.config.max_concurrent - self._semaphore._value,
        }


class RateLimitContext:
    """Context manager for rate limit slot."""
    
    def __init__(self, limiter: RateLimiter, tokens: int = 0):
        self.limiter = limiter
        self.tokens = tokens
        self._released = False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
    
    def release(self):
        """Release the rate limit slot."""
        if not self._released:
            self.limiter.release()
            self._released = True
    
    def update_tokens(self, actual_tokens: int):
        """Update token count after response received.
        
        Useful for TPM tracking when actual tokens differ from estimate.
        """
        # Could adjust TPM tracking here if needed
        self.tokens = actual_tokens


# Pre-configured rate limit profiles for common providers
PROVIDER_RATE_LIMITS: Dict[str, RateLimitConfig] = {
    "openai": RateLimitConfig(
        requests_per_minute=500,
        requests_per_second=None,
        tokens_per_minute=None,  # Varies by tier
        max_concurrent=50,
    ),
    "anthropic": RateLimitConfig(
        requests_per_minute=1000,
        requests_per_second=None,
        tokens_per_minute=None,
        max_concurrent=50,
    ),
    "openrouter": RateLimitConfig(
        requests_per_minute=200,  # Conservative default
        requests_per_second=10,
        tokens_per_minute=None,
        max_concurrent=20,
    ),
    "openrouter_high": RateLimitConfig(
        requests_per_minute=500,
        requests_per_second=20,
        tokens_per_minute=None,
        max_concurrent=50,
    ),
}


def get_rate_limiter(provider_name: str, custom_config: Optional[RateLimitConfig] = None) -> RateLimiter:
    """Get a rate limiter for a provider.
    
    Args:
        provider_name: Name of the provider
        custom_config: Optional custom configuration
        
    Returns:
        Configured RateLimiter instance
    """
    if custom_config:
        return RateLimiter(custom_config)
    
    config = PROVIDER_RATE_LIMITS.get(
        provider_name.lower(),
        PROVIDER_RATE_LIMITS["openrouter"]  # Default to conservative
    )
    return RateLimiter(config)

