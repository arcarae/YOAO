"""Retry utilities with exponential backoff, jitter, and circuit breaker.

Provides robust retry logic for handling transient failures in LLM API calls.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Type, Any, Dict
from enum import Enum
import functools


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class RetryConfig:
    """Configuration for retry behavior.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Jitter factor (0-1, fraction of delay to randomize)
        strategy: Retry strategy type
        retryable_exceptions: Exception types to retry on
        retryable_status_codes: HTTP status codes to retry on
    """
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.5  # More jitter to spread out retries
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retryable_exceptions: Set[Type[Exception]] = field(default_factory=lambda: {
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,  # Includes network errors
    })
    retryable_status_codes: Set[int] = field(default_factory=lambda: {
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
        520,  # Cloudflare errors
        521,
        522,
        523,
        524,
    })


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.
    
    Attributes:
        failure_threshold: Number of failures before opening
        success_threshold: Number of successes to close from half-open
        timeout: Seconds before trying again (open -> half-open)
        half_open_max_calls: Max concurrent calls in half-open state
    """
    failure_threshold: int = 15  # Higher threshold for parallel workloads
    success_threshold: int = 2
    timeout: float = 10.0  # Faster recovery
    half_open_max_calls: int = 3  # Allow more test calls


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures.
    
    Opens after consecutive failures, closes after successful recovery.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    @property
    def is_available(self) -> bool:
        """Whether requests can proceed."""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            # Check if timeout elapsed
            if self._last_failure_time:
                if time.time() - self._last_failure_time >= self.config.timeout:
                    return True  # Will transition to half-open
            return False
        # Half-open: limited availability
        return self._half_open_calls < self.config.half_open_max_calls
    
    async def acquire(self) -> bool:
        """Try to acquire permission to make a call.
        
        Returns:
            True if call is allowed, False otherwise
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check timeout
                if self._last_failure_time:
                    if time.time() - self._last_failure_time >= self.config.timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 1
                        return True
                return False
            
            # Half-open
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    async def record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
            else:
                # In closed state, reset failure count on success
                self._failure_count = 0
    
    async def record_failure(self):
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Immediately reopen on any failure
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._half_open_calls = 0
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "time_since_last_failure": (
                time.time() - self._last_failure_time
                if self._last_failure_time else None
            ),
        }


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a retry attempt.
    
    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    if config.strategy == RetryStrategy.CONSTANT:
        delay = config.base_delay
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.base_delay * (attempt + 1)
    else:  # Exponential
        delay = config.base_delay * (config.exponential_base ** attempt)
    
    # Apply max delay cap
    delay = min(delay, config.max_delay)
    
    # Apply jitter
    if config.jitter > 0:
        jitter_amount = delay * config.jitter
        delay = delay + random.uniform(-jitter_amount, jitter_amount)
        delay = max(0.1, delay)  # Ensure positive delay
    
    return delay


def is_retryable_exception(
    exception: Exception,
    config: RetryConfig,
) -> bool:
    """Check if an exception is retryable.
    
    Args:
        exception: The exception that occurred
        config: Retry configuration
        
    Returns:
        True if the exception should be retried
    """
    # Check exception type
    for exc_type in config.retryable_exceptions:
        if isinstance(exception, exc_type):
            return True
    
    # Check for HTTP status codes in common API exception patterns
    status_code = getattr(exception, 'status_code', None)
    if status_code is None:
        # Try common patterns
        status_code = getattr(exception, 'status', None)
        if status_code is None and hasattr(exception, 'response'):
            response = getattr(exception, 'response', None)
            if response:
                status_code = getattr(response, 'status_code', None)
    
    if status_code and status_code in config.retryable_status_codes:
        return True
    
    # Check exception message for common retryable patterns
    error_msg = str(exception).lower()
    retryable_patterns = [
        'rate limit',
        'too many requests',
        'quota exceeded',
        'overloaded',
        'capacity',
        'connection',
        'timeout',
        'temporarily unavailable',
        'service unavailable',
        'server error',
        'bad gateway',
        'network',
        'reset by peer',
        'broken pipe',
    ]
    if any(pattern in error_msg for pattern in retryable_patterns):
        return True
    
    # Check exception class name for common patterns
    exc_name = type(exception).__name__.lower()
    retryable_exc_patterns = [
        'connection',
        'timeout',
        'network',
        'api',
    ]
    if any(pattern in exc_name for pattern in retryable_exc_patterns):
        return True
    
    return False


async def retry_with_backoff(
    func: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs,
) -> Any:
    """Execute a function with retry and exponential backoff.
    
    Args:
        func: Async function to call
        *args: Positional arguments for func
        config: Retry configuration
        circuit_breaker: Optional circuit breaker
        on_retry: Callback called on each retry (attempt, exception, delay)
        **kwargs: Keyword arguments for func
        
    Returns:
        Result of successful function call
        
    Raises:
        Last exception if all retries exhausted
    """
    config = config or RetryConfig()
    last_exception: Optional[Exception] = None
    
    for attempt in range(config.max_retries + 1):
        # Check circuit breaker
        if circuit_breaker:
            if not await circuit_breaker.acquire():
                raise RuntimeError(
                    f"Circuit breaker open: {circuit_breaker.get_stats()}"
                )
        
        try:
            result = await func(*args, **kwargs)
            
            # Record success
            if circuit_breaker:
                await circuit_breaker.record_success()
            
            return result
            
        except Exception as e:
            last_exception = e
            
            # Record failure
            if circuit_breaker:
                await circuit_breaker.record_failure()
            
            # Check if retryable
            if not is_retryable_exception(e, config):
                raise
            
            # Check if more retries available
            if attempt >= config.max_retries:
                raise
            
            # Calculate delay
            delay = calculate_delay(attempt, config)
            
            # Call retry callback
            if on_retry:
                on_retry(attempt, e, delay)
            
            # Wait before retry
            await asyncio.sleep(delay)
    
    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop exited unexpectedly")


def with_retry(
    config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
):
    """Decorator for adding retry behavior to async functions.
    
    Args:
        config: Retry configuration
        circuit_breaker: Optional circuit breaker
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(
                func, *args,
                config=config,
                circuit_breaker=circuit_breaker,
                **kwargs
            )
        return wrapper
    return decorator


# Pre-configured retry profiles
RETRY_PROFILES: Dict[str, RetryConfig] = {
    "aggressive": RetryConfig(
        max_retries=5,
        base_delay=0.5,
        max_delay=30.0,
        jitter=0.3,
    ),
    "conservative": RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.25,
    ),
    "patient": RetryConfig(
        max_retries=7,
        base_delay=2.0,
        max_delay=120.0,
        jitter=0.2,
    ),
}


def get_retry_config(profile: str = "conservative") -> RetryConfig:
    """Get a pre-configured retry config.
    
    Args:
        profile: Profile name (aggressive, conservative, patient)
        
    Returns:
        RetryConfig instance
    """
    return RETRY_PROFILES.get(profile, RETRY_PROFILES["conservative"])

