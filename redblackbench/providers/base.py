"""Base LLM provider interface for RedBlackBench.

Enhanced with:
- Rate limiting
- Response caching  
- Retry with exponential backoff
- Connection pooling configuration
- Circuit breaker for fault tolerance
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from redblackbench.providers.rate_limiter import (
    RateLimiter, 
    RateLimitConfig, 
    get_rate_limiter,
)
from redblackbench.providers.cache import (
    ResponseCache, 
    CacheConfig, 
    get_global_cache,
)
from redblackbench.providers.retry import (
    RetryConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    retry_with_backoff,
    get_retry_config,
)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider.
    
    Attributes:
        model: Model identifier (e.g., 'gpt-4', 'claude-3-opus-20240229')
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in response
        api_key: API key (if not using environment variable)
    """
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: Optional[str] = None


@dataclass
class EnhancedProviderConfig(ProviderConfig):
    """Extended configuration with reliability features.
    
    Attributes:
        enable_rate_limiting: Enable rate limiting
        enable_caching: Enable response caching
        enable_retry: Enable retry with backoff
        enable_circuit_breaker: Enable circuit breaker
        rate_limit_config: Custom rate limit configuration
        cache_config: Custom cache configuration
        retry_config: Custom retry configuration
        circuit_breaker_config: Custom circuit breaker configuration
        connection_pool_size: HTTP connection pool size
        request_timeout: Request timeout in seconds
    """
    enable_rate_limiting: bool = True
    enable_caching: bool = False  # Disabled by default (non-deterministic)
    enable_retry: bool = True
    enable_circuit_breaker: bool = True
    rate_limit_config: Optional[RateLimitConfig] = None
    cache_config: Optional[CacheConfig] = None
    retry_config: Optional[RetryConfig] = None
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    connection_pool_size: int = 100
    request_timeout: float = 120.0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    Defines the interface that all LLM providers must implement.
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g., 'openai', 'anthropic')."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response text
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"


class EnhancedLLMProvider(BaseLLMProvider):
    """Enhanced base provider with rate limiting, caching, retry, and circuit breaker.
    
    Provides a robust foundation for LLM API interactions with:
    - Rate limiting to prevent quota exhaustion
    - Response caching for repeated prompts
    - Retry with exponential backoff for transient failures
    - Circuit breaker for cascading failure prevention
    """
    
    def __init__(self, config: EnhancedProviderConfig):
        """Initialize enhanced provider.
        
        Args:
            config: Enhanced provider configuration
        """
        super().__init__(config)
        self.enhanced_config = config
        
        # Initialize rate limiter
        self._rate_limiter: Optional[RateLimiter] = None
        if config.enable_rate_limiting:
            self._rate_limiter = get_rate_limiter(
                self.provider_name,
                config.rate_limit_config
            )
        
        # Initialize cache
        self._cache: Optional[ResponseCache] = None
        if config.enable_caching:
            cache_config = config.cache_config or CacheConfig()
            self._cache = ResponseCache(cache_config)
        
        # Initialize retry config
        self._retry_config: Optional[RetryConfig] = None
        if config.enable_retry:
            self._retry_config = config.retry_config or get_retry_config("conservative")
        
        # Initialize circuit breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if config.enable_circuit_breaker:
            cb_config = config.circuit_breaker_config or CircuitBreakerConfig()
            self._circuit_breaker = CircuitBreaker(cb_config)
        
        # Metrics
        self._total_requests = 0
        self._cached_responses = 0
        self._failed_requests = 0
        self._total_latency = 0.0
    
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Generate a response with all reliability features.
        
        Args:
            system_prompt: The system message
            messages: List of message dicts
            **kwargs: Additional parameters (e.g., max_tokens, chat_template_kwargs)
            
        Returns:
            Generated response text
        """
        import time
        start_time = time.time()
        self._total_requests += 1
        
        # Check cache first
        if self._cache:
            cached = await self._cache.get_async(
                self.config.model,
                system_prompt,
                messages,
                self.config.temperature,
            )
            if cached:
                self._cached_responses += 1
                return cached
        
        # Acquire rate limit slot
        rate_limit_ctx = None
        if self._rate_limiter:
            # Estimate tokens (rough: ~4 chars per token)
            estimated_tokens = (
                len(system_prompt) + 
                sum(len(m.get("content", "")) for m in messages)
            ) // 4 + self.config.max_tokens
            
            rate_limit_ctx = await self._rate_limiter.acquire(
                estimated_tokens=estimated_tokens,
                timeout=self.enhanced_config.request_timeout,
            )
        
        try:
            # Execute with retry and circuit breaker
            if self._retry_config:
                response = await retry_with_backoff(
                    lambda sp, msgs: self._do_generate(sp, msgs, **kwargs),
                    system_prompt,
                    messages,
                    config=self._retry_config,
                    circuit_breaker=self._circuit_breaker,
                    on_retry=self._on_retry,
                )
            else:
                response = await self._do_generate(system_prompt, messages, **kwargs)
            
            # Cache response
            if self._cache:
                await self._cache.put_async(
                    self.config.model,
                    system_prompt,
                    messages,
                    self.config.temperature,
                    response,
                )
            
            self._total_latency += time.time() - start_time
            return response
            
        except Exception as e:
            self._failed_requests += 1
            raise
        finally:
            if rate_limit_ctx:
                rate_limit_ctx.release()
    
    @abstractmethod
    async def _do_generate(
        self,
        system_prompt: str,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Actual generation implementation (to be overridden).
        
        Args:
            system_prompt: The system message
            messages: List of message dicts
            **kwargs: Additional parameters (e.g., max_tokens, chat_template_kwargs)
            
        Returns:
            Generated response text
        """
        pass
    
    def _on_retry(self, attempt: int, exception: Exception, delay: float):
        """Callback for retry events."""
        print(
            f"[{self.provider_name}] Retry {attempt + 1}: {type(exception).__name__} "
            f"- {str(exception)[:100]}. Waiting {delay:.1f}s..."
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics.
        
        Returns:
            Dict of statistics
        """
        stats = {
            "provider": self.provider_name,
            "model": self.config.model,
            "total_requests": self._total_requests,
            "cached_responses": self._cached_responses,
            "failed_requests": self._failed_requests,
            "cache_hit_rate": (
                self._cached_responses / max(1, self._total_requests)
            ),
            "avg_latency_ms": (
                (self._total_latency / max(1, self._total_requests - self._cached_responses)) * 1000
            ),
        }
        
        if self._rate_limiter:
            stats["rate_limiter"] = self._rate_limiter.get_stats()
        if self._cache:
            stats["cache"] = self._cache.get_stats()
        if self._circuit_breaker:
            stats["circuit_breaker"] = self._circuit_breaker.get_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics."""
        self._total_requests = 0
        self._cached_responses = 0
        self._failed_requests = 0
        self._total_latency = 0.0


def create_http_client_config(
    pool_size: int = 100,
    timeout: float = 120.0,
    keepalive_expiry: float = 30.0,
) -> Dict[str, Any]:
    """Create httpx client configuration for connection pooling.
    
    Args:
        pool_size: Maximum connections in pool
        timeout: Request timeout in seconds
        keepalive_expiry: Keepalive expiry time
        
    Returns:
        Dict of httpx client kwargs
    """
    try:
        import httpx
        
        return {
            "timeout": httpx.Timeout(
                timeout,
                connect=30.0,
                read=timeout,
                write=30.0,
            ),
            "limits": httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size // 2,
                keepalive_expiry=keepalive_expiry,
            ),
        }
    except ImportError:
        # httpx not available, return basic config
        return {"timeout": timeout}

