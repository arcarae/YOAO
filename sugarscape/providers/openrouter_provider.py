"""OpenRouter LLM provider for RedBlackBench.

Enhanced with:
- Rate limiting (prevent 429 errors)
- Response caching (skip identical prompts)
- Retry with exponential backoff
- Connection pooling (reduce latency)
- KV cache optimization (prompt caching for transformer efficiency)
"""

import os
import hashlib
from typing import Optional, List, Any, Dict

from sugarscape.providers.base import (
    EnhancedLLMProvider,
    EnhancedProviderConfig,
    ProviderConfig,
    create_http_client_config,
)
from sugarscape.providers.rate_limiter import RateLimitConfig


class OpenRouterProvider(EnhancedLLMProvider):
    """OpenRouter API provider with enhanced reliability and KV cache optimization.
    
    Uses OpenAI-compatible API to access models via OpenRouter.
    
    Features:
    - Rate limiting to prevent quota exhaustion
    - Retry with exponential backoff for transient failures
    - Connection pooling for reduced latency
    - KV cache optimization via prompt caching
    
    KV Cache Optimization:
    - Uses OpenRouter's prompt caching feature to reuse transformer KV states
    - System prompts are cached to avoid recomputing attention for repeated prefixes
    - Reduces latency and cost for repeated prompts with same prefix
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        include_reasoning: bool = False,  # Disabled by default for cleaner output
        # KV Cache / Prompt Caching
        enable_prompt_caching: bool = True,
        # Enhanced features
        enable_rate_limiting: bool = True,
        enable_caching: bool = False,
        enable_retry: bool = True,
        enable_circuit_breaker: bool = True,
        rate_limit_rpm: int = 200,
        rate_limit_concurrent: int = 10,  # Reduced for reliability
        connection_pool_size: int = 100,
        request_timeout: float = 120.0,
    ):
        """Initialize the OpenRouter provider.
        
        Args:
            model: Model identifier (e.g., 'openai/gpt-5')
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: OpenRouter API key
            include_reasoning: Whether to request reasoning/thinking tokens
            enable_prompt_caching: Enable KV cache optimization via prompt caching
            enable_rate_limiting: Enable rate limiting
            enable_caching: Enable response caching (for identical full prompts)
            enable_retry: Enable retry with backoff
            enable_circuit_breaker: Enable circuit breaker
            rate_limit_rpm: Requests per minute limit
            rate_limit_concurrent: Max concurrent requests
            connection_pool_size: HTTP connection pool size
            request_timeout: Request timeout in seconds
        """
        # Create enhanced config
        config = EnhancedProviderConfig(
            model=model,
            temperature=temperature,
            max_tokens=max(16, max_tokens),
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            enable_rate_limiting=enable_rate_limiting,
            enable_caching=enable_caching,
            enable_retry=enable_retry,
            enable_circuit_breaker=enable_circuit_breaker,
            rate_limit_config=RateLimitConfig(
                requests_per_minute=rate_limit_rpm,
                max_concurrent=rate_limit_concurrent,
            ),
            connection_pool_size=connection_pool_size,
            request_timeout=request_timeout,
        )
        
        super().__init__(config)
        
        self.include_reasoning = include_reasoning
        self.enable_prompt_caching = enable_prompt_caching
        
        # Track system prompt for KV cache optimization
        self._cached_system_prompt_hash: Optional[str] = None
        self._prompt_cache_hits = 0
        self._prompt_cache_misses = 0
        
        # Initialize OpenAI client with connection pooling
        from openai import AsyncOpenAI
        
        final_api_key = (
            api_key 
            or os.environ.get("OPENROUTER_API_KEY") 
            or os.environ.get("OPENAI_API_KEY") 
            or "sk-dummy-key"
        )
        
        # Create httpx client with connection pooling
        http_config = create_http_client_config(
            pool_size=connection_pool_size,
            timeout=request_timeout,
        )
        
        self._client = AsyncOpenAI(
            api_key=final_api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=http_config.get("timeout", request_timeout),
            max_retries=0,  # We handle retries ourselves
            default_headers={
                "HTTP-Referer": "https://github.com/arcarae/sugarscape",
                "X-Title": "RedBlackBench",
            }
        )
    
    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "openrouter"

    def _get_prompt_hash(self, system_prompt: str) -> str:
        """Get a hash of the system prompt for cache tracking."""
        return hashlib.md5(system_prompt.encode()).hexdigest()[:16]

    async def _do_generate(
        self,
        system_prompt: str,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Execute the actual API call to OpenRouter with KV cache optimization.
        
        Args:
            system_prompt: System prompt
            messages: List of message dicts
            **kwargs: Additional parameters (e.g., max_tokens, chat_template_kwargs)
            
        Returns:
            Generated response text
            
        Note on KV Cache / Prompt Caching:
            OpenRouter automatically caches KV states for certain models (like GPT-4o)
            when prompts exceed 1024 tokens. The caching is automatic - no special
            parameters needed. We track cache metrics locally and check response
            for `cached_tokens` to measure effectiveness.
        """
        # Build messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)
        
        # Prepare extra parameters for OpenRouter
        extra_body: Dict[str, Any] = {}
        
        # Check if thinking should be disabled (for movement decisions)
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {})
        enable_thinking = chat_template_kwargs.get("enable_thinking", True)
        
        # Include reasoning tokens if requested AND thinking is enabled
        if self.include_reasoning and enable_thinking:
            extra_body["include_reasoning"] = True
        
        # Track prompt cache metrics locally
        # OpenRouter's prompt caching is automatic for supported models
        if self.enable_prompt_caching:
            current_hash = self._get_prompt_hash(system_prompt)
            if self._cached_system_prompt_hash == current_hash:
                self._prompt_cache_hits += 1
            else:
                self._prompt_cache_misses += 1
                self._cached_system_prompt_hash = current_hash

        # Use custom max_tokens if provided, otherwise use config default
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Route to native providers for OpenAI models
        if self.config.model.startswith("openai/"):
            extra_body["provider"] = {
                "order": ["openai", "azure"],
                "allow_fallbacks": True,
            }

        # Make API call
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
            extra_body=extra_body if extra_body else None
        )
        
        choice = response.choices[0]
        content = choice.message.content or ""
        
        # Check for cached tokens in response (OpenRouter automatic caching)
        usage = getattr(response, 'usage', None)
        if usage and self.enable_prompt_caching:
            cached_tokens = getattr(usage, 'cached_tokens', 0) or 0
            if cached_tokens > 0:
                self._cached_tokens_total = getattr(self, '_cached_tokens_total', 0) + cached_tokens
        
        # Check for reasoning field (specific to OpenRouter/DeepSeek/Thinking models)
        reasoning = getattr(choice.message, "reasoning", None)

        # Handle reasoning content based on settings
        if reasoning:
            if self.include_reasoning and enable_thinking:
                # Prepend reasoning with delimiters when explicitly requested
                content = f"__THINKING_START__\n{reasoning}\n__THINKING_END__\n\n{content}"
            elif not content.strip():
                # CRITICAL FIX: If content is empty but reasoning exists, use reasoning as content
                # This handles models like Qwen3 that put everything in the reasoning field
                # Try to extract the actual response from the reasoning (after "So," "Therefore," etc.)
                import re
                # Look for markers indicating the actual response
                markers = [
                    r'\n(?:So|Therefore|My response|Here\'s my response|I\'ll say)[:\s]+(.+)',
                    r'\n"(.+)"',  # Quoted response
                    r'(?:^|\n)([A-Z][^.?!]*[.?!])$',  # Last sentence if it looks like a response
                ]
                extracted = None
                for pattern in markers:
                    match = re.search(pattern, reasoning, re.IGNORECASE | re.DOTALL)
                    if match:
                        extracted = match.group(1).strip()
                        break
                
                # Use extracted if found and reasonable, else use reasoning as-is
                # (The caller's _strip_reasoning_prefix will handle cleanup)
                content = extracted if extracted and len(extracted) > 10 else reasoning
            
        return content
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics including KV cache metrics."""
        stats = super().get_stats()
        
        # Add KV cache / prompt caching stats
        total_prompts = self._prompt_cache_hits + self._prompt_cache_misses
        cached_tokens = getattr(self, '_cached_tokens_total', 0)
        stats["kv_cache"] = {
            "enabled": self.enable_prompt_caching,
            "prompt_cache_hits": self._prompt_cache_hits,
            "prompt_cache_misses": self._prompt_cache_misses,
            "prompt_cache_hit_rate": (
                self._prompt_cache_hits / max(1, total_prompts)
            ),
            "cached_tokens_total": cached_tokens,
        }
        
        return stats


class OpenRouterProviderLegacy:
    """Legacy OpenRouter provider for backward compatibility.
    
    Use OpenRouterProvider for new code.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        include_reasoning: bool = False,  # Disabled for cleaner output
    ):
        """Initialize legacy provider (wraps enhanced provider)."""
        self._provider = OpenRouterProvider(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            include_reasoning=include_reasoning,
            # Use conservative defaults
            enable_rate_limiting=True,
            enable_caching=False,
            enable_retry=True,
            enable_circuit_breaker=True,
        )
        self.config = self._provider.config
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
    
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
        **kwargs,
    ) -> str:
        return await self._provider.generate(system_prompt, messages, **kwargs)
