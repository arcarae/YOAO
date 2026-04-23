"""AIHubMix LLM provider for RedBlackBench.

AIHubMix provides access to various models through an OpenAI-compatible API.
Supports models like kimi-k2-thinking, qwen3-vl-30b-a3b-thinking, and others.

Enhanced with rate limiting, retry, and connection pooling for reliability.
"""

import os
from typing import List, Optional, Dict, Any

from redblackbench.providers.base import (
    EnhancedLLMProvider,
    EnhancedProviderConfig,
    create_http_client_config,
)
from redblackbench.providers.rate_limiter import RateLimitConfig


class AIHubMixProvider(EnhancedLLMProvider):
    """AIHubMix API provider with enhanced reliability features.

    Uses OpenAI-compatible API to access models through AIHubMix.
    Includes rate limiting, retry with backoff, and connection pooling.

    Supported models include:
    - kimi-k2-thinking: Kimi k2 with thinking/reasoning capabilities
    - qwen3-vl-30b-a3b-thinking: Qwen3 VL with vision and thinking capabilities
    - And other models available through AIHubMix
    """

    def __init__(
        self,
        model: str = "kimi-k2-thinking",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        # Enhanced features
        enable_rate_limiting: bool = True,
        enable_caching: bool = False,
        enable_retry: bool = True,
        enable_circuit_breaker: bool = True,
        rate_limit_rpm: int = 200,
        rate_limit_concurrent: int = 10,
        connection_pool_size: int = 100,
        request_timeout: float = 120.0,
    ):
        """Initialize the AIHubMix provider.

        Args:
            model: Model identifier (default: 'kimi-k2-thinking')
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 1024)
            api_key: AIHubMix API key (uses AIHUBMIX_API_KEY env var if not provided)
            enable_rate_limiting: Enable rate limiting
            enable_caching: Enable response caching
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
            api_key=api_key or os.environ.get("AIHUBMIX_API_KEY"),
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

        # Import here to avoid requiring openai if not used
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for AIHubMixProvider. "
                "Install it with: pip install openai"
            )

        # Initialize async client with AIHubMix base URL
        final_api_key = (
            api_key
            or os.environ.get("AIHUBMIX_API_KEY")
            or "sk-dummy-key"
        )
        
        # Create httpx client with connection pooling
        http_config = create_http_client_config(
            pool_size=connection_pool_size,
            timeout=request_timeout,
        )

        self._client = AsyncOpenAI(
            api_key=final_api_key,
            base_url="https://aihubmix.com/v1",
            timeout=http_config.get("timeout", request_timeout),
            max_retries=0,  # We handle retries ourselves
        )

    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "aihubmix"

    async def _do_generate(
        self,
        system_prompt: str,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Execute the actual API call to AIHubMix.

        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters (e.g., max_tokens, chat_template_kwargs)

        Returns:
            The generated response text
        """
        # Build messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        # Allow per-call max_tokens override
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # Make API call
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""


# Convenience factory functions for common models
def create_kimi_k2_thinking_provider(
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> AIHubMixProvider:
    """Create an AIHubMix provider using Kimi k2 thinking model.

    Args:
        temperature: Sampling temperature
        api_key: AIHubMix API key (optional, uses env var)

    Returns:
        Configured AIHubMix provider
    """
    return AIHubMixProvider(
        model="kimi-k2-thinking",
        temperature=temperature,
        api_key=api_key,
    )


def create_qwen3_vl_thinking_provider(
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> AIHubMixProvider:
    """Create an AIHubMix provider using Qwen3 VL thinking model.

    Args:
        temperature: Sampling temperature
        api_key: AIHubMix API key (optional, uses env var)

    Returns:
        Configured AIHubMix provider
    """
    return AIHubMixProvider(
        model="qwen3-vl-30b-a3b-thinking",
        temperature=temperature,
        api_key=api_key,
    )
