"""vLLM LLM provider for RedBlackBench.

Provides integration with vLLM inference servers through OpenAI-compatible API.
"""

import asyncio
import os
from typing import List, Optional

from redblackbench.providers.base import BaseLLMProvider, ProviderConfig


class VLLMProvider(BaseLLMProvider):
    """vLLM API provider for local model serving.

    Uses vLLM's OpenAI-compatible API to serve models locally with high throughput.
    Supports features like PagedAttention, continuous batching, and prefix caching.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        """Initialize the vLLM provider.

        Args:
            model: Model identifier (model name or path on the vLLM server)
            base_url: Base URL for vLLM server (default: 'http://localhost:8000/v1')
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 1024)
            api_key: API key (vLLM doesn't require one, defaults to 'EMPTY')
        """
        config = ProviderConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or "EMPTY",
        )
        super().__init__(config)

        self.base_url = base_url

        # Import here to avoid requiring openai if not used
        try:
            from openai import AsyncOpenAI
            self._AsyncOpenAI = AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for VLLMProvider. "
                "Install it with: pip install openai"
            )

        # Shared client for connection reuse and proper batching
        # Lazily initialized on first generate() call
        self._client = None
        self._owns_client = True  # Track if we created the client

    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "vllm"

    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
        chat_template_kwargs: Optional[dict] = None,
    ) -> str:
        """Generate a response from vLLM server.

        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys
            chat_template_kwargs: Optional kwargs for chat template (e.g., {"enable_thinking": False} for Qwen3)

        Returns:
            The generated response text
        """
        # Build messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)

        # Estimate input tokens and adjust max_tokens to stay within context limit
        total_chars = len(system_prompt) + sum(len(m.get("content", "")) for m in messages)
        estimated_input_tokens = total_chars // 3  # ~3 chars per token for English

        context_limit = 32768  # Qwen3-14B supports 32k context
        available_tokens = context_limit - estimated_input_tokens - 100  # buffer
        effective_max_tokens = min(self.config.max_tokens, max(256, available_tokens))

        # Build extra_body for vLLM-specific parameters
        extra_body = {}
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        # Retry with client recreation on connection errors
        max_retries = 3
        for attempt in range(max_retries):
            if self._client is None:
                import httpx
                self._client = self._AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.base_url,
                    timeout=httpx.Timeout(timeout=600.0, connect=60.0),
                )
            try:
                response = await self._client.chat.completions.create(
                    model=self.config.model,
                    messages=api_messages,
                    temperature=self.config.temperature,
                    max_tokens=effective_max_tokens,
                    extra_body=extra_body if extra_body else None,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_name = type(e).__name__
                is_connection_err = any(k in err_name for k in ("Connect", "Transport", "Timeout", "Remote"))
                if is_connection_err and attempt < max_retries - 1:
                    # Destroy broken client, back off, retry with fresh connection
                    try:
                        await self._client.close()
                    except Exception:
                        pass
                    self._client = None
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                raise

    async def aclose(self) -> None:
        """Async close the underlying HTTP client.

        Call this when done with the provider to properly clean up connections.
        """
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    def close(self) -> None:
        """Synchronous close - schedules async cleanup.

        For use when an event loop is available but we're in sync context.
        """
        if self._client is not None and self._owns_client:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup for later
                    loop.create_task(self.aclose())
                elif not loop.is_closed():
                    loop.run_until_complete(self.aclose())
                else:
                    # Loop is closed, force cleanup
                    self._client = None
            except RuntimeError:
                # No event loop available, just clear the reference
                self._client = None


# Convenience factory functions
def create_vllm_qwen3_14b_provider(
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    model_path: Optional[str] = None,
) -> VLLMProvider:
    """Create a vLLM provider for the unmodified Qwen3-14B base model.

    Args:
        base_url: vLLM server base URL (default: $YOAO_VLLM_URL or http://localhost:8000/v1)
        temperature: Sampling temperature
        model_path: Served model name (default: $YOAO_BASE_MODEL or Qwen/Qwen3-14B)

    Returns:
        Configured vLLM provider
    """
    from redblackbench import defaults

    return VLLMProvider(
        model=model_path or defaults.BASE_MODEL,
        base_url=base_url or defaults.VLLM_URL,
        temperature=temperature,
    )


def create_vllm_lora_provider(
    lora_adapter: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
) -> VLLMProvider:
    """Create a vLLM provider for the cooperative LoRA adapter.

    Args:
        lora_adapter: Adapter name on the server (default: $YOAO_SFT_MODEL or
            redblackbench-qwen3-14b-sft-v2)
        base_url: vLLM server base URL
        temperature: Sampling temperature

    Returns:
        Configured vLLM provider
    """
    from redblackbench import defaults

    return VLLMProvider(
        model=lora_adapter or defaults.SFT_MODEL,
        base_url=base_url or defaults.VLLM_URL,
        temperature=temperature,
    )
