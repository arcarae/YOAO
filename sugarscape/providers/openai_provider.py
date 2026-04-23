"""OpenAI LLM provider for RedBlackBench."""

import os
from typing import List, Optional

from sugarscape.providers.base import BaseLLMProvider, ProviderConfig


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider for GPT models.
    
    Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, and other OpenAI chat models.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            model: OpenAI model identifier (default: 'gpt-4')
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 512, reduced from 1024 to prevent context overflow)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            base_url: Base URL for API endpoint (for vLLM or other OpenAI-compatible servers)
        """
        config = ProviderConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        super().__init__(config)

        # Import here to avoid requiring openai if not used
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            )

        # Initialize async client
        # Use a dummy key if none provided to allow instantiation (calls will fail)
        # For vLLM, any dummy key works since it doesn't validate API keys
        final_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"

        # Create client with base_url if provided (for vLLM)
        client_kwargs = {"api_key": final_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**client_kwargs)
    
    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "openai"
    
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Generate a response from OpenAI.
        
        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response text
        """
        # Build messages list with system prompt
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(messages)
        
        # Make API call
        response = await self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.choices[0].message.content or ""


# Convenience factory functions
def create_gpt4_provider(
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> OpenAIProvider:
    """Create an OpenAI provider using GPT-4.
    
    Args:
        temperature: Sampling temperature
        api_key: OpenAI API key (optional, uses env var)
        
    Returns:
        Configured OpenAI provider
    """
    return OpenAIProvider(
        model="gpt-4",
        temperature=temperature,
        api_key=api_key,
    )


def create_gpt4_turbo_provider(
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> OpenAIProvider:
    """Create an OpenAI provider using GPT-4 Turbo.
    
    Args:
        temperature: Sampling temperature
        api_key: OpenAI API key (optional, uses env var)
        
    Returns:
        Configured OpenAI provider
    """
    return OpenAIProvider(
        model="gpt-4-turbo",
        temperature=temperature,
        api_key=api_key,
    )

