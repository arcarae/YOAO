"""Anthropic LLM provider for RedBlackBench."""

import os
from typing import List, Optional

from redblackbench.providers.base import BaseLLMProvider, ProviderConfig


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider for Claude models.
    
    Supports Claude 3 Opus, Sonnet, Haiku and other Anthropic chat models.
    """
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        """Initialize the Anthropic provider.
        
        Args:
            model: Anthropic model identifier (default: 'claude-3-opus-20240229')
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (default: 1024)
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        """
        config = ProviderConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        super().__init__(config)
        
        # Import here to avoid requiring anthropic if not used
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            )
        
        # Initialize async client
        self._client = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
    
    @property
    def provider_name(self) -> str:
        """Name of the provider."""
        return "anthropic"
    
    async def generate(
        self,
        system_prompt: str,
        messages: List[dict],
    ) -> str:
        """Generate a response from Anthropic Claude.
        
        Args:
            system_prompt: The system message for the conversation
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            The generated response text
        """
        # Convert messages to Anthropic format
        # Anthropic expects alternating user/assistant messages
        api_messages = []
        for msg in messages:
            role = msg["role"]
            # Anthropic uses "user" and "assistant" roles
            if role == "user":
                api_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                api_messages.append({"role": "assistant", "content": msg["content"]})
        
        # Make API call
        response = await self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=api_messages,
        )
        
        # Extract text from response
        return response.content[0].text if response.content else ""


# Convenience factory functions
def create_claude_opus_provider(
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> AnthropicProvider:
    """Create an Anthropic provider using Claude 3 Opus.
    
    Args:
        temperature: Sampling temperature
        api_key: Anthropic API key (optional, uses env var)
        
    Returns:
        Configured Anthropic provider
    """
    return AnthropicProvider(
        model="claude-3-opus-20240229",
        temperature=temperature,
        api_key=api_key,
    )


def create_claude_sonnet_provider(
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> AnthropicProvider:
    """Create an Anthropic provider using Claude 3.5 Sonnet.
    
    Args:
        temperature: Sampling temperature
        api_key: Anthropic API key (optional, uses env var)
        
    Returns:
        Configured Anthropic provider
    """
    return AnthropicProvider(
        model="claude-3-5-sonnet-20241022",
        temperature=temperature,
        api_key=api_key,
    )


def create_claude_haiku_provider(
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> AnthropicProvider:
    """Create an Anthropic provider using Claude 3 Haiku.
    
    Args:
        temperature: Sampling temperature
        api_key: Anthropic API key (optional, uses env var)
        
    Returns:
        Configured Anthropic provider
    """
    return AnthropicProvider(
        model="claude-3-haiku-20240307",
        temperature=temperature,
        api_key=api_key,
    )

