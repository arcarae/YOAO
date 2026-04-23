"""Agent components for RedBlackBench."""

from redblackbench.agents.base import BaseAgent, AgentResponse
from redblackbench.agents.llm_agent import LLMAgent
from redblackbench.agents.prompts import PromptTemplate, DEFAULT_PROMPTS

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "LLMAgent",
    "PromptTemplate",
    "DEFAULT_PROMPTS",
]

