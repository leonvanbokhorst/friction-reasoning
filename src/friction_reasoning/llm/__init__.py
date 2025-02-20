"""LLM integration package."""

from .client import LLMClient
from .prompts import get_agent_prompt, get_synthesis_prompt, get_direct_prompt

__all__ = ['LLMClient', 'get_agent_prompt', 'get_synthesis_prompt', 'get_direct_prompt'] 