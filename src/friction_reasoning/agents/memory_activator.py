"""Memory Activator Agent - Surfaces and processes relevant memories."""

from typing import Dict, Optional, List
from .base_agent import BaseAgent
from ..llm.client import LLMClient

class MemoryActivator(BaseAgent):
    """Agent that surfaces and processes memories related to the topic.
    
    Characteristics:
    - Surfaces memories gradually
    - Shows incomplete recalls
    - Uses active waiting
    - Includes sensory memories
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the Memory Activator agent."""
        super().__init__("memory_activator", llm_client)
        
    async def think(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Generate memory-based thought stream.
        
        Pattern:
        1. Echo previous thought
        2. Memory trigger words
        3. Memory surfacing action
        4. Partial recall
        5. Sensory fragments
        
        Args:
            prompt: The question or topic to reason about
            context: Previous agent's response for continuity
            
        Returns:
            Dict containing the agent's response with thought stream and friction points
        """
        return await self._generate_thought_stream(prompt, context)