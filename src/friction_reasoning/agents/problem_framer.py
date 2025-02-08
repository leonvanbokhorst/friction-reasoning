"""Problem Framer Agent - Initial problem exploration and framing."""

from typing import Dict, Optional
from .base_agent import BaseAgent
from ..llm.client import LLMClient

class ProblemFramer(BaseAgent):
    """Agent that performs initial problem framing and exploration.
    
    Characteristics:
    - Uses verbal markers (Hmm..., wait...)
    - Questions cascade naturally
    - Shows initial reactions
    - Mental noting
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the Problem Framer agent."""
        super().__init__("problem_framer", llm_client)
    
    async def think(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Generate initial problem framing thought stream.
        
        Pattern:
        1. Initial reaction
        2. Word play/breakdown
        3. First questions emerge
        4. Physical/mental shift
        5. Reframing attempt
        
        Args:
            prompt: The question or topic to reason about
            context: Not used by this agent as it's always first
            
        Returns:
            Dict containing the agent's response with thought stream and friction points
        """
        return await self._generate_thought_stream(prompt, context) 