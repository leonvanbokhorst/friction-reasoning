"""Base class for reasoning agents with friction points."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..llm.client import LLMClient

class BaseAgent(ABC):
    """Base class for all reasoning agents.
    
    Each agent implements a specific thinking pattern and generates
    friction points as part of their reasoning process.
    """
    
    def __init__(self, name: str, llm_client: Optional[LLMClient] = None):
        """Initialize the agent.
        
        Args:
            name: The name/type of the agent
            llm_client: Optional LLM client for generation (default: creates new client)
        """
        self.name = name
        self.thought_stream: List[str] = []
        self.friction_points: List[Dict] = []
        self.llm_client = llm_client or LLMClient()
    
    @abstractmethod
    async def think(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Generate a stream of consciousness response to the prompt.
        
        Args:
            prompt: The question or topic to reason about
            context: Optional context from previous agent responses
            
        Returns:
            Dict containing the agent's response with thought stream and friction points
        """
        pass
    
    async def _generate_thought_stream(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Generate thought stream using LLM client.
        
        This is separated to make testing easier through mocking.
        """
        response = await self.llm_client.generate_thought_stream(prompt, context)
        
        # Extract thought stream and friction points
        self.thought_stream = response.thought_stream.split("\n")
        self.friction_points = response.friction_points
        
        return self.get_response()
    
    def add_friction_point(self, type_: str, marker: str):
        """Record a friction point in the thinking process.
        
        Args:
            type_: The type of friction (e.g., "pause", "doubt", "shift")
            marker: The text marker where the friction occurred
        """
        self.friction_points.append({
            "type": type_,
            "marker": marker
        })
    
    def get_response(self) -> Dict:
        """Get the complete agent response including thought stream and friction points.
        
        Returns:
            Dict containing the agent's name, thought stream, and friction points
        """
        return {
            "agent_type": self.name,
            "thinking_pattern": {
                "raw_thought_stream": "\n".join(self.thought_stream),
                "friction_moments": self.friction_points
            }
        } 