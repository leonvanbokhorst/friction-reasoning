"""Mechanism Explorer agent that investigates physical and embodied aspects of friction."""

from typing import Dict, Optional, List
from .base_agent import BaseAgent
from ..llm.client import LLMClient

class MechanismExplorer(BaseAgent):
    """Agent that explores and traces mechanisms and processes.
    
    Characteristics:
    - Uses visualization markers
    - Traces processes
    - Maps physical understanding
    - Shows technical wonderings
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the Mechanism Explorer agent."""
        super().__init__("mechanism_explorer", llm_client)
        
    async def think(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Generate mechanism-focused thought stream.
        
        Pattern:
        1. Pick up on detail
        2. Physical visualization
        3. Technical questions
        4. Process tracing
        5. New understanding emerges
        
        Args:
            prompt: The question or topic to reason about
            context: Previous agent's response for continuity
            
        Returns:
            Dict containing the agent's response with thought stream and friction points
        """
        return await self._generate_thought_stream(prompt, context)

    def _generate_thought_stream(self, prompt: str, context: Optional[Dict] = None) -> Dict:
        """Generate mechanism-focused thought stream.
        
        Pattern:
        1. Pick up on detail
        2. Physical visualization
        3. Technical questions
        4. Process tracing
        5. New understanding emerges
        
        Args:
            prompt: The question or topic to reason about
            context: Previous agent's response for continuity
            
        Returns:
            Dict containing the agent's response with thought stream and friction points
        """
        # Clear previous state
        self.thought_stream = []
        self.friction_points = []
        
        # Pick up detail from context if available
        if context and "raw_thought_stream" in context.get("thinking_pattern", {}):
            last_thoughts = context["thinking_pattern"]["raw_thought_stream"].split("\n")
            for thought in reversed(last_thoughts):
                if "..." in thought and not thought.startswith("*"):
                    detail = thought.split("...")[-2].strip()
                    pickup = f"{detail}... I need to trace this physically..."
                    self.thought_stream.append(pickup)
                    break
        
        # Physical visualization
        vis_marker = "*mentally constructs a diagram*"
        self.thought_stream.append(vis_marker)
        self.add_friction_point("visualization", vis_marker)
        
        # Technical questioning
        self.thought_stream.append("But how does it actually WORK? Let me break this down...")
        
        # Process tracing with friction
        trace_marker = "*follows the flow with fingers*"
        self.thought_stream.append(trace_marker)
        self.add_friction_point("physical_tracing", trace_marker)
        
        # Hit a snag
        snag_marker = "*encounters resistance*"
        self.thought_stream.append(snag_marker)
        self.add_friction_point("conceptual_snag", snag_marker)
        self.thought_stream.append("Wait... this part doesn't make sense... unless...")
        
        # New understanding
        insight_marker = "*mental model shifts*"
        self.thought_stream.append(insight_marker)
        self.add_friction_point("insight_emergence", insight_marker)
        self.thought_stream.append("Oh! What if it's more like...")
        
        return {
            "thought_stream": "\n".join(self.thought_stream),
            "friction_points": self.friction_points
        } 