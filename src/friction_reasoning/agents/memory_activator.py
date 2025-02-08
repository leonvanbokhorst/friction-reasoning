"""Memory Activator Agent - Surfaces and processes relevant memories."""

from typing import Dict, Optional, List
from .base_agent import BaseAgent

class MemoryActivator(BaseAgent):
    """Agent that surfaces and processes memories related to the topic.
    
    Characteristics:
    - Surfaces memories gradually
    - Shows incomplete recalls
    - Uses active waiting
    - Includes sensory memories
    """
    
    def __init__(self):
        """Initialize the Memory Activator agent."""
        super().__init__("memory_activator")
        
    def think(self, prompt: str, context: Optional[Dict] = None) -> str:
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
            The raw thought stream
        """
        # Clear previous state
        self.thought_stream = []
        self.friction_points = []
        
        # Echo from context if available
        if context and "raw_thought_stream" in context.get("thinking_pattern", {}):
            last_thought = context["thinking_pattern"]["raw_thought_stream"].split("\n")[-1]
            key_phrase = last_thought.split("...")[-1].strip()
            echo = f"{key_phrase}... that's tugging at something..."
            self.thought_stream.append(echo)
        
        # Memory surfacing action
        surface_marker = "*closes eyes, letting memory float up*"
        self.thought_stream.append(surface_marker)
        self.add_friction_point("active_waiting", surface_marker)
        
        # Partial recall with uncertainty
        recall = "There was this... something about... wait..."
        self.thought_stream.append(recall)
        
        # Memory arrangement
        arrange_marker = "*feels memories arranging themselves*"
        self.thought_stream.append(arrange_marker)
        self.add_friction_point("memory_organization", arrange_marker)
        
        # Sensory fragment
        self.thought_stream.append("Like pieces of a puzzle... some clear, some fuzzy...")
        
        return "\n".join(self.thought_stream) 