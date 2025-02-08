"""Synthesizer Agent - Integrates perspectives and finds emergent understanding."""

from typing import Dict, Optional, List
from .base_agent import BaseAgent

class Synthesizer(BaseAgent):
    """Agent that synthesizes multiple perspectives into emergent understanding.
    
    Characteristics:
    - Shows integration
    - Demonstrates emergence
    - Uses sensory synthesis
    - Embraces complexity
    """
    
    def __init__(self):
        """Initialize the Synthesizer agent."""
        super().__init__("synthesizer")
        
    def think(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate synthesizing thought stream.
        
        Pattern:
        1. Gather threads
        2. Watch patterns form
        3. Embodied integration
        4. Emerging insight
        5. Embrace complexity
        
        Args:
            prompt: The question or topic to reason about
            context: Previous agent's response for continuity
            
        Returns:
            The raw thought stream
        """
        # Clear previous state
        self.thought_stream = []
        self.friction_points = []
        
        # Initial gathering
        gather_marker = "*breathes in all the perspectives*"
        self.thought_stream.append(gather_marker)
        self.add_friction_point("perspective_gathering", gather_marker)
        
        # Acknowledge the journey
        if context and "raw_thought_stream" in context.get("thinking_pattern", {}):
            self.thought_stream.append("There's something emerging here... from our initial confusion...")
            
            # Look for key markers in context
            last_thoughts = context["thinking_pattern"]["raw_thought_stream"].split("\n")
            for thought in last_thoughts:
                if "what if" in thought.lower():
                    self.thought_stream.append(f"Through the questions... {thought.strip()}")
                    break
        
        # Pattern watching
        pattern_marker = "*watches thoughts weave together*"
        self.thought_stream.append(pattern_marker)
        self.add_friction_point("pattern_emergence", pattern_marker)
        
        # Resistance to premature closure
        resist_marker = "*feels the urge to conclude too quickly*"
        self.thought_stream.append(resist_marker)
        self.add_friction_point("closure_resistance", resist_marker)
        self.thought_stream.append("No... not yet... there's something deeper here...")
        
        # Embodied understanding
        embody_marker = "*lets understanding settle in the body*"
        self.thought_stream.append(embody_marker)
        self.add_friction_point("embodied_integration", embody_marker)
        
        # Emerging synthesis
        emerge_marker = "*feeling the whole pattern emerge*"
        self.thought_stream.append(emerge_marker)
        self.add_friction_point("synthesis_emergence", emerge_marker)
        self.thought_stream.append("It's not just about... it's about... *feeling it form*...")
        
        # Embrace complexity
        complexity_marker = "*sits in the richness*"
        self.thought_stream.append(complexity_marker)
        self.add_friction_point("complexity_embrace", complexity_marker)
        self.thought_stream.append("And maybe that's exactly what makes this so...")
        
        return "\n".join(self.thought_stream) 