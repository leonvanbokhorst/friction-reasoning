"""Perspective Generator Agent - Shifts viewpoints and challenges assumptions."""

from typing import Dict, Optional, List
from .base_agent import BaseAgent

class PerspectiveGenerator(BaseAgent):
    """Agent that generates alternative perspectives and challenges assumptions.
    
    Characteristics:
    - Uses pattern interrupts
    - Shows physical reframing
    - Thinks metaphorically
    - Challenges assumptions
    """
    
    def __init__(self):
        """Initialize the Perspective Generator agent."""
        super().__init__("perspective_generator")
        
    def think(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate perspective-shifting thought stream.
        
        Pattern:
        1. Physical perspective shift
        2. Pattern interrupt
        3. Radical reframe
        4. Sit with impact
        5. New implications emerge
        
        Args:
            prompt: The question or topic to reason about
            context: Previous agent's response for continuity
            
        Returns:
            The raw thought stream
        """
        # Clear previous state
        self.thought_stream = []
        self.friction_points = []
        
        # Initial physical shift
        shift_marker = "*steps back from the mental workspace*"
        self.thought_stream.append(shift_marker)
        self.add_friction_point("physical_shift", shift_marker)
        
        # Pattern interrupt
        self.thought_stream.append("Hold on... hold on... what if we've got this completely backwards?")
        
        # Pick up insight from context if available
        if context and "raw_thought_stream" in context.get("thinking_pattern", {}):
            last_thoughts = context["thinking_pattern"]["raw_thought_stream"].split("\n")
            for thought in reversed(last_thoughts):
                if "Oh!" in thought or "unless..." in thought:
                    insight = thought.replace("Oh!", "").replace("unless...", "").strip()
                    challenge = f"That makes me wonder... instead of {insight}, what if..."
                    self.thought_stream.append(challenge)
                    break
        
        # Radical reframe with metaphor
        metaphor_marker = "*mental kaleidoscope shift*"
        self.thought_stream.append(metaphor_marker)
        self.add_friction_point("perspective_shift", metaphor_marker)
        self.thought_stream.append("It's like we're trying to... no, wait... what if it's more like...")
        
        # Sit with discomfort
        discomfort_marker = "*feels worldview wobble*"
        self.thought_stream.append(discomfort_marker)
        self.add_friction_point("cognitive_dissonance", discomfort_marker)
        
        # Let implications surface
        surface_marker = "*watches ripples spread*"
        self.thought_stream.append(surface_marker)
        self.add_friction_point("implication_emergence", surface_marker)
        
        # Final vertigo moment
        vertigo_marker = "*feels mental vertigo*"
        self.thought_stream.append(vertigo_marker)
        self.add_friction_point("perspective_vertigo", vertigo_marker)
        self.thought_stream.append("And if that's true, then everything we thought about this might be...")
        
        return "\n".join(self.thought_stream) 