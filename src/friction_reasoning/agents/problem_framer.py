"""Problem Framer Agent - Initial problem exploration and framing."""

from typing import Dict, Optional
from .base_agent import BaseAgent

class ProblemFramer(BaseAgent):
    """Agent that performs initial problem framing and exploration.
    
    Characteristics:
    - Uses verbal markers (Hmm..., wait...)
    - Questions cascade naturally
    - Shows initial reactions
    - Mental noting
    """
    
    def __init__(self):
        """Initialize the Problem Framer agent."""
        super().__init__("problem_framer")
    
    def think(self, prompt: str, context: Optional[Dict] = None) -> str:
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
            The raw thought stream
        """
        # Clear previous state
        self.thought_stream = []
        self.friction_points = []
        
        # Initial reaction with word play
        initial_reaction = f"{prompt}... *feels the words resonate*... "
        self.thought_stream.append(initial_reaction)
        
        # Add natural pause friction
        self.add_friction_point("natural_pause", "*feels the words resonate*")
        
        # Word breakdown and questioning
        words = prompt.split()
        key_word = words[-1]  # Often the main concept is at the end
        word_play = f"When I think '{key_word}' I think... "
        self.thought_stream.append(word_play)
        
        # Mental shift and reframing
        shift_marker = "*shifts mental position*"
        self.thought_stream.append(shift_marker)
        self.add_friction_point("perspective_shift", shift_marker)
        
        # Final reframing
        reframe = "Wait... maybe we need to look at this differently..."
        self.thought_stream.append(reframe)
        
        return "\n".join(self.thought_stream) 