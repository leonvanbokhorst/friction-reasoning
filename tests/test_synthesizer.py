"""Tests for the Synthesizer agent."""

from friction_reasoning.agents.synthesizer import Synthesizer
from friction_reasoning.agents.perspective_generator import PerspectiveGenerator

def test_synthesizer_basic():
    """Test basic functionality of Synthesizer."""
    agent = Synthesizer()
    response = agent.think("How do trees communicate?")
    
    # Check that response contains key elements
    assert "*breathes in all the perspectives*" in response
    assert "*watches thoughts weave together*" in response
    assert "*feels the urge to conclude too quickly*" in response
    assert "*lets understanding settle in the body*" in response
    assert "*feeling the whole pattern emerge*" in response
    assert "*sits in the richness*" in response

def test_synthesizer_with_context():
    """Test Synthesizer with context from Perspective Generator."""
    # First get context from Perspective Generator
    generator = PerspectiveGenerator()
    generator.think("What is time?")  # Need to think first to generate response
    generator_response = generator.get_response()
    
    # Then use it with Synthesizer
    synthesizer = Synthesizer()
    response = synthesizer.think("What is time?", context=generator_response)
    
    # Check that response builds on context
    assert "There's something emerging here" in response
    assert "Through the questions" in response
    
def test_synthesizer_friction_points():
    """Test that friction points are properly recorded."""
    agent = Synthesizer()
    agent.think("What is time?")
    
    # Get complete response with friction points
    response = agent.get_response()
    
    # Check response structure
    assert "agent_type" in response
    assert response["agent_type"] == "synthesizer"
    assert "thinking_pattern" in response
    assert "friction_moments" in response["thinking_pattern"]
    
    # Check friction points
    friction_points = response["thinking_pattern"]["friction_moments"]
    assert len(friction_points) >= 6  # Should have gathering, pattern, closure, embodied, synthesis, complexity
    
    # Check specific friction points
    gather_point = next(p for p in friction_points if p["type"] == "perspective_gathering")
    assert "*breathes in all the perspectives*" in gather_point["marker"]
    
    pattern_point = next(p for p in friction_points if p["type"] == "pattern_emergence")
    assert "*watches thoughts weave together*" in pattern_point["marker"]
    
    closure_point = next(p for p in friction_points if p["type"] == "closure_resistance")
    assert "*feels the urge to conclude too quickly*" in closure_point["marker"]
    
    embody_point = next(p for p in friction_points if p["type"] == "embodied_integration")
    assert "*lets understanding settle in the body*" in embody_point["marker"]
    
    synthesis_point = next(p for p in friction_points if p["type"] == "synthesis_emergence")
    assert "*feeling the whole pattern emerge*" in synthesis_point["marker"]
    
    complexity_point = next(p for p in friction_points if p["type"] == "complexity_embrace")
    assert "*sits in the richness*" in complexity_point["marker"] 