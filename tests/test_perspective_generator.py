"""Tests for the Perspective Generator agent."""

from friction_reasoning.agents.perspective_generator import PerspectiveGenerator
from friction_reasoning.agents.mechanism_explorer import MechanismExplorer

def test_perspective_generator_basic():
    """Test basic functionality of Perspective Generator."""
    agent = PerspectiveGenerator()
    response = agent.think("How do trees communicate?")
    
    # Check that response contains key elements
    assert "*steps back from the mental workspace*" in response
    assert "what if we've got this completely backwards?" in response
    assert "*mental kaleidoscope shift*" in response
    assert "*feels worldview wobble*" in response
    assert "*watches ripples spread*" in response
    assert "*feels mental vertigo*" in response

def test_perspective_generator_with_context():
    """Test Perspective Generator with context from Mechanism Explorer."""
    # First get context from Mechanism Explorer
    explorer = MechanismExplorer()
    explorer.think("What is time?")  # Need to think first to generate response
    explorer_response = explorer.get_response()
    
    # Then use it with Perspective Generator
    generator = PerspectiveGenerator()
    response = generator.think("What is time?", context=explorer_response)
    
    # Check that response builds on context
    assert "That makes me wonder... instead of" in response
    
def test_perspective_generator_friction_points():
    """Test that friction points are properly recorded."""
    agent = PerspectiveGenerator()
    agent.think("What is time?")
    
    # Get complete response with friction points
    response = agent.get_response()
    
    # Check response structure
    assert "agent_type" in response
    assert response["agent_type"] == "perspective_generator"
    assert "thinking_pattern" in response
    assert "friction_moments" in response["thinking_pattern"]
    
    # Check friction points
    friction_points = response["thinking_pattern"]["friction_moments"]
    assert len(friction_points) >= 5  # Should have physical, perspective, cognitive, implication, vertigo
    
    # Check specific friction points
    physical_point = next(p for p in friction_points if p["type"] == "physical_shift")
    assert "*steps back from the mental workspace*" in physical_point["marker"]
    
    perspective_point = next(p for p in friction_points if p["type"] == "perspective_shift")
    assert "*mental kaleidoscope shift*" in perspective_point["marker"]
    
    cognitive_point = next(p for p in friction_points if p["type"] == "cognitive_dissonance")
    assert "*feels worldview wobble*" in cognitive_point["marker"]
    
    implication_point = next(p for p in friction_points if p["type"] == "implication_emergence")
    assert "*watches ripples spread*" in implication_point["marker"]
    
    vertigo_point = next(p for p in friction_points if p["type"] == "perspective_vertigo")
    assert "*feels mental vertigo*" in vertigo_point["marker"] 