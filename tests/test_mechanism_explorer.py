"""Tests for the Mechanism Explorer agent."""

from friction_reasoning.agents.mechanism_explorer import MechanismExplorer
from friction_reasoning.agents.memory_activator import MemoryActivator

def test_mechanism_explorer_basic():
    """Test basic functionality of Mechanism Explorer."""
    agent = MechanismExplorer()
    response = agent.think("How do trees communicate?")
    
    # Check that response contains key elements
    assert "*mentally constructs a diagram*" in response
    assert "how does it actually WORK?" in response
    assert "*follows the flow with fingers*" in response
    assert "*encounters resistance*" in response
    assert "*mental model shifts*" in response

def test_mechanism_explorer_with_context():
    """Test Mechanism Explorer with context from Memory Activator."""
    # First get context from Memory Activator
    memory = MemoryActivator()
    memory.think("What is time?")  # Need to think first to generate response
    memory_response = memory.get_response()
    
    # Then use it with Mechanism Explorer
    explorer = MechanismExplorer()
    response = explorer.think("What is time?", context=memory_response)
    
    # Check that response builds on context
    assert "trace this physically" in response
    
def test_mechanism_explorer_friction_points():
    """Test that friction points are properly recorded."""
    agent = MechanismExplorer()
    agent.think("What is time?")
    
    # Get complete response with friction points
    response = agent.get_response()
    
    # Check response structure
    assert "agent_type" in response
    assert response["agent_type"] == "mechanism_explorer"
    assert "thinking_pattern" in response
    assert "friction_moments" in response["thinking_pattern"]
    
    # Check friction points
    friction_points = response["thinking_pattern"]["friction_moments"]
    assert len(friction_points) >= 4  # Should have visualization, tracing, snag, and insight
    
    # Check specific friction points
    vis_point = next(p for p in friction_points if p["type"] == "visualization")
    assert "*mentally constructs a diagram*" in vis_point["marker"]
    
    trace_point = next(p for p in friction_points if p["type"] == "physical_tracing")
    assert "*follows the flow with fingers*" in trace_point["marker"]
    
    snag_point = next(p for p in friction_points if p["type"] == "conceptual_snag")
    assert "*encounters resistance*" in snag_point["marker"]
    
    insight_point = next(p for p in friction_points if p["type"] == "insight_emergence")
    assert "*mental model shifts*" in insight_point["marker"] 