"""Tests for the Memory Activator agent."""

from friction_reasoning.agents.memory_activator import MemoryActivator
from friction_reasoning.agents.problem_framer import ProblemFramer

def test_memory_activator_basic():
    """Test basic functionality of Memory Activator."""
    agent = MemoryActivator()
    response = agent.think("How do trees communicate?")
    
    # Check that response contains key elements
    assert "*closes eyes, letting memory float up*" in response
    assert "memories arranging themselves" in response
    assert "pieces of a puzzle" in response

def test_memory_activator_with_context():
    """Test Memory Activator with context from Problem Framer."""
    # First get context from Problem Framer
    framer = ProblemFramer()
    framer_response = framer.get_response()
    
    # Then use it with Memory Activator
    activator = MemoryActivator()
    response = activator.think("What is time?", context=framer_response)
    
    # Check that response builds on context
    assert "tugging at something" in response
    
def test_memory_activator_friction_points():
    """Test that friction points are properly recorded."""
    agent = MemoryActivator()
    agent.think("What is time?")
    
    # Get complete response with friction points
    response = agent.get_response()
    
    # Check response structure
    assert "agent_type" in response
    assert response["agent_type"] == "memory_activator"
    assert "thinking_pattern" in response
    assert "friction_moments" in response["thinking_pattern"]
    
    # Check friction points
    friction_points = response["thinking_pattern"]["friction_moments"]
    assert len(friction_points) >= 2  # Should have at least active_waiting and memory_organization
    
    # Check specific friction points
    waiting_point = next(p for p in friction_points if p["type"] == "active_waiting")
    assert "*closes eyes, letting memory float up*" in waiting_point["marker"]
    
    org_point = next(p for p in friction_points if p["type"] == "memory_organization")
    assert "memories arranging themselves" in org_point["marker"] 