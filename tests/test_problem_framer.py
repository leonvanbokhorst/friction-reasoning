"""Tests for the Problem Framer agent."""

from friction_reasoning.agents.problem_framer import ProblemFramer

def test_problem_framer_basic():
    """Test basic functionality of Problem Framer."""
    agent = ProblemFramer()
    response = agent.think("How do trees communicate?")
    
    # Check that response contains key elements
    assert "How do trees communicate?" in response
    assert "*feels the words resonate*" in response
    assert "communicate" in response
    assert "*shifts mental position*" in response
    
def test_problem_framer_friction_points():
    """Test that friction points are properly recorded."""
    agent = ProblemFramer()
    agent.think("What is time?")
    
    # Get complete response with friction points
    response = agent.get_response()
    
    # Check response structure
    assert "agent_type" in response
    assert response["agent_type"] == "problem_framer"
    assert "thinking_pattern" in response
    assert "raw_thought_stream" in response["thinking_pattern"]
    assert "friction_moments" in response["thinking_pattern"]
    
    # Check friction points
    friction_points = response["thinking_pattern"]["friction_moments"]
    assert len(friction_points) > 0
    
    # Check specific friction point
    pause_point = next(p for p in friction_points if p["type"] == "natural_pause")
    assert pause_point["marker"] == "*feels the words resonate*" 