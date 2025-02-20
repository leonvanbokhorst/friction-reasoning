"""Prompt template handling for agent reasoning."""

from pathlib import Path
from typing import Dict

def load_prompt_template(agent_type: str) -> str:
    """Load a prompt template for a specific agent type.
    
    Args:
        agent_type: Type of agent (e.g., "problem_framer", "memory_activator")
        
    Returns:
        The prompt template string
        
    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_dir = Path(__file__).parent / "templates"
    template_path = template_dir / f"{agent_type}.txt"
    
    if not template_path.exists():
        raise FileNotFoundError(f"No template found for agent type: {agent_type}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

def get_agent_prompt(agent_type: str, question: str, thought_pattern: Dict[str, str], previous_thoughts: str = "", focus: str = "general") -> str:
    """Get the prompt for an agent's stream of consciousness.
    
    Args:
        agent_type: Type of agent
        question: User's question
        thought_pattern: Pattern and style for thinking
        previous_thoughts: Previous agent responses
        focus: Focus area (e.g., "general", "disagreement")
    """
    template_name = f"{focus}_{agent_type}" if focus != "general" else agent_type
    
    try:
        base_prompt = load_prompt_template(template_name)
    except FileNotFoundError:
        # Fallback to general template if specific one not found
        base_prompt = load_prompt_template(agent_type)
        
    formatted_prompt = base_prompt.format(
        question=question,
        context=previous_thoughts,
        pattern=thought_pattern.get("pattern", ""),
        style=thought_pattern.get("style", "")
    )
    
    return formatted_prompt

def get_synthesis_prompt(question: str, agent_responses: str) -> str:
    """Get the prompt for synthesizing agent responses."""
    return f"""Having explored this question through multiple perspectives, but still sitting with uncertainty:

Your previous explorations and thoughts:
{agent_responses}

The user's question:
"{question}"

Generate a SHORT response (3-4 sentences) that:
1. Shows genuine uncertainty about the situation
2. Uses hesitant language ("maybe", "I wonder if", "not sure but")
3. Leaves the question open-ended

Keep it brief and uncertain. Share just one quick moment of reflection."""

def get_direct_prompt(question: str) -> str:
    """Get the prompt for direct answer without friction."""
    return f"""Consider this question with uncertainty and doubt:
"{question}"

Respond with a SHORT, uncertain answer that raises more questions than it answers (1-2 sentences max)."""

def get_final_answer_prompt(question: str, agent_responses: str) -> str:
    """Get the prompt for synthesizing a final answer."""
    return f"""Having explored this question through multiple perspectives, but still feeling uncertain:

Your previous explorations and thoughts:
{agent_responses}

The user's question:
"{question}"

Generate a SHORT response (3-4 sentences) that:
1. Directly addresses the user with empathy ("you know?", "I hear you", "I wonder if you...")
2. Uses hesitant language ("maybe", "I wonder if", "not sure but")
3. Leaves room for multiple possibilities or a vulnerable question

Keep it brief and reflective. Share just one quick moment of connection, speaking directly TO the user."""

def get_disagreement_synthesis_prompt(question: str, agent_responses: str) -> str:
    """Get the prompt for synthesizing disagreement-focused responses."""
    return f"""Having explored this question through multiple perspectives of friction and resistance:

Your previous explorations and thoughts:
{agent_responses}

The user's question:
"{question}"

Synthesize your thoughts into a firm disagreement with the user's question. Respond in first person like you are confronting the user.
"""