"""Prompt template handling for agent reasoning."""

import os
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

def get_agent_prompt(agent_type: str, variables: Dict[str, str]) -> str:
    """Get a formatted prompt for an agent.
    
    Args:
        agent_type: Type of agent
        variables: Dictionary of variables to insert into template
        
    Returns:
        Formatted prompt string
    """
    template = load_prompt_template(agent_type)
    return template.format(**variables) 