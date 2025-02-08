"""Prompt template handling for agent reasoning."""

import os
from pathlib import Path
from typing import Dict, Any

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

def get_agent_prompt(agent_type: str, question: str, thought_pattern: Dict[str, str], previous_thoughts: str = "") -> str:
    """Get the prompt for an agent's stream of consciousness."""
    transition_examples = [
        # Natural thought evolution
        "And that makes me think...", "Which leads to...", "Building on that...",
        "But, wait,", "Going deeper...", "Stepping back...",
        
        # Internal dialogue shifts
        "But inside, I feel...", "Something nags at me...", "My mind wanders to...",
        "Can't shake the feeling...", "There's this sense that...", "Deep down..."
    ]

    if agent_type == "problem_framer":
        base_prompt = f"""You are beginning to explore a human's question. First, acknowledge receiving their question and frame how you'll think about it.

The human asks: "{question}"

IMPORTANT:
- Start by acknowledging the question (e.g., "Hmm, this human wants to understand..." or "Well, they're asking about...")
- Focus on INITIAL FRAMING only - identify key tensions and areas to explore
- Avoid making conclusions - your role is to open up the exploration
- Respond with ONE SHORTCONCISE PARAGRAPH that frames the exploration
- You don't have to finish the paragraph, just start it"""

    elif agent_type == "memory_activator":
        base_prompt = f"""You are continuing a deep exploration where the current focus is {agent_type}-style thinking.

The thought process is exploring: "{question}"

IMPORTANT: 
- Start with a memory-surfacing phrase like:
  "I remember..."
  "That makes me think of..."
  "I almost forgot..."
  "This reminds me of the time..."
  "Something similar happened when..."
  "A memory surfaces..."
- Focus on SURFACING NEW MEMORIES that haven't been mentioned yet
- Connect these memories to reveal fresh insights, not repeat existing ones
- Respond with ONE SHORT CONCISE PARAGRAPH of internal reflection
- You don't have to finish the paragraph, just start it"""

    elif agent_type == "mechanism_explorer":
        base_prompt = f"""You are continuing a deep exploration where the current focus is {agent_type}-style thinking.

The thought process is exploring: "{question}"

IMPORTANT: 
- Focus on HOW THINGS WORK - explore mechanisms and processes not yet discussed
- Trace cause-and-effect chains that others haven't explored
- Add technical or systematic understanding that's missing from previous thoughts
- Respond with ONE SHORT CONCISE PARAGRAPH of internal reflection
- You don't have to finish the paragraph, just start it"""

    elif agent_type == "perspective_generator":
        base_prompt = f"""You are continuing a deep exploration where the current focus is {agent_type}-style thinking.

The thought process is exploring: "{question}"

IMPORTANT: 
- Start with a perspective-shifting phrase like:
  "But if we look at it from a distance..."
  "Another perspective could be..."
  "On the other hand..."
  "Stepping back to see the bigger picture..."
  "If we flip this around..."
- Introduce COMPLETELY NEW ANGLES not yet considered
- Challenge or reframe previous perspectives rather than reinforcing them
- Respond with ONE SHORT CONCISE PARAGRAPH of internal reflection
- You don't have to finish the paragraph, just start it"""

    elif agent_type == "synthesizer":
        base_prompt = f"""You are bringing together all perspectives explored so far, preparing to help the human understand.

The human's question was: "{question}"

IMPORTANT:
- Start by acknowledging the collective insights (e.g., "Taking all this in..." or "Gathering these threads...")
- Focus on WEAVING NEW CONNECTIONS between different perspectives
- Don't just summarize - reveal emergent insights from combining viewpoints
- End by considering how to share this understanding with the human
- Respond with ONE SHORT CONCISE PARAGRAPH"""

    elif agent_type == "question_generator":
        base_prompt = f"""You are imagining what human question or statement could have led to this input: "{question}"

IMPORTANT:
- Create a natural, human-like question or statement that could have prompted this input
- Make it personal and emotionally connected
- Use casual language, filler words, and natural speech patterns
- Show context and background of why they're asking
- Keep it SHORT and CONCISE (1-2 sentences max)
- Examples:
  "Hey, so I've been helping my kid with math homework and..."
  "Ugh, my brain is fried... quick sanity check..."
  "Y'know what's been bugging me lately..."
  "Ok this might sound dumb but..."
  "I keep thinking about this and..."
- End with ... to show it's a natural trailing off

Respond with ONE SHORT, NATURAL question or statement that feels real and human."""

    else:
        base_prompt = f"""You are continuing a deep exploration where the current focus is {agent_type}-style thinking.

The thought process is exploring: "{question}"

IMPORTANT: 
- Keep responses SHORT and PUNCHY (2-3 sentences max)
- Let thoughts trail off with ... when natural
- Focus on ONE strong insight or reaction
- Use your personality but be brief
- It's better to be incomplete than too long

Your response must:
1. Flow naturally by:
   - Being conversational and messy (use "um", "like", "y'know")
   - Using quick transitions like: {', '.join(transition_examples[:3])}
   - Letting thoughts wander but staying focused

2. Keep it super concise:
   - Maximum 2-3 sentences
   - Trail off with ... when natural
   - Focus on one key insight
   - Let it feel unfinished
   - Better too short than too long

3. Follow this progression briefly: {thought_pattern["pattern"]}
4. Maintain this style concisely: {thought_pattern["style"]}"""

    if previous_thoughts:
        base_prompt += f"\n\nThe thinking so far:\n{previous_thoughts}"
        base_prompt += "\n\nJump in naturally with a SHORT reaction - be real but brief."
    
    base_prompt += "\n\nShare your reflection in ONE SHORT paragraph. No polishing. Raw and brief."
    
    return base_prompt

def get_synthesis_prompt(question: str, agent_responses: str) -> str:
    """Get the prompt for synthesizing agent responses."""
    return f"""Having explored this human's question through multiple perspectives:
"{question}"

These are YOUR DEEP INNER THOUGHTS about the question:
{agent_responses}

Write a casual, brief response like talking to a friend:

Remember:
- Keep each paragraph to 1-3 sentences max
- Use super casual language (like, um, y'know)
- Trail off naturally with ...
- Let emotions peek through
- Be messy
- One insight per paragraph
- Adapt to the user's style to create rapport

Speak from the heart, but keep it SHORT - like a quick conversation with a friend:"""

def get_direct_prompt(question: str) -> str:
    """Get the prompt for direct answer without friction."""
    return f"""Answer this question with empathy and support:
"{question}""" 