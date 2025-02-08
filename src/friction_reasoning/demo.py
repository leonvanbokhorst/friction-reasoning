"""Demo script for friction-based reasoning."""

import random
from typing import Dict, Any, List
from friction_reasoning.llm import LLMClient, get_agent_prompt, get_synthesis_prompt

class Agent:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.thought_patterns = {
            "problem_framer": {
                "style": "Skeptical overthinker, questions everything, bit anxious, uses sarcasm",
                "pattern": "Initial doubt → Sarcastic observation → *nervous fidget* → Overthinking spiral → Reluctant insight"
            },
            "memory_activator": {
                "style": "Emotional, dramatic, interrupts self, gets carried away with details",
                "pattern": "Random tangent → Emotional trigger → *physical reaction* → Oversharing → Sudden connection"
            },
            "mechanism_explorer": {
                "style": "Blunt, technical, slightly condescending, gets lost in details",
                "pattern": "Technical nitpick → *dismissive gesture* → Actually-well-technically → Process obsession → Grudging insight"
            },
            "perspective_generator": {
                "style": "Contrarian, challenges assumptions, bit aggressive, plays devil's advocate",
                "pattern": "*confrontational stance* → Challenge norms → Push buttons → *intense stare* → Provocative reframe"
            },
            "synthesizer": {
                "style": "Chaotic connector, jumps between ideas, gets excited, sometimes judgmental",
                "pattern": "Scattered gathering → Random connection → *excited bounce* → Messy insight → Opinionated conclusion"
            }
        }

def generate_agent_reasoning(llm: LLMClient, question: str, agent: Agent, previous_thoughts: str = "") -> Dict[str, Any]:
    """Generate authentic stream-of-consciousness reasoning."""
    prompt = get_agent_prompt(
        agent.agent_type, 
        question, 
        agent.thought_patterns[agent.agent_type],
        previous_thoughts
    )
    llm.temperature = random.uniform(0.7, 1.0)
    response = llm.complete(prompt)
    
    return {
        "agent_type": agent.agent_type,
        "thinking_pattern": {
            "raw_thought_stream": response,
            "friction_moments": []  # In a full implementation, we'd parse and identify friction points
        }
    }

def synthesize_final_answer(llm: LLMClient, question: str, agent_responses: List[Dict[str, Any]]) -> str:
    """Synthesize a final answer based on all responses."""
    thoughts = chr(10).join(f"{resp['thinking_pattern']['raw_thought_stream']}" 
                           for resp in agent_responses)
    prompt = get_synthesis_prompt(question, thoughts)
    return llm.complete(prompt)

def generate_random_question(llm: LLMClient) -> str:
    """Generate a random human-like question."""
    prompts = [
        "Why do people...",
        "What if we could...",
        "I keep wondering about...",
        "Sometimes I think about...",
        "Is it weird that...",
        "How come whenever...",
        "Do you ever feel like...",
        "I can't stop thinking about...",
    ]
    base = random.choice(prompts)
    prompt = f"""Create a natural human question that starts with something like "{base}" but different.
    
    Rules:
    - Must be personal and emotionally resonant
    - End with ... or something similar
    - Use casual very human-like language
    - Make it feel unfinished/trailing off or like a question that was cut off
    - Or ponder a question that's not really a question
    
    Examples:
    "Why do people get so weird about money... (but not about love?)"
    "Is it weird that I love the smell of rain, and the sound of rain?
    "Sometimes I think about my childhood bedroom and then... I'm not sure what I think about it. What do you think?"
    """
    
    llm.temperature = 1.0  # High creativity for question generation
    return llm.complete(prompt)

def main():
    # Initialize LLM client
    llm = LLMClient(model="gpt-4o", temperature=0.7)
    
    # Generate a random human-like question
    question = generate_random_question(llm)
    print(f"\nExploring human question:")
    print("-" * 40)
    print(question)
    print("-" * 40)
    
    # Create agents in the correct sequence for building understanding
    agents = [
        Agent("problem_framer"),
        Agent("memory_activator"),
        Agent("mechanism_explorer"),
        Agent("perspective_generator"),
        Agent("synthesizer")
    ]
    
    # Collect all agent responses
    agent_responses = []
    previous_thoughts = ""
    
    print("\nThinking process:")
    print("-" * 80)
    
    # Process all agents
    for agent in agents:
        print(f"\n{agent.agent_type.replace('_', ' ').title()}:")
        print("-" * 40)
        
        result = generate_agent_reasoning(llm, question, agent, previous_thoughts)
        print(result["thinking_pattern"]["raw_thought_stream"])
        
        agent_responses.append(result)
        # Update previous thoughts for next agent - without including agent type prefix
        previous_thoughts = "\n".join(
            resp["thinking_pattern"]["raw_thought_stream"]
            for resp in agent_responses
        )
    
    # Generate and display final answer
    print("\nFinal Answer:")
    print("-" * 80)
    final_answer = synthesize_final_answer(llm, question, agent_responses)
    print(final_answer)
    print("-" * 80)

if __name__ == "__main__":
    import sys
    import os
    # Add the src directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    main() 