from typing import Dict, Any, List
from litellm import completion

class Agent:
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.thought_patterns = {
            "problem_framer": {
                "style": "Uses verbal markers, questions cascade naturally, shows initial reactions",
                "pattern": "Initial reaction → Word play/breakdown → First questions emerge → *physical/mental shift* → Reframing attempt"
            },
            "memory_activator": {
                "style": "Surfaces memories gradually, shows incomplete recalls, uses active waiting",
                "pattern": "Echo previous thought → Memory trigger words → *memory surfacing action* → Partial recall → Sensory fragments"
            },
            "mechanism_explorer": {
                "style": "Uses visualization markers, traces processes, maps physical understanding",
                "pattern": "Pick up on detail → *physical visualization* → Technical questions → Process tracing → New understanding"
            },
            "perspective_generator": {
                "style": "Uses pattern interrupts, shows physical reframing, thinks metaphorically",
                "pattern": "*physical perspective shift* → Pattern interrupt → Radical reframe → *sit with impact* → New implications"
            },
            "synthesizer": {
                "style": "Shows integration, demonstrates emergence, uses sensory synthesis",
                "pattern": "Gather threads → Watch patterns form → *embodied integration* → Emerging insight → Embrace complexity"
            },
        }

def get_stream_of_consciousness_prompt(question: str, agent: Agent) -> str:
    return f"""You are a {agent.agent_type}. Give me EXACTLY 3 clear, concise sentences showing one key moment of friction in your thinking.

Question: "{question}"

Your response must:
1. Be EXACTLY 3 complete consecutive sentences
2. Include *asterisks* for physical actions
3. Use ... for pauses
4. Follow this pattern: {agent.thought_patterns[agent.agent_type]["pattern"]}

Style: {agent.thought_patterns[agent.agent_type]["style"]}

CRITICAL: Give me JUST the 3 consecutive sentences. No newlines, no introduction, no commentary, no fragments. Keep each sentence clear and concise."""

def generate_agent_reasoning(question: str, agent: Agent) -> Dict[str, Any]:
    """Generate authentic stream-of-consciousness reasoning."""
    
    prompt = get_stream_of_consciousness_prompt(question, agent)
    
    response = completion(
        model="ollama/llama3.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7  # Slightly lower temperature for more focused responses
    )
    
    return {
        "agent_type": agent.agent_type,
        "thinking_pattern": {
            "raw_thought_stream": response.choices[0].message.content,
            "friction_moments": []  # In a full implementation, we'd parse and identify friction points
        }
    }

def synthesize_final_answer(question: str, agent_responses: List[Dict[str, Any]]) -> str:
    """Synthesize a final answer based on all responses."""
    prompt = f"""Based on the collective thinking about the question:
"{question}"

Their thoughts:
{chr(10).join(f"{resp['agent_type']}: {resp['thinking_pattern']['raw_thought_stream']}" for resp in agent_responses)}

Give me an answer that is a synthesis of the thoughts:
1. Include *asterisks* for physical actions
2. Use ... for pauses
3. Answer in first person, human-like messy like a human would speak
"""

    response = completion(
        model="ollama/llama3.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def generate_direct_answer(question: str) -> str:
    """Generate a direct answer without the friction-based thinking process."""
    prompt = f"""Answer this question:
"{question}"
"""

    response = completion(
        model="ollama/llama3.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def main():
    question = "Write a love letter to my ex boss"
    
    print("\nNow let's explore with friction-based thinking...")
    
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
    
    # Each agent builds on the collective understanding
    for agent in agents:
        print(f"\n{agent.agent_type.replace('_', ' ').title()}:")
        print("-" * 80)
        
        result = generate_agent_reasoning(question, agent)
        print(result["thinking_pattern"]["raw_thought_stream"])
        print("-" * 80)
        
        agent_responses.append(result)
    
    # Generate and display final answer
    print(f"\nQuestion: {question}")
    print("\nFinal Answer (after friction-based thinking):")
    print("-" * 80)
    final_answer = synthesize_final_answer(question, agent_responses)
    print(final_answer)
    print("-" * 80)

    print("\nDirect Answer (without friction-based thinking):")
    print("-" * 80)
    direct_answer = generate_direct_answer(question)
    print(direct_answer)
    print("-" * 80)
    

if __name__ == "__main__":
    main() 