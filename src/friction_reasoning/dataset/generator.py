"""Dataset generation for friction-based reasoning."""

from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from ..agents import (
    ProblemFramer,
    MemoryActivator,
    MechanismExplorer,
    PerspectiveGenerator,
    Synthesizer
)

class DatasetGenerator:
    """Generates datasets of multi-agent reasoning with friction points."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize dataset generator.
        
        Args:
            output_dir: Directory to save generated data (default: "data")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize agents
        self.agents = {
            "problem_framer": ProblemFramer(),
            "memory_activator": MemoryActivator(),
            "mechanism_explorer": MechanismExplorer(),
            "perspective_generator": PerspectiveGenerator(),
            "synthesizer": Synthesizer()
        }
    
    async def generate_example(self, prompt: str) -> Dict[str, Any]:
        """Generate a single example of multi-agent reasoning.
        
        Args:
            prompt: The question or topic to reason about
            
        Returns:
            Dictionary containing the full reasoning process
        """
        responses = []
        context = None
        
        # Generate responses from each agent in sequence
        for agent_type, agent in self.agents.items():
            response = await agent.think(prompt, context)
            responses.append({
                "agent_type": agent_type,
                "response": response
            })
            context = response
        
        # Structure the example
        return {
            "user_input": prompt,
            "agents": responses,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "0.1.0"
            }
        }
    
    async def generate_dataset(self, prompts: List[str], output_file: str = None) -> str:
        """Generate a dataset from a list of prompts.
        
        Args:
            prompts: List of questions/topics to reason about
            output_file: Optional filename for the dataset
            
        Returns:
            Path to the generated dataset file
        """
        if output_file is None:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = f"friction_reasoning_{timestamp}.jsonl"
            
        output_path = self.output_dir / output_file
        
        # Generate examples
        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in prompts:
                example = await self.generate_example(prompt)
                f.write(json.dumps(example) + "\n")
                
        return str(output_path) 