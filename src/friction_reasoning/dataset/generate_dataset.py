"""Script to generate a dataset of friction-based reasoning examples."""

import asyncio
import json
import random
import time
import traceback
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from statistics import mean, median

from ..llm import LLMClient
from ..demo import Agent, generate_agent_reasoning, synthesize_final_answer

# Base prompts that can be combined and modified
BASE_PROMPTS = {
    "emotional": [
        "Why do I feel...",
        "Sometimes when I'm alone...",
        "I can't stop thinking about...",
        "Do you ever get that feeling when...",
        "It's weird but lately...",
    ],
    "philosophical": [
        "What if consciousness is...",
        "Maybe reality isn't...",
        "I keep wondering about time and...",
        "The universe makes me feel...",
        "Ever think about how small we are...",
    ],
    "social": [
        "Why do people always...",
        "Is it normal when friends...",
        "Social media makes me...",
        "Dating these days feels...",
        "Family gatherings are so...",
    ],
    "existential": [
        "Sometimes I question if...",
        "What's the point of...",
        "Life feels like a...",
        "Does anyone else worry about...",
        "I'm scared that maybe...",
    ],
    "memory": [
        "Remember when we used to...",
        "Childhood memories make me...",
        "Looking back at photos...",
        "That song reminds me...",
        "Growing up, I always...",
    ]
}

# Emotional states to influence question generation
EMOTIONS = [
    "nostalgic", "anxious", "curious", "hopeful", "confused",
    "overwhelmed", "peaceful", "restless", "grateful", "uncertain",
    "frustrated", "excited", "sad", "happy", "angry", "bored", "sleepy",
    "hungry", "thirsty", "tired", "sick", "happy", "sad", "angry", "bored",
    "sleepy", "hungry", "thirsty", "tired", "sick"
]

def generate_question(llm: LLMClient) -> str:
    """Generate a unique, emotionally resonant question."""
    # Pick random category and base prompt
    category = random.choice(list(BASE_PROMPTS.keys()))
    base = random.choice(BASE_PROMPTS[category])
    emotion = random.choice(EMOTIONS)
    
    prompt = f"""Create a natural, {emotion} question or thought that starts similarly to: "{base}"

Rules:
- Must feel deeply personal and emotional
- Use very casual, human-like language
- Make it feel unfinished/uncertain
- Maximum 21 words
- Can be a statement that implies a question
- Sometimes end with ... or similar trailing off
Examples:
"Sometimes I look at old photos and just... y'know?"
"Why do people pretend everything's fine when clearly they're not?"
"Been thinking about my childhood room lately and..."
"""
    
    llm.temperature = random.uniform(0.7, 1.0)  # High creativity for variety
    response = llm.complete(prompt)
    
    # Clean up the response
    response = response.strip()
    # Remove surrounding quotes if present
    if (response.startswith('"') and response.endswith('"')) or \
       (response.startswith("'") and response.endswith("'")):
        response = response[1:-1]
    # Replace any escaped quotes
    response = response.replace('\\"', '"').replace("\\'", "'")
    
    return response

class DatasetStats:
    """Track statistics for dataset generation."""
    def __init__(self):
        self.question_lengths = []
        self.response_lengths = {"total": []}
        self.errors = []
        self.categories_used = {}
        self.emotions_used = {}
        self.start_time = time.time()
    
    def add_datapoint(self, datapoint: Dict):
        """Update stats with a new datapoint."""
        # Track question stats
        self.question_lengths.append(len(datapoint["question"].split()))
        
        # Track response lengths
        for resp in datapoint["agent_responses"]:
            agent_type = resp["agent_type"]
            if agent_type not in self.response_lengths:
                self.response_lengths[agent_type] = []
            length = len(resp["thought_stream"].split())
            self.response_lengths[agent_type].append(length)
            self.response_lengths["total"].append(length)
    
    def add_error(self, error: str, batch_num: int):
        """Track an error occurrence."""
        self.errors.append({"batch": batch_num, "error": error, "time": time.strftime("%H:%M:%S")})
    
    def print_summary(self):
        """Print current statistics."""
        runtime = time.time() - self.start_time
        print("\n=== Dataset Generation Statistics ===")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"\nQuestions:")
        print(f"- Average length: {mean(self.question_lengths):.1f} words")
        print(f"- Median length: {median(self.question_lengths):.1f} words")
        print("\nResponses by agent:")
        for agent, lengths in self.response_lengths.items():
            if lengths:  # Check if we have data
                print(f"- {agent}:")
                print(f"  - Average: {mean(lengths):.1f} words")
                print(f"  - Median: {median(lengths):.1f} words")
        if self.errors:
            print(f"\nErrors encountered: {len(self.errors)}")
            for err in self.errors[-3:]:  # Show last 3 errors
                print(f"- Batch {err['batch']}: {err['error']} at {err['time']}")

async def generate_datapoint(llm: LLMClient, question: str, stats: DatasetStats) -> Dict:
    """Generate a single datapoint with question and agent responses."""
    try:
        # Create agents
        agents = [
            Agent("problem_framer"),
            Agent("memory_activator"),
            Agent("mechanism_explorer"),
            Agent("perspective_generator"),
            Agent("synthesizer")
        ]
        
        # Collect responses
        agent_responses = []
        previous_thoughts = ""
        
        for agent in agents:
            result = generate_agent_reasoning(llm, question, agent, previous_thoughts)
            agent_responses.append(result)
            previous_thoughts = "\n".join(
                resp["thinking_pattern"]["raw_thought_stream"]
                for resp in agent_responses
            )
        
        # Generate final synthesis
        final_answer = synthesize_final_answer(llm, question, agent_responses)
        
        # Structure the datapoint
        datapoint = {
            "id": str(int(time.time() * 1000)),  # Unique ID
            "question": question,
            "agent_responses": [
                {
                    "agent_type": resp["agent_type"],
                    "thought_stream": resp["thinking_pattern"]["raw_thought_stream"],
                    "friction_moments": resp["thinking_pattern"]["friction_moments"]
                }
                for resp in agent_responses
            ],
            "final_answer": final_answer,
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": llm.model
            }
        }
        
        # Update statistics
        stats.add_datapoint(datapoint)
        return datapoint
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        stats.add_error(error_msg, -1)  # -1 as batch number not known here
        raise

async def generate_dataset(num_examples: int = 1000, batch_size: int = 3) -> List[Dict]:
    """Generate the full dataset in batches."""
    llm = LLMClient(model="gpt-4o", temperature=0.7)
    dataset = []
    seen_ids = set()  # Track unique IDs
    stats = DatasetStats()
    
    # Create output directory
    output_dir = Path("data/friction_reasoning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the highest existing batch number
    existing_batches = list(output_dir.glob("batch_*.jsonl"))
    next_batch_num = 0
    if existing_batches:
        batch_numbers = [int(f.stem.split('_')[1]) for f in existing_batches]
        next_batch_num = max(batch_numbers) + 1
        print(f"\nResuming from batch {next_batch_num} (found {len(existing_batches)} existing batches)")
        
        # Load existing IDs to prevent duplicates
        for batch_file in existing_batches:
            with open(batch_file, 'r') as f:
                for line in f:
                    try:
                        datapoint = json.loads(line)
                        seen_ids.add(datapoint["id"])
                        dataset.append(datapoint)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON in {batch_file}")
        print(f"Loaded {len(dataset)} existing examples")
    
    # Generate in batches with progress bar
    with tqdm(total=num_examples, initial=len(dataset)) as pbar:
        batch_num = next_batch_num
        while len(dataset) < num_examples:
            try:
                batch = []
                current_batch_file = output_dir / f"batch_{batch_num}.jsonl"
                
                # Generate batch
                for _ in range(min(batch_size, num_examples - len(dataset))):
                    try:
                        # Generate unique question and datapoint
                        question = generate_question(llm)
                        datapoint = await generate_datapoint(llm, question, stats)
                        
                        # Ensure unique ID
                        while datapoint["id"] in seen_ids:
                            datapoint["id"] = str(int(time.time() * 1000))
                        seen_ids.add(datapoint["id"])
                        
                        batch.append(datapoint)
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)}"
                        stats.add_error(error_msg, batch_num)
                        print(f"\nError in example generation: {error_msg}")
                        print("Continuing with next example...")
                        continue
                
                if batch:  # Only save if we have data
                    # Save batch (overwrite instead of append)
                    with open(current_batch_file, "w") as f:
                        for dp in batch:
                            f.write(json.dumps(dp) + "\n")
                    
                    dataset.extend(batch)
                    pbar.update(len(batch))
                    
                    # Print batch statistics every 10 batches
                    if batch_num % 10 == 0:
                        stats.print_summary()
                
                batch_num += 1
                # Optional: Sleep between batches to avoid rate limits
                await asyncio.sleep(1)
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                stats.add_error(error_msg, batch_num)
                print(f"\nError in batch {batch_num}: {error_msg}")
                print("Retrying after 5 seconds...")
                await asyncio.sleep(5)
                continue
    
    # Combine all batches into final dataset
    final_file = output_dir / "friction_reasoning_dataset.jsonl"
    with open(final_file, "w") as f:
        for datapoint in dataset:
            f.write(json.dumps(datapoint) + "\n")
    
    # Print final statistics
    stats.print_summary()
    print(f"\nDataset generated at: {final_file}")
    print(f"Total examples: {len(dataset)}")
    print(f"Unique examples: {len(seen_ids)}")
    print(f"Total batches: {batch_num}")
    return dataset

async def main():
    """Generate the dataset."""
    print("\nGenerating 1000 examples in small batches of 3...")
    try:
        await generate_dataset(num_examples=1000, batch_size=3)
        print("\nDataset generation complete.")
    except Exception as e:
        print(f"\nFatal error in dataset generation: {type(e).__name__}: {str(e)}")
        print("\nStacktrace:")
        traceback.print_exc()
        print("\nDataset generation failed.")

if __name__ == "__main__":
    asyncio.run(main()) 