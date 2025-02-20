"""Script to generate a dataset of friction-based reasoning examples."""

import asyncio
import json
import random
import time
import traceback
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from statistics import mean, median
from dotenv import load_dotenv
import argparse

from ..llm import LLMClient
from ..demo import Agent, generate_agent_reasoning, synthesize_final_answer
from ..llm.base_prompts import (
    BASE_PROMPTS,
    EMOTIONS,
    VULNERABILITY_CONFIG,
    DISAGREEMENT_CONFIG
)

# Load environment variables
load_dotenv()

async def generate_question(llm: LLMClient) -> str:
    """Generate a unique, emotionally resonant question."""
    # Pick random category and base prompt
    category = random.choice(list(BASE_PROMPTS.keys()))
    base = random.choice(BASE_PROMPTS[category])
    base2 = random.choice(BASE_PROMPTS[category])
    emotion = random.choice(EMOTIONS)
    emotion2 = random.choice(EMOTIONS)

    if base2 == base:
        base2 = random.choice(BASE_PROMPTS[category])

    if emotion2 == emotion:
        emotion2 = random.choice(EMOTIONS)
    
    prompt = f"""Create a SHORT natural, {emotion} and {emotion2} question or thought that starts similarly to: "{base}".

Rules:
- Must feel deeply personal and emotional
- Use very casual, human-like language
- Make it feel unfinished/uncertain
- Can be a statement that implies a question or a question to the AI
- Sometimes end with ... or similar trailing off
"""
    
    llm.temperature = random.uniform(0.6, 0.8)  # High creativity for variety
    response = await llm.complete(prompt)
    
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

async def inject_vulnerability(response: str) -> str:
    """Occasionally inject expressions of uncertainty or limitation acknowledgment."""
    if random.random() < VULNERABILITY_CONFIG["injection_probability"]:
        prefix = random.choice(
            VULNERABILITY_CONFIG["uncertainty_phrases"] + 
            VULNERABILITY_CONFIG["limitation_acknowledgments"]
        )
        # Insert the vulnerability expression at a natural breaking point
        sentences = response.split(". ")
        insert_point = random.randint(0, len(sentences) - 1)
        sentences.insert(insert_point, prefix)
        return ". ".join(sentences)
    return response

async def generate_datapoint(llm: LLMClient, question: str, stats: DatasetStats) -> Dict:
    """Generate a single datapoint with question and agent responses."""
    try:
        # Create agents
        agents = [
            Agent("problem_framer"),
            Agent("memory_activator"),
            Agent("mechanism_explorer"),  
            Agent("perspective_generator"),
            Agent("limitation_acknowledger"),
            Agent("synthesizer")
        ]
        
        # Collect responses
        agent_responses = []
        previous_thoughts = ""
        
        for agent in agents:
            result = await generate_agent_reasoning(llm, question, agent, previous_thoughts)
            # Inject vulnerability into the thought stream
            result["thinking_pattern"]["raw_thought_stream"] = await inject_vulnerability(
                result["thinking_pattern"]["raw_thought_stream"]
            )
            agent_responses.append(result)
            previous_thoughts = "\n".join(
                resp["thinking_pattern"]["raw_thought_stream"]
                for resp in agent_responses
            )
        
        # Generate final synthesis with vulnerability
        final_answer = await synthesize_final_answer(llm, question, agent_responses)
        final_answer = await inject_vulnerability(final_answer)
        
        # Structure the datapoint
        datapoint = {
            "id": str(int(time.time() * 1000)),  # Unique ID
            "question": question,
            "agent_responses": [
                {
                    "agent_type": resp["agent_type"],
                    "thought_stream": resp["thinking_pattern"]["raw_thought_stream"]
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

async def generate_batch(
    llm: LLMClient, 
    batch_size: int, 
    stats: DatasetStats, 
    seen_ids: set
) -> List[Dict]:
    """Generate a batch of examples concurrently."""
    # First generate all questions concurrently
    question_tasks = [generate_question(llm) for _ in range(batch_size)]
    questions = await asyncio.gather(*question_tasks, return_exceptions=True)
    
    # Filter out any failed question generations
    valid_questions = [q for q in questions if not isinstance(q, Exception)]
    
    # Process examples concurrently
    datapoint_tasks = [
        generate_datapoint(llm, question, stats)
        for question in valid_questions
    ]
    
    # Gather results, filtering out any failed generations
    results = []
    completed = await asyncio.gather(*datapoint_tasks, return_exceptions=True)
    
    for datapoint in completed:
        if isinstance(datapoint, Exception):
            stats.add_error(str(datapoint), -1)
            continue
            
        # Ensure unique ID
        while datapoint["id"] in seen_ids:
            datapoint["id"] = str(int(time.time() * 1000))
        seen_ids.add(datapoint["id"])
        results.append(datapoint)
    
    return results

async def test_generation():
    """Test dataset generation with a single batch."""
    print("\nTesting dataset generation with one batch of 3 examples...")
    try:
        llm = LLMClient(model="openai/gpt-4o-mini", temperature=0.8)  # Using gpt-4o-mini
        stats = DatasetStats()
        
        # Generate test questions
        questions = []
        for _ in range(3):
            question = await generate_question(llm)
            print(f"\nGenerated question: {question}")
            questions.append(question)
        
        # Generate responses for each question
        for i, question in enumerate(questions, 1):
            print(f"\nGenerating responses for question {i}...")
            datapoint = await generate_datapoint(llm, question, stats)
            
            print(f"\nAgent responses:")
            for resp in datapoint["agent_responses"]:
                print(f"\n{resp['agent_type']}:")
                print(f"Thought stream preview: {resp['thought_stream'][:150]}...")
            
            print(f"\nFinal answer preview: {datapoint['final_answer'][:150]}...")
        
        print("\nTest generation complete!")
        stats.print_summary()
        
    except Exception as e:
        print(f"\nError in test generation: {type(e).__name__}: {str(e)}")
        print("\nStacktrace:")
        traceback.print_exc()
        print("\nTest failed.")

async def generate_dataset(num_examples: int = 1200, batch_size: int = 10) -> List[Dict]:
    """Generate the full dataset in batches."""
    llm = LLMClient(model="openai/gpt-4o-mini", temperature=0.7)  # Using gpt-4o-mini
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
                        stats.add_datapoint(datapoint)  # Update stats with existing data
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON in {batch_file}")
        print(f"Loaded {len(dataset)} existing examples")
    
    # If we already have enough examples, return early
    if len(dataset) >= num_examples:
        print("\nTarget number of examples already generated.")
        stats.print_summary()
        return dataset
        
    # Generate in batches with progress bar
    with tqdm(total=num_examples, initial=len(dataset)) as pbar:
        batch_num = next_batch_num
        while len(dataset) < num_examples:
            try:
                current_batch_file = output_dir / f"batch_{batch_num}.jsonl"
                
                # Generate batch concurrently
                batch = await generate_batch(
                    llm, 
                    min(batch_size, num_examples - len(dataset)),
                    stats,
                    seen_ids
                )
                
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
    """Generate the dataset and upload to HuggingFace Hub."""
    parser = argparse.ArgumentParser(description='Generate friction reasoning dataset')
    parser.add_argument('--num_examples', type=int, default=3,
                      help='Number of examples to generate (default: 3 for testing)')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='Batch size for generation (default: 10)')
    parser.add_argument('--test', action='store_true',
                      help='Run in test mode with 3 examples')
    args = parser.parse_args()
    
    if args.test:
        # Run test generation
        await test_generation()
    else:
        # Generate full dataset
        print(f"\nGenerating {args.num_examples} examples in batches of {args.batch_size}...")
        dataset = await generate_dataset(num_examples=args.num_examples, batch_size=args.batch_size)
        print("\nDataset generation complete!")

if __name__ == "__main__":
    asyncio.run(main())

def generate_disagreement_dataset(
    base_prompts: Dict[str, str],
    output_path: Path,
    num_samples: int = 1000
) -> None:
    """Generate dataset focused on disagreement as productive friction.
    
    Args:
        base_prompts: Dictionary of base prompts for each agent
        output_path: Where to save the dataset
        num_samples: Number of samples to generate
    """
    dataset = []
    
    for _ in range(num_samples):
        sample = {
            "metadata": {
                "focus": "disagreement",
                "interaction_type": random.choice(
                    DISAGREEMENT_CONFIG["interaction_patterns"]
                )["type"]
            },
            "agents": []
        }
        
        # Generate agent responses with disagreement focus
        for agent_type, config in DISAGREEMENT_CONFIG["agent_configs"].items():
            response = generate_agent_response(
                agent_type=agent_type,
                base_prompt=base_prompts[agent_type],
                temperature=config["temperature"],
                focus=config["focus"],
                thought_style=config["thought_style"]
            )
            sample["agents"].append({
                "type": agent_type,
                "response": response,
                "config": config
            })
            
        dataset.append(sample)
    
    # Save dataset
    output_path.write_text(json.dumps(dataset, indent=2)) 