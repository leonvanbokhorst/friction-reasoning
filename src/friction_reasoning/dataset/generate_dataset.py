"""Script to generate a dataset of friction-based reasoning examples."""

import asyncio
import json
import random
import time
import traceback
import os
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from statistics import mean, median
from dotenv import load_dotenv
from huggingface_hub import HfApi
from datasets import Dataset, load_dataset
import argparse

from ..llm import LLMClient
from ..demo import Agent, generate_agent_reasoning, synthesize_final_answer
from .upload import upload_to_hub

# Load environment variables
load_dotenv()

# Base prompts that can be combined and modified
BASE_PROMPTS = {
    "emotional": [
        "Ugh, my heart's doing that thing again where...",
        "Just had one of those moments where everything inside me...",
        "Why does my brain always pick 3am to...",
        "Trying to breathe through this feeling of...",
        "Anyone else get randomly blindsided by...",
        "My therapist would probably say it's because...",
        "Funny how your body tells you things like...",
        "That moment when your emotions just...",
    ],
    "relationships": [
        "Saw their name pop up and my stomach just...",
        "Why do we keep dancing around this thing where...",
        "Text conversations hit different when...",
        "Kinda hurts how some people just casually...",
        "Getting tired of being the friend who always...",
        "Maybe I'm toxic for thinking this but...",
        "The way they looked at me made me wonder if...",
        "Starting to notice a pattern in how I...",
    ],
    "identity": [
        "Okay but who even am I when nobody's...",
        "Getting real with myself about how I might be...",
        "Lowkey freaking out about becoming...",
        "Used to think I was the type of person who...",
        "Trying on different versions of myself like...",
        "Keep catching glimpses of who I could be if...",
        "What if this isn't even my real personality but...",
        "The gap between who I am online and offline...",
    ],
    "existential": [
        "3am thoughts hitting different like...",
        "Do you ever just stop and get scared about...",
        "Having an existential crisis about how we...",
        "My brain broke a little thinking about how...",
        "Kinda terrifying how we're all just...",
        "Getting lost in the void thinking about...",
        "Reality feels glitchy when you realize...",
        "Trying to process how we're all just...",
    ],
    "society": [
        "Is anyone else exhausted by how we're all...",
        "The way social media makes us think we need to...",
        "Getting weird vibes from how everyone's...",
        "Can we talk about this pressure to always...",
        "Why are we all pretending it's normal to...",
        "The algorithm knows me better than my friends and...",
        "Society really got us thinking we gotta...",
        "Living through *gestures at everything* while...",
    ],
    "growth": [
        "Oof, just caught myself doing that thing where...",
        "Past me would be shook seeing me now...",
        "Having to unlearn so much about how I...",
        "Growing pains hit different when you're...",
        "Plot twist: maybe I was the toxic one when...",
        "Healing is weird because suddenly you're...",
        "That moment when you realize you're becoming...",
        "Kind of scary how much I've changed since...",
    ],
    "memory": [
        "Brain randomly decided to replay that time...",
        "A song came on and suddenly I'm back in...",
        "Memory unlocked: that random moment when...",
        "Why do I keep thinking about that one time...",
        "Getting emotional about how we used to...",
        "Found an old photo and now I'm spiraling about...",
        "That core memory just resurfaced where...",
        "Weird how your mind suddenly throws you back to...",
    ],
    "dreams": [
        "Lowkey manifesting a reality where...",
        "Living in my head rent free: that scenario where...",
        "Daydreaming again about how life could be if...",
        "Anyone else build entire futures around...",
        "Stuck between wanting to dream big and...",
        "In my imaginary perfect life I'm always...",
        "Keep fantasizing about dropping everything to...",
        "That alternate timeline where I actually...",
    ]
}

# Emotional states to influence question generation
EMOTIONS = [
    "nostalgic", "anxious", "curious", "hopeful", "confused",
    "overwhelmed", "peaceful", "restless", "grateful", "uncertain",
    "frustrated", "excited", "sad", "happy", "angry", "bored", "sleepy",
    "hungry", "thirsty", "tired", "sick", "happy", "sad", "angry", "bored",
    "sleepy", "hungry", "thirsty", "tired", "sick", 
]

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
            result = await generate_agent_reasoning(llm, question, agent, previous_thoughts)
            agent_responses.append(result)
            previous_thoughts = "\n".join(
                resp["thinking_pattern"]["raw_thought_stream"]
                for resp in agent_responses
            )
        
        # Generate final synthesis
        final_answer = await synthesize_final_answer(llm, question, agent_responses)
        
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

async def generate_dataset(num_examples: int = 1200, batch_size: int = 10) -> List[Dict]:
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

async def test_generation():
    """Test dataset generation with a single batch."""
    print("\nTesting dataset generation with one batch of 3 examples...")
    try:
        dataset = await generate_dataset(num_examples=3, batch_size=3)
        print("\nTest batch generation complete.")
        print(f"Generated {len(dataset)} examples")
        
        # Print a sample example for inspection
        if dataset:
            print("\nSample example:")
            sample = dataset[0]
            print(f"Question: {sample['question']}")
            print("\nAgent responses:")
            for resp in sample['agent_responses']:
                print(f"\n{resp['agent_type']}:")
                print(f"Thought stream: {resp['thought_stream'][:200]}...")
            print(f"\nFinal answer: {sample['final_answer'][:200]}...")
    except Exception as e:
        print(f"\nError in test generation: {type(e).__name__}: {str(e)}")
        print("\nStacktrace:")
        traceback.print_exc()
        print("\nTest failed.")

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