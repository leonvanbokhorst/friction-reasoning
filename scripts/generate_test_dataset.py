#!/usr/bin/env python3
"""Test script for dataset generation."""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from friction_reasoning.dataset.generator import DatasetGenerator

# Test prompts (subset of the full dataset)
TEST_PROMPTS = [
    "Are AI agents a threat to humanity?",
    "Write a love letter to my ex boss",
    "Should we colonize Mars, or focus on saving Earth first?"
]

async def main():
    """Generate a test dataset."""
    # Initialize generator
    generator = DatasetGenerator(output_dir="data/test")
    
    # Generate dataset
    output_file = "friction_reasoning_test.jsonl"
    dataset_path = await generator.generate_dataset(TEST_PROMPTS, output_file)
    
    print(f"\nTest dataset generated at: {dataset_path}")
    print(f"Number of examples: {len(TEST_PROMPTS)}")

if __name__ == "__main__":
    asyncio.run(main()) 