"""Main entry point for dataset operations."""

import os
import asyncio
import argparse
from pathlib import Path
from .upload import upload_to_hub
from .generate_dataset import generate_dataset, test_generation

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate and/or upload friction reasoning dataset')
    parser.add_argument('--test', action='store_true',
                      help='Run in test mode with 3 examples')
    parser.add_argument('--num_examples', type=int, default=1200,
                      help='Number of examples to generate (default: 1200)')
    parser.add_argument('--upload', action='store_true',
                      help='Upload dataset to HuggingFace Hub after generation')
    args = parser.parse_args()
    
    # Set up paths
    data_dir = Path("data/friction_reasoning")
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "friction_reasoning_dataset.jsonl"
    
    if args.test:
        print("\nRunning test generation...")
        await test_generation()
        return
        
    print(f"\nGenerating {args.num_examples} examples...")
    dataset = await generate_dataset(num_examples=args.num_examples)
    
    if args.upload:
        print("\nUploading dataset to HuggingFace Hub...")
        try:
            await upload_to_hub(
                dataset_path=str(dataset_path),
                repo_id="leonvanbokhorst/friction-disagreement-v2",
                description="A dataset of multi-agent reasoning exploring friction and disagreement in human experience.",
                private=False
            )
            print("Upload complete!")
        except Exception as e:
            print(f"Error uploading dataset: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(main()) 