"""Main entry point for dataset operations."""

import os
import asyncio
from pathlib import Path
from .upload import upload_to_hub
from .generate_dataset import generate_dataset

async def main():
    """Main entry point."""
    # Set up paths
    data_dir = Path("data/friction_reasoning")
    dataset_path = data_dir / "friction_reasoning_dataset.jsonl"
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return
        
    # Upload dataset
    try:
        await upload_to_hub(
            dataset_path=str(dataset_path),
            repo_id="leonvanbokhorst/friction-disagreement-v2",
            description="A dataset of multi-agent reasoning exploring friction and disagreement in human experience.",
            private=False
        )
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 