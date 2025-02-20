#!/usr/bin/env python3
"""Push LoRA model to HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi

def push_lora_to_hub(
    hub_repo="leonvanbokhorst/deepseek-r1-mixture-of-friction",
    model_dir="models/friction_reasoning/lora_model",
    model_card="MODEL_CARD.md"
):
    """Push LoRA model to HuggingFace Hub."""
    # Set up working directory
    work_dir = Path(__file__).parent.absolute()
    os.chdir(work_dir)
    print(f"Working directory: {work_dir}")
    
    # Check model directory
    model_dir = work_dir.parent.parent.parent / model_dir
    if not model_dir.exists():
        raise ValueError(f"Model directory {model_dir} does not exist!")
    
    # Check model card
    model_card = work_dir.parent.parent.parent / model_card
    if not model_card.exists():
        raise ValueError(f"Model card {model_card} does not exist!")
    
    print(f"Model directory: {model_dir}")
    print(f"Model card: {model_card}")
    
    # Create API client
    api = HfApi()
    
    # Create or get repo
    print(f"\nCreating/accessing repository: {hub_repo}")
    api.create_repo(hub_repo, exist_ok=True)
    
    # Upload model card
    print("\nUploading model card...")
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=hub_repo,
    )
    
    # Upload all files from model directory
    print("\nUploading model files...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=hub_repo,
        ignore_patterns=[".*", "__pycache__", "*.pyc", "*.pyo", "*.pyd"],
    )
    
    print("\nDone! Model and model card have been pushed to hub.")

if __name__ == "__main__":
    push_lora_to_hub() 