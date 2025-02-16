#!/usr/bin/env python3
"""Push GGUF models to HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi

def push_gguf_to_hub(
    hub_repo="leonvanbokhorst/deepseek-r1-mixture-of-friction",
    model_dir="model_gguf"
):
    """Push GGUF models to HuggingFace Hub."""
    # Set up working directory
    work_dir = Path(__file__).parent.absolute()
    os.chdir(work_dir)
    print(f"Working directory: {work_dir}")
    
    model_dir = work_dir / model_dir
    if not model_dir.exists():
        raise ValueError(f"Model directory {model_dir} does not exist!")
    
    quantizations = [
        "q4_k_m",  # Recommended balance of size/speed
        #"q5_k_m",  # Higher quality than q4_k_m
        #"q8_0",    # High resource but high quality
    ]
    
    api = HfApi()
    api.create_repo(hub_repo, exist_ok=True)
    
    for quant in quantizations:
        src_path = model_dir / quant / f"unsloth.{quant.upper()}.gguf"
        if not src_path.exists():
            print(f"Skipping {quant} - file not found at {src_path}")
            continue
            
        # Create a renamed copy for upload
        hub_filename = f"deepseek-r1-mixture-of-friction-{quant}.gguf"
        temp_path = model_dir / quant / hub_filename
        print(f"\nCopying {src_path.name} to {hub_filename}...")
        try:
            import shutil
            shutil.copy2(src_path, temp_path)
            
            print(f"Pushing {quant} version to hub...")
            api.upload_file(
                path_or_fileobj=str(temp_path),
                path_in_repo=hub_filename,
                repo_id=hub_repo,
            )
            print(f"Successfully pushed {quant} version")
            
            # Clean up temp file
            temp_path.unlink()
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            print(f"Failed to push {quant}: {str(e)}")
            continue
    
    print("\nDone! All available GGUF versions have been pushed to hub.")

if __name__ == "__main__":
    push_gguf_to_hub() 