#!/usr/bin/env python3
"""Merge LoRA weights and prepare for vLLM serving."""

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from huggingface_hub import login
import yaml
from pathlib import Path
import os
from peft import PeftModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def load_config(config_path: str = None) -> dict:
    """Load configuration from yaml file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def merge_and_push():
    """Merge LoRA weights and push to hub."""
    print("Loading configuration...")
    config = load_config()
    
    # Check for HF token
    token = os.getenv("HUGGINGFACE_API_KEY")
    if not token:
        print("\nSkipping - HUGGINGFACE_API_KEY not found in .env")
        print("Please add your token to .env file as:")
        print("HUGGINGFACE_API_KEY=your_token_here")
        return
    
    print("\nLogging into Hugging Face...")
    login(token=token)
    
    print("\nLoading base model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_config"]["base_model"],
        max_seq_length=config["model_config"]["model_max_length"],
        load_in_4bit=True,  # Keep 4-bit quantization from Unsloth
        trust_remote_code=False,
        device_map="auto",
    )
    
    # Load the trained adapter
    adapter_path = Path(config["output_config"]["output_dir"]) / "lora_model"
    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_path}")
    
    print(f"\nLoading trained adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge weights
    print("\nMerging weights...")
    merged_model = model.merge_and_unload()
    
    # Push directly to hub
    hub_name = "leonvanbokhorst/deepseek-r1-mixture-of-friction-4bit"
    print(f"\nPushing merged model to {hub_name}...")
    merged_model.push_to_hub(hub_name, use_auth_token=token)
    tokenizer.push_to_hub(hub_name, use_auth_token=token)
    
    print(f"\nModel uploaded to: https://huggingface.co/{hub_name}")
    print("\nTo use with vLLM:")
    print(f"""
    docker run --gpus all -p 8000:8000 --shm-size 1g \\
        ghcr.io/vllm-project/vllm:latest \\
        --host 0.0.0.0 \\
        --model {hub_name}
    """)

def main():
    print("Starting model merge and push process...")
    merge_and_push()
    print("\nProcess complete! 🎉")

if __name__ == "__main__":
    main() 