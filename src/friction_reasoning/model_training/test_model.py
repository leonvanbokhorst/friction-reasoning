"""Test script for the trained friction reasoning model."""

import torch
from unsloth import FastLanguageModel
import yaml
from pathlib import Path
import sys
from typing import Iterator
from transformers import TextStreamer

def load_config(config_path: str = None) -> dict:
    """Load configuration from yaml file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_response(model, tokenizer, prompt: str, system_prompt: str = None, stream: bool = False) -> str:
    """Generate a response using the trained model."""
    if system_prompt:
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    else:
        formatted_prompt = ""
        
    formatted_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        if stream:
            streamer = TextStreamer(tokenizer, skip_special_tokens=False)
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                streamer=streamer
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
            
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    try:
        response = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0]
    except:
        response = "Error: Could not parse response"
    return response

def main():
    print("Loading configuration...")
    config = load_config()
    
    print("\nLoading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_config"]["base_model"],
        max_seq_length=config["model_config"]["model_max_length"],
        load_in_4bit=True,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Load the trained adapter
    adapter_path = Path(config["output_config"]["output_dir"]) / "lora_model"
    if adapter_path.exists():
        print(f"Loading trained adapter from {adapter_path}")
        model.load_adapter(adapter_path)
    else:
        raise FileNotFoundError(f"No adapter found at {adapter_path}")
    
    # Set up tokenizer
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Put model in evaluation mode
    model.eval()
    
    # Wrap model for inference
    model = FastLanguageModel.for_inference(model)
    
    # Test prompts that require multi-agent reasoning
    test_prompts = [
        "What are the potential risks and benefits of implementing a universal basic income?",
        "How might artificial general intelligence impact human creativity and artistic expression?",
        "What role should government play in regulating emerging technologies?",
        "How can we balance individual privacy with national security in the digital age?",
        "What are the ethical implications of human genetic engineering?",
    ]
    
    print("\nRunning test prompts (with streaming)...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        print("-" * 80)
        print("Response:")
        _ = generate_response(model, tokenizer, prompt, stream=True)
        print("-" * 80)
    
    print("\nEntering interactive mode (press Ctrl+C to exit)")
    print("Type your questions to test the model's reasoning capabilities:")
    
    try:
        while True:
            prompt = input("\nYour question: ").strip()
            if not prompt:
                break
            print("-" * 80)
            print("Response:")
            _ = generate_response(model, tokenizer, prompt, stream=True)
            print("-" * 80)
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 