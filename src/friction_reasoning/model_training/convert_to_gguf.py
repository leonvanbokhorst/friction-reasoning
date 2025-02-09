#!/usr/bin/env python3
"""Merge LoRA weights and export to GGUF format using Unsloth."""

import os
import subprocess
from unsloth import FastLanguageModel
from peft import PeftConfig
from pathlib import Path

def setup_llama_cpp():
    """Set up llama.cpp with correct binary names."""
    work_dir = Path.cwd()
    llama_cpp_dir = work_dir / "llama.cpp"
    
    if not llama_cpp_dir.exists():
        print("\nCloning llama.cpp...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    
    print("\nBuilding llama.cpp...")
    os.chdir(llama_cpp_dir)
    subprocess.run(["mkdir", "-p", "build"], check=True)
    os.chdir("build")
    subprocess.run(["cmake", ".."], check=True)
    subprocess.run(["cmake", "--build", ".", "--config", "Release"], check=True)
    os.chdir(work_dir)
    
    # Create symlinks for Unsloth compatibility
    if (llama_cpp_dir / "build/bin/quantize").exists():
        if not (llama_cpp_dir / "quantize").exists():
            os.symlink("build/bin/quantize", llama_cpp_dir / "quantize")
    if (llama_cpp_dir / "build/bin/llama-quantize").exists():
        if not (llama_cpp_dir / "llama-quantize").exists():
            os.symlink("build/bin/llama-quantize", llama_cpp_dir / "llama-quantize")

def merge_lora_weights(
    base_model="unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit",
    lora_path="lora_model",  # Relative to work_dir
):
    """Merge LoRA weights and convert to GGUF format."""
    # Set up working directory
    work_dir = Path(__file__).parent.absolute()
    os.chdir(work_dir)
    print(f"Working directory: {work_dir}")
    
    # Set up llama.cpp first
    setup_llama_cpp()
    
    # Convert path to absolute
    lora_path = "/home/lonn/repo/neural-notebook-ai/output/lora_model"
    output_dir = work_dir / "Deepseek-R1-distill-Qwen-7B-Friction"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\nLoading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,
        dtype="bfloat16",  # Keep bfloat16 as model was trained with it
        load_in_4bit=True,  # Keep 4-bit as trained
        device_map="auto",
    )
    
    print("\nLoading LoRA weights...")
    config = PeftConfig.from_pretrained(lora_path)
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # LoRA rank: Determines the size of the trainable adapters
        target_modules=[  # List of transformer layers where LoRA adapters will be applied
            "q_proj",   # Query projection in self-attention
            "k_proj",   # Key projection in self-attention
            "v_proj",   # Value projection in self-attention
            "o_proj",   # Output projection from attention
            "gate_proj",  # Used in feed-forward layers (MLP)
            "up_proj",    # Part of transformer's feed-forward network
            "down_proj",  # Another part of transformer's FFN
        ],
        lora_alpha=128,  # Scaling factor for LoRA updates
    )

    model.load_adapter(lora_path, adapter_name="default")
    
    print("\nMerging weights and saving to GGUF...")
    # Save directly to GGUF formats
    output_dir = work_dir / "Deepseek-R1-distill-Qwen-7B-Friction"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    quantizations = [
        "q4_k_m",  # Recommended balance of size/speed
        "q5_k_m",  # Higher quality than q4_k_m
        "q8_0",    # High resource but high quality
    ]
    
    for quant in quantizations:
        print(f"\nCreating {quant} version...")
        try:
            output_path = output_dir / quant
            output_path.mkdir(exist_ok=True)
            model.save_pretrained_gguf(
                str(output_path),
                tokenizer,
                quantization_method=quant
            )
            # Rename the output file to match Unsloth's naming
            gguf_path = output_path / "model.gguf"
            new_path = output_path / f"unsloth.{quant.upper()}.gguf"
            if gguf_path.exists():
                gguf_path.rename(new_path)
            print(f"Saved {quant} version to {new_path}")
        except Exception as e:
            print(f"Failed to create {quant}: {str(e)}")
            # Print more debug info
            print("\nChecking llama.cpp binaries:")
            llama_cpp_dir = work_dir / "llama.cpp"
            print(f"llama-quantize exists: {(llama_cpp_dir / 'llama-quantize').exists()}")
            print(f"quantize exists: {(llama_cpp_dir / 'quantize').exists()}")
            print(f"build/bin/quantize exists: {(llama_cpp_dir / 'build/bin/quantize').exists()}")
            print(f"build/bin/llama-quantize exists: {(llama_cpp_dir / 'build/bin/llama-quantize').exists()}")
            continue
    
    print("\nDone! GGUF versions have been created.")

if __name__ == "__main__":
    merge_lora_weights() 