"""Training script for the friction reasoning model using Unsloth's optimized training."""

import torch
from pathlib import Path
import os
import yaml
from datetime import datetime
from typing import Dict, Any, List
import random
from statistics import mean

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import (
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
)

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent


def print_step(message: str, char: str = "=") -> None:
    """Print a step message with nice formatting."""
    print(f"\n{char * 40}")
    print(f"🚀 {message}")
    print(f"{char * 40}\n")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def format_conversation(examples: Dict[str, Any], config: Dict[str, Any], indices: List[int]) -> Dict[str, str]:
    """Format multiple conversation turns into a single conversation.
    
    Args:
        examples: Dictionary containing multiple examples
        config: Configuration dictionary
        indices: List of indices for this batch
        
    Returns:
        Dictionary with concatenated conversations for the batch
    """
    # Get the relevant columns
    question_col = config["dataset_config"]["columns"]["question"]
    agent_responses_col = config["dataset_config"]["columns"]["agent_responses"]
    final_answer_col = config["dataset_config"]["columns"]["final_answer"]
    
    # Number of examples in dataset
    n_examples = len(examples[question_col])
    
    # Process each batch index
    all_texts = []
    for batch_idx in indices:
        # Build conversation by concatenating 4 turns
        conversation_parts = []
        for i in range(4):  # Always combine 4 turns
            # Get current example index, wrapping around if needed
            current_idx = (batch_idx + i) % n_examples
            
            # Get the components for this turn
            question = examples[question_col][current_idx]
            thought_stream = "\n\n".join(
                agent["thought_stream"]
                for agent in examples[agent_responses_col][current_idx]
            )
            final_answer = examples[final_answer_col][current_idx]
            
            # Format this turn using the template
            # Only include system prompt for first turn
            if i == 0:
                turn = config["dataset_config"]["format_template"].format(
                    question=question,
                    thought_stream=thought_stream,
                    final_answer=final_answer
                )
            else:
                # Skip system prompt for subsequent turns
                turn = (
                    "<|im_start|>user\n"
                    f"{question}\n"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    "<think>\n"
                    f"{thought_stream}\n"
                    "</think>\n\n"
                    f"{final_answer}\n"
                    "<|im_end|>"
                )
            conversation_parts.append(turn)
        
        # Join all turns for this conversation
        text = "".join(conversation_parts)
        all_texts.append(text)
    
    return {"text": all_texts}


def format_dataset_example(example: Dict[str, Any], dataset_name: str, config: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """Format a dataset example based on its source dataset.
    
    Args:
        example: The dataset example to format
        dataset_name: Name of the source dataset
        config: Configuration dictionary
        tokenizer: The tokenizer for length calculation
        
    Returns:
        Dictionary with formatted text and token count
    """
    if "reluctance" in dataset_name:
        # Convert <judging> tags to <think> tags for consistency
        text = example["text"].replace("<judging>", "<think>").replace("</judging>", "</think>")
        return {
            "text": text,
            "token_count": len(tokenizer.encode(text))
        }
    else:
        # Format friction datasets
        question = example[config["dataset_config"]["columns"]["question"]]
        thought_stream = "\n\n".join(
            agent["thought_stream"]
            for agent in example[config["dataset_config"]["columns"]["agent_responses"]]
        )
        final_answer = example[config["dataset_config"]["columns"]["final_answer"]]
        
        formatted_text = config["dataset_config"]["format_template"].format(
            question=question,
            thought_stream=thought_stream,
            final_answer=final_answer
        )
        return {
            "text": formatted_text,
            "token_count": len(tokenizer.encode(formatted_text))
        }


def stack_examples(examples: List[Dict[str, Any]], max_tokens: int = 4096) -> List[str]:
    """Stack examples up to max token limit.
    
    Args:
        examples: List of examples with text and token count
        max_tokens: Maximum number of tokens per stacked example
        
    Returns:
        List of stacked examples
    """
    stacked_examples = []
    current_stack = []
    current_tokens = 0
    
    for example in examples:
        # If adding this example would exceed max tokens, save current stack and start new one
        if current_tokens + example["token_count"] > max_tokens:
            if current_stack:
                stacked_examples.append("".join(current_stack))
                current_stack = []
                current_tokens = 0
        
        # Add example to current stack
        current_stack.append(example["text"])
        current_tokens += example["token_count"]
    
    # Add any remaining examples in the last stack
    if current_stack:
        stacked_examples.append("".join(current_stack))
    
    return stacked_examples


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    print_step("Loading configuration")

    if config_path is None:
        config_path = SCRIPT_DIR / "config.yaml"

    print(f"• Loading config from: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Debug print
    print("\nConfig structure:")
    for key in config:
        print(f"• {key}:")
        if isinstance(config[key], dict):
            for subkey in config[key]:
                print(f"  - {subkey}: {config[key][subkey]}")

    # Convert numeric values
    training_config = config.get("training_config", {})
    training_config["learning_rate"] = float(training_config.get("learning_rate", 2e-4))
    training_config["num_train_epochs"] = int(training_config.get("num_train_epochs", 3))
    training_config["per_device_train_batch_size"] = int(
        training_config.get("per_device_train_batch_size", 4)
    )
    training_config["gradient_accumulation_steps"] = int(
        training_config.get("gradient_accumulation_steps", 4)
    )
    training_config["warmup_ratio"] = float(training_config.get("warmup_ratio", 0.1))

    print("✓ Configuration loaded successfully")
    return config


def setup_model_and_tokenizer(config: Dict[str, Any]):
    """Set up the model and tokenizer."""
    print_step("Setting up model and tokenizer")
    
    print(f"• Loading base model: {config['model_config']['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_config"]["base_model"],
        max_seq_length=config["model_config"]["model_max_length"],
        dtype=config["model_config"]["torch_dtype"],
        load_in_4bit=True,  # Always use 4-bit quantization for efficiency
        device_map=config["model_config"]["device_map"],
        trust_remote_code=config["model_config"]["trust_remote_code"],
    )
    
    # Add LoRA adapters
    print("• Adding LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_config"]["r"],
        target_modules=config["lora_config"]["target_modules"],
        lora_alpha=config["lora_config"]["lora_alpha"],
        lora_dropout=config["lora_config"]["lora_dropout"],
        bias=config["lora_config"]["bias"],
        inference_mode=config["lora_config"]["inference_mode"],
    )
    
    # Set up chat template
    print("• Setting up chat template")
    tokenizer = get_chat_template(tokenizer)
    print("✓ Chat template configured")
    
    return model, tokenizer


def analyze_dataset_lengths(dataset: Dataset, dataset_name: str, tokenizer) -> Dict[str, float]:
    """Analyze token lengths of a dataset.
    
    Args:
        dataset: The dataset to analyze
        dataset_name: Name of the dataset for logging
        tokenizer: Tokenizer for length calculation
        
    Returns:
        Dictionary with length statistics
    """
    lengths = [len(tokenizer.encode(example["text"])) for example in dataset]
    stats = {
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
        "total_tokens": sum(lengths)
    }
    print(f"\n{dataset_name} Statistics:")
    print(f"• Examples: {len(dataset)}")
    print(f"• Min tokens: {stats['min']}")
    print(f"• Max tokens: {stats['max']}")
    print(f"• Mean tokens: {stats['mean']:.1f}")
    print(f"• Total tokens: {stats['total_tokens']:,}")
    return stats


def calculate_token_based_weights(datasets: List[Dataset], dataset_configs: List[Dict], tokenizer) -> List[float]:
    """Calculate dataset weights based on token counts to ensure balanced token distribution.
    
    Args:
        datasets: List of datasets
        dataset_configs: List of dataset configurations
        tokenizer: Tokenizer for length calculation
        
    Returns:
        List of adjusted weights
    """
    print_step("Analyzing dataset token distributions")
    
    # Analyze each dataset
    stats = []
    total_tokens = 0
    for dataset, config in zip(datasets, dataset_configs):
        dataset_stats = analyze_dataset_lengths(dataset, config["name"], tokenizer)
        stats.append(dataset_stats)
        total_tokens += dataset_stats["total_tokens"]
    
    # Calculate target tokens per dataset for balanced distribution
    target_tokens_per_dataset = total_tokens / len(datasets)
    
    # Calculate adjustment factors
    adjustments = [target_tokens_per_dataset / stat["total_tokens"] for stat in stats]
    adjustment_sum = sum(adjustments)
    
    # Normalize to get final weights
    weights = [adj / adjustment_sum for adj in adjustments]
    
    print("\nAdjusted weights for token balance:")
    for config, weight in zip(dataset_configs, weights):
        print(f"• {config['name']}: {weight:.2%}")
    
    return weights


def load_and_prepare_dataset(config: Dict[str, Any], tokenizer) -> Dataset:
    """Load and prepare the dataset for training."""
    print_step("Loading and preparing dataset")

    # Load and combine datasets
    print("• Loading and mixing datasets")
    datasets = []
    
    for dataset_config in config["dataset_config"]["datasets"]:
        print(f"  - Loading {dataset_config['name']}")
        dataset = load_dataset(dataset_config["name"])["train"]
        
        # Format the dataset based on its source
        formatted_dataset = dataset.map(
            lambda x: format_dataset_example(x, dataset_config["name"], config, tokenizer),
            remove_columns=dataset.column_names,
            desc=f"Formatting {dataset_config['name']}"
        )
        
        datasets.append(formatted_dataset)
    
    # Calculate optimal number of examples to take from each dataset
    total_examples = sum(len(d) for d in datasets)
    # Target around 4000 examples total, but no more than 80% of available examples
    target_size = min(6000, int(total_examples * 0.9))
    print(f"\n• Target size: {target_size:,} examples (from total {total_examples:,})")
    
    samples_per_dataset = []
    for dataset, dataset_config in zip(datasets, config["dataset_config"]["datasets"]):
        # Calculate target samples using the pre-calculated weights
        target_samples = min(
            int(target_size * dataset_config["weight"]),
            len(dataset)
        )
        samples_per_dataset.append(target_samples)
    
    # Print sampling plan
    print("\nSampling plan:")
    for config_item, samples in zip(config["dataset_config"]["datasets"], samples_per_dataset):
        print(f"• {config_item['name']}: {samples:,} examples")
    
    # Sample from each dataset
    combined_examples = []
    for dataset, n_samples in zip(datasets, samples_per_dataset):
        # Use the same seed for all shuffling
        sampled = dataset.shuffle(seed=42).select(range(n_samples))
        combined_examples.extend(sampled)
    
    # Shuffle combined examples
    random.seed(42)  # Use consistent seed
    random.shuffle(combined_examples)
    
    # Stack examples up to max token limit
    print("\n• Stacking examples up to token limit")
    print("\nDebug - Config keys:", list(config.keys()))
    print("Debug - Config type:", type(config))
    if "model_config" in config:
        print("Debug - model_config keys:", list(config["model_config"].keys()))
    else:
        print("Debug - model_config not found!")
        print("Debug - Full config:", config)
    
    stacked_texts = stack_examples(
        combined_examples,
        max_tokens=config["model_config"]["model_max_length"]
    )
    
    # Create dataset from stacked examples
    stacked_dataset = Dataset.from_dict({"text": stacked_texts})
    
    # Create train/validation split
    print("\n• Creating train/validation split")
    split_dataset = stacked_dataset.train_test_split(
        test_size=0.1,
        seed=42,  # Use consistent seed
        shuffle=True
    )
    
    dataset_dict = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    # Print final statistics
    print("\nFinal Dataset Statistics:")
    print(f"• Original examples: {len(combined_examples):,}")
    print(f"• Stacked examples: {len(stacked_texts):,}")
    print(f"• Training examples: {len(dataset_dict['train']):,}")
    print(f"• Validation examples: {len(dataset_dict['validation']):,}")
    
    # Print average tokens per stack
    avg_tokens = mean(len(tokenizer.encode(text)) for text in stacked_texts)
    print(f"• Average tokens per stack: {avg_tokens:.1f}")
    
    # Calculate and print total tokens
    total_tokens = sum(len(tokenizer.encode(text)) for text in stacked_texts)
    print(f"• Total tokens in dataset: {total_tokens:,}")

    return dataset_dict


def create_trainer(model, tokenizer, dataset, config: Dict[str, Any]):
    """Create and configure the trainer."""
    print_step("Initializing trainer")

    # Initialize trainer
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        max_seq_length=config["model_config"]["model_max_length"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=config["model_config"]["model_max_length"],
            return_tensors="pt",
            label_pad_token_id=-100,  # Keep -100 as it's used by train_on_responses_only
        ),
        dataset_num_proc=1,  # Changed from 2 to 1 to avoid multiprocessing issues
        packing=False,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=config["training_config"]["per_device_train_batch_size"],
            per_device_eval_batch_size=config["training_config"]["per_device_train_batch_size"],
            gradient_accumulation_steps=config["training_config"]["gradient_accumulation_steps"],
            warmup_ratio=config["training_config"]["warmup_ratio"],
            num_train_epochs=config["training_config"]["num_train_epochs"],
            learning_rate=config["training_config"]["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=config["training_config"]["logging_steps"],
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=config["dataset_config"].get("shuffle_seed", 42),  # Use consistent seed
            output_dir=config["output_config"]["output_dir"],
            logging_dir=config["output_config"]["logging_dir"],
            group_by_length=True,
            max_grad_norm=config["training_config"].get("max_grad_norm", 0.5),
            weight_decay=config["training_config"].get("weight_decay", 0.01),
            remove_unused_columns=True,
            prediction_loss_only=True,
            evaluation_strategy="steps",
            eval_steps=config["training_config"]["eval_steps"],
            save_strategy=config["training_config"]["save_strategy"],
            save_steps=config["training_config"]["save_steps"],
            save_total_limit=config["training_config"]["save_total_limit"],
            load_best_model_at_end=config["training_config"]["load_best_model_at_end"],
            metric_for_best_model=config["training_config"]["metric_for_best_model"],
            greater_is_better=config["training_config"]["greater_is_better"],
            dataloader_num_workers=2, 
            dataloader_pin_memory=True,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config["training_config"]["early_stopping_patience"],
                early_stopping_threshold=config["training_config"]["early_stopping_threshold"],
            )
        ]
    )

    # Apply response-only training after trainer is created
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("✓ Trainer initialized")
    return trainer


def main():
    start_time = datetime.now()
    print_step(f"Starting training process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load configuration
    config = load_config()

    # Create output directories
    print("\n• Creating output directories")
    output_dir = Path(config["output_config"]["output_dir"])
    logging_dir = Path(config["output_config"]["logging_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    print("✓ Output directories created")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(config, tokenizer)

    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset, config)

    # Print GPU stats before training
    print_step("GPU Information")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"• GPU Model: {gpu_stats.name}")
    print(f"• Total GPU Memory: {max_memory} GB")
    print(f"• Initially Reserved Memory: {start_gpu_memory} GB")

    # Start training
    print_step("Starting training")
    trainer_stats = trainer.train()

    # Print final stats
    print_step("Training Complete!")

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    end_time = datetime.now()
    training_duration = end_time - start_time

    print("\nTraining Statistics:")
    print(f"• Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"• End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"• Total duration: {training_duration}")
    print(f"• Training time: {trainer_stats.metrics['train_runtime']} seconds")
    print(f"• Training time (min): {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    
    print("\nMemory Usage:")
    print(f"• Peak reserved memory: {used_memory} GB")
    print(f"• Memory used for training: {used_memory_for_lora} GB")
    print(f"• Peak memory % of total: {used_percentage}%")
    print(f"• Training memory % of total: {lora_percentage}%")

    # Save the model
    print_step("Saving model")
    output_dir = Path(config["output_config"]["output_dir"])
    save_path = output_dir / "lora_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✓ Model saved to: {save_path}")

    print_step("All done! 🎉", char="*")


if __name__ == "__main__":
    main() 