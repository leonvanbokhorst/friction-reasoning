"""Training script for the friction reasoning model using Unsloth's optimized training."""

import torch
from pathlib import Path
import os
import yaml
from datetime import datetime
from typing import Dict, Any, List

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import load_dataset, Dataset, DatasetDict
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


def format_conversation(example: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, str]:
    """Format a single conversation example according to the template."""
    # Concatenate all agent thought streams with double newlines
    thought_stream = "\n\n".join(
        agent["thought_stream"] 
        for agent in example[config["dataset_config"]["columns"]["agent_responses"]]
    )
    
    # Format the conversation using the template
    text = config["dataset_config"]["format_template"].format(
        question=example[config["dataset_config"]["columns"]["question"]],
        thought_stream=thought_stream,
        final_answer=example[config["dataset_config"]["columns"]["final_answer"]]
    )
    
    return {"text": text}


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


def load_and_prepare_dataset(config: Dict[str, Any], tokenizer) -> Dataset:
    """Load and prepare the dataset for training."""
    print_step("Loading and preparing dataset")

    # Load dataset from HuggingFace
    print(f"• Loading dataset: {config['dataset_config']['dataset_name']}")
    dataset = load_dataset(config["dataset_config"]["dataset_name"])
    
    # Format conversations
    print("• Formatting conversations")
    formatted_dataset = dataset.map(
        lambda x: format_conversation(x, config),
        remove_columns=dataset["train"].column_names,  # Remove original columns
        desc="Formatting conversations"
    )

    # Shuffle dataset with fixed seed for reproducibility
    print("\n• Shuffling dataset")
    shuffle_seed = config["dataset_config"].get("shuffle_seed", 3407)
    formatted_dataset = formatted_dataset.shuffle(seed=shuffle_seed)

    # Create train/validation split with stratification
    print("\n• Creating train/validation split")
    split_dataset = formatted_dataset["train"].train_test_split(
        test_size=0.1,  # 10% for validation
        seed=shuffle_seed,  # Use same seed for reproducibility
        shuffle=True,  # Ensure data is shuffled before splitting
    )
    
    # Rename the test split to validation
    formatted_dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    # Additional shuffling of training data
    formatted_dataset["train"] = formatted_dataset["train"].shuffle(
        seed=shuffle_seed  # Just use seed for reproducibility
    )

    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"• Number of training examples: {len(formatted_dataset['train'])}")
    print(f"• Number of validation examples: {len(formatted_dataset['validation'])}")

    # Sample and print conversations
    print("\nVerifying conversation formats:")
    for i in range(2):  # Show 2 random examples
        random_idx = torch.randint(0, len(formatted_dataset["train"]), (1,)).item()
        print(f"\n=== Random conversation example {i+1} (idx={random_idx}) ===")
        print(formatted_dataset["train"][random_idx]["text"])
        print("=" * 50)

    return formatted_dataset


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
        dataset_num_proc=2,
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
            seed=config["dataset_config"].get("shuffle_seed", 3407),  # Use consistent seed
            output_dir=config["output_config"]["output_dir"],
            logging_dir=config["output_config"]["logging_dir"],
            group_by_length=True,
            max_grad_norm=config["training_config"].get("max_grad_norm", 0.5),
            weight_decay=config["training_config"].get("weight_decay", 0.01),
            remove_unused_columns=True,
            prediction_loss_only=True,
            eval_strategy="steps",
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