# Phase 05 · Fine-Tuning Trials

## Story Beat — Feeding the Model a Friction Diet

After collecting and publishing the disagreement-focused dataset, we rolled straight into training. Goal: infuse DeepSeek-R1 with the multi-agent disagreement voice without overbaking agreement. Phase 05 is about preparing balanced batches from `friction-disagreement-v2`, configuring LoRA adapters, and monitoring training signals so the model internalizes argument, vulnerability, and hesitation.

## Configuration Scroll

Everything starts with a trimmed `config.yaml`, which we updated for this run to keep only the disagreement dataset.

```1:40:src/friction_reasoning/model_training/config.yaml
model_config:
  base_model: "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
  ...
  model_max_length: 4096
lora_config:
  r: 32
  lora_alpha: 64
  target_modules: ["q_proj", "k_proj", ...]
...
dataset_config:
  datasets:
    - name: "leonvanbokhorst/friction-disagreement-v2"
      weight: 1.0
  format_template: |
    <|im_start|>system
    You are a human-like AI assistant.
    ...
training_config:
  learning_rate: 2e-4
  num_train_epochs: 7
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  warmup_ratio: 0.1
  ...
  report_to: ["wandb"]
```

Highlights for the audience:

- Base model is already distilled + quantized, ideal for LoRA fine-tuning on commodity GPUs.
- Only `friction-disagreement-v2` feeds the run, so every batch reinforces productive pushback.
- Chat template with `<think>` tag keeps thought streams explicit for instruction tuning.

## Data Preparation Pipeline

`train.py` handles dataset loading, formatting, stacking, and splitting.

```320:365:src/friction_reasoning/model_training/train.py
for dataset_config in config["dataset_config"]["datasets"]:
    dataset = load_dataset(dataset_config["name"]) ["train"]
    formatted_dataset = dataset.map(
        lambda x: format_dataset_example(x, dataset_config["name"], config, tokenizer),
        remove_columns=dataset.column_names,
        desc=f"Formatting {dataset_config['name']}"
    )
    datasets.append(formatted_dataset)
...
stacked_texts = stack_examples(
    combined_examples,
    max_tokens=config["model_config"]["model_max_length"]
)
stacked_dataset = Dataset.from_dict({"text": stacked_texts})
split_dataset = stacked_dataset.train_test_split(test_size=0.1, seed=42)
```

- Formatting path now only sees disagreement-style records, but the helper still works if we add other flavors later.
- We stack four single-turn conversations into longer training samples so the model practices carrying context across consecutive exchanges.
- Train/validation split ensures evaluation without leaking test prompts.

### Formatting Helpers

`format_dataset_example` and `format_conversation` provide precise control over the prompt structure (they still support additional datasets, but in this run we only feed disagreement examples).

```36:109:src/friction_reasoning/model_training/train.py
turn = config["dataset_config"]["format_template"].format(
    question=question,
    thought_stream=thought_stream,
    final_answer=final_answer
)
...
formatted_text = config["dataset_config"]["format_template"].format(
    question=question,
    thought_stream=thought_stream,
    final_answer=final_answer
)
```

## Token Balance & Sampling Weights

With a single dataset, weighting is trivial (`weight: 1.0`), but the pipeline still logs token counts for transparency.

## Model & Trainer Setup

`setup_model_and_tokenizer()` loads the base model with 4-bit quantization and applies LoRA adapters.

```213:244:src/friction_reasoning/model_training/train.py
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(...)
```

The trainer uses Unsloth’s optimized `UnslothTrainer` to boost speed and handle response-only training.

```413:480:src/friction_reasoning/model_training/train.py
trainer = UnslothTrainer(...)
trainer = train_on_responses_only(...)
```

- Response-only training focuses gradients on assistant turns—the parts we want to improve.
- Early stopping prevents overfitting; LoRA keeps VRAM usage ~5–6 GB on a single 24 GB GPU.

## Training Orchestration & Monitoring

The `main()` function handles logging and GPU stats for storytelling.

```483:553:src/friction_reasoning/model_training/train.py
print_step("Starting training")
trainer_stats = trainer.train()
...
print("\nTraining Statistics:")
print(f"• Training time: {trainer_stats.metrics['train_runtime']} seconds")
...
print_step("Saving model")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

- Provides start/end timestamps, runtime in minutes, GPU memory usage (peak, percentage), and final save location.
- Perfect for talk slides showing the “battle wounds” of the training run.

## Testing the Adapter

`test_model.py` reloads the LoRA adapter and provides an interactive CLI for spot checks.

```61:118:src/friction_reasoning/model_training/test_model.py
model, tokenizer = FastLanguageModel.from_pretrained(...)
model.load_adapter(adapter_path)
model = FastLanguageModel.for_inference(model)
...
for prompt in test_prompts:
    _ = generate_response(model, tokenizer, prompt, stream=True)
...
while True:
    prompt = input("\nYour question: ").strip()
    ...
```

Tell the audience: we validated the model’s disagreement tone live before shipping it downstream.

## Quickstart Checklist

- Install CUDA dependencies (or set `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon) before launching training.
- Run a dry configuration check: `python -m friction_reasoning.model_training.train --config src/friction_reasoning/model_training/config.yaml --dry-run`.
- Kick off training with `python -m friction_reasoning.model_training.train --config ... --run-name disagreement-v2-lora`.
- Watch the console or W&B dashboard for spikes in loss; if it plateaus too early, lower `gradient_accumulation_steps` or tweak learning rate.
- After training, reload the adapter via `python -m friction_reasoning.model_training.test_model --adapter-path models/friction_reasoning/lora_model` to sanity check tone.

## Outputs & Artifacts

| Path                                    | Description                                  |
| --------------------------------------- | -------------------------------------------- |
| `models/friction_reasoning/lora_model/` | Saved LoRA weights + tokenizer               |
| `logs/friction_reasoning/`              | Training logs (for W&B or manual inspection) |
| `model_training/test_model.py`          | Post-training evaluation harness             |

## Troubleshooting Notes

- **VRAM crashes mid-run**: reduce `per_device_train_batch_size` to 1 or enable gradient checkpointing in the config.
- **Adapter feels too combative**: mix in a lighter dataset and adjust weighting, or drop total epochs and rerun.
- **Response-only training errors**: confirm your tokenizer has `bos_token`/`eos_token` defined; missing tokens lead to misaligned labels.

## Lessons Learned

- Training purely on the disagreement corpus gives the fine-tuned model a consistent argumentative stance.
- Response-only LoRA training is efficient yet powerful—quick to iterate, easy to merge later.
- Instrumentation (timings, GPU stats) adds texture to the story and proves the work wasn’t just “hit run & pray.”

## Next Phase Preview

Phase 06 wraps the journey: merging adapters into full GGUF builds, quantizing for edge devices, and publishing the final Pushback-AI model to both Hugging Face and Ollama. Dive into `06_deployment.md` next.
