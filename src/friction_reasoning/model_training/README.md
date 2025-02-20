# 🤖 Model Training Tools

This directory contains tools for training, testing, and deploying friction reasoning models. Each tool is designed to handle a specific part of the model lifecycle.

## 🗂️ Quick Overview

```bash
model_training/
├── train.py           # Main training script
├── config.yaml        # Training configuration
├── test_model.py      # Model evaluation tools
├── analyze_tokens.py  # Token usage analysis
├── push_to_hub.py    # HuggingFace Hub upload
└── convert_to_gguf.py # GGUF conversion for llama.cpp
```

## 🚀 Getting Started

1. **Setup Configuration**
   ```bash
   # Edit config.yaml to set your training parameters
   nano config.yaml
   ```

2. **Run Training**
   ```bash
   # Start training with default config
   python -m friction_reasoning.model_training.train
   
   # Override config with command line args
   python -m friction_reasoning.model_training.train \
       --batch_size 8 \
       --learning_rate 2e-5 \
       --num_epochs 3
   ```

3. **Test Your Model**
   ```bash
   # Run basic tests
   python -m friction_reasoning.model_training.test_model
   
   # Run specific test cases
   python -m friction_reasoning.model_training.test_model --test_case uncertainty
   ```

## 📊 Token Analysis

Analyze token usage in your training data:

```bash
# Basic token analysis
python -m friction_reasoning.model_training.analyze_tokens

# Detailed analysis with visualization
python -m friction_reasoning.model_training.analyze_tokens --visualize
```

## 🔄 Model Conversion

Convert models to GGUF format for llama.cpp:

```bash
# Convert to GGUF
python -m friction_reasoning.model_training.convert_to_gguf \
    --input_model path/to/model \
    --output_path model_gguf/
```

## ⬆️ Model Upload

Push trained models to HuggingFace Hub:

```bash
# Upload model
python -m friction_reasoning.model_training.push_to_hub \
    --model_path path/to/model \
    --repo_id your-username/model-name
```

## 🔧 Configuration Options

Key settings in `config.yaml`:

```yaml
training:
  batch_size: 8
  learning_rate: 2e-5
  num_epochs: 3
  warmup_steps: 500
  gradient_accumulation: 4

model:
  base_model: "mistralai/Mistral-7B-v0.1"
  tokenizer: "mistralai/Mistral-7B-v0.1"
  max_length: 2048
  
data:
  train_path: "data/friction_reasoning_train.jsonl"
  eval_path: "data/friction_reasoning_eval.jsonl"
  test_path: "data/friction_reasoning_test.jsonl"
```

## 🎯 Training Tips

1. **Start Small**
   - Begin with a small dataset to verify everything works
   - Use `--test` flag for quick iteration
   - Monitor token usage with `analyze_tokens.py`

2. **Optimize Training**
   - Use gradient accumulation for larger effective batch sizes
   - Enable mixed precision training for speed
   - Monitor GPU memory usage and adjust accordingly

3. **Evaluate Properly**
   - Use `test_model.py` for comprehensive evaluation
   - Check both quantitative metrics and qualitative outputs
   - Test specifically for uncertainty and friction points

## 🐛 Troubleshooting

Common issues and solutions:

1. **Out of Memory**
   ```bash
   # Reduce batch size
   python -m friction_reasoning.model_training.train --batch_size 4
   
   # Enable gradient checkpointing
   python -m friction_reasoning.model_training.train --gradient_checkpointing
   ```

2. **Slow Training**
   ```bash
   # Enable mixed precision
   python -m friction_reasoning.model_training.train --mixed_precision
   
   # Use more workers for data loading
   python -m friction_reasoning.model_training.train --num_workers 4
   ```

3. **Poor Results**
   - Check token distribution with `analyze_tokens.py`
   - Verify training data quality
   - Adjust learning rate and batch size
   - Increase number of epochs

## 📝 Notes

- Always backup your models before converting to GGUF
- Use version control for your config files
- Monitor training with wandb or tensorboard
- Test thoroughly before pushing to production

## 🔗 Related Tools

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - For running converted models
- [HuggingFace Hub](https://huggingface.co/) - For model hosting
- [Weights & Biases](https://wandb.ai/) - For experiment tracking 