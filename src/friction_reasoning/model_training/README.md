# Friction Reasoning Model Training

This module provides functionality for training a LoRA-adapted language model on the friction reasoning dataset using Unsloth's optimized training pipeline.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU with at least 16GB VRAM
- Dependencies from `requirements.txt`

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your training data in `data/friction_reasoning/train.jsonl`
   - Place your validation data in `data/friction_reasoning/validation.jsonl`
   - Data should be in JSONL format with a "text" field containing the conversation

3. Configure training parameters:
   - Review and modify `config.yaml` according to your needs
   - Key parameters to consider:
     - `model_config.base_model`: The base model to fine-tune
     - `model_config.model_max_length`: Maximum sequence length
     - `training_config.learning_rate`: Learning rate for training
     - `training_config.num_train_epochs`: Number of training epochs

## Training

To start training:

```bash
python -m src.friction_reasoning.model_training.train
```

The script will:
1. Load the configuration from `config.yaml`
2. Set up the model and tokenizer
3. Load and prepare the dataset
4. Initialize the trainer with LoRA adapters
5. Train the model
6. Save the trained model and logs

## Output

The training process will create:
- Trained model checkpoints in `models/friction_reasoning/`
- Training logs in `logs/friction_reasoning/`
- The final LoRA model in `models/friction_reasoning/lora_model/`

## Monitoring

Training progress can be monitored through:
- Console output showing training metrics
- TensorBoard logs in the logging directory
- Model checkpoints saved according to the configuration

## Configuration

Key configuration sections in `config.yaml`:

```yaml
model_config:
  base_model: "mistralai/Mistral-7B-v0.1"
  model_max_length: 4096
  ...

lora_config:
  r: 8
  target_modules: [...]
  ...

training_config:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  ...
```

See `config.yaml` for all available options and their descriptions. 