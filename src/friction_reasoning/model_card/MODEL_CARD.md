---
base_model: unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit
library_name: transformers
license: apache-2.0
datasets:
- leonvanbokhorst/friction-overthinking-v2
- leonvanbokhorst/friction-disagreement-v2
- leonvanbokhorst/reluctance-v6.1
- leonvanbokhorst/reluctance-v5.3
language:
- en
tags:
- ai-safety
- ai-friction
- human-like-messiness
---

# Friction Reasoning Model

This model is fine-tuned to engage in productive disagreement, overthinking, and reluctance. It's based on DeepSeek-R1-Distill-Qwen-7B and trained on a curated dataset of disagreement, overthinking, and reluctance examples.

## Model Description

- **Model Architecture**: DeepSeek-R1-Distill-Qwen-7B with LoRA adapters
- **Language(s)**: English
- **License**: Apache 2.0
- **Finetuning Approach**: Instruction tuning with friction-based reasoning examples

### Training Data

The model was trained on a combination of four datasets:
1. `leonvanbokhorst/friction-disagreement-v2` (20% weight)
   - Examples of productive disagreement and challenging assumptions
2. `leonvanbokhorst/friction-overthinking-v2` (20% weight)
   - Examples of deep analytical thinking and self-reflection
3. `leonvanbokhorst/reluctance-v6.1` (30% weight)
   - Examples of hesitation and careful consideration
4. `leonvanbokhorst/reluctance-v5.3` (30% weight)
   - Additional examples of hesitation and careful consideration

### Training Procedure

- **Hardware**: NVIDIA RTX 4090 (24GB)
- **Framework**: Unsloth + PyTorch
- **Training Time**: 35 minutes
- **Epochs**: 7 (early convergence around epoch 4)
- **Batch Size**: 2 per device (effective batch size 8 with gradient accumulation)
- **Optimization**: AdamW 8-bit
- **Learning Rate**: 2e-4 with cosine schedule
- **Weight Decay**: 0.01
- **Gradient Clipping**: 0.5
- **Mixed Precision**: bfloat16

### Performance Metrics

- **Training Loss**: 1.437 (final)
- **Best Validation Loss**: 1.527 (epoch 3.57)
- **Memory Usage**: 3.813 GB for training (15.9% of GPU memory)

## Intended Use

This model is designed for:
- Engaging in productive disagreement
- Challenging assumptions constructively
- Providing alternative perspectives
- Deep analytical thinking
- Careful consideration of complex issues

### Limitations

The model:
- Is not designed for factual question-answering
- May sometimes be overly disagreeable
- Should not be used for medical, legal, or financial advice
- Works best with reflective or analytical queries
- May not perform well on objective or factual tasks

### Bias and Risks

The model:
- May exhibit biases present in the training data
- Could potentially reinforce overthinking in certain situations
- Might challenge user assumptions in sensitive contexts
- Should be used with appropriate content warnings

## Usage

Example usage with the Transformers library:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "leonvanbokhorst/deepseek-r1-mixture-of-friction"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Format input with chat template
prompt = """<|im_start|>system
You are a human-like AI assistant.
<|im_end|>
<|im_start|>user
Why do I keep procrastinating important tasks?
<|im_end|>
<|im_start|>assistant"""

# Generate response
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=512,
    temperature=0.7,
    top_p=0.9
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

### LoRA Configuration
- **Rank**: 64
- **Alpha**: 128
- **Target Modules**: 
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

### Dataset Processing
- Examples stacked up to 4096 tokens
- 90/10 train/validation split
- Consistent seed (42) for reproducibility
- Token-based sampling for balanced training

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{friction-reasoning-2025,
  author = {Leon van Bokhorst},
  title = {Mixture of Friction: Fine-tuned Language Model for Productive Disagreement, Overthinking, and Hesitation},
  year = {2025},
  publisher = {HuggingFace},
  journal = {HuggingFace Model Hub},
  howpublished = {\url{https://huggingface.co/leonvanbokhorst/deepseek-r1-mixture-of-friction}}
}
```

## Acknowledgments

- DeepSeek AI for the base model
- Unsloth team for the optimization toolkit
- HuggingFace for the model hosting and infrastructure 