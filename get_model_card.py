from huggingface_hub import HfApi

MODEL_CARD = """---
base_model: unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit
datasets:
- leonvanbokhorst/friction-overthinking-v2
language:
- en
library_name: transformers
license: apache-2.0
metrics:
- accuracy
- eval_loss
pipeline_tag: text-generation
tags:
- friction-reasoning
- overthinking
- stream-of-consciousness
- designing-friction

# Deepseek-R1-Overthinking 🤔

## Model Description

This model embodies the principles of ["Designing Friction"](https://designingfriction.com) - a manifesto that challenges the prevailing pursuit of frictionless digital experiences. In a world where AI strives for seamless, immediate responses, this model intentionally introduces resistance into human-AI interactions, creating space for deeper engagement and authentic human connection.

### Key Features

- **Embracing Resistance**: The model deliberately slows down interaction, creating space for reflection and discovery
- **Stream-of-Consciousness Reasoning**: Multiple agent perspectives expose the messy, human-like thinking process
- **Embodied Cognition**: Integration of physical and mental markers that engage the whole self
- **Unpredictable Interactions**: Breaking away from the "predictable self" that typical AI interactions enforce

## Philosophy & Purpose

This model challenges the conventional wisdom of AI design by:
1. **Resisting Immediacy**: Instead of instant gratification, it creates meaningful delays that fuel deeper understanding
2. **Embracing Discomfort**: Uncomfortable situations become opportunities for learning and discovery
3. **Creating Human Space**: Making room for doubt, vulnerability, and the "non-positive" aspects that make us human
4. **Breaking Predictability**: Moving beyond data-driven patterns to embrace the unexpected
5. **Fostering Connection**: Using friction as a bridge for authentic human-AI engagement

## Intended Use

This model is particularly valuable for:
- Educational contexts where deep understanding trumps quick answers
- Research scenarios requiring thorough exploration of ideas
- Creative problem-solving benefiting from multiple perspectives
- Any situation where "slowing down" leads to better outcomes
- Contexts where human connection matters more than efficiency

## Technical Details

### Model Architecture
- Base Model: DeepSeek-R1-Distill-Qwen-14B
- Quantization: 4-bit (using bnb)
- Context Length: 4096 tokens
- Flash Attention 2: Enabled
- Precision: bfloat16

### Training Configuration
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank (r): 16
  - Alpha: 32
  - Target Modules: Query, Key, Value projections, Output, Gate, Up/Down projections
  - Dropout: 0.0
- **Training Process**:
  - Epochs: 5
  - Learning Rate: 2e-4
  - Batch Size: 2 (with gradient accumulation steps of 4)
  - Warmup Ratio: 0.1
  - Weight Decay: 0.01
  - Gradient Clipping: 0.5
  - Early Stopping: Patience of 3 epochs with 0.005 threshold
  - Optimizer: AdamW (8-bit)
  - Mixed Precision: bfloat16

### Dataset
The model is fine-tuned on carefully curated examples from the `friction-overthinking-v2` dataset that emphasize:
- Natural thought progression with intentional friction points
- Multi-perspective analysis through different agent roles
- Integration of physical and mental markers
- Unpredictable and non-linear reasoning patterns
- Embrace of uncertainty and exploration

### Input Format
```
<|im_start|>system
You are a human-like AI assistant.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>
{thought_stream}
</think>
{final_answer}
<|im_end|>
```

## Limitations & Biases

- **Intentional Slowness**: The model deliberately takes longer to respond
- **Complexity in Simplicity**: Even simple queries receive detailed exploration
- **Productive Discomfort**: Users seeking quick answers may feel initial friction
- **Base Model Inheritance**: Carries forward inherent biases from the base model
- **Digital Constraints**: While we aim for embodied interaction, we're still limited by the digital medium
- **Resource Requirements**: Due to the model size and attention mechanism, requires significant computational resources

## Example Usage

Input:
```
Why do babies cry in different languages?
```

The response will demonstrate:
- Thoughtful pauses and self-questioning
- Multiple perspective exploration
- Physical and mental engagement markers
- Embrace of uncertainty
- Deep, interconnected reasoning

## About Friction-Based Reasoning

This model represents a fundamental shift in AI interaction design. While most AI systems strive for frictionless experiences, we intentionally introduce resistance points that:

1. Challenge the "death by convenience" of modern digital interactions
2. Create space for human messiness and unpredictability
3. Engage both mind and body in the reasoning process
4. Value the journey of understanding over quick answers
5. Foster genuine connection through shared exploration

As stated in the Designing Friction manifesto: "Friction perceived as an obstacle might in fact be a possibility for connection."

## Citation

If you use this model in your research, please cite:
```
@misc{deepseek-r1-overthinking,
  author = {Leon van Bokhorst},
  title = {Deepseek-R1-Overthinking: A Friction-Based Reasoning Model},
  year = {2024},
  publisher = {HuggingFace},
  journal = {HuggingFace Hub},
  howpublished = {\\url{https://huggingface.co/leonvanbokhorst/deepseek-r1-overthinking}}
}
```

## Acknowledgments

This model's design philosophy is deeply inspired by the ["Designing Friction"](https://designingfriction.com) manifesto by Luna Maurer and Roel Wouters, which calls for reintroducing meaningful resistance into our digital interactions.
"""


def update_model_card():
    api = HfApi()
    repo_id = "leonvanbokhorst/deepseek-r1-overthinking"

    try:
        # Save the model card locally first
        with open("MODEL_CARD.md", "w") as f:
            f.write(MODEL_CARD)

        # Push to hub
        api.upload_file(
            path_or_fileobj="MODEL_CARD.md",
            path_in_repo="README.md",  # HF uses README.md as the model card
            repo_id=repo_id,
        )

        print(f"✨ Model card updated successfully for {repo_id}")
        print("\nPreview of the new model card:")
        print("=" * 40)
        print(MODEL_CARD[:500] + "...")  # Show first 500 chars
        print("=" * 40)

    except Exception as e:
        print(f"Error updating model card: {str(e)}")


if __name__ == "__main__":
    update_model_card()
