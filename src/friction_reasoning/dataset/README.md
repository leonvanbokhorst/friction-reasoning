# Dataset Generation and Upload

This directory contains the code for generating and uploading the friction reasoning dataset. The code is organized into several modules, each with a specific purpose.

## 🗂️ File Structure

- `generate_dataset.py` - Main dataset generation functionality (active)
- `upload.py` - Functions for uploading to HuggingFace Hub (active)
- `__main__.py` - CLI entry point for dataset operations

## 🚀 Quick Start

Generate and optionally upload a dataset:

```bash
# Run a quick test with 3 examples
python -m friction_reasoning.dataset --test

# Generate 1200 examples
python -m friction_reasoning.dataset --num_examples 1200

# Generate and upload to HuggingFace Hub
python -m friction_reasoning.dataset --num_examples 1200 --upload
```

## 💻 Command Line Arguments

```bash
--test            # Run in test mode (3 examples)
--num_examples N  # Number of examples to generate (default: 1200)
--upload          # Upload to HuggingFace Hub after generation
--repo_id ID      # HuggingFace repo ID (default: leonvanbokhorst/friction-uncertainty-v2)
--data_dir DIR    # Output directory (default: data/friction_reasoning)
```

## 🔧 Configuration

The dataset generation uses several configuration objects defined in `base_prompts.py`:

- `BASE_PROMPTS` - Template prompts for different emotional categories
- `EMOTIONS` - List of emotional states for generation
- `VULNERABILITY_CONFIG` - Settings for vulnerability injection
- `DISAGREEMENT_CONFIG` - Settings for disagreement-focused generation

## 📊 Generated Data Format

Each example in the dataset has this structure:

```json
{
    "id": "unique_id",
    "question": "Generated question/prompt",
    "agent_responses": [
        {
            "agent_type": "problem_framer",
            "thought_stream": "Agent's reasoning process"
        },
        // ... more agent responses
    ],
    "final_answer": "Synthesized response",
    "metadata": {
        "timestamp": "2024-03-XX",
        "model": "model_name"
    }
}
```

## 🔄 Generation Process

1. **Question Generation**: Creates emotionally resonant questions using templates
2. **Agent Reasoning**: Multiple agents process the question:
   - Problem Framer
   - Memory Activator
   - Mechanism Explorer
   - Perspective Generator
   - Limitation Acknowledger
   - Synthesizer
3. **Vulnerability Injection**: Adds uncertainty/limitation acknowledgments
4. **Final Synthesis**: Combines agent perspectives into a final answer

## 📤 Upload Process

The upload functionality:
1. Creates/verifies the HuggingFace repository
2. Generates a dataset card (README.md)
3. Uploads the dataset in HuggingFace's format
4. Updates repository metadata

## 🔑 Environment Setup

Required environment variables (in `.env`):
```bash
HUGGINGFACE_API_KEY=your_api_key_here
```
