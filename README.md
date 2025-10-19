# Pushback-AI

End-to-end friction lab for disagreement-driven AI: synthetic dialogue generation, persona orchestration, dataset management, fine-tuning, GGUF export, and deployment automation.

---

## Overview

Pushback-AI orchestrates six contrasting agent personas to manufacture productive disagreement. The toolkit covers the full lifecycle: synthetic dialogue generation, dataset packaging, fine-tuning (LoRA + DeepSeek-R1), GGUF conversion, and distribution via Hugging Face and Ollama.

> **Platform note**
> The end-to-end workflow (generation → training → export) is tested on **Windows 11 WSL2** with Python 3.10–3.12. macOS builds may require alternative GPU backends (e.g., replacing `bitsandbytes`).

---

## Key Capabilities

- **Disagreement-first dataset synthesis** powered by six agent personas (skeptical, memory-rich, mechanical, contrarian, humble, synthesizer).
- **Prompt templating system** that enforces gestures, hesitation, and persona tone.
- **LoRA fine-tuning pipeline** targeting DeepSeek-R1 distilled models using Unsloth.
- **Deployment tooling** for GGUF export and Hugging Face/Ollama publication.


---

## Documentation Hub

| Phase | Description                        | Link                                                                                                 |
| ----- | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 01    | Synthetic dataset pipeline         | [`docs/pushback_ai/01_synthetic_pipeline.md`](docs/pushback_ai/01_synthetic_pipeline.md)             |
| 02    | Prompt arsenal & persona templates | [`docs/pushback_ai/02_prompt_design.md`](docs/pushback_ai/02_prompt_design.md)                       |
| 03    | Multi-turn orchestration           | [`docs/pushback_ai/03_multi_turn_orchestration.md`](docs/pushback_ai/03_multi_turn_orchestration.md) |
| 04    | Hugging Face delivery workflow     | [`docs/pushback_ai/04_hf_delivery.md`](docs/pushback_ai/04_hf_delivery.md)                           |
| 05    | Disagreement-only fine-tuning      | [`docs/pushback_ai/05_finetuning.md`](docs/pushback_ai/05_finetuning.md)                             |
| 06    | GGUF creation & Ollama release     | [`docs/pushback_ai/06_deployment.md`](docs/pushback_ai/06_deployment.md)                             |
| 07    | Benchmarking & evaluation results | [`docs/pushback_ai/07_benchmarking.md`](docs/pushback_ai/07_benchmarking.md)                         |

Additional references:

- Thinking patterns & persona guidance: [`docs/thinking-pattern.md`](docs/thinking-pattern.md)

---

## Project Layout

```
friction-reasoning/
├── src/friction_reasoning/
│   ├── dataset/        # Generation, batching, HF upload scripts
│   ├── llm/            # Prompt templates, LiteLLM client
│   ├── agents/         # Persona base classes
│   └── model_training/ # Unsloth training, GGUF export, HF push scripts
├── docs/pushback_ai/   # Phase docs (01–06)
├── README.md           # This file
└── pyproject.toml      # Project + dependency metadata
```

---

## Getting Started

```bash
# Clone
git clone https://github.com/leonvanbokhorst/friction-reasoning-data-gen.git
cd friction-reasoning-data-gen

# Install (creates .venv with uv)
uv sync

# Activate
source .venv/bin/activate  # or .venv\Scripts\activate on Windows / WSL
```

### Environment variables

Duplicate `.env.example` → `.env`, then provide the required API tokens (e.g., OpenAI/LiteLLM, Hugging Face).

---

## Core Workflows

### 1. Generate disagreement dataset

```bash
uv run python -m friction_reasoning.dataset --num_examples 3 --batch_size 3
```

Outputs staged JSONL batches under `data/friction_reasoning/` and a consolidated dataset file.

### 2. Upload to Hugging Face

```bash
uv run python -m friction_reasoning.dataset.upload \
  --dataset_path data/friction_reasoning/friction_reasoning_dataset.jsonl \
  --repo_id leonvanbokhorst/friction-disagreement-v2
```

For details, see Phase 04 documentation.

### 3. Fine-tune DeepSeek-R1 (disagreement corpus)

```bash
uv run python src/friction_reasoning/model_training/train.py
```

Configuration lives in `src/friction_reasoning/model_training/config.yaml` (Phase 05).

### 4. Convert to GGUF & publish

```bash
uv run python src/friction_reasoning/model_training/convert_to_gguf.py
uv run python src/friction_reasoning/model_training/push_to_hub.py \
  --hub_repo leonvanbokhorst/deepseek-r1-disagreement
```

See Phase 06 for Ollama packaging guidance.

### 5. Benchmark the tuned model
- Sample evaluation prompts from your dataset or `data/eval/prompts_disagreement.jsonl` (create if needed).
- Compare baseline vs fine-tuned outputs using the checklist in Phase 07.
- Log hedging, gestures, and explicit pushback to confirm improvements.

---

## Contributing

1. Fork & branch (`git checkout -b feature/pushback-upgrade`)
2. Sync dependencies (`uv sync`)
3. Run tests / linters (`pytest`, `pylint`, `mypy`, `black`)
4. Open a PR referencing the relevant phase documentation.

For bug reports or feature requests, file an issue on GitHub.

---

Made with 🧠 and ❤️ by the Pushback-AI team.
