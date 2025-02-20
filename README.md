# 🤔 Friction Reasoning Dataset Generator

Generate a dataset of multi-agent reasoning dialogues that deliberately incorporate cognitive friction, uncertainty, and emotional resonance. This project uses multiple AI agents with distinct personalities to explore questions from different angles, creating rich, human-like thought processes.

## 🌟 Features

- **Multi-Agent System**: 6 unique agents with distinct personalities:
  - 🤨 Problem Framer (skeptical overthinker)
  - 💭 Memory Activator (emotional, dramatic)
  - 🔍 Mechanism Explorer (technical, detailed)
  - 🎭 Perspective Generator (contrarian, challenging)
  - 🤷 Limitation Acknowledger (humble, uncertain)
  - 🎯 Synthesizer (chaotic connector)

- **Emotional Intelligence**:
  - Generates emotionally resonant questions
  - Incorporates vulnerability and uncertainty
  - Mimics human thought patterns
  - Uses casual, relatable language

- **Dataset Generation**:
  - Configurable batch generation
  - Progress tracking and statistics
  - Automatic HuggingFace upload
  - JSONL format for easy processing

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/leonvanbokhorst/friction-reasoning-data-gen.git
cd friction-reasoning-data-gen

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Generate a test dataset (3 examples)
python -m friction_reasoning.dataset --test

# Generate full dataset
python -m friction_reasoning.dataset --num_examples 1200
```

## 📊 Generated Data Format

Each example in the dataset looks like this:

```json
{
    "id": "unique_id",
    "question": "Why do we keep dancing around this thing where...",
    "agent_responses": [
        {
            "agent_type": "problem_framer",
            "thought_stream": "Hmm *fidgets nervously* isn't it weird how..."
        },
        // ... more agent responses
    ],
    "final_answer": "Maybe we're all just trying to...",
    "metadata": {
        "timestamp": "2024-03-XX",
        "model": "model_name"
    }
}
```

## 🗂️ Project Structure

```bash
friction-reasoning-data-gen/
├── src/
│   └── friction_reasoning/
│       ├── agent_reasoning/    # Core agent system
│       ├── dataset/           # Dataset generation
│       ├── llm/              # LLM interaction
│       └── model_training/   # Model training tools
├── data/                    # Generated datasets
├── tests/                  # Test suite
└── examples/              # Usage examples
```

## 📚 Documentation

- [Dataset Generation Guide](src/friction_reasoning/dataset/README.md)
- [Model Training Guide](src/friction_reasoning/model_training/README.md)
- [Agent System Documentation](docs/agents.md)

## 🛠️ Development

### Prerequisites

- Python 3.8+
- OpenAI API key (or compatible LLM API)
- HuggingFace account (for dataset upload)
- 16GB+ RAM recommended

### Setup for Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
```

## 🎯 Use Cases

1. **Training Data Generation**
   - Create datasets for training emotional intelligence
   - Generate examples of productive uncertainty
   - Build friction-aware dialogue systems

2. **Research**
   - Study multi-agent interactions
   - Analyze patterns in human-like reasoning
   - Explore cognitive friction points

3. **Education**
   - Demonstrate different thinking styles
   - Practice emotional awareness
   - Learn about cognitive biases

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Configuration

Key configuration files:
- `.env`: Environment variables and API keys
- `config.yaml`: Generation settings
- `pyproject.toml`: Project metadata and dependencies

## 🐛 Troubleshooting

Common issues and solutions:

1. **API Rate Limits**
   ```bash
   # Use batch processing with delays
   python -m friction_reasoning.dataset --batch_size 10 --delay 1
   ```

2. **Memory Usage**
   ```bash
   # Reduce parallel processing
   python -m friction_reasoning.dataset --workers 2
   ```

3. **Dataset Quality**
   ```bash
   # Run analysis tools
   python -m friction_reasoning.dataset.analyze --check-quality
   ```

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- OpenAI for the base LLM technology
- HuggingFace for dataset hosting
- The amazing open-source community

## 📫 Contact

- GitHub Issues: For bugs and features
- Discussions: For questions and ideas
- Email: your.email@example.com

---
Made with 🧠 and ❤️ by the Friction Reasoning team 