# Friction Reasoning Dataset Generator

A dataset generator for multi-agent reasoning with designed friction points. This project creates datasets of thought processes that incorporate natural friction points in reasoning, inspired by human cognitive patterns.

## Features

- Multiple reasoning agents with unique thinking patterns
- Natural friction points in thought processes
- Stream-of-consciousness style reasoning
- Structured output format for machine learning

## Installation

```bash
# Clone the repository
git clone https://github.com/leonvanbokhorst/friction-reasoning-data-gen.git
cd friction-reasoning-data-gen

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

```python
from friction_reasoning.agents import ProblemFramer

# Create an agent
agent = ProblemFramer()

# Generate thought stream
response = agent.think("How do trees communicate?")
print(response)

# Get structured response with friction points
structured = agent.get_response()
print(structured)
```

## Project Structure

```
friction-reasoning-data-gen/
├── docs/                    # Documentation
│   ├── thinking-pattern.md  # Thinking pattern documentation
│   ├── friction.md         # Friction design principles
│   └── dataset-format.md   # Dataset format specification
├── src/                    # Source code
│   └── friction_reasoning/
│       ├── agents/         # Reasoning agents
│       ├── core/           # Core functionality
│       └── utils/          # Utility functions
├── tests/                  # Test files
├── requirements.txt        # Project dependencies
└── setup.py               # Package setup file
```

## Documentation

See the `docs/` directory for detailed documentation on:
- Thinking patterns and agent types
- Friction design principles
- Dataset format specifications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 