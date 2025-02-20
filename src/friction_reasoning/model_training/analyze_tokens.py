from transformers import AutoTokenizer
from datasets import load_dataset
import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit')

# Dataset configs with new weights
DATASET_CONFIGS = [
    {"name": "leonvanbokhorst/friction-disagreement-v2", "key": "disagreement", "weight": 0.085},
    {"name": "leonvanbokhorst/friction-overthinking-v2", "key": "overthinking", "weight": 0.045},
    {"name": "leonvanbokhorst/reluctance-v6.1", "key": "reluctance", "weight": 1.0},
    {"name": "leonvanbokhorst/friction-uncertainty-v2", "key": "uncertainty", "weight": 0.15}
]

# Load datasets
datasets = {
    config["key"]: load_dataset(config["name"])["train"]
    for config in DATASET_CONFIGS
}

# Analyze token lengths
print("\nAnalyzing Original Datasets:")
stats = {}
total_tokens = 0
for name, dataset in datasets.items():
    # Take first 100 examples for quick analysis
    sample = dataset.select(range(min(100, len(dataset))))
    lengths = [len(tokenizer.encode(ex['text'] if 'text' in ex else json.dumps(ex, cls=DateTimeEncoder))) for ex in sample]
    avg_len = sum(lengths) / len(lengths)
    stats[name] = {
        'examples': len(dataset),
        'avg_tokens': avg_len,
        'estimated_total': len(dataset) * avg_len
    }
    total_tokens += stats[name]['estimated_total']

for name, data in stats.items():
    print(f'\n{name.title()}:')
    print(f'• Total examples: {data["examples"]:,}')
    print(f'• Avg tokens/example: {data["avg_tokens"]:.1f}')
    print(f'• Estimated total tokens: {data["estimated_total"]:.0f}')
    print(f'• % of total tokens: {(data["estimated_total"] / total_tokens * 100):.1f}%')

# Simulate sampling with new weights
print("\nSimulating Sampling with New Weights:")
target_size = 4000  # Target total examples
samples_per_dataset = []

for config in DATASET_CONFIGS:
    name = config["key"]
    # Calculate target samples, but don't exceed dataset size
    target_samples = min(
        int(target_size * config["weight"]),
        len(datasets[name])
    )
    samples_per_dataset.append({
        "name": name,
        "samples": target_samples,
        "avg_tokens": stats[name]["avg_tokens"]
    })

# Print sampling simulation results
print("\nSampling Plan:")
total_sampled_tokens = 0
for data in samples_per_dataset:
    estimated_tokens = data["samples"] * data["avg_tokens"]
    total_sampled_tokens += estimated_tokens
    print(f'\n{data["name"].title()}:')
    print(f'• Samples: {data["samples"]:,}')
    print(f'• Estimated tokens: {estimated_tokens:,.0f}')

# Print final token distribution
print("\nEstimated Final Token Distribution:")
for data in samples_per_dataset:
    estimated_tokens = data["samples"] * data["avg_tokens"]
    percentage = (estimated_tokens / total_sampled_tokens) * 100
    print(f'• {data["name"].title()}: {percentage:.1f}% of tokens')

print(f'\nTotal sampled tokens: {total_sampled_tokens:,.0f}') 