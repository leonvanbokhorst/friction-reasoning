"""Generate paired baseline vs pushback responses using local Ollama models."""

import json
import os
from pathlib import Path
from typing import Iterable, Tuple, List, Dict

import requests
from tqdm import tqdm

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
BASE_MODEL = os.environ.get("OLLAMA_BASELINE_MODEL", "erwan2/DeepSeek-R1-Distill-Qwen-7B:latest")
TUNED_MODEL = os.environ.get("OLLAMA_TUNED_MODEL", "leonvanbokhorst/deepseek-r1-disagreement:latest")
PROMPT_FILE = Path(os.environ.get("OLLAMA_EVAL_FILE", "data/eval/prompts_disagreement.jsonl"))
OUTPUT_DIR = Path(os.environ.get("OLLAMA_EVAL_OUTPUT_DIR", "benchmark/results"))
TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.8"))
MAX_TOKENS = int(os.environ.get("OLLAMA_MAX_TOKENS", "4096"))
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))


def load_prompts(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation file {path} not found. Create it with disagreement prompts before running."
        )
    prompts: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            prompts.append((data["id"], data["prompt"]))
    return prompts


def generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


def unload(model: str) -> None:
    try:
        requests.post(
            f"{OLLAMA_HOST}/api/unload",
            json={"model": model},
            timeout=TIMEOUT,
        )
    except requests.RequestException:
        # Unload failures are non-fatal; continue benchmark.
        pass


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(PROMPT_FILE)

    baseline_outputs: Dict[str, str] = {}
    for example_id, prompt in tqdm(prompts, desc="Baseline", unit="prompt"):
        baseline_outputs[example_id] = generate(BASE_MODEL, prompt)

    unload(BASE_MODEL)

    tuned_outputs: Dict[str, str] = {}
    for example_id, prompt in tqdm(prompts, desc="Tuned", unit="prompt"):
        tuned_outputs[example_id] = generate(TUNED_MODEL, prompt)

    unload(TUNED_MODEL)

    for example_id, prompt in prompts:
        data = {
            "id": example_id,
            "prompt": prompt,
            "baseline_model": BASE_MODEL,
            "baseline_response": baseline_outputs.get(example_id, ""),
            "tuned_model": TUNED_MODEL,
            "tuned_response": tuned_outputs.get(example_id, ""),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
        output_path = OUTPUT_DIR / f"{example_id}.json"
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(prompts)} evaluation files under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
