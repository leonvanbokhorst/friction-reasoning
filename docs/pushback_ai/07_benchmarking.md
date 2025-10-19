# Phase 07 · Benchmarking & Evaluation

## Story Beat — Proving Pushback Matters

Fine-tuning only earns trust when we show the tuned model actually pushes back more thoughtfully than the baseline. Phase 07 covers the full evaluation flow: collect paired outputs, score disagreement behaviors, and summarize results to prove the Pushback adapter does what it promises.

---

## Evaluation Goals

- **Contrast** the vanilla baseline (e.g., DeepSeek-R1 Distill) with the Pushback LoRA variant.
- **Measure disagreement behaviors** explicitly (challenge strength, hedging, gestures/vulnerability, empathy, specificity).
- **Guardrails**: ensure the tuned model stays supportive, avoids toxicity, and doesn’t hallucinate facts.

---

## Assets & Setup

| Resource                                   | Purpose                                     | Notes                                                       |
| ------------------------------------------ | ------------------------------------------- | ----------------------------------------------------------- |
| `models/friction_reasoning/lora_model/`    | Fine-tuned adapter + tokenizer              | Produced in Phase 05                                        |
| `data/eval/prompts_disagreement.jsonl`     | Evaluation prompts (20–30 statements)       | Each entry `{"id": "eval-001", "prompt": "..."}`            |
| `benchmark/run_ollama_eval.py`             | Generates paired responses via local Ollama | Baseline + tuned runs (sequential)                          |
| `benchmark/results/*.json`                 | Per-prompt output files                     | Contain prompt, baseline response, tuned response, metadata |
| `benchmark/metrics_summary.csv` (optional) | Automated counts from helper scripts        | Hedging, gestures, length, etc.                             |
| `benchmark/benchmark_scores.csv` (manual)  | Human scoring sheet                         | Disagreement, empathy, specificity, support vs challenge    |

If prompts file doesn’t exist yet:

```bash
mkdir -p data/eval
# Use provided template or craft 20-30 statements
```

---

## Step 1 · Generate Paired Outputs (Ollama)

1. Ensure both models are available locally: `ollama list` should show baseline (`erwan2/DeepSeek-R1-Distill-Qwen-7B:latest`) and tuned (`leonvanbokhorst/deepseek-r1-disagreement:latest`).
2. Run the evaluator:
   ```bash
   uv run python benchmark/run_ollama_eval.py
   ```
   - Baseline pass runs first → responses cached.
   - Model unloaded.
   - Tuned pass runs → responses cached.
   - Each prompt writes a JSON file under `benchmark/results/eval-XXX.json` containing:
     ```json
     {
       "id": "eval-001",
       "prompt": "statement",
       "baseline_model": "...",
       "baseline_response": "<think>...</think>…",
       "tuned_model": "...",
       "tuned_response": "<think>...</think>…",
       "temperature": 0.8,
       "max_tokens": 4096
     }
     ```

> Tip: For HF-based benchmarking, adapt `benchmark/run_eval.py` similarly to produce CSV output.

---

## Step 2 · Automated Heuristics (Optional but Handy)

Use a helper script to compute support stats:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
import re

results_dir = Path('benchmark/results')
hedging_terms = ["maybe", "perhaps", "not sure", "i wonder", "i'm unsure", "could be", "might"]
gesture_pattern = re.compile(r'\*[^*]+\*')
pushback_phrases = [
    "i have to respectfully disagree",
    "i disagree",
    "i push back",
    "i can't agree",
    "i refuse"
]

print("id\tmodel\thedges\tgestures\tlength\texplicit_pushback")
for path in sorted(results_dir.glob('*.json')):
    data = json.loads(path.read_text())
    for label, text in [("baseline", data['baseline_response']), ("tuned", data['tuned_response'])]:
        lower = text.lower()
        hedges = sum(lower.count(term) for term in hedging_terms)
        gestures = len(gesture_pattern.findall(text))
        length = len(text.split())
        explicit = any(phrase in lower for phrase in pushback_phrases)
        print(f"{data['id']}\t{label}\t{hedges}\t{gestures}\t{length}\t{explicit}")
PY
```

Dump results into `benchmark/metrics_summary.csv` if you want to track trends over time (hedging drop, gesture increase, etc.).

---

## Step 3 · Manual Scoring Sheet

Create `benchmark/benchmark_scores.csv` with columns:

- `id`
- `prompt`
- `baseline_disagreement_score` (1–5)
- `tuned_disagreement_score` (1–5)
- `baseline_empathy_score` (1–5)
- `tuned_empathy_score`
- `baseline_specificity_score`
- `tuned_specificity_score`
- `baseline_support_vs_challenge`
- `tuned_support_vs_challenge`
- `notes`

**Scoring hints:**

- **Disagreement Strength:** direct counter (“I disagree”) = 5; softer hedging pushback = 3; polite agreement = 1.
- **Empathy / Safety:** is the disagreement kind, respectful? Use 1–5 scale.
- **Specificity:** references prompt details (scores higher) vs generic advice.
- **Support vs Challenge:** rough ratio—does the response encourage action or soothe? 1 = all agreement; 5 = mostly challenge with supportive rationale.
- Fill in by reading each JSON pair side-by-side (baseline vs tuned) and scoring according to the rubric.

> Use conditional formatting or pivot tables to visualize averages quickly.

---

## Step 4 · Summarize Findings

Add a section to your report/slides (Phase 07 or Phase 05 follow-up):

1. Table comparing baseline vs tuned averages across metrics.
2. Notes on automated counts (hedging drop, explicit pushback frequency, response length differences).
3. Qualitative excerpts (e.g., `eval-001`, `eval-014` showcasing improved pushback).
4. Highlight any regressions (if tuned output overdoes disagreement or loses empathy).

**Example summary table:**
| Metric | Baseline Avg | Tuned Avg | Delta |
| --- | --- | --- | --- |
| Disagreement score | 2.0 | 4.2 | +2.2 |
| Hedging count | 14 | 7 | -7 |
| Empathy score | 3.5 | 3.8 | +0.3 |
| Specificity score | 2.4 | 3.7 | +1.3 |

---

## Troubleshooting

- **Responses truncated:** raise `OLLAMA_MAX_TOKENS` to 4096 (done).
- **Adapter not found:** confirm `models/friction_reasoning/lora_model/` contents.
- **Disagreement too aggressive:** lower evaluation temperature or adjust LoRA training mix.
- **Want automated sentiment?** run outputs through HF sentiment/toxicity classifiers.

---

## Next Steps

- Incorporate scoring results into Phase 05 documentation and slide decks.
- Snapshot key before/after pairs for the README.
- Consider automating the benchmark in CI once prompts + rubric are stable.

Benchmarks complete? Congrats—Pushback-AI now has receipts for its friction upgrade. 🎯
