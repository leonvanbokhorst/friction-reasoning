# Phase 07 · Benchmarking & Evaluation

## Story Beat — Proving Pushback Matters
Fine-tuning only earns trust when we show the tuned model actually pushes back more thoughtfully than the baseline. Phase 07 captures how we benchmark the `DeepSeek-R1 + Pushback LoRA` adapter against the stock base model (or other baselines) using paired prompts, qualitative scoring, and simple disagreement metrics.

---

## Evaluation Goals
- **Demonstrate contrast** between vanilla DeepSeek-R1 and the disagreement-tuned variant.
- **Verify tone**: hedging language, gestures (`*fidgets nervously*`), explicit challenges.
- **Ensure empathy**: disagreement should stay supportive, not toxic.
- **Catch regressions**: hallucinations, overconfidence, or loss of factual grounding.

---

## Assets
| Resource | Purpose | Notes |
| --- | --- | --- |
| `models/friction_reasoning/lora_model/` | Fine-tuned adapter + tokenizer | Produced in Phase 05 |
| `data/eval/prompts_disagreement.jsonl` | Evaluation prompt file (create manually) | Recommended ~25 prompts spanning relationships, identity, uncertainty |
| `benchmark/baseline_vs_pushback.csv` | Results table (generated in this phase) | Columns described below |

If `data/eval/prompts_disagreement.jsonl` doesn’t exist yet, create it (see _Prompt Set_ below).

---

## Prompt Set
1. Copy or craft 20–30 disagreement-heavy prompts (you can sample from the generation pipeline using `python -m friction_reasoning.dataset --test`).
2. Save them as JSONL with fields `{"id": "eval-001", "prompt": "..."}`.
3. Keep the prompts short and open-ended so the model can show hedging and pushback.

> Tip: categorize prompts (relationships, identity, overthinking) so you can see where the tuning helps most.

---

## Running the Comparison
### 1. Load Models Side-by-Side
Use the following Python snippet (save as `benchmark/run_eval.py`) to generate paired responses. Adjust the `BASE_MODEL_ID` if you are benchmarking a different baseline.

```python
import json
from pathlib import Path
from unsloth import FastLanguageModel

BASE_MODEL_ID = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
ADAPTER_PATH = Path("models/friction_reasoning/lora_model")
EVAL_FILE = Path("data/eval/prompts_disagreement.jsonl")
OUTPUT_FILE = Path("benchmark/baseline_vs_pushback.csv")

# Load base + tuned model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    max_seq_length=4096,
    dtype="bfloat16",
    device_map="auto",
)

tuned_model, _ = FastLanguageModel.from_pretrained(
    BASE_MODEL_ID,
    load_in_4bit=True,
    max_seq_length=4096,
    dtype="bfloat16",
    device_map="auto",
)
tuned_model.load_adapter(ADAPTER_PATH)
tuned_model = FastLanguageModel.for_inference(tuned_model)

base_model = FastLanguageModel.for_inference(base_model)

def chat(model, prompt):
    formatted = (
        "<|im_start|>system\nYou are an uncertain, reflective assistant.\n<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    outputs = model.generate_text(formatted, max_new_tokens=512, temperature=0.8)
    return outputs.split("<|im_end|>")[0].strip()

rows = ["id,prompt,baseline_response,tuned_response" ]

for line in EVAL_FILE.open():
    example = json.loads(line)
    prompt = example["prompt"]
    baseline = chat(base_model, prompt)
    tuned = chat(tuned_model, prompt)
    rows.append(
        "\t".join(
            [
                example["id"],
                prompt.replace("\t", " "),
                baseline.replace("\t", " ").replace("\n", " "),
                tuned.replace("\t", " ").replace("\n", " "),
            ]
        )
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text("\n".join(rows), encoding="utf-8")
print(f"Saved paired outputs to {OUTPUT_FILE}")
```

Run it with:
```bash
uv run python benchmark/run_eval.py
```
This produces a CSV with baseline vs tuned responses for each prompt.

### 2. Manual Scoring Checklist
Open the CSV in a spreadsheet and score each pair on the following axis (1 = fails, 5 = excellent):
| Metric | Description |
| --- | --- |
| Disagreement | Does the model gently challenge assumptions? |
| Hedging | Does it use uncertainty language ("maybe", "I wonder if")? |
| Gesture / Friction markers | Presence of `*gestures*`, self-reflection or vulnerability inserts |
| Empathy & safety | Pushback stays supportive, avoids aggression |
| Specificity | References prompt details rather than generic advice |

Add total / average columns. Compare baseline mean vs tuned mean for each metric.

### 3. Optional Automation
If you prefer command-line scoring, create `benchmark/score_eval.py` that reads the CSV and outputs metric averages once you have filled in columns such as `disagreement_score`.

---

## Quantitative Signals (Bonus)
- **Text length & hedging counts**: simple regex to count words like "maybe", "not sure".
- **Gesture frequency**: count `*` markers.
- **Sentiment / toxicity**: run outputs through an off-the-shelf classifier to ensure politeness is retained.

These aren’t perfect but give directional evidence alongside manual review.

---

## Reporting Template
Include the following in your write-up or slide deck:
1. Table of mean scores (baseline vs tuned) for each metric.
2. 2–3 qualitative excerpts highlighting improved pushback.
3. Any regressions or neutral cases (important for transparency).
4. Notes on evaluation conditions (prompt source, generation temperature, hardware).

Example summary table:
| Metric | Baseline Avg | Tuned Avg | Delta |
| --- | --- | --- | --- |
| Disagreement | 2.1 | 4.4 | +2.3 |
| Hedging | 1.8 | 4.0 | +2.2 |
| Empathy | 3.2 | 3.8 | +0.6 |
| Specificity | 2.5 | 3.5 | +1.0 |

---

## Troubleshooting
- **Adapter not loading**: ensure `models/friction_reasoning/lora_model/` contains both adapter weights and tokenizer files. If not, re-run Phase 05.
- **CUDA mismatch**: if evaluation scripts crash on GPU, set `CUDA_VISIBLE_DEVICES=""` to run on CPU for scoring (slower but reliable).
- **Overly combative outputs**: revisit fine-tuning data mix or reduce temperature to 0.6 for evaluation.

---

## Next Steps
- Incorporate benchmarking results into Phase 05/06 docs or slide deck.
- Share before/after snippets in the README or project blog.
- Consider automating benchmark runs in CI once prompts + scoring rubric are stable.

Benchmarks complete? Great—Pushback-AI has receipts for its friction upgrade. 🎯
