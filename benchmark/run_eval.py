import json
from pathlib import Path
from unsloth import FastLanguageModel

BASE_MODEL_ID = "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
ADAPTER_PATH = Path("models/friction_reasoning/lora_model")
EVAL_FILE = Path("data/eval/prompts_disagreement.jsonl")
OUTPUT_FILE = Path("benchmark/baseline_vs_pushback.csv")

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

rows = ["id\tprompt\tbaseline_response\ttuned_response"]

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
