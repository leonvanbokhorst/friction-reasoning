# Phase 04 · Hugging Face Relay

## Story Beat — Shipping Friction to the Community

By this point the pipeline is producing rich, multi-agent transcripts. Phase 04 is the return journey: packaging those dialogues, documenting their flavor, and delivering them to the community via Hugging Face. The tale isn’t just “we uploaded a file”—it’s about preserving tension, vulnerability, and uncertainty in a shareable format while honoring platform expectations.

## Environment & Secrets Checklist

Before the upload script runs, we load environment variables and confirm credentials.

```10:113:src/friction_reasoning/dataset/upload.py
load_dotenv()
...
env_path = Path(__file__).parents[3] / ".env"
if not env_path.exists():
    raise ValueError(f".env file not found at {env_path}")
load_dotenv(env_path)
hf_token = os.getenv("HUGGINGFACE_API_KEY")
if not hf_token:
    raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
```

- Two layers: a generic `.env` load at import time, plus a specific lookup three directories up (`/friction-reasoning/.env`).
- Explicit error messages mean talks can highlight “watch for missing tokens” as a friction point we embraced.

## Dataset Card Creation

We auto-generate a Markdown dataset card, embedding metadata and schema details.

```13:84:src/friction_reasoning/dataset/upload.py
def create_dataset_card(...):
    return f"""---
annotations_creators:
- machine-generated
...
# Dataset Card for {dataset_name}
...
- `agent_responses`: List of agent reasoning processes
  - `agent_type`
  - `thought_stream`
  - `friction_moments`
..."""
```

- Ensures every upload explains why multi-agent friction matters.
- The card explicitly lists `friction_moments` even if empty today—invites future work.

## Upload Workflow Breakdown

The async `upload_to_hub` function handles everything end-to-end.

```86:191:src/friction_reasoning/dataset/upload.py
print("\nUploading dataset to Hugging Face Hub: {repo_id}")
dataset = Dataset.from_json(dataset_path)
num_examples = len(dataset)
...
api = HfApi(token=hf_token)
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=private,
    exist_ok=True
)
api.upload_file(
    path_or_fileobj=dataset_card.encode(),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset"
)
dataset.push_to_hub(
    repo_id,
    token=hf_token,
    private=private,
    commit_message="Update friction reasoning dataset"
)
```

Key steps we can narrate:

1. Load the consolidated JSONL produced in Phase 01.
2. Create/verify the target dataset repo (idempotent thanks to `exist_ok=True`).
3. Upload the dataset card as `README.md`—the storytelling piece.
4. Push the actual records via `Dataset.push_to_hub` with a descriptive commit.

## CLI Moments Worth Sharing

Running the upload script peppers the console with human-friendly messages: number of examples loaded, dataset card upload status, final URL. Perfect material for talk screenshots.

## Quickstart Checklist

- Refresh your `.env` to include `HUGGINGFACE_API_KEY` before running any upload commands.
- Preview the dataset card output with `python -m friction_reasoning.dataset.upload --preview-card` to confirm tone and formatting.
- Run the upload script with `python -m friction_reasoning.dataset.upload --repo-id leonvanbokhorst/friction-disagreement-v2 --private False` (adjust flags for new datasets or staging runs).
- After upload, visit the Hugging Face repo page and skim the rendered README to ensure tables, links, and bullet lists look correct.
- Tag the release on the Hugging Face UI so collaborators get notified.

## Data Path Recap

| Source                                                     | Description                        |
| ---------------------------------------------------------- | ---------------------------------- |
| `data/friction_reasoning/friction_reasoning_dataset.jsonl` | Combined dataset (Phase 01 output) |
| `src/friction_reasoning/dataset/upload.py`                 | Upload orchestrator                |

## Troubleshooting Notes

- **Missing token errors**: double-check both the project `.env` and your shell session; the script intentionally halts rather than attempting an anonymous upload.
- **README formatting glitches**: inspect the generated Markdown for triple backticks or tables that might need escaping; rerun with the `--dry-run` flag to iterate quickly.
- **Dataset push timeouts**: Hugging Face SDK retries automatically, but you can split a massive dataset across multiple pushes by chunking the JSONL ahead of time.

## Storytelling Angle

- **Preserve the friction**: highlight that the dataset card explains disagreement & vulnerability so downstream users know what they’re getting (e.g., `leonvanbokhorst/friction-disagreement-v2`).
- **Designing Friction tie-in**: we aren’t chasing frictionless API interactions; we log progress, caution about tokens, and accept the upload isn’t “one click.” It matches the manifesto’s call to embrace deliberate slowness and transparency.
- **Community bridge**: once the dataset lives at `https://huggingface.co/datasets/leonvanbokhorst/friction-disagreement-v2` (and siblings), others can analyze or fine-tune, spreading the Pushback ethos.

## Lessons Learned

- Automating the dataset card ensures every release ships with context, not just raw JSONL.
- Explicit environment checks save time and become part of the story (“we forgot the token once; the script caught it”).
- Hugging Face’s `push_to_hub` handles versioning, letting us iterate on new friction flavors (uncertainty, disagreement, reluctance).

## Next Phase Preview

Phase 05 dives into training: how we mix these datasets, stack examples into longer conversations, and fine-tune DeepSeek-R1 with LoRA while keeping the pushback spirit alive. Continue with `05_finetuning.md` for the training deep dive.
