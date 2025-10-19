# Phase 06 · Deployment & Sharing

## Story Beat — Carrying Pushback Beyond Our Workspace

With LoRA adapters trained (Phase 05), the final quest is delivering Pushback-AI in forms the wider community can actually run. Phase 06 covers merging adapters, producing GGUF artifacts, pushing them to Hugging Face, and packaging everything for Ollama. This is when the friction ethos leaves our workstation and lands in other people’s terminals.

## Merging LoRA + Base Model

`convert_to_gguf.py` handles the heavy lifting: clone/build llama.cpp, merge adapters with the base model, and export quantized GGUF variants.

```1:119:src/friction_reasoning/model_training/convert_to_gguf.py
setup_llama_cpp()
...
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model,
    max_seq_length=4096,
    dtype="bfloat16",
    load_in_4bit=True,
    device_map="auto",
)
...
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
)
model.load_adapter(lora_path, adapter_name="default")
...
model.save_pretrained_gguf(
    str(output_path),
    tokenizer,
    quantization_method=quant
)
```

Key beats to narrate:

- **`setup_llama_cpp()`** clones and builds llama.cpp if it isn’t already around, creating symlinks for Unsloth compatibility. Cue the montage of compiling C++—friction is literal here.
- LoRA + base weights load in 4-bit mode, then `model.save_pretrained_gguf` writes out quantized models (q5_k_m by default).
- Renaming to `unsloth.Q5_K_M.gguf` ensures consistent artifact naming.

### Output Directory Layout

| Path                                                   | Purpose                                       |
| ------------------------------------------------------ | --------------------------------------------- |
| `model_training/model_gguf/q5_k_m/unsloth.Q5_K_M.gguf` | Quantized merged model ready for distribution |
| `model_training/llama.cpp/`                            | Local build of llama.cpp used for conversion  |

## Publishing GGUF to Hugging Face

`push_to_hub.py` uploads the GGUF artifacts so others can download them without cloning the repo.

```1:61:src/friction_reasoning/model_training/push_to_hub.py
api = HfApi()
api.create_repo(hub_repo, exist_ok=True)
...
for quant in quantizations:
    src_path = model_dir / quant / f"unsloth.{quant.upper()}.gguf"
    hub_filename = f"deepseek-r1-mixture-of-friction-{quant}.gguf"
    shutil.copy2(src_path, temp_path)
    api.upload_file(
        path_or_fileobj=str(temp_path),
        path_in_repo=hub_filename,
        repo_id=hub_repo,
    )
    temp_path.unlink()
```

- Creates/updates the dataset repo `leonvanbokhorst/deepseek-r1-mixture-of-friction`.
- Uploads each quantized file with a descriptive name so users know the precision trade-off.
- Leaves the repo containing both dataset (Phase 04) and model (Phase 06) assets.

## Ollama Packaging (Narrative Guidance)

While Ollama packaging isn’t scripted in this repo, the typical pipeline after GGUF export is:

1. Copy the GGUF file to your Ollama models directory (defaults to `~/.ollama/models`). A quick `cp model_training/model_gguf/q5_k_m/unsloth.Q5_K_M.gguf ~/.ollama/models/` does the trick.
2. Write a `Modelfile` referencing the GGUF and describing metadata (name, parameters, template). Minimal example:

   ```
   FROM ./unsloth.Q5_K_M.gguf
   PARAMETER num_ctx 4096
   TEMPLATE """
   {{ .System }}

   {{ .Prompt }}
   """
   ```

3. Run `ollama create pushback-ai -f Modelfile` to generate the local model.
4. Test with `ollama run pushback-ai`. When it sounds like it belongs in the dojo, move on to distribution.

### Publishing to the Ollama Library

Getting the model into the shared Ollama registry takes a couple of extra beats:

1. `ollama login` so the CLI can push under your namespace.
2. Rename the model in the `Modelfile` (or during `ollama create`) to match `<username>/<model-name>`—e.g. `ollama create leonvanbokhorst/pushback-ai -f Modelfile`.
3. Attach a concise `LICENSE` and `README` alongside the `Modelfile`; Ollama surfaces these in the library UI.
4. Run `ollama push leonvanbokhorst/pushback-ai` to publish the artifact.
5. Double-check the listing at https://ollama.ai/library to confirm metadata, downloadable size, and that the GGUF variant (e.g. Q5_K_M) shows up correctly.

Emphasize during the talk: producing GGUF is the bridge that allows friction-trained behavior to run on laptops via Ollama—and pushing it upstream makes sparring partners discoverable outside the project walls.

## Quickstart Checklist

- Run `python -m friction_reasoning.model_training.convert_to_gguf --quant q5_k_m` after training to generate the merged GGUF.
- Verify the artifact (`model_training/model_gguf/q5_k_m/unsloth.Q5_K_M.gguf`) before uploading—file corruption usually means llama.cpp build issues.
- Upload to Hugging Face via `python -m friction_reasoning.model_training.push_to_hub --repo leonvanbokhorst/deepseek-r1-mixture-of-friction` to keep models and datasets together.
- Craft your `Modelfile` and create an Ollama build; test with `ollama run pushback-ai` to confirm prompt templates still resonate.
- Log the Hugging Face model URL and Ollama listing in the README so teammates can find the latest artifacts.

## Troubleshooting Notes

- **llama.cpp build failures**: rerun `setup_llama_cpp()` with a clean environment; outdated compilers are the usual culprit.
- **Upload script missing files**: ensure the GGUF naming matches `unsloth.{quant}.gguf`; the script builds filenames dynamically.
- **Ollama runs out of memory**: drop to a smaller quant (`q4_0`) or prune template defaults to reduce prompt tokens.

## Lessons Learned

- Building llama.cpp is a frictiony step on purpose—we document it so future contributors know exactly which commands run under the hood.
- Merging adapters and exporting quantized binaries ensures the Pushback personality survives outside the training stack.
- Publishing to Hugging Face + guiding Ollama packaging makes the project accessible: folks can download the dataset, run the fine-tuned model, or remix it further.

## Storytelling Angle

Tie back to [Designing Friction](https://designingfriction.com): distribution isn’t about smoothing everything over. We expose prerequisites (compiling llama.cpp, managing quantizations) because they embody the balance between accessibility and meaningful resistance.

## Closing the Journey

With Phase 06 complete, the Pushback-AI saga covers the full arc—from intentionally designing disagreement to sharing a runnable model that keeps asking uncomfortable questions. Encourage the audience to run the Ollama build, explore the dataset, and contribute new friction personas.
