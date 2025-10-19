# Pushback-AI Slide Storyboard

Presentation target: 12–15 minute walkthrough of the Pushback-AI pipeline, tailored for builders curious about disagreement-driven AI. Each slide pairs narrative intent with suggested visuals/demos.

---

## Slide 1 — Title: "Welcome to Pushback-AI"

- **Purpose**: Set the tone; introduce Pushback-AI as an intentional break from "agreeable" assistants.
- **Talking points**: Pain of nodding assistants; hero’s journey arc across six phases.
- **Visual cues**: Bold typography, timeline graphic, project logo or Hugging Face banner.

## Slide 2 — Problem Statement: "Death by Convenience"

- **Purpose**: Explain why friction matters.
- **Talking points**: Designing Friction manifesto; risk of frictionless UX; need for productive disagreement.
- **Visual cues**: Quote from designingfriction.com; contrasting smooth vs. jagged interaction diagram.

## Slide 3 — Solution Overview: "Pushback-AI Playbook"

- **Purpose**: Summarize the end-to-end workflow before diving deep.
- **Talking points**: Dataset generation → HF packaging → LoRA fine-tuning → GGUF/Ollama distribution.
- **Visual cues**: Pipeline diagram with six phases; icons for dataset, training, deployment.

## Slide 4 — Personas Primer: "Six Agents, Six Attitudes"

- **Purpose**: Introduce the agent roster forged in Phase 02.
- **Talking points**: Persona styles and gestures; role of templates; how gestures mark friction moments.
- **Visual cues**: Table or character cards for each persona; callout to `agent_reasoning.py` thought patterns.

## Slide 5 — Phase 01: "Manufacturing Raw Disagreement"

- **Purpose**: Highlight the synthetic pipeline mechanics.
- **Talking points**: Question generator randomness; six-agent relay; resumable batching.
- **Visual cues**: Annotated code snippet from `generate_dataset.py`; flowchart of question → agents → dataset.

## Slide 6 — Phase 02: "Prompt Arsenal Mechanics"

- **Purpose**: Show how tone control is enforced.
- **Talking points**: Template snippets; focus/temperature overrides; vulnerability injections.
- **Visual cues**: Side-by-side prompt fragments; persona-specific gestures; mention of `DISAGREEMENT_CONFIG`.

## Slide 7 — Phase 03: "Sequential Relay Orchestration"

- **Purpose**: Explain context passing and multi-turn choreography.
- **Talking points**: `previous_thoughts` baton; per-turn temperature jitter; vulnerability injection hook.
- **Visual cues**: Sequence diagram showing agents reading/writing transcript; highlight `generate_datapoint()` loop.

## Slide 8 — Phase 04: "Hugging Face Delivery Ritual"

- **Purpose**: Outline dataset distribution workflow.
- **Talking points**: Environment checks; auto-generated dataset card; upload steps via `push_to_hub`.
- **Visual cues**: CLI screenshot; dataset card excerpt; HF repo URL.

## Slide 9 — Phase 05: "Feeding DeepSeek-R1 a Friction Diet"

- **Purpose**: Dive into training pragmatics.
- **Talking points**: LoRA config; stacking 4K-token windows; response-only training; instrumentation metrics.
- **Visual cues**: Snippet of `config.yaml`; training stats screenshot; W&B chart or console output.

## Slide 10 — Phase 06: "From LoRA to Laptop"

- **Purpose**: Show deployment path to GGUF/Ollama.
- **Talking points**: llama.cpp setup; GGUF exports; push to Hugging Face; Ollama Modelfile guidance.
- **Visual cues**: Artifact directory listing; GGUF filename callouts; Ollama command snippet.

## Slide 11 — Live Demo or Sample Dialogue

- **Purpose**: Make the friction tangible.
- **Talking points**: Walk through a dialogue from `friction-disagreement-v2`; highlight gestures and uncertainty phrases.
- **Visual cues**: Highlighted transcript; callouts showing persona hand-offs and injected hesitations.

## Slide 12 — Lessons & Takeaways

- **Purpose**: Reflect on what the team learned.
- **Talking points**: Importance of intentional friction; benefits of templates; resilience in batching & deployment.
- **Visual cues**: Bullet list with icons; tie back to manifesto.

## Slide 13 — Call to Action

- **Purpose**: Invite collaboration and experimentation.
- **Talking points**: Run the dataset generator; fine-tune variants; contribute new personas; explore Ollama build.
- **Visual cues**: Links/QR codes; GitHub + HF badges; “Build your own friction” tagline.

## Slide 14 (Optional) — Appendix: Deep Dives

- **Purpose**: Provide backup material for Q&A.
- **Talking points**: Dataset stats; LoRA hyperparameters; troubleshooting tips (missing tokens, compile errors).
- **Visual cues**: Tabular stats; flow diagrams; error log snippets.

---

### Speaker Flow Tips

- Open with the friction manifesto story, then map each phase to a hero’s journey beat to keep narrative cohesion.
- When showing code, zoom into small snippets to avoid overwhelming the audience; focus on how each piece enforces disagreement.
- Anchor every technical detail with the mantra: "Does this keep the assistant from nodding along?" If not, skip it.
- Close by emphasizing how the dataset + model + deployment toolkit empower others to script their own productive tension experiments.
