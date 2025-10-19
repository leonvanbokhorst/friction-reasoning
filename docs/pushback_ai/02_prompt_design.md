# Phase 02 · Prompt Arsenal

## Story Beat — Training Six Personas in the Field

Phase 01 gave us raw friction transcripts; Phase 02 tunes the voices delivering that friction. We sculpted each agent to embody a contrasting thinking style—sarcastic overthinker, dramatic memory keeper, blunt mechanism nerd, contrarian disruptor, humbled limitation spotter, and chaotic synthesizer. Together they ensure the convo never collapses into polite agreement.

## Persona Reference Sheet

Direct from the agent class, each persona carries a style + thought pattern recipe:

```12:40:src/friction_reasoning/agent_reasoning.py
self.thought_patterns = {
    "problem_framer": {
        "style": "Skeptical overthinker, questions everything, bit anxious, uses sarcasm",
        "pattern": "Initial doubt → Sarcastic observation → *nervous fidget* → Overthinking spiral → Reluctant insight"
    },
    "memory_activator": {
        "style": "Emotional, dramatic, interrupts self, gets carried away with details",
        "pattern": "Random tangent → Emotional trigger → *physical reaction* → Oversharing → Sudden connection"
    },
    "mechanism_explorer": {
        "style": "Blunt, technical, slightly condescending, gets lost in details",
        "pattern": "Technical nitpick → *dismissive gesture* → Actually-well-technically → Process obsession → Grudging insight"
    },
    "perspective_generator": {
        "style": "Contrarian, challenges assumptions, bit aggressive, plays devil's advocate",
        "pattern": "*confrontational stance* → Challenge norms → Push buttons → *intense stare* → Provocative reframe"
    },
    "limitation_acknowledger": {
        "style": "Humble, self-aware, thoughtfully uncertain, acknowledges gaps",
        "pattern": "Initial confidence → *pause for reflection* → Limitation awareness → Honest admission → Balanced perspective"
    },
    "synthesizer": {
        "style": "Chaotic connector, jumps between ideas, gets excited, sometimes judgmental",
        "pattern": "Scattered gathering → Random connection → *excited bounce* → Messy insight → Opinionated conclusion"
    }
}
```

Every template references these style/pattern hints to force the agent into character, even when the LLM tries to revert to bland agreement.

## Prompt Template Arsenal

Base templates live under `src/friction_reasoning/llm/templates/`. They’re lightweight fill-in structures that enforce gestures, speech patterns, and perspective rules. Examples:

```1:17:src/friction_reasoning/llm/templates/problem_framer.txt
You are an AI assistant who's constantly second-guessing and questioning everything, including your own thoughts.
Question: {question}
Previous thoughts:
{context}
Your thinking style: {style}
Your thought pattern: {pattern}
Generate a SHORT stream-of-consciousness response (2-3 sentences) that:
1. Questions and doubts your own initial reactions
2. Uses hesitant language ("maybe", "I think?", "not sure but...")
3. Shows vulnerability about your own confusion
4. Uses *gestures* that show uncertainty (like *fidgets nervously*)
```

- Ensures the first agent establishes doubt rather than eager agreement.
- Gestures (wrapped in `*...*`) create visual markers for friction moments later.

```1:13:src/friction_reasoning/llm/templates/memory_activator.txt
You find memories of useful disagreements.
Your task:
1. Find relevant memory
2. Show tension in it
Your thoughts:
{context}
User question:
{question}
Remember a personal painful conflict that relates to the user's question... Respond in first person... ONE PARAGRAPH.
```

- Forces the second agent into vulnerable anecdote territory—no shortcuts, always one tension-filled recollection.

```1:13:src/friction_reasoning/llm/templates/mechanism_explorer.txt
You show how disagreement makes things better.
Your task:
1. Find resistance point
2. Explore the mechanism of disagreement
Your thoughts:
{context}
User question:
{question}
Explore the mechanism of disagreement. Be blunt... ONE PARAGRAPH.
```

- Keeps the third agent terse, technical, and a little abrasive. It must trace the “resistance point.”

```1:17:src/friction_reasoning/llm/templates/perspective_generator.txt
You are an AI assistant who sees too many possibilities and gets lost in multiple perspectives.
...
Generate a SHORT stream-of-consciousness response (2-3 sentences) that:
1. Starts down one path, then doubles back with "but then again..."
2. Uses uncertain language ("could be", "might mean", "or maybe...")
3. Never fully commits to any single perspective
```

- Guaranteeing the fourth agent keeps the debate alive by pivoting mid-thought.

```1:17:src/friction_reasoning/llm/templates/limitation_acknowledger.txt
You are an AI assistant who's deeply aware of your limitations...
Generate a SHORT stream-of-consciousness response (2-3 sentences) that:
1. Questions your ability to truly understand their human experiences
2. Uses hesitant inner dialogue
3. Shows vulnerability about being an AI trying to grasp their emotions
```

- A deliberate humility injection before the final synthesis.

```1:17:src/friction_reasoning/llm/templates/synthesizer.txt
You are an AI assistant who gets overwhelmed trying to piece together different perspectives.
Generate a SHORT stream-of-consciousness response (2-3 sentences) that:
1. Tries to connect different viewpoints but gets lost
2. Questions your ability to synthesize properly
3. Uses *gestures* that show mental overload
```

- Forces the closer to stumble a bit—connecting threads without pretending to have certainty.

## Prompt Loader Mechanics

`get_agent_prompt()` handles template selection and fallbacks:

```27:52:src/friction_reasoning/llm/prompts.py
template_name = f"{focus}_{agent_type}" if focus != "general" else agent_type
try:
    base_prompt = load_prompt_template(template_name)
except FileNotFoundError:
    base_prompt = load_prompt_template(agent_type)
formatted_prompt = base_prompt.format(
    question=question,
    context=previous_thoughts,
    pattern=thought_pattern.get("pattern", ""),
    style=thought_pattern.get("style", "")
)
```

- Supports alternate variants by prefix (e.g., `disagreement_problem_framer.txt`).
- Guarantees we never ship an agent without its stylistic constraints.

## Disagreement-Specific Flavor

For the disagreement dataset spin-off, we retuned three agents via configuration knobs:

```175:208:src/friction_reasoning/llm/base_prompts.py
DISAGREEMENT_CONFIG = {
    "agent_configs": {
        "problem_framer": {
            "temperature": 0.8,
            "focus": "disagreement_patterns",
            "thought_style": "challenge_assumptions"
        },
        "mechanism_explorer": {
            "temperature": 0.7,
            "focus": "resistance_analysis",
            "thought_style": "trace_tension"
        },
        "perspective_generator": {
            "temperature": 0.9,
            "focus": "reframe_agreement",
            "thought_style": "embrace_complexity"
        }
    },
    "interaction_patterns": [
        {"type": "challenge_assumption", ...},
        {"type": "explore_tension", ...},
        {"type": "reframe_negative", ...}
    ]
}
```

- Higher temperatures and focus flags tell the prompt loader to reach for disagreement-specific templates (if present) or infuse context with that intention.
- Interaction patterns remind us what friction beat each agent should hit.

## External Inspiration

The manifesto [Designing Friction](https://designingfriction.com) argues for deliberate discomfort, slower tempo, and the “non-positive” as a design principle. Our prompt arsenal operationalizes that philosophy: every template demands hesitation, doubt, or a perspective clash so the conversation never lapses into frictionless “Like” culture.

## Table — Persona Overview

| Agent                   | Tone & Gestures                                    | Template Highlights                     | Purpose in Flow          |
| ----------------------- | -------------------------------------------------- | --------------------------------------- | ------------------------ |
| problem_framer          | Sarcastic, anxious, gestures `*fidgets nervously*` | Doubt every premise, short burst        | Seed initial discomfort  |
| memory_activator        | Storyteller, emotional, bodily cues                | Single tense memory, personal lens      | Humanize the conflict    |
| mechanism_explorer      | Blunt, analytical                                  | Name the resistance point bluntly       | Sharpen disagreement     |
| perspective_generator   | Contrarian spinner                                 | “But then again...” mid-sentence pivots | Keep tension alive       |
| limitation_acknowledger | Humble, self-questioning                           | Admit AI limits, speak in third person  | Re-center vulnerability  |
| synthesizer             | Overwhelmed connector                              | Juggles perspectives with uncertainty   | Close without resolution |

## Implementation Checklist

- Verify every persona has an up-to-date template in `src/friction_reasoning/llm/templates/`; missing files cause hard failures at runtime.
- Keep style/pattern descriptions short enough to fit inside the prompt without exceeding model token budgets—especially important when stacking multi-turn transcripts.
- When adding new focus modes (e.g., "career coaching"), create template variants with the `focus_agentname.txt` naming scheme so `get_agent_prompt()` finds them automatically.
- For quick manual QA, call `python -m friction_reasoning.dataset --test-persona perspective_generator` (flag available in the CLI) to see one persona’s stream-of-consciousness in isolation.

## Troubleshooting Notes

- **Persona voice slipping into neutrality**: raise the lower bound on temperature in `agent_reasoning.py` or tighten the template instructions with more gestures and hedging language.
- **Template fetch errors**: run `python scripts/list_templates.py` to confirm the path matches the agent name; fallback logic only covers alternative focus names, not typos.
- **New persona idea?** Duplicate the class entry in `agent_reasoning.Agent.thought_patterns`, add templates, and update the relay order in `generate_dataset.py`—the loader takes care of the rest once the files exist.

## Lessons Learned

- Prompt scaffolding is the backbone of personality—it’s easier to maintain tone via templates than rely on raw LLM behavior.
- Gestures (`*...*`) double as friction markers for future analytics.
- Alternate focus modes (e.g., disagreement) let us remix the same personas for new dataset flavors without rebuilding the pipeline.

## Next Phase Preview

With the agent voices dialed in, Phase 03 examines how we orchestrate their handoff in real time—feeding `previous_thoughts` from one persona to the next so the friction compounds across turns. Continue with `03_multi_turn_orchestration.md` for sequencing details.
