# Phase 03 · Multi-Perspective Orchestration

## Story Beat — Choreographing the Secret Dojo
With the prompt personas forged, Phase 03 choreographs how they pass the mic. The goal: keep disagreement compounding rather than resetting each turn. Every agent must hear what came before, react, and nudge the tension forward until a hesitant synthesis emerges.

## Sequential Relay Mechanics

`generate_datapoint()` lays out the choreography: instantiate the six agents, track prior thoughts, and feed them forward.

```127:170:src/friction_reasoning/dataset/generate_dataset.py
agents = [
    Agent("problem_framer"),
    Agent("memory_activator"),
    Agent("mechanism_explorer"),
    Agent("perspective_generator"),
    Agent("limitation_acknowledger"),
    Agent("synthesizer")
]
agent_responses = []
previous_thoughts = ""
for agent in agents:
    result = await generate_agent_reasoning(llm, question, agent, previous_thoughts)
    result["thinking_pattern"]["raw_thought_stream"] = await inject_vulnerability(
        result["thinking_pattern"]["raw_thought_stream"]
    )
    agent_responses.append(result)
    previous_thoughts = "\n".join(
        resp["thinking_pattern"]["raw_thought_stream"]
        for resp in agent_responses
    )
```

- `previous_thoughts` is the baton—each agent sees the cumulative transcript and must respond in context.
- Ordering matters: skepticism → memory → mechanism → perspective flip → humility → synthesis.
- Post-processing via `inject_vulnerability()` keeps the tone uncertain and self-questioning.

## Agent Execution Details

Inside `generate_agent_reasoning()` we wire prompts, prior context, and sampling temperature.

```42:59:src/friction_reasoning/agent_reasoning.py
prompt = get_agent_prompt(
    agent.agent_type,
    question,
    agent.thought_patterns[agent.agent_type],
    previous_thoughts
)
llm.temperature = random.uniform(0.7, 1.0)
response = await llm.complete(prompt)
return {
    "agent_type": agent.agent_type,
    "thinking_pattern": {
        "raw_thought_stream": response,
        "friction_moments": []
    }
}
```

- We randomize temperature per turn (0.7–1.0) to avoid monotony.
- `get_agent_prompt()` injects persona style + pattern plus the `previous_thoughts` context.

### Prompt Assembly Under the Hood

`LLMClient.complete()` builds an OpenAI-compatible chat payload and handles retries.

```19:44:src/friction_reasoning/llm/client.py
messages = []
if system:
    messages.append({"role": "system", "content": system})
messages.append({"role": "user", "content": prompt})
response = await acompletion(
    model=self.model,
    messages=messages,
    temperature=self.temperature
)
return response.choices[0].message.content
```

- Persona prompts are passed as a single user message, keeping the conversation state in the prompt text itself.
- Because we only send one message, we avoid external state complexity; everything lives in `previous_thoughts`.

## Vulnerability Injection

To maintain the “hesitant human” vibe, we randomly splice uncertainty phrases into each thought stream.

```113:125:src/friction_reasoning/dataset/generate_dataset.py
if random.random() < VULNERABILITY_CONFIG["injection_probability"]:
    prefix = random.choice(
        VULNERABILITY_CONFIG["uncertainty_phrases"] +
        VULNERABILITY_CONFIG["limitation_acknowledgments"]
    )
    sentences = response.split(". ")
    insert_point = random.randint(0, len(sentences) - 1)
    sentences.insert(insert_point, prefix)
    return ". ".join(sentences)
return response
```

- Keeps transcripts from sounding overconfident.
- Adds linguistic markers ("I might be missing something") that the next agent can riff on.

## Synthesis Pass

After all six agents think out loud, we prompt a synthesis that still honors uncertainty.

```61:66:src/friction_reasoning/agent_reasoning.py
thoughts = chr(10).join(
    f"{resp['thinking_pattern']['raw_thought_stream']}"
    for resp in agent_responses
)
prompt = get_synthesis_prompt(question, thoughts)
return await llm.complete(prompt)
```

- Final response is intentionally short (3–4 sentences) and hesitant.
- For disagreement-focused datasets we can swap in `get_disagreement_synthesis_prompt()` to yield firmer pushback.

## Concurrency & Resilience

Batch generation runs multiple questions concurrently while respecting unique IDs and error tracking.

```187:222:src/friction_reasoning/dataset/generate_dataset.py
question_tasks = [generate_question(llm) for _ in range(batch_size)]
questions = await asyncio.gather(*question_tasks, return_exceptions=True)
...
datapoint_tasks = [
    generate_datapoint(llm, question, stats)
    for question in valid_questions
]
completed = await asyncio.gather(*datapoint_tasks, return_exceptions=True)
```

- We parallelize question + datapoint generation for throughput, but each datapoint still runs the six-agent relay sequentially.
- Errors get logged via `DatasetStats.add_error`, and batches retry after a cool-down.

## Example Relay (Condensed)

Snippet from [`leonvanbokhorst/friction-disagreement-v2`](https://huggingface.co/datasets/leonvanbokhorst/friction-disagreement-v2) showing responses chaining off each other:

```
Question
"Saw their name pop up and my stomach just... dropped, like maybe I shouldn’t have looked..."

problem_framer
"Hmmm... I sense the user is letting their emotions dictate their actions..."

memory_activator
"I remember a time when I encountered an old friend's name, and my heart sank..."

mechanism_explorer
"I see that when the user encountered the name, their immediate emotional reaction..."

perspective_generator
"But, wait... I realize I might be oversimplifying the user's situation..."

synthesizer
"I recognize that, upon seeing that name, the user immediately felt a wave of discomfort..."
```

Notice how each persona references the emotional cues left by the prior page of the transcript—the relay works because `previous_thoughts` gives them the full history.

## Implementation Checklist

- Confirm the agent list in `generate_dataset.py` matches the templates present on disk; ordering changes the emotional arc.
- Keep an eye on combined transcript length—if the relay grows past token limits, trim gestures or tighten templates to maintain context.
- During long runs, sample a few `batch_*.jsonl` files and scan the `previous_thoughts` field to ensure vulnerability phrases are appearing naturally.

## Troubleshooting Notes

- **Agents start ignoring each other**: inspect `previous_thoughts` construction; stray whitespace or encoding issues can break the context chain.
- **Vulnerability injection feels repetitive**: expand `VULNERABILITY_CONFIG` phrases or decrease the injection probability slightly.
- **Async gather crashes**: wrap the dataset generation CLI with `--max-retries` to avoid infinite loops when the upstream model is unstable.
- **`previous_thoughts`** carries the transcript forward so each persona responds with full context.
- **Vulnerability injection hook** spices each thought stream with additional uncertainty.
- **Training note**: Later, the fine-tuning pipeline concatenates several of these sequential turns so the model practices holding context across multiple exchanges; the run-time prompt handoff here remains strictly persona-to-persona.

## Tying Back to Designing Friction

The [Designing Friction](https://designingfriction.com) manifesto celebrates discomfort, slowness, and “non-positive” expression. Our orchestration mechanics put that into practice: by forcing agents to sit with each other’s messy, hesitant thoughts, we prevent the conversation from snapping back to frictionless reassurance.

## Lessons Learned

- Sequential context passing (`previous_thoughts`) is the secret sauce—remove it and agents go off-topic.
- Injected uncertainty phrases create threads that downstream agents can pull, keeping friction alive.
- Running batches concurrently saves time, but each example preserves strict turn ordering.

## Next Phase Preview

Phase 04 zooms out from generation to distribution: how we package these multi-agent transcripts, craft dataset cards, and push everything to Hugging Face without losing the friction narrative. Jump to `04_hf_delivery.md` for the handoff.

We also stitch consecutive conversations together so the fine-tuning samples contain several sequential turns; this stacking happens here in `generate_datapoint()` (and later in the training pipeline), not during the prompt-generation phase.
