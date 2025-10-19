# Phase 01 · Synthetic Pipeline

## Story Beat — From Agreeable AI to Engineered Friction

We started with a simple pain: our assistants nodded along, never challenging assumptions. Pushback-AI begins when we decide to manufacture tension on purpose, zeroing in on disagreement-heavy dialogues.

- **Call to adventure**: tame the comfort monster by scripting arguments
- **New rule**: every dataset example must feel raw, emotional, unfinished
- **Goal**: produce disagreement-centered multi-agent dialogues that lean into friction rather than smoothing it out

## Why Friction Matters

Designers Luna Maurer and Roel Wouters remind us that friction is a “lively, intrinsic experience” that keeps digital interactions human, resisting the “death by convenience” of frictionless design [**Designing Friction**](https://designingfriction.com). Their manifesto argues for deliberate discomfort, slower pacing, and non-positive expression so people re-engage with nuance and unpredictability. Pushback-AI literalizes that call by scripting agents who resist smoothing over tension and instead surface messy emotional truth.

## Technical Walkthrough — How the Pipeline Works

### Ingredient 1: Question Generator

We spin up unfinished, emotionally charged prompts by remixing template starters and affective tags.

```28:64:src/friction_reasoning/dataset/generate_dataset.py
async def generate_question(llm: LLMClient) -> str:
    category = random.choice(list(BASE_PROMPTS.keys()))
    base = random.choice(BASE_PROMPTS[category])
    base2 = random.choice(BASE_PROMPTS[category])
    emotion = random.choice(EMOTIONS)
    emotion2 = random.choice(EMOTIONS)
    # ...
    llm.temperature = random.uniform(0.6, 0.8)
    response = await llm.complete(prompt)
    return response.strip()
```

- Randomized emotional palette ensures variety.
- Cleanup logic strips quotes and escapes, so JSONL stays tidy.

### Ingredient 2: Six-Agent Gauntlet

Each persona replies in sequence, hearing all previous thoughts and optionally injecting uncertainty.

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
    result["thinking_pattern"]["raw_thought_stream"] = await inject_vulnerability(...)
    agent_responses.append(result)
    previous_thoughts = "\n".join(resp["thinking_pattern"]["raw_thought_stream"] for resp in agent_responses)
```

- Ordering nudges the conversation from doubt → memory → mechanics → contrarian push → humility → synthesis.
- `inject_vulnerability()` randomly inserts uncertainty phrases so transcripts sound hesitant.

### Ingredient 3: Stats and Resumable Batching

We track dataset health and write batches safely, allowing reruns without duplicates.

```259:347:src/friction_reasoning/dataset/generate_dataset.py
output_dir = Path("data/friction_reasoning")
existing_batches = list(output_dir.glob("batch_*.jsonl"))
seen_ids = set()
# resume existing batches, update stats, avoid duplicate IDs
with tqdm(total=num_examples, initial=len(dataset)) as pbar:
    while len(dataset) < num_examples:
        batch = await generate_batch(...)
        with open(current_batch_file, "w") as f:
            for dp in batch:
                f.write(json.dumps(dp) + "\n")
```

- `DatasetStats` records question/response lengths and recent errors.
- `seen_ids` prevents ID collisions if generation restarts.
- Progress bar + periodic summaries let us monitor distribution drift mid-run.

### Ingredient 4: Test Harness

A quick sanity check keeps hallucinated batches out of production.

```224:258:src/friction_reasoning/dataset/generate_dataset.py
async def test_generation():
    llm = LLMClient(model="openai/gpt-4o-mini", temperature=0.8)
    for _ in range(3):
        question = await generate_question(llm)
        datapoint = await generate_datapoint(llm, question, stats)
        # prints preview snippets
```

- Run via `python -m friction_reasoning.dataset --test` to demo the agent voices live.
- Use during talks to show the raw thought streams before training.

## Data Artifacts

| File                                                       | Description                                      |
| ---------------------------------------------------------- | ------------------------------------------------ |
| `data/friction_reasoning/batch_*.jsonl`                    | Intermediate batches (safe to resume or inspect) |
| `data/friction_reasoning/friction_reasoning_dataset.jsonl` | Consolidated output ready for upload             |
| `logs/*`                                                   | Console logs capturing stats and error messages  |

## Example Dialogue — Disagreement in the Wild

Lifted straight from the published dataset [`leonvanbokhorst/friction-disagreement-v2`](https://huggingface.co/datasets/leonvanbokhorst/friction-disagreement-v2), this sample shows the six-agent relay in action:

```
Question
"That alternate timeline where I actually followed my gut and moved to that little beach town instead of staying here... I wonder what my life would be like now..."

problem_framer
"Hmmm... I sense the user is fixated on the idea that moving to that beach town would have led to a better life, but that's a pretty narrow view of what could have been. Life is full of unexpected twists and turns, and staying put might have opened up opportunities that are just as valuable, if not more so, than what they imagine would have happened by the beach. It’s easy to romanticize the road not taken, but the reality is that every choice has its own set of challenges and rewards, and it’s possible that the user’s current path holds potential for growth and fulfillment that they’re overlooking."

memory_activator
"I remember when I faced a similar decision about moving to a city that seemed vibrant and full of promise, much like that beach town the user fantasizes about. I was torn between the allure of a fresh start and the comfort of my familiar surroundings. In the end, I chose to stay, but there was a persistent nagging feeling of what might have been—an ache that accompanied my choice. Yet, as time passed, I began to see the unique opportunities that unfolded in my current path, often overshadowed by the romanticized version of the life I could have had. It’s a reminder that while the user may feel drawn to that alternate timeline, the present holds its own potential for joy and growth, even if it’s not what they initially envisioned."

mechanism_explorer
"It's clear the user is fixated on the fantasy of that beach town, but honestly, it's just a daydream without substance. They need to realize that life isn’t about chasing an idealized version of what could have been; it’s about making the most of where they are now. Every choice has its consequences—staying here might lead to new experiences that can be just as fulfilling, if not more so. Romanticizing the road not taken can blind them to the value of their current path, which could be bursting with opportunities they’re completely overlooking."

perspective_generator
"But wait, the idea that moving to that beach town would have guaranteed a better life is fundamentally flawed. Sure, the allure of a sun-soaked paradise is enticing, but it glosses over the reality that every decision comes with its own set of challenges. The user seems caught up in a fantasy that overlooks the richness of their current experiences and the potential for growth right where they are. Rather than pining for an imagined life, they should recognize the unique opportunities and unexpected joy that can arise from embracing their present circumstances."

synthesizer
"I can see that the user is overly fixated on the allure of that beach town, romanticizing what their life could have been if they had made that move. However, this perspective is fundamentally flawed; it overlooks the inherent challenges that come with any decision, including relocating. While it's tempting to envision a perfect life by the beach, the reality is that every path has its own hurdles and rewards. By focusing solely on an idealized alternate timeline, the user risks dismissing the unique opportunities and growth potential available in their current situation. It's crucial for them to understand that fulfillment comes from embracing the present, rather than pining for a fantasy that may not have been as idyllic as imagined."

Answer:
I have to disagree with your fixation on that alternate timeline where you moved to the beach town. You seem to be caught up in a fantasy that suggests everything would have been perfect if you’d just followed your gut. But let’s be real: life is never that straightforward. Sure, the beach sounds idyllic and the thought of a fresh start is enticing, but that path wouldn’t have come without its own set of challenges.  The truth is that every decision you make—whether it’s staying put or moving somewhere new—comes with its own hurdles and uncertainties.

You’re romanticizing what could have been, but that alternate life may not have been as perfect as you envision. You could have faced unexpected struggles, and the reality might not have lived up to the dream you’ve constructed in your mind.  Instead of mulling over what might have been, I encourage you to focus on the opportunities and richness of your current life.

Every moment, every experience has the potential to contribute to your growth and fulfillment. By fixating on an idealized version of a life you didn’t choose, you could be overlooking the unique joys and lessons that your present circumstances have to offer. Embracing where you are now might just lead to unexpected happiness and growth that far exceeds what you imagine could have been by the beach.
```

## Lessons Learned

- Friction emerges when personas hear each other; multi-turn context is non-negotiable.
- Random vulnerability injections keep transcripts from sounding overconfident.
- Resumable batching saved hours when rate limits or network hiccups struck mid-run.

## Next Phase Preview

With the synthetic pipeline humming, we turned to sculpting each agent’s voice—dialing sarcasm, drama, and contrarian energy through prompt templates.
