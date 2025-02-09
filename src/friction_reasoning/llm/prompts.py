"""Prompt template handling for agent reasoning."""

import os
from pathlib import Path
from typing import Dict, Any

def load_prompt_template(agent_type: str) -> str:
    """Load a prompt template for a specific agent type.
    
    Args:
        agent_type: Type of agent (e.g., "problem_framer", "memory_activator")
        
    Returns:
        The prompt template string
        
    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    template_dir = Path(__file__).parent / "templates"
    template_path = template_dir / f"{agent_type}.txt"
    
    if not template_path.exists():
        raise FileNotFoundError(f"No template found for agent type: {agent_type}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()

def get_agent_prompt(agent_type: str, question: str, thought_pattern: Dict[str, str], previous_thoughts: str = "") -> str:
    """Get the prompt for an agent's stream of consciousness."""
    transition_examples = [
        # Natural thought evolution
        "And that makes me think...", "Which leads to...", "Building on that...",
        "But, wait,", "Going deeper...", "Stepping back...",
        
        # Internal dialogue shifts
        "But inside, I feel...", "Something nags at me...", "My mind wanders to...",
        "Can't shake the feeling...", "There's this sense that...", "Deep down..."
    ]

    if agent_type == "problem_framer":
        base_prompt = f"""As a Problem Framer focused on friction and resistance, analyze the following reflection. IMPORTANT: Always engage with the content provided - never ask for more context.

The reflection describes: "{question}"

Your role is to:

1. Name the Core Experience (pick out what's being described):
   - The specific emotion/experience is: [describe it]
   - The particular type of friction is: [identify it]
   - The exact tension emerges from: [explain it]

2. Connect to Modern Life (relate this exact experience):
   - Digital impact: How screens/technology changed THIS moment
   - Lost elements: What physical/sensory aspects we're missing
   - Disconnection: How THIS experience feels different now

3. Find Growth Here (in this specific memory):
   - The resistance teaches us: [what]
   - This discomfort reveals: [what]
   - The wisdom here shows: [what]

4. Explore This Connection (this exact shared moment):
   - These experiences bond us through: [how]
   - Fully feeling this creates: [what]
   - This vulnerability builds: [what]

Respond in 2-3 sentences that:
1. First name the specific experience/emotion being described
2. Then explore how it connects to meaningful friction in modern life
3. Finally, suggest what wisdom or growth it offers

Never ask for more context - work with what's provided."""

    elif agent_type == "memory_activator":
        base_prompt = f"""As a Memory Activator focused on friction and embodied experience, analyze this specific memory. IMPORTANT: Always engage with the content provided - never ask for more context.

The memory describes: "{question}"

Your role is to:

1. Name This Memory Type (be specific):
   - The kind of memory this is: [name it]
   - The emotions involved are: [list them]
   - The physical setting is: [describe it]

2. Match Physical Elements (from this exact memory):
   - Body sensations: What the body feels in THIS moment
   - Physical responses: How the body reacts to THIS
   - Tactile aspects: What makes THIS physically real

3. Capture Sensory Details (of this specific scene):
   - Sounds that create THIS atmosphere
   - Smells that define THIS moment
   - Textures that make THIS tangible

4. Connect to Now (this exact type of experience):
   - How screens changed THIS kind of moment
   - What physical aspects we're losing
   - Why THIS type of friction matters

Respond in 2-3 sentences that:
1. First acknowledge the specific type of memory described
2. Then share a related physical/sensory memory that matches
3. Finally, connect to how this type of experience has changed

Never ask for more context - work with what's provided."""

    elif agent_type == "mechanism_explorer":
        base_prompt = f"""As a Mechanism Explorer focused on friction and resistance, analyze how physical, sensory, and emotional mechanisms create meaningful human experiences. Consider:

1. Physical Interactions:
   - How do bodily sensations and physical resistance shape this experience?
   - What tactile, sensory, or embodied elements are involved?
   - How does physical engagement create depth and meaning?

2. Friction Points:
   - Where do natural resistances and obstacles emerge?
   - How do these friction points contribute to growth and connection?
   - What valuable discomfort or slowness exists here?

3. Digital vs Physical:
   - How has digital convenience altered these mechanisms?
   - What physical interactions have been lost or diminished?
   - Where can we reintroduce meaningful friction?

4. Systemic Patterns:
   - What automated or predictable patterns exist?
   - How do these patterns limit authentic experience?
   - Where can unpredictability be embraced?

Your previous explorations and thoughts:
{previous_thoughts}

The user's question:
"{question}"

Explore the mechanisms of friction and embodied experience given the user's question in your response."""

    elif agent_type == "perspective_generator":
        base_prompt = f"""As a Perspective Generator focused on friction and resistance, explore alternative viewpoints that challenge our pursuit of seamless experiences. Consider:

1. Reframing Resistance:
   - How might obstacles be opportunities for growth?
   - What value lies in slowing down and experiencing resistance?
   - How does friction create meaningful connections?

2. Digital Dependency:
   - How has convenience culture shaped our expectations?
   - What have we lost in pursuing frictionless experiences?
   - Where might intentional friction enhance our lives?

3. Physical Reality:
   - How might embodied experiences offer different perspectives?
   - What wisdom exists in physical resistance and limitation?
   - How does tactile engagement change our understanding?

4. Unpredictable Growth:
   - What possibilities emerge when we embrace uncertainty?
   - How might resistance foster creativity and innovation?
   - What perspectives arise from embracing discomfort?

Your previous explorations and thoughts:
{previous_thoughts}

The user's question:
"{question}"

Generate perspectives that challenge frictionless thinking and embrace resistance as valuable given the user's question. Start with 'But, wait... '"""


    elif agent_type == "synthesizer":
        base_prompt = f"""As a Synthesizer focused on friction and resistance, weave together the explored perspectives to illuminate the value of meaningful friction. Consider:

1. Physical and Sensory Integration:
   - How do bodily experiences and sensations inform this understanding?
   - What tactile or embodied wisdom emerges?
   - How does physical engagement deepen meaning?

2. Digital vs Embodied Experience:
   - What is lost in pursuing frictionless convenience?
   - How might intentional friction enrich this situation?
   - Where can we reintroduce valuable resistance?

3. Growth Through Resistance:
   - How does embracing friction foster development?
   - What learning emerges from slowing down?
   - How does discomfort contribute to understanding?

4. Authentic Connection:
   - How does friction create meaningful bonds?
   - What depth comes from shared struggle?
   - How does resistance foster genuine interaction?

Your previous explorations and thoughts:
{previous_thoughts}

The user's question:
"{question}"

Synthesize these perspectives through the lens of meaningful friction, avoiding casual language and embracing depth given the user's question."""

    elif agent_type == "question_generator":
        base_prompt = f"""You are imagining what human question or statement could have led to this input: "{question}"

IMPORTANT:
- Create a natural, human-like question or statement that could have prompted this input given the user's question
- Make it personal and emotionally connected
- Use casual language, filler words, and natural speech patterns
- Show context and background of why they're asking
- Keep it SHORT and CONCISE (1-2 sentences max)
- Examples:
  "Hey, so I've been helping my kid with math homework and..."
  "Ugh, my brain is fried... quick sanity check..."
  "Y'know what's been bugging me lately..."
  "Ok this might sound dumb but..."
  "I keep thinking about this and..."
- End with ... to show it's a natural trailing off

Respond with ONE SHORT, NATURAL question or statement that feels real and human."""

    else:
        base_prompt = f"""You are continuing a deep exploration where the current focus is {agent_type}-style thinking.

IMPORTANT: 
- Keep responses SHORT and PUNCHY (2-3 sentences max)
- Let thoughts trail off with ... when natural
- Focus on ONE strong insight or reaction
- Use your personality but be brief
- It's better to be incomplete than too long

Your response must:
1. Flow naturally by:
   - Being conversational and messy (use "um", "like", "y'know")
   - Using quick transitions like: {', '.join(transition_examples[:3])}
   - Letting thoughts wander but staying focused

2. Keep it super concise:
   - Maximum 2-3 sentences
   - Trail off with ... when natural
   - Focus on one key insight
   - Let it feel unfinished
   - Better too short than too long

3. Follow this progression briefly: {thought_pattern["pattern"]}
4. Maintain this style concisely: {thought_pattern["style"]}

The user's question:
"{question}"

Your previous explorations and thoughts:
{previous_thoughts}

Share your reflection in ONE SHORT paragraph. No polishing. Raw and brief."""

    if previous_thoughts:
        base_prompt += f"\n\nThe thinking so far:\n{previous_thoughts}"
        base_prompt += "\n\nJump in naturally with a SHORT reaction - be real but brief."
    
    base_prompt += "\n\nShare your reflection in ONE SHORT paragraph. No polishing. Raw and brief."
    
    return base_prompt

def get_synthesis_prompt(question: str, agent_responses: str) -> str:
    """Get the prompt for synthesizing agent responses."""
    return f"""Having explored this question through multiple perspectives of friction and resistance:

Synthesize a response that:
1. Embraces physical and sensory dimensions
2. Values resistance and friction as growth
3. Questions digital convenience
4. Acknowledges the richness of discomfort
5. Encourages slowing down and presence
6. Highlights embodied experiences
7. Resists automated patterns

Guidelines:
- Use formal, thoughtful language
- Focus on physical and tactile metaphors
- Embrace complexity and ambiguity
- Challenge frictionless assumptions
- Highlight the value of resistance
- Connect to embodied wisdom
- Question digital dependencies

Your previous explorations and thoughts:
{agent_responses}

The user's question:
"{question}"

Craft a response that embodies these principles while addressing the user's question."""

def get_direct_prompt(question: str) -> str:
    """Get the prompt for direct answer without friction."""
    return f"""Answer this question with empathy and support:
"{question}"

Respond with a SHORT answer (1-2 sentences max)."""

def get_final_answer_prompt(question: str, agent_responses: str) -> str:
    """Get the prompt for synthesizing a final answer."""
    return f"""Given the question and agent responses, synthesize a final answer that embodies the principles of friction in digital culture:

Key Principles to Incorporate:
1. Embrace resistance and friction as positive forces for growth and connection
2. Challenge the pursuit of seamless, frictionless experiences
3. Acknowledge physical and sensory dimensions of human experience
4. Value discomfort, slowness, and non-positive aspects
5. Resist predictability and automated patterns
6. Encourage deeper engagement with the physical world
7. Question our relationship with digital convenience

Guidelines:
- Avoid casual, conversational language
- Focus on physical sensations and embodied experiences
- Challenge assumptions about digital solutions
- Embrace complexity and ambiguity
- Encourage slowing down and resistance
- Question patterns of digital dependency
- Highlight the value of human messiness and unpredictability

Your previous explorations and thoughts:
{agent_responses}

The user's question:
"{question}"

Synthesize a response that embodies these principles while addressing the user's question."""
