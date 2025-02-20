"""Base prompts for generating friction reasoning dataset."""

from typing import Dict, List

# Base prompts that can be combined and modified
BASE_PROMPTS: Dict[str, List[str]] = {
    "emotional": [
        "Ugh, my heart's doing that thing again where...",
        "Just had one of those moments where everything inside me...",
        "Why does my brain always pick 3am to...",
        "Trying to breathe through this feeling of...",
        "Anyone else get randomly blindsided by...",
        "My therapist would probably say it's because...",
        "Funny how your body tells you things like...",
        "That moment when your emotions just...",
    ],
    "relationships": [
        "Saw their name pop up and my stomach just...",
        "Why do we keep dancing around this thing where...",
        "Text conversations hit different when...",
        "Kinda hurts how some people just casually...",
        "Getting tired of being the friend who always...",
        "Maybe I'm toxic for thinking this but...",
        "The way they looked at me made me wonder if...",
        "Starting to notice a pattern in how I...",
    ],
    "identity": [
        "Okay but who even am I when nobody's...",
        "Getting real with myself about how I might be...",
        "Lowkey freaking out about becoming...",
        "Used to think I was the type of person who...",
        "Trying on different versions of myself like...",
        "Keep catching glimpses of who I could be if...",
        "What if this isn't even my real personality but...",
        "The gap between who I am online and offline...",
    ],
    "existential": [
        "3am thoughts hitting different like...",
        "Do you ever just stop and get scared about...",
        "Having an existential crisis about how we...",
        "My brain broke a little thinking about how...",
        "Kinda terrifying how we're all just...",
        "Getting lost in the void thinking about...",
        "Reality feels glitchy when you realize...",
        "Trying to process how we're all just...",
    ],
    "society": [
        "Is anyone else exhausted by how we're all...",
        "The way social media makes us think we need to...",
        "Getting weird vibes from how everyone's...",
        "Can we talk about this pressure to always...",
        "Why are we all pretending it's normal to...",
        "The algorithm knows me better than my friends and...",
        "Society really got us thinking we gotta...",
        "Living through *gestures at everything* while...",
    ],
    "growth": [
        "Just caught myself doing that thing where...",
        "Past me would be shook seeing me now...",
        "Having to unlearn so much about how I...",
        "Growing pains hit different when you're...",
        "Plot twist: maybe I was the toxic one when...",
        "Healing is weird because suddenly you're...",
        "That moment when you realize you're becoming...",
        "Kind of scary how much I've changed since...",
    ],
    "memory": [
        "Brain randomly decided to replay that time...",
        "A song came on and suddenly I'm back in...",
        "Memory unlocked: that random moment when...",
        "Why do I keep thinking about that one time...",
        "Getting emotional about how we used to...",
        "Found an old photo and now I'm spiraling about...",
        "That core memory just resurfaced where...",
        "Weird how your mind suddenly throws you back to...",
    ],
    "dreams": [
        "Lowkey manifesting a reality where...",
        "Living in my head rent free: that scenario where...",
        "Daydreaming again about how life could be if...",
        "Anyone else build entire futures around...",
        "Stuck between wanting to dream big and...",
        "In my imaginary perfect life I'm always...",
        "Keep fantasizing about dropping everything to...",
        "That alternate timeline where I actually...",
    ],
    "uncertainty": [
        "I might be wrong about this, but...",
        "Not entirely sure, though I think...",
        "From what I understand, although I could be missing something...",
        "Based on my current knowledge, while acknowledging my limitations...",
        "This is just my perspective, but...",
        "I'm wrestling with this idea where...",
        "Still learning about this, but my current thinking is...",
        "Take this with a grain of salt, but...",
        "Something feels off about my thinking here...",
        "Maybe I'm overthinking, but there's this part where...",
        "The more I think about it, the less sure I am that...",
        "Is it weird that I keep second-guessing...",
        "Part of me thinks one thing, but then another part...",
        "My brain keeps going back and forth between...",
        "There's this nagging doubt I can't shake about...",
        "The line gets blurry when I try to figure out...",
    ],
    "vulnerability": [  # New category for explicitly vulnerable moments
        "Sometimes I worry that I'm not qualified to...",
        "Feel kinda exposed admitting this, but...",
        "My imposter syndrome is screaming because...",
        "Trying to be honest with myself about how...",
        "It's scary to admit, but I might be...",
        "Never told anyone this, but I sometimes...",
        "The real reason I hesitate is probably...",
        "Behind all my confidence, I'm actually...",
    ]
}

# Emotional states to influence question generation
EMOTIONS: List[str] = [
    "nostalgic", "anxious", "curious", "hopeful", "confused",
    "overwhelmed", "peaceful", "restless", "grateful", "uncertain",
    "frustrated", "excited", "sad", "happy", "angry", "bored", "sleepy",
    "hungry", "thirsty", "tired", "sick", "happy", "sad", "angry", "bored", 
    
    # New nuanced emotional states
    "vulnerable", "hesitant", "ambivalent", "conflicted", "doubtful",
    "self-conscious", "introspective", "unsure", "contemplative",
    "questioning", "wavering", "tentative", "humble", "candid",
    "raw", "exposed", "authentic", "unguarded"
]

# Configuration for vulnerability-aware responses
VULNERABILITY_CONFIG: Dict[str, Any] = {
    "uncertainty_phrases": [
        "I'm not entirely certain about",
        "I might be missing something regarding",
        "My understanding is limited when it comes to",
        "I could be wrong, but",
        "This is just my perspective on",
        "Take this with appropriate skepticism, but",
        "Based on what I know, though that may be incomplete,",
        
        # New more nuanced uncertainty phrases
        "I find myself questioning my assumptions about",
        "There's a complexity here I'm still grappling with regarding",
        "The more I consider this, the more I realize I might not fully understand",
        "I'm sitting with some uncertainty about",
        "Perhaps I need to reconsider my thinking on",
        "I'm trying to hold space for multiple possibilities about",
        "This is a tentative understanding, but",
        "I'm learning to be comfortable with not knowing everything about"
    ],
    "limitation_acknowledgments": [
        "As an AI, I might not fully grasp",
        "My experience with this is limited to",
        "I'm still learning about",
        "This is beyond my direct experience, but",
        "I can only speak from my training, not personal experience about",
        "While I don't have firsthand experience with",
        "Acknowledging my limitations regarding",
        
        # New more nuanced limitation acknowledgments
        "Given the complexity of human experience, I may not fully capture",
        "There are nuances to this that might escape my understanding of",
        "I want to be transparent about my limitations when it comes to",
        "Human experiences are rich and varied, and I might miss subtle aspects of",
        "I'm aware that my perspective might be incomplete when discussing",
        "This touches on depths of human experience I can only approximate in",
        "I aim to be helpful while acknowledging my constraints around",
        "There's a depth to human understanding that I might not fully reach in"
    ],
    "injection_probability": 0.3  # 30% chance to inject vulnerability
}

# Configuration for disagreement-focused dataset generation
DISAGREEMENT_CONFIG: Dict[str, Any] = {
    "dataset_name": "deepseek-r1-disagree-v1",
    "description": "A dataset focused on productive disagreement as digital friction",
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
        {
            "type": "challenge_assumption",
            "description": "Agent challenges a common assumption about digital agreement"
        },
        {
            "type": "explore_tension",
            "description": "Agent explores the value in disagreement and tension"
        },
        {
            "type": "reframe_negative",
            "description": "Agent reframes seemingly negative interactions as valuable"
        }
    ]
} 