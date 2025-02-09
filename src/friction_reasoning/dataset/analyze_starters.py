#!/usr/bin/env python3
"""Analyze conversation starters in the dataset."""

import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

def extract_starter(text: str) -> str:
    """Extract the conversation starter from a response."""
    # Remove quotes if present
    text = text.strip('"')
    
    # Common patterns to look for
    patterns = [
        r'^(Oh\s+[a-z]+,?\s+)',  # Oh man, Oh gosh, etc.
        r'^(Hey\s+[a-z]*,?\s+)',  # Hey, Hey there, etc.
        r'^(You\s+know\s+[a-z]*,?\s+)',  # You know what, etc.
        r'^([Hh]mm+,?\s+)',  # Hmm, Hmmm, etc.
        r'^(Well,?\s+)',  # Well, etc.
        r'^(So,?\s+)',  # So, etc.
        r'^(Yeah,?\s+)',  # Yeah, etc.
        r'^(I\s+get\s+[a-z]*,?\s+)',  # I get it, I get that, etc.
        r'^([Ll]ike,?\s+)',  # Like, etc.
        r'^(Honestly,?\s+)',  # Honestly, etc.
        r'^(Look,?\s+)',  # Look, etc.
        r'^(Right,?\s+)',  # Right, etc.
        r'^(It\'s\s+[a-z]+,?\s+)',  # It's like, It's weird, etc.
        r'^(That\'s\s+[a-z]+,?\s+)',  # That's tough, That's hard, etc.
    ]
    
    # Try to match each pattern
    for pattern in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no pattern matches, return first few words
    words = text.split()
    if words:
        return " ".join(words[:2]).strip()
    return ""

def analyze_starters(data_dir: Path) -> Tuple[Counter, List[Dict]]:
    """Analyze conversation starters in all dataset files."""
    starters_counter = Counter()
    examples = []
    
    # Process all batch files
    for batch_file in sorted(data_dir.glob("**/*.jsonl")):
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    final_answer = data.get('final_answer', '')
                    if final_answer:
                        starter = extract_starter(final_answer)
                        if starter:
                            starters_counter[starter.lower()] += 1
                            # Store example if it's one of first 3 for this starter
                            if starters_counter[starter.lower()] <= 3:
                                examples.append({
                                    'starter': starter,
                                    'answer': final_answer[:200] + "...",  # First 200 chars
                                    'file': batch_file.name
                                })
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing {batch_file}: {str(e)}")
    
    return starters_counter, examples

def main():
    """Analyze and print statistics about conversation starters."""
    data_dir = Path("data/friction_reasoning/v1")
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found!")
        return
    
    print("Analyzing conversation starters...")
    starters_counter, examples = analyze_starters(data_dir)
    
    # Print statistics
    total = sum(starters_counter.values())
    print(f"\nFound {total} responses with starters")
    print("\nTop 20 most common starters:")
    print("-" * 50)
    for starter, count in starters_counter.most_common(20):
        percentage = (count / total) * 100
        print(f"{starter:<20} {count:>5} ({percentage:>5.1f}%)")
    
    # Print examples
    print("\nExample responses for top 5 starters:")
    print("-" * 50)
    top_starters = [s for s, _ in starters_counter.most_common(5)]
    for starter in top_starters:
        print(f"\n{starter.upper()}:")
        starter_examples = [e for e in examples if e['starter'].lower() == starter.lower()]
        for ex in starter_examples[:2]:  # Show up to 2 examples per starter
            print(f"- {ex['answer']}")
            print(f"  (from {ex['file']})")

if __name__ == "__main__":
    main() 