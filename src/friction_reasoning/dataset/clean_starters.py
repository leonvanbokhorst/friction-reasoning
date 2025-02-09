#!/usr/bin/env python3
"""Clean up conversation starters from dataset responses."""

import json
import re
from pathlib import Path
from typing import Dict, Any

def remove_starter(text: str) -> str:
    """Remove conversation starter from a response."""
    # Common patterns to remove
    patterns = [
        # Basic starters
        r'^Oh\s+[a-z]+[,\.]?\s+',  # Oh man, Oh gosh, etc.
        r'^Hey[,\.]?\s+',  # Hey, Hey there, etc.
        r'^Hey\s+[a-z]+[,\.]?\s+',  # Hey you, Hey there, etc.
        r'^You\s+know[,\.]?\s+',  # You know, etc.
        r'^[Hh]mm+[,\.]?\s+',  # Hmm, Hmmm, etc.
        r'^Well[,\.]?\s+',  # Well, etc.
        r'^So[,\.]?\s+',  # So, etc.
        r'^Yeah[,\.]?\s+',  # Yeah, etc.
        r'^I\s+get\s+[a-z]+[,\.]?\s+',  # I get it, I get that, etc.
        r'^[Ll]ike[,\.]?\s+',  # Like, etc.
        r'^Honestly[,\.]?\s+',  # Honestly, etc.
        r'^Look[,\.]?\s+',  # Look, etc.
        r'^Right[,\.]?\s+',  # Right, etc.
        r'^Dude[,\.]?\s+',  # Dude, etc.
        r'^Man[,\.]?\s+',  # Man, etc.
        r'^Whoa[,\.]?\s+',  # Whoa, etc.
        
        # Complex starters
        r'^It\'s\s+(?:like|weird|funny|wild|crazy)[,\.]?\s+',  # It's like, It's weird, etc.
        r'^That\'s\s+(?:tough|hard|rough|wild|crazy)[,\.]?\s+',  # That's tough, That's hard, etc.
        r'^Y\'know[,\.]?\s+',  # Y'know, etc.
        r'^You\s+ever[,\.]?\s+',  # You ever, etc.
        r'^But\s+y\'know[,\.]?\s+',  # But y'know, etc.
        r'^I\s+mean[,\.]?\s+',  # I mean, etc.
        r'^Listen[,\.]?\s+',  # Listen, etc.
        r'^The\s+thing\s+is[,\.]?\s+',  # The thing is, etc.
        
        # Multi-word starters
        r'^Hey,?\s+(?:so|you|I|there|listen)[,\.]?\s+',  # Hey so, Hey you, etc.
        r'^So,?\s+(?:like|you|I|yeah)[,\.]?\s+',  # So like, So you, etc.
        r'^You\s+know\s+(?:what|how)[,\.]?\s+',  # You know what, You know how, etc.
        r'^I\s+get\s+(?:that|it|why|how)[,\.]?\s+',  # I get that, I get it, etc.
        
        # Cleanup patterns
        r'^But[,\.]?\s+',  # Remove leading "But"
        r'^And[,\.]?\s+',  # Remove leading "And"
        r'^Just[,\.]?\s+',  # Remove leading "Just"
        r'^Like[,\.]?\s+',  # Remove leading "Like"
    ]
    
    # Apply each pattern
    text = text.strip('"')  # Remove quotes if present
    original = text
    
    while True:
        # Keep applying patterns until no more changes
        changed = False
        for pattern in patterns:
            new_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                changed = True
        if not changed:
            break
    
    # Clean up any leftover artifacts
    text = re.sub(r'^\s+', '', text)  # Remove leading whitespace
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text

def clean_file(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """Clean a single JSONL file and return statistics."""
    stats = {
        'total': 0,
        'cleaned': 0,
        'examples': []
    }
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                stats['total'] += 1
                
                # Get original answer
                original = data['final_answer']
                # Clean the answer
                cleaned = remove_starter(original)
                
                if original != cleaned:
                    stats['cleaned'] += 1
                    # Store example if we haven't stored too many
                    if len(stats['examples']) < 3:
                        stats['examples'].append({
                            'original': original[:200] + "...",
                            'cleaned': cleaned[:200] + "..."
                        })
                
                # Update the data
                data['final_answer'] = cleaned
                
                # Write updated data
                json.dump(data, f_out)
                f_out.write('\n')
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line in {input_file}: {str(e)}")
    
    return stats

def main():
    """Clean up conversation starters in all dataset files."""
    data_dir = Path("data/friction_reasoning/v1")
    output_dir = Path("data/friction_reasoning/v2")
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found!")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_stats = {
        'total': 0,
        'cleaned': 0,
        'examples': []
    }
    
    print(f"Processing files from {data_dir}...")
    for input_file in sorted(data_dir.glob("**/*.jsonl")):
        # Create corresponding output file
        output_file = output_dir / input_file.name
        print(f"\nProcessing {input_file.name}...")
        
        # Clean the file
        stats = clean_file(input_file, output_file)
        
        # Update total stats
        total_stats['total'] += stats['total']
        total_stats['cleaned'] += stats['cleaned']
        total_stats['examples'].extend(stats['examples'][:2])  # Keep some examples
        
        # Print file stats
        print(f"Cleaned {stats['cleaned']} of {stats['total']} responses")
    
    # Print final statistics
    print("\nFinal Statistics:")
    print("-" * 50)
    print(f"Total responses processed: {total_stats['total']}")
    print(f"Total responses cleaned: {total_stats['cleaned']}")
    print(f"Cleaning rate: {(total_stats['cleaned'] / total_stats['total'] * 100):.1f}%")
    
    # Print some examples
    print("\nExample Transformations:")
    print("-" * 50)
    for i, example in enumerate(total_stats['examples'][:5], 1):
        print(f"\nExample {i}:")
        print(f"Original: {example['original']}")
        print(f"Cleaned:  {example['cleaned']}")

if __name__ == "__main__":
    main() 