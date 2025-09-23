#!/usr/bin/env python3

import sys
sys.path.append('.')

from src.eval.reasoning_extractor import ReasoningExtractor
from src.metrics.accuracy import extract_final_answer, gsm8k_em
import re

# Mock OpenAI-style vs Claude-style responses for the same problems
test_cases = [
    {
        "problem": "Test 1: Janet's eggs (Answer: 18)",
        "openai_style": """I need to calculate Janet's daily revenue from egg sales.

Given:
- 16 eggs laid per day
- 3 eggs for breakfast
- 4 eggs for muffins
- $2 per egg sold

Step 1: Calculate eggs used = 3 + 4 = 7
Step 2: Calculate eggs for sale = 16 - 7 = 9
Step 3: Calculate revenue = 9 × $2 = $18

Therefore, Janet makes $18 per day.""",
        
        "claude_style": """Certainly! I'll solve this problem step by step, showing all calculations and explaining my reasoning.

Step 1: Determine the total number of eggs Janet's ducks lay per day.
Total eggs per day = 16

Step 2: Calculate how many eggs Janet uses herself each day.
Eggs for breakfast = 3
Eggs for muffins = 4
Total eggs used by Janet = 3 + 4 = 7

Step 3: Calculate the number of eggs remaining for sale.
Eggs for sale = Total eggs - Eggs used by Janet
Eggs for sale = 16 - 7 = 9

Step 4: Calculate the revenue from selling the remaining eggs.
Price per egg = $2
Revenue = Eggs for sale × Price per egg
Revenue = 9 × $2 = $18

Reasoning: Janet's ducks produce 16 eggs daily. She uses 7 of these eggs (3 for breakfast and 4 for muffins), leaving 9 eggs available for sale. Each egg is sold for $2 at the farmers' market."""
    },
    {
        "problem": "Test 4: James sprints (Answer: 540)",
        "openai_style": """Looking at this step by step:

- 3 sprints per session
- 3 sessions per week  
- 60 meters per sprint

Distance per session: 3 × 60 = 180 meters
Total weekly distance: 180 × 3 = 540 meters

Answer: 540""",
        
        "claude_style": """Certainly! I'll solve this problem step by step, showing all my work and reasoning.

Step 1: Understand the given information
- James runs 3 sprints per session
- He does this 3 times a week
- Each sprint is 60 meters

Step 2: Calculate the distance James runs in one session
Distance per session = Number of sprints × Distance per sprint
Distance per session = 3 × 60 meters
Distance per session = 180 meters

Step 3: Calculate the total distance James runs in a week
Weekly distance = Distance per session × Number of sessions per week
Weekly distance = 180 meters × 3
Weekly distance = 540 meters

Reasoning: 
James runs 180 meters in each session because he does 3 sprints of 60 meters each. Since he does this 3 times a week, we multiply the distance per session (180 meters) by 3 to get the total weekly distance.

Final answer: James runs a total of 540 meters a week."""
    }
]

def analyze_extraction_patterns():
    print("=== Comparing OpenAI vs Claude Answer Extraction ===\n")
    
    extractor = ReasoningExtractor()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"TEST CASE {i}: {test_case['problem']}")
        print(f"{'='*60}")
        
        for model_type in ['openai_style', 'claude_style']:
            response = test_case[model_type]
            print(f"\n--- {model_type.upper().replace('_', ' ')} ---")
            print(f"Response length: {len(response)} chars")
            print(f"Response preview: {response[:100]}...")
            
            # Test different extraction methods
            print("\n🔍 EXTRACTION RESULTS:")
            
            # Method 1: ReasoningExtractor (original problematic method)
            extracted_reasoning, summary = extractor.extract_math_answer(response)
            print(f"1. ReasoningExtractor: '{extracted_reasoning}'")
            
            # Method 2: GSM8K extract_final_answer (the fix)
            gsm8k_answer = extract_final_answer(response)
            print(f"2. GSM8K extract_final_answer: '{gsm8k_answer}'")
            
            # Method 3: Manual analysis of patterns
            print(f"3. Manual pattern analysis:")
            
            # Show what each regex pattern matches
            reasoning_patterns = [
                (r"(?:final answer|answer|result|solution)(?:\s*is)?:?\s*([+-]?\d+(?:\.\d+)?)", "Final answer patterns"),
                (r"(?:therefore|thus|so),?\s*(?:the\s+)?(?:answer\s+is\s*)?([+-]?\d+(?:\.\d+)?)", "Conclusion patterns"),
                (r"####\s*([+-]?\d+(?:\.\d+)?)", "GSM8K hash patterns"),
                (r"=\s*([+-]?\d+(?:\.\d+)?)\s*$", "Equation end patterns"),
                (r"([+-]?\d+(?:\.\d+)?)(?:\s*\.?\s*$)", "Last number patterns")
            ]
            
            for pattern, description in reasoning_patterns:
                matches = list(re.finditer(pattern, response, re.IGNORECASE | re.MULTILINE))
                if matches:
                    match_values = [m.group(1) for m in matches]
                    print(f"   - {description}: {match_values} (using last: '{match_values[-1]}')")
            
            # Show all numbers found
            all_numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", response)
            print(f"   - All numbers found: {all_numbers[:10]}{'...' if len(all_numbers) > 10 else ''}")
            print(f"   - Last number: '{all_numbers[-1] if all_numbers else 'None'}'")
            
        print("\n")

def analyze_response_structure():
    print("\n=== Response Structure Analysis ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"TEST {i}: {test_case['problem']}")
        
        for model_type in ['openai_style', 'claude_style']:
            response = test_case[model_type]
            lines = response.split('\n')
            
            print(f"\n{model_type.upper().replace('_', ' ')} Structure:")
            print(f"  - Total lines: {len(lines)}")
            print(f"  - Non-empty lines: {len([l for l in lines if l.strip()])}")
            
            # Count calculation lines
            calc_lines = [l for l in lines if '=' in l and any(d.isdigit() for d in l)]
            print(f"  - Calculation lines: {len(calc_lines)}")
            
            # Show final lines
            final_lines = [l.strip() for l in lines[-3:] if l.strip()]
            print(f"  - Last 3 lines: {final_lines}")
            
            # Count step markers
            step_markers = len([l for l in lines if 'step' in l.lower() or 'stage' in l.lower()])
            print(f"  - Step markers: {step_markers}")
            
        print("-" * 40)

if __name__ == "__main__":
    analyze_extraction_patterns()
    analyze_response_structure()