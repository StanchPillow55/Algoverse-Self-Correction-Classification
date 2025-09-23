#!/usr/bin/env python3

import sys
sys.path.append('.')

from src.eval.reasoning_extractor import ReasoningExtractor
from src.metrics.accuracy import extract_final_answer, gsm8k_em

# Test cases from Claude responses
test_cases = [
    {
        "name": "Test 3 (House flipping)",
        "claude_response": """Certainly! I'll walk you through the problem step by step, showing all calculations and explaining the reasoning.

Step 1: Understand the given information
- Josh bought a house for $80,000
- He spent $50,000 on repairs
- The repairs increased the value of the house by 150%

Step 2: Calculate the total investment
Total investment = Purchase price + Repair costs
Total investment = $80,000 + $50,000 = $130,000

Step 3: Calculate the increase in value
The value increased by 150% of the original purchase price.
Value increase = 150% of $80,000
Value increase = 1.5 × $80,000 = $120,000

Step 4: Calculate the new value of the house
New value = Original purchase price + Value increase
New value = $80,000 + $120,000 = $200,000

Step 5: Calculate the profit
Profit = New value - Total investment
Profit = $200,000 - $130,000 = $70,000

Reasoning:
Josh's total investment includes both the purchase price""",
        "expected_answer": "65000",  # From ground truth
        "correct_calculation": "70000"
    },
    {
        "name": "Test 4 (Sprint distance)",
        "claude_response": """Certainly! I'll solve this problem step by step, showing all my work and reasoning.

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

Final answer: James runs a total of 540 meters a week.""",
        "expected_answer": "540"
    }
]

def test_extraction():
    print("=== Testing Answer Extraction for Claude Responses ===\n")
    
    extractor = ReasoningExtractor()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['name']}")
        print(f"Expected: {test_case['expected_answer']}")
        
        # Test ReasoningExtractor
        extracted_answer, summary = extractor.extract_math_answer(test_case['claude_response'])
        print(f"ReasoningExtractor result: '{extracted_answer}'")
        
        # Test GSM8K extract_final_answer
        gsm8k_answer = extract_final_answer(test_case['claude_response'])
        print(f"GSM8K extract_final_answer result: '{gsm8k_answer}'")
        
        # Test accuracy calculation
        if extracted_answer:
            accuracy = gsm8k_em(extracted_answer, test_case['expected_answer'])
            print(f"Accuracy (extracted vs expected): {accuracy}")
        
        if gsm8k_answer:
            accuracy_gsm8k = gsm8k_em(gsm8k_answer, test_case['expected_answer'])
            print(f"Accuracy (GSM8K vs expected): {accuracy_gsm8k}")
        
        print(f"Reasoning summary: {summary[:200]}...")
        print("-" * 80)

if __name__ == "__main__":
    test_extraction()