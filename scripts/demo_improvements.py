#!/usr/bin/env python3
"""
Demonstration of the Full Reasoning Traces improvements to the teacher-learner pipeline.

This script showcases the key improvements made to capture full reasoning instead of just final answers.
"""

import os
import sys
from pathlib import Path
import re

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.reasoning_extractor import ReasoningExtractor


def demonstrate_improvements():
    """Demonstrate the key improvements to the reasoning trace system."""
    
    print("🧠 FULL REASONING TRACES - KEY IMPROVEMENTS")
    print("=" * 60)
    
    print("\n📋 BEFORE vs AFTER:")
    print("-" * 40)
    
    print("\n❌ BEFORE (Limited Output):")
    print("  • Math problems: 'Provide only the final numeric answer'")
    print("  • Code problems: 'Return only the function code'")
    print("  • Lost reasoning: No insight into how the model solved the problem")
    print("  • Limited debugging: Hard to understand errors")
    
    print("\n✅ AFTER (Full Reasoning Traces):")
    print("  • Math problems: 'Show your complete reasoning process'")
    print("  • Code problems: 'Explain your approach and implementation'")
    print("  • Full reasoning: Complete problem-solving process captured")
    print("  • Better debugging: Can see where models make mistakes")
    
    print("\n🔧 TECHNICAL IMPROVEMENTS:")
    print("-" * 40)
    
    improvements = [
        "✅ New ReasoningExtractor module separates answers from reasoning",
        "✅ Full reasoning traces saved as individual .txt files per turn",
        "✅ Updated prompts encourage detailed reasoning explanations", 
        "✅ CSV outputs include reasoning trace file paths for analysis",
        "✅ Increased token limits to accommodate longer reasoning",
        "✅ Multi-turn reasoning traces for iterative corrections",
        "✅ Preserved evaluation accuracy using extracted answers"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print("\n🧮 REASONING EXTRACTION DEMO:")
    print("-" * 40)
    
    extractor = ReasoningExtractor()
    
    # Demo math reasoning extraction
    math_reasoning = """
To solve this problem, I need to calculate step by step:

First, let me understand what we know:
- Natalia sold 48 clips in April
- She sold half as many in May as in April

Step 1: Calculate May sales
May sales = April sales ÷ 2 = 48 ÷ 2 = 24 clips

Step 2: Calculate total sales
Total = April + May = 48 + 24 = 72 clips

Therefore, Natalia sold 72 clips altogether.

Final answer: 72
"""
    
    math_answer, math_summary = extractor.extract_math_answer(math_reasoning)
    
    print("\n📊 MATH REASONING EXAMPLE:")
    print(f"  • Extracted Answer: {math_answer}")
    print(f"  • Reasoning Summary: {math_summary[:100]}...")
    print(f"  • Full Reasoning Length: {len(math_reasoning)} characters")
    
    # Demo code reasoning extraction
    code_reasoning = """
Looking at this problem, I need to return the absolute value of a number.

My approach:
1. Check if the number is negative
2. If negative, return the positive version
3. If positive or zero, return as-is

This is a straightforward implementation of absolute value logic.

def abs_value(x):
    '''Return the absolute value of x'''
    if x < 0:
        return -x
    else:
        return x
"""
    
    code_answer, code_summary = extractor.extract_code_answer(code_reasoning, "abs_value")
    
    print("\n💻 CODE REASONING EXAMPLE:")  
    print(f"  • Extracted Function: {len(code_answer)} characters of code")
    print(f"  • Reasoning Summary: {code_summary[:100]}...")
    print(f"  • Full Reasoning Length: {len(code_reasoning)} characters")
    
    print("\n📁 FILE ORGANIZATION:")
    print("-" * 40)
    
    file_structure = """
outputs/
├── reasoning_traces/          # Full reasoning text files
│   ├── math/
│   │   └── {problem_id}/
│   │       ├── turn_0_reasoning.txt
│   │       └── turn_1_reasoning.txt
│   └── code/
│       └── {problem_id}/
│           └── turn_0_reasoning.txt
├── csv_results/              # Analysis CSV files
│   ├── {experiment}_results_{timestamp}.csv
│   ├── {experiment}_summary_{timestamp}.csv
│   ├── turn_analysis_{timestamp}.csv
│   └── analysis_dashboard.txt
└── enhanced_traces/          # Formatted trace summaries
    └── {experiment}_full_traces/
        └── {problem_id}_full_trace.txt
"""
    
    print(file_structure)
    
    print("\n📈 ANALYSIS BENEFITS:")
    print("-" * 40)
    
    benefits = [
        "🔍 Debug model reasoning: See exactly where models make errors",
        "📊 Analyze reasoning quality: Correlate reasoning depth with accuracy", 
        "🎯 Improve prompts: Understand which prompt styles work best",
        "🔄 Multi-turn insights: Track how reasoning evolves across corrections",
        "📋 Template effectiveness: Measure which reprompting templates help most",
        "🧠 Confidence calibration: Compare model confidence to reasoning quality"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print("\n🚀 READY FOR PRODUCTION:")
    print("-" * 40)
    print("  • Full backward compatibility with existing experiments")
    print("  • No changes needed to existing analysis scripts")
    print("  • Enhanced CSV outputs for deeper insights")
    print("  • Individual reasoning traces for qualitative analysis")
    print("  • Preserved accuracy metrics using answer extraction")
    
    print("\n🎯 USAGE:")
    print("-" * 40)
    print("  1. Run experiments as before - no config changes needed")
    print("  2. Full reasoning automatically captured in .txt files")
    print("  3. Analyze CSV results with reasoning trace file paths")
    print("  4. Review individual reasoning files for qualitative insights")
    print("  5. Use enhanced traces for model comparison studies")
    
    print(f"\n✨ Full reasoning traces system ready!")


if __name__ == "__main__":
    demonstrate_improvements()