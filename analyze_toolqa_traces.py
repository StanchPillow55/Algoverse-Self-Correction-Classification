#!/usr/bin/env python3
"""
ToolQA Trace Analysis Script

Analyzes the relationship between tool success and answer accuracy
to identify patterns and improvement opportunities.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def analyze_traces(results_file: str):
    """Analyze tool traces and their relationship to accuracy"""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("📊 DETAILED TOOLQA TRACE ANALYSIS")
    print("=" * 80)
    
    # Overall stats
    models = data["results"]
    experiment_info = data["experiment_info"]
    
    print(f"Dataset: {experiment_info['dataset']}")
    print(f"Models: {len(models)}")
    print(f"Questions per model: {experiment_info['total_questions']}")
    print()
    
    # Analysis by model
    for model_name, results in models.items():
        print(f"🤖 MODEL: {model_name.upper()}")
        print("-" * 50)
        
        # Categorize results
        tool_success_correct = []      # Tool worked + answer correct
        tool_success_incorrect = []    # Tool worked + answer incorrect  
        tool_failure_incorrect = []    # Tool failed + answer incorrect
        no_tools_correct = []          # No tools + answer correct
        no_tools_incorrect = []        # No tools + answer incorrect
        
        for result in results:
            has_tools = result["tool_augmented"]
            is_correct = result["is_correct"]
            tool_results = result["tool_results"]
            
            # Check if tools succeeded
            tool_success = False
            if has_tools and tool_results:
                tool_success = any(tr.get("success", False) for tr in tool_results)
            
            # Categorize
            if has_tools:
                if tool_success:
                    if is_correct:
                        tool_success_correct.append(result)
                    else:
                        tool_success_incorrect.append(result)
                else:
                    tool_failure_incorrect.append(result)
            else:
                if is_correct:
                    no_tools_correct.append(result)
                else:
                    no_tools_incorrect.append(result)
        
        # Print summary stats
        total = len(results)
        print(f"Total Questions: {total}")
        print(f"Overall Accuracy: {sum(r['is_correct'] for r in results)}/{total} ({100*sum(r['is_correct'] for r in results)/total:.1f}%)")
        print()
        
        print("📋 BREAKDOWN BY TOOL SUCCESS:")
        print(f"✅ Tool Success + Correct Answer: {len(tool_success_correct)}")
        print(f"❌ Tool Success + Wrong Answer:   {len(tool_success_incorrect)}")
        print(f"🔧 Tool Failure + Wrong Answer:   {len(tool_failure_incorrect)}")
        print(f"📝 No Tools + Correct Answer:     {len(no_tools_correct)}")
        print(f"📝 No Tools + Wrong Answer:       {len(no_tools_incorrect)}")
        print()
        
        # Analyze tool success rate impact
        if len(tool_success_correct) + len(tool_success_incorrect) > 0:
            tool_success_accuracy = len(tool_success_correct) / (len(tool_success_correct) + len(tool_success_incorrect))
            print(f"🎯 Tool Success → Accuracy: {len(tool_success_correct)}/{len(tool_success_correct) + len(tool_success_incorrect)} ({100*tool_success_accuracy:.1f}%)")
        else:
            print(f"🎯 Tool Success → Accuracy: 0/0 (N/A)")
        
        if len(no_tools_correct) + len(no_tools_incorrect) > 0:
            no_tool_accuracy = len(no_tools_correct) / (len(no_tools_correct) + len(no_tools_incorrect))
            print(f"📝 No Tools → Accuracy: {len(no_tools_correct)}/{len(no_tools_correct) + len(no_tools_incorrect)} ({100*no_tool_accuracy:.1f}%)")
        else:
            print(f"📝 No Tools → Accuracy: 0/0 (N/A)")
        print()
        
        # Detailed analysis of tool success but wrong answers
        if tool_success_incorrect:
            print("🔍 DETAILED ANALYSIS: Tool Success but Wrong Answers")
            print("-" * 40)
            
            for i, result in enumerate(tool_success_incorrect, 1):
                print(f"{i}. Question: {result['question_id']}")
                print(f"   Q: {result['question'][:60]}...")
                print(f"   Expected: {result['expected_answer']}")
                print(f"   Extracted: {result['extracted_answer']}")
                
                # Show tool results
                for tr in result['tool_results']:
                    if tr['success']:
                        print(f"   Tool: {tr['tool_name']} → {tr['result']}")
                    else:
                        print(f"   Tool: {tr['tool_name']} → FAILED: {tr['error']}")
                
                print(f"   Model Response: {result['model_response'][:100]}...")
                print()
        
        # Analysis of successful cases
        if tool_success_correct:
            print("✅ SUCCESSFUL CASES (Tool Success + Correct Answer)")
            print("-" * 40)
            
            for result in tool_success_correct:
                print(f"• {result['question_id']}: {result['question'][:50]}...")
                print(f"  Expected: {result['expected_answer']} | Got: {result['extracted_answer']}")
                for tr in result['tool_results']:
                    if tr['success']:
                        print(f"  Tool: {tr['tool_name']} → {tr['result']}")
                print()
        
        print("=" * 60)
        print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_toolqa_traces.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not Path(results_file).exists():
        print(f"Error: File {results_file} not found")
        sys.exit(1)
    
    analyze_traces(results_file)

if __name__ == "__main__":
    main()