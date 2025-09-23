#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from src.agents.learner import LearnerBot

# Test with the same problem using both providers
test_problem = """You are a math problem solver. Show your complete reasoning and work.

Think through the problem step by step. Show all calculations and explain your reasoning.
Work through the problem completely and provide your final answer.

Question:
Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Please show all your work and reasoning, then state your final answer."""

def test_both_models():
    print("=== Comparing OpenAI vs Claude with Identical Prompts ===\n")
    
    # Test OpenAI first (if API key available)
    print("🔍 TESTING OPENAI GPT-4O:")
    if os.getenv("OPENAI_API_KEY"):
        try:
            openai_bot = LearnerBot(provider="openai", model="gpt-4o")
            openai_answer, openai_conf, openai_full = openai_bot.answer(test_problem, [])
            
            print(f"Answer: '{openai_answer}'")
            print(f"Confidence: {openai_conf}")
            print(f"Response length: {len(openai_full)} chars")
            print(f"Full response preview:\n{openai_full[:300]}...")
            
        except Exception as e:
            print(f"OpenAI Error: {e}")
    else:
        print("No OpenAI API key available")
    
    print("\n" + "="*60)
    
    # Test Claude
    print("\n🔍 TESTING CLAUDE:")
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            claude_bot = LearnerBot(provider="anthropic", model="claude-3-sonnet")
            claude_answer, claude_conf, claude_full = claude_bot.answer(test_problem, [])
            
            print(f"Answer: '{claude_answer}'")
            print(f"Confidence: {claude_conf}")
            print(f"Response length: {len(claude_full)} chars")
            print(f"Full response preview:\n{claude_full[:300]}...")
            
        except Exception as e:
            print(f"Claude Error: {e}")
    else:
        print("No Anthropic API key available")
    
    print("\n" + "="*60)
    print("\n📊 EXTRACTION COMPARISON:")
    
    # Test extraction on both if we got responses
    if 'openai_full' in locals() and 'claude_full' in locals():
        from src.eval.reasoning_extractor import ReasoningExtractor
        from src.metrics.accuracy import extract_final_answer
        
        extractor = ReasoningExtractor()
        
        print("\nOpenAI Extraction:")
        openai_reasoning_extract, _ = extractor.extract_math_answer(openai_full)
        openai_gsm8k_extract = extract_final_answer(openai_full)
        print(f"  - ReasoningExtractor: '{openai_reasoning_extract}'")
        print(f"  - GSM8K extract: '{openai_gsm8k_extract}'")
        
        print("\nClaude Extraction:")
        claude_reasoning_extract, _ = extractor.extract_math_answer(claude_full)
        claude_gsm8k_extract = extract_final_answer(claude_full)
        print(f"  - ReasoningExtractor: '{claude_reasoning_extract}'")
        print(f"  - GSM8K extract: '{claude_gsm8k_extract}'")

if __name__ == "__main__":
    test_both_models()