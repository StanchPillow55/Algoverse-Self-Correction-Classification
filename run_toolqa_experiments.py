#!/usr/bin/env python3
"""
Enhanced ToolQA Experiment Runner with Tool Integration

This script runs ToolQA experiments with tool augmentation for improved accuracy.

Usage:
    python run_toolqa_experiments.py --dataset data/scaling/toolqa_deterministic_100.json --model claude-3-5-sonnet-20241022
"""

import json
import logging
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import anthropic

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from tools.experiment_integration import (
    ToolAugmentedExperimentRunner, 
    AnthropicToolIntegration,
    ModelProvider,
    extract_answer_with_tool_context
)
from src.data.scaling_datasets import ScalingDatasetManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('toolqa_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_toolqa_experiment(dataset_path: str, model_name: str = "claude-3-5-sonnet-20241022", 
                         max_questions: int = None, use_tools: bool = True) -> Dict[str, Any]:
    """Run ToolQA experiment with tool augmentation"""
    
    logger.info(f"Starting ToolQA experiment")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Tools: {'enabled' if use_tools else 'disabled'}")
    
    # Initialize tool system
    tool_runner = ToolAugmentedExperimentRunner()
    anthropic_integration = AnthropicToolIntegration(tool_runner)
    client = anthropic.Anthropic()
    
    logger.info(f"Tool system initialized: {tool_runner.tool_system.get_system_stats()}")
    
    # Load dataset
    try:
        # Load JSON dataset directly
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Handle both formats: samples or questions
        if "samples" in dataset:
            questions = dataset["samples"]
        elif "questions" in dataset:
            questions = dataset["questions"]
        else:
            # Fallback - treat entire dataset as questions
            questions = dataset if isinstance(dataset, list) else []
        
        if max_questions:
            questions = questions[:max_questions]
        
        logger.info(f"Loaded {len(questions)} questions from {dataset.get('name', dataset_path)}")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"error": f"Dataset loading failed: {e}"}
    
    # Run experiment
    results = []
    correct_count = 0
    tool_usage_count = 0
    
    for i, question_data in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}")
        
        question = question_data.get("question", "")
        expected_answer = question_data.get("answer", "")
        question_id = question_data.get("id", str(i))
        
        start_time = time.time()
        
        # Initialize result
        result = {
            "question_id": question_id,
            "question": question,
            "expected_answer": expected_answer,
            "model_response": None,
            "extracted_answer": None,
            "is_correct": False,
            "response_time": 0,
            "tool_augmented": False,
            "tools_used": [],
            "error": None
        }
        
        try:
            # Check if this question needs tools
            if use_tools and tool_runner.is_toolqa_question(question):
                result["tool_augmented"] = True
                tool_usage_count += 1
                
                # Create enhanced messages with tools
                messages = [{"role": "user", "content": question}]
                enhanced_messages, available_tools = anthropic_integration.enhance_anthropic_call(
                    messages, question
                )
                
                # Make API call with tools
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    messages=enhanced_messages,
                    tools=available_tools if available_tools else None
                )
                
                # Extract text response
                text_response = ""
                if hasattr(response, 'content'):
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            text_response += content_block.text + "\n"
                
                result["model_response"] = text_response.strip()
                
                # Process tool calls if any
                tool_response_data = anthropic_integration.process_anthropic_response(
                    response, question
                )
                
                if tool_response_data.get("used_tools"):
                    result["tools_used"] = [tool["tool"] for tool in tool_response_data.get("tool_calls_executed", [])]
                    # Extract answer considering tool results
                    result["extracted_answer"] = extract_answer_with_tool_context(
                        text_response, 
                        tool_response_data.get("tool_results", [])
                    )
                else:
                    result["extracted_answer"] = extract_numeric_answer(text_response)
            else:
                # Run without tools
                messages = [{"role": "user", "content": question}]
                
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    messages=messages
                )
                
                # Extract text response
                text_response = ""
                if hasattr(response, 'content'):
                    for content_block in response.content:
                        if hasattr(content_block, 'text'):
                            text_response += content_block.text + "\n"
                
                result["model_response"] = text_response.strip()
                result["extracted_answer"] = extract_numeric_answer(text_response)
            
            # Check correctness
            result["is_correct"] = check_answer_correctness(
                result["extracted_answer"], 
                expected_answer
            )
            
            if result["is_correct"]:
                correct_count += 1
            
            result["response_time"] = time.time() - start_time
            
            logger.info(f"Question {question_id}: {'✓' if result['is_correct'] else '✗'} "
                       f"(Tools: {result['tool_augmented']}, Time: {result['response_time']:.2f}s)")
            
        except Exception as e:
            result["error"] = str(e)
            result["response_time"] = time.time() - start_time
            logger.error(f"Error processing question {question_id}: {e}")
        
        results.append(result)
        
        # Progress logging
        if i % 10 == 0:
            current_accuracy = (correct_count / i) * 100
            logger.info(f"Progress: {i}/{len(questions)}, Accuracy: {current_accuracy:.1f}%")
    
    # Calculate final metrics
    total_questions = len(questions)
    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    tool_usage_rate = (tool_usage_count / total_questions) * 100 if total_questions > 0 else 0
    
    # Get tool statistics
    tool_stats = tool_runner.get_usage_statistics()
    
    # Compile results
    experiment_results = {
        "experiment_info": {
            "model": model_name,
            "dataset": dataset_path,
            "total_questions": total_questions,
            "use_tools": use_tools,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": {
            "accuracy": accuracy,
            "correct_answers": correct_count,
            "total_questions": total_questions,
            "tool_usage_rate": tool_usage_rate,
            "questions_with_tools": tool_usage_count
        },
        "tool_statistics": tool_stats,
        "individual_results": results
    }
    
    # Log final results
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT COMPLETED")
    logger.info(f"Model: {model_name}")
    logger.info(f"Questions: {total_questions}")
    logger.info(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")
    logger.info(f"Tool Usage: {tool_usage_rate:.1f}% ({tool_usage_count} questions)")
    logger.info(f"Tool Success Rate: {tool_stats.get('success_rate', 0):.1f}%")
    logger.info("=" * 60)
    
    # Log tool usage summary
    tool_runner.log_tool_usage_summary()
    
    return experiment_results

def extract_numeric_answer(text: str) -> str:
    """Extract numeric answer from text"""
    import re
    
    # Look for numbers in the text
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    if numbers:
        # Return the last number found (often the final answer)
        return numbers[-1]
    
    # Fallback
    return text.strip()

def check_answer_correctness(extracted_answer: str, expected_answer: str) -> bool:
    """Check if extracted answer matches expected answer"""
    import re
    
    def normalize_answer(ans: str) -> str:
        # Convert to string and normalize
        ans = str(ans).lower().strip()
        # Remove common punctuation
        ans = re.sub(r'[^\w\s.]', '', ans)
        return ans
    
    extracted_norm = normalize_answer(extracted_answer)
    expected_norm = normalize_answer(expected_answer)
    
    # Exact match
    if extracted_norm == expected_norm:
        return True
    
    # Numeric comparison
    try:
        extracted_num = float(extracted_answer)
        expected_num = float(expected_answer)
        return abs(extracted_num - expected_num) < 1e-6
    except (ValueError, TypeError):
        pass
    
    # Substring match
    if extracted_norm in expected_norm or expected_norm in extracted_norm:
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="Run ToolQA experiments with tool augmentation")
    parser.add_argument("--dataset", required=True, help="Path to ToolQA dataset JSON file")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Claude model to use")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to process")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool augmentation")
    parser.add_argument("--output", help="Output file for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Run experiment
    use_tools = not args.no_tools
    results = run_toolqa_experiment(
        dataset_path=args.dataset,
        model_name=args.model,
        max_questions=args.max_questions,
        use_tools=use_tools
    )
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        dataset_name = Path(args.dataset).stem
        model_safe = args.model.replace("/", "_").replace(":", "_")
        tool_suffix = "with_tools" if use_tools else "no_tools"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"results/toolqa_{dataset_name}_{model_safe}_{tool_suffix}_{timestamp}.json"
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()