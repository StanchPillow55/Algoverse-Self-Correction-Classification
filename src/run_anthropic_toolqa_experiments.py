#!/usr/bin/env python3
"""
Enhanced Anthropic ToolQA Experiment Runner with Tool Integration

This script runs ToolQA experiments with the Anthropic Claude models, now enhanced
with tool augmentation capabilities for improved performance on data-dependent questions.

Features:
- Tool-augmented reasoning for ToolQA domains
- Multi-turn conversation support with tools
- Enhanced answer extraction considering tool results
- Comprehensive tool usage metrics and logging

Usage:
    python run_anthropic_toolqa_experiments.py --dataset data/scaling/toolqa_deterministic_100.json --model claude-3-5-sonnet-20241022
"""

import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import anthropic

# Enhanced imports with tool integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append('src')

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
        logging.FileHandler('toolqa_anthropic_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedAnthropicToolQARunner:
    """Enhanced Anthropic ToolQA experiment runner with tool augmentation"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", max_tokens: int = 4096):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Initialize tool integration
        self.tool_runner = ToolAugmentedExperimentRunner()
        self.anthropic_tools = AnthropicToolIntegration(self.tool_runner)
        
        logger.info(f"Enhanced Anthropic ToolQA Runner initialized with {model_name}")
        logger.info(f"Tool system status: {self.tool_runner.tool_system.get_system_stats()}")
    
    def run_single_question(self, question: str, expected_answer: str, 
                          question_id: str = None, use_tools: bool = True) -> Dict[str, Any]:
        """Run a single ToolQA question with tool augmentation"""
        
        start_time = time.time()
        
        # Initialize result structure
        result = {
            "question_id": question_id or "unknown",
            "question": question,
            "expected_answer": expected_answer,
            "model_response": None,
            "extracted_answer": None,
            "is_correct": False,
            "response_time": 0,
            "tool_augmented": False,
            "tools_used": [],
            "tool_results": [],
            "error": None
        }
        
        try:
            # Check if this question needs tools
            if use_tools and self.tool_runner.is_toolqa_question(question):
                result["tool_augmented"] = True
                response_data = self._run_with_tools(question)
            else:
                response_data = self._run_without_tools(question)
            
            # Process response
            result.update(response_data)
            
            # Extract answer
            if result.get("tool_results"):
                result["extracted_answer"] = extract_answer_with_tool_context(
                    result["model_response"], 
                    result["tool_results"]
                )
            else:
                result["extracted_answer"] = self._extract_answer(result["model_response"])
            
            # Check correctness
            result["is_correct"] = self._check_answer_correctness(
                result["extracted_answer"], 
                expected_answer
            )
            
            result["response_time"] = time.time() - start_time
            
            logger.info(f"Question {question_id}: {'✓' if result['is_correct'] else '✗'} "
                       f"(Tools: {result['tool_augmented']}, Time: {result['response_time']:.2f}s)")
            
        except Exception as e:
            result["error"] = str(e)
            result["response_time"] = time.time() - start_time
            logger.error(f"Error processing question {question_id}: {e}")
        
        return result
    
    def _run_with_tools(self, question: str) -> Dict[str, Any]:
        """Run question with tool augmentation"""
        
        # Create initial conversation
        messages = [{"role": "user", "content": question}]
        
        # Enhance with tools
        enhanced_messages, available_tools = self.anthropic_tools.enhance_anthropic_call(
            messages, question
        )
        
        try:
            # Make API call with tools
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=enhanced_messages,
                tools=available_tools if available_tools else None
            )
            
            # Process response with tool execution
            tool_response_data = self.anthropic_tools.process_anthropic_response(
                response, question
            )
            
            return {
                "model_response": self._extract_text_from_response(response),
                "raw_response": response,
                "tools_used": [tool["tool"] for tool in tool_response_data.get("tool_calls_executed", [])],
                "tool_results": tool_response_data.get("tool_results", []),
                "tool_execution_data": tool_response_data
            }
            
        except Exception as e:
            logger.error(f"Tool-augmented API call failed: {e}")
            # Fallback to non-tool version
            return self._run_without_tools(question)
    
    def _run_without_tools(self, question: str) -> Dict[str, Any]:
        """Run question without tool augmentation"""
        
        messages = [{"role": "user", "content": question}]
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=messages
            )
            
            return {
                "model_response": self._extract_text_from_response(response),
                "raw_response": response,
                "tools_used": [],
                "tool_results": []
            }
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def _extract_text_from_response(self, response) -> str:
        """Extract text content from Anthropic response"""
        if hasattr(response, 'content'):
            text_parts = []
            for content_block in response.content:
                if hasattr(content_block, 'text'):
                    text_parts.append(content_block.text)
            return '\n'.join(text_parts)
        return str(response)
    
    def _extract_answer(self, response_text: str) -> str:
        """Extract answer from response text"""
        import re
        
        # Look for numeric answers first
        numbers = re.findall(r'-?\d+\.?\d*', response_text)
        if numbers:
            return numbers[-1]
        
        # Look for explicit answers
        answer_patterns = [
            r'(?i)(?:answer|result|solution)(?:\s*is|\s*:)\s*(.+?)(?:\.|$)',
            r'(?i)the\s+answer\s+is\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1).strip()
        
        # Fallback to last sentence
        sentences = response_text.strip().split('.')
        if sentences:
            return sentences[-1].strip()
        
        return response_text.strip()
    
    def _check_answer_correctness(self, extracted_answer: str, expected_answer: str) -> bool:
        """Check if extracted answer matches expected answer"""
        
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
    
    def run_experiment(self, dataset_path: str, max_questions: int = None, 
                      use_tools: bool = True) -> Dict[str, Any]:
        """Run full ToolQA experiment"""
        
        logger.info(f"Starting ToolQA experiment with tools={'enabled' if use_tools else 'disabled'}")
        logger.info(f"Dataset: {dataset_path}")
        
        # Load dataset
        try:
            dataset_manager = ScalingDatasetManager()
            dataset = dataset_manager.load_dataset(dataset_path)
            questions = dataset.get("questions", [])
            
            if max_questions:
                questions = questions[:max_questions]
            
            logger.info(f"Loaded {len(questions)} questions")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {"error": f"Dataset loading failed: {e}"}
        
        # Run questions
        results = []
        correct_count = 0
        tool_usage_count = 0
        
        for i, question_data in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            
            question = question_data.get("question", "")
            expected_answer = question_data.get("answer", "")
            question_id = question_data.get("id", str(i))
            
            result = self.run_single_question(
                question, expected_answer, question_id, use_tools
            )
            results.append(result)
            
            if result["is_correct"]:
                correct_count += 1
            
            if result["tool_augmented"]:
                tool_usage_count += 1
            
            # Progress logging
            if i % 10 == 0:
                current_accuracy = (correct_count / i) * 100
                logger.info(f"Progress: {i}/{len(questions)}, Accuracy: {current_accuracy:.1f}%")
        
        # Calculate final metrics
        total_questions = len(questions)
        accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        tool_usage_rate = (tool_usage_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Tool usage statistics
        tool_stats = self.tool_runner.get_usage_statistics()
        
        # Compile experiment results
        experiment_results = {
            "experiment_info": {
                "model": self.model_name,
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
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Questions: {total_questions}")
        logger.info(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_questions})")
        logger.info(f"Tool Usage: {tool_usage_rate:.1f}% ({tool_usage_count} questions)")
        logger.info(f"Tool Success Rate: {tool_stats.get('success_rate', 0):.1f}%")
        logger.info("=" * 60)
        
        return experiment_results

def main():
    parser = argparse.ArgumentParser(description="Run enhanced ToolQA experiments with Anthropic Claude")
    parser.add_argument("--dataset", required=True, help="Path to ToolQA dataset JSON file")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Claude model to use")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to process")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool augmentation")
    parser.add_argument("--output", help="Output file for results (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create runner
    runner = EnhancedAnthropicToolQARunner(model_name=args.model)
    
    # Run experiment
    use_tools = not args.no_tools
    results = runner.run_experiment(
        dataset_path=args.dataset,
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
    
    # Log tool usage summary
    runner.tool_runner.log_tool_usage_summary()

if __name__ == "__main__":
    main()