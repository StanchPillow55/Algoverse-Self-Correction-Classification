"""
Trace Evaluation for Multi-Pass Self-Correction

This module provides evaluation functions for measuring the performance
of multi-pass self-correction chains, including:
- Exact match accuracy
- F1 score for partial matches
- Cost efficiency (accuracy per token)
- Multi-turn analysis

Key metrics:
- exact_match: Binary correctness of final answer
- f1: Token-level F1 score between prediction and reference
- num_prompts: Total reprompts used in the chain
- accuracy_per_cost: Efficiency metric (exact_match / total_tokens)
"""

import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from collections import Counter
import json

logger = logging.getLogger(__name__)

@dataclass
class TraceMetrics:
    """Metrics for a single trace evaluation"""
    exact_match: float          # 0 or 1
    f1: float                  # Token-level F1 score
    num_prompts: int           # Number of reprompts used
    accuracy_per_cost: float   # exact_match / total_tokens
    total_tokens: int          # Total tokens consumed
    final_answer: str          # Final answer from trace
    reference_answer: str      # Ground truth answer
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "exact_match": self.exact_match,
            "f1": self.f1,
            "num_prompts": self.num_prompts,
            "accuracy_per_cost": self.accuracy_per_cost,
            "total_tokens": self.total_tokens,
            "final_answer": self.final_answer,
            "reference_answer": self.reference_answer
        }

def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    
    Handles common variations in mathematical answers, yes/no responses,
    and removes extra whitespace/punctuation.
    """
    if not answer or not isinstance(answer, str):
        return ""
    
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()
    
    # Remove common punctuation at the end
    answer = re.sub(r'[.!?]+$', '', answer)
    
    # Handle common answer formats
    # Yes/No normalization
    if answer in ['yes', 'y', 'true', '1']:
        return 'yes'
    elif answer in ['no', 'n', 'false', '0']:
        return 'no'
    
    # Numerical answer extraction (for math problems)
    # Extract final number if present
    number_match = re.search(r'-?\d+\.?\d*', answer)
    if number_match:
        number_str = number_match.group()
        # If the answer contains mostly digits and common words, extract just the number
        # This handles cases like "The answer is 42" -> "42"
        clean_answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
        if len(clean_answer.split()) <= 5 and any(word in answer for word in ['answer', 'is', 'equals', 'result']):
            return number_str
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer

def compute_f1_score(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 score between prediction and reference.
    
    This is useful for partial credit when answers are close but not exact matches.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    if not pred_tokens:
        return 0.0
    
    # Count common tokens
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    # Calculate overlap
    common_tokens = sum((pred_counter & ref_counter).values())
    
    if common_tokens == 0:
        return 0.0
    
    precision = common_tokens / len(pred_tokens)
    recall = common_tokens / len(ref_tokens)
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def extract_final_answer(trace: Dict) -> str:
    """
    Extract the final answer from a trace.
    
    Looks for the last answer in the trace turns.
    """
    if not trace or 'turns' not in trace:
        logger.warning("Invalid trace format: missing 'turns'")
        return ""
    
    turns = trace['turns']
    if not turns:
        logger.warning("Empty trace: no turns found")
        return ""
    
    # Get the last turn's answer
    last_turn = turns[-1]
    
    # Try different possible keys for the answer
    answer_keys = ['answer', 'revised_answer', 'response', 'output']
    
    for key in answer_keys:
        if key in last_turn and last_turn[key]:
            return str(last_turn[key])
    
    logger.warning(f"Could not find answer in final turn. Keys available: {list(last_turn.keys())}")
    return ""

def count_total_tokens(trace: Dict) -> int:
    """
    Count total tokens consumed across all turns in the trace.
    
    Sums up input_tokens, output_tokens, and reprompt_tokens if available.
    """
    if not trace or 'turns' not in trace:
        return 0
    
    total_tokens = 0
    
    for turn in trace['turns']:
        # Sum various token counts that might be present
        token_fields = [
            'input_tokens', 'output_tokens', 'reprompt_tokens',
            'prompt_tokens', 'completion_tokens', 'total_tokens'
        ]
        
        for field in token_fields:
            if field in turn and isinstance(turn[field], (int, float)):
                total_tokens += int(turn[field])
    
    # If no token counts found, estimate based on text length
    if total_tokens == 0:
        total_tokens = estimate_tokens_from_text(trace)
    
    return max(total_tokens, 1)  # Avoid division by zero

def estimate_tokens_from_text(trace: Dict) -> int:
    """
    Estimate token count from text length when exact counts unavailable.
    
    Uses rough approximation: ~4 characters per token.
    """
    total_chars = 0
    
    if 'turns' not in trace:
        return 1
    
    for turn in trace['turns']:
        # Count characters in common text fields
        text_fields = ['question', 'answer', 'prompt', 'response', 'reprompt']
        
        for field in text_fields:
            if field in turn and isinstance(turn[field], str):
                total_chars += len(turn[field])
    
    # Rough estimation: 4 characters per token
    estimated_tokens = max(total_chars // 4, 1)
    
    return estimated_tokens

def count_reprompts(trace: Dict) -> int:
    """
    Count the number of reprompts used in the trace.
    
    A reprompt is any turn after the initial answer generation.
    """
    if not trace or 'turns' not in trace:
        return 0
    
    turns = trace['turns']
    
    # Method 1: Count turns with reprompt indicators
    reprompt_count = 0
    for turn in turns:
        if any(key in turn for key in ['reprompt', 'reprompt_id', 'is_reprompt']):
            if turn.get('reprompt') or turn.get('reprompt_id') != 'none' or turn.get('is_reprompt'):
                reprompt_count += 1
    
    # Method 2: If no explicit reprompt markers, assume all turns after first are reprompts
    if reprompt_count == 0 and len(turns) > 1:
        reprompt_count = len(turns) - 1
    
    return reprompt_count

def evaluate_trace(trace: Dict, reference: Union[str, Dict]) -> Dict:
    """
    Evaluate a single trace against a reference answer.
    
    Args:
        trace: Dictionary containing trace data with turns
        reference: Reference answer (string) or dict with 'answer' key
        
    Returns:
        Dictionary with evaluation metrics:
        {
            "exact_match": 0/1,
            "f1": float,
            "num_prompts": int,
            "accuracy_per_cost": float
        }
    """
    try:
        # Extract reference answer
        if isinstance(reference, dict):
            ref_answer = reference.get('answer', '')
        else:
            ref_answer = str(reference)
        
        # Extract final answer from trace
        final_answer = extract_final_answer(trace)
        
        # Compute exact match
        normalized_pred = normalize_answer(final_answer)
        normalized_ref = normalize_answer(ref_answer)
        exact_match = 1.0 if normalized_pred == normalized_ref else 0.0
        
        # Compute F1 score
        f1_score = compute_f1_score(final_answer, ref_answer)
        
        # Count reprompts and tokens
        num_prompts = count_reprompts(trace)
        total_tokens = count_total_tokens(trace)
        
        # Calculate accuracy per cost
        accuracy_per_cost = exact_match / total_tokens if total_tokens > 0 else 0.0
        
        # Create metrics object
        metrics = TraceMetrics(
            exact_match=exact_match,
            f1=f1_score,
            num_prompts=num_prompts,
            accuracy_per_cost=accuracy_per_cost,
            total_tokens=total_tokens,
            final_answer=final_answer,
            reference_answer=ref_answer
        )
        
        return metrics.to_dict()
        
    except Exception as e:
        logger.error(f"Error evaluating trace: {str(e)}")
        return {
            "exact_match": 0.0,
            "f1": 0.0,
            "num_prompts": 0,
            "accuracy_per_cost": 0.0,
            "error": str(e)
        }

def evaluate_batch(traces: List[Dict], references: List[Union[str, Dict]]) -> Dict:
    """
    Evaluate a batch of traces and compute aggregate statistics.
    
    Args:
        traces: List of trace dictionaries
        references: List of reference answers
        
    Returns:
        Dictionary with aggregate metrics and per-trace results
    """
    if len(traces) != len(references):
        raise ValueError(f"Number of traces ({len(traces)}) must match number of references ({len(references)})")
    
    results = []
    total_metrics = {
        'exact_match': 0.0,
        'f1': 0.0,
        'num_prompts': 0,
        'total_tokens': 0,
        'accuracy_per_cost': 0.0
    }
    
    for trace, reference in zip(traces, references):
        result = evaluate_trace(trace, reference)
        results.append(result)
        
        # Accumulate metrics
        if 'error' not in result:
            total_metrics['exact_match'] += result['exact_match']
            total_metrics['f1'] += result['f1']
            total_metrics['num_prompts'] += result['num_prompts']
            total_metrics['total_tokens'] += result.get('total_tokens', 0)
            total_metrics['accuracy_per_cost'] += result['accuracy_per_cost']
    
    # Compute averages
    n_traces = len(traces)
    avg_metrics = {
        'avg_exact_match': total_metrics['exact_match'] / n_traces,
        'avg_f1': total_metrics['f1'] / n_traces,
        'avg_num_prompts': total_metrics['num_prompts'] / n_traces,
        'avg_tokens_per_trace': total_metrics['total_tokens'] / n_traces,
        'avg_accuracy_per_cost': total_metrics['accuracy_per_cost'] / n_traces,
        'total_traces': n_traces
    }
    
    return {
        'aggregate_metrics': avg_metrics,
        'per_trace_results': results
    }

# Utility functions for common evaluation scenarios
def evaluate_math_trace(trace: Dict, reference_answer: Union[str, int, float]) -> Dict:
    """Specialized evaluation for mathematical reasoning traces"""
    return evaluate_trace(trace, str(reference_answer))

def evaluate_qa_trace(trace: Dict, reference_answer: str) -> Dict:
    """Specialized evaluation for question answering traces"""
    return evaluate_trace(trace, reference_answer)

def evaluate_bool_trace(trace: Dict, reference_answer: Union[bool, str]) -> Dict:
    """Specialized evaluation for boolean question traces"""
    ref_str = 'yes' if reference_answer in [True, 'true', 'yes', '1'] else 'no'
    return evaluate_trace(trace, ref_str)

# Example usage and testing
def create_sample_trace(question: str, turns_data: List[Dict]) -> Dict:
    """Helper to create sample traces for testing"""
    return {
        'question': question,
        'turns': turns_data,
        'metadata': {
            'dataset': 'test',
            'question_id': 'sample'
        }
    }

def demonstrate_evaluation():
    """Demonstrate the evaluation system with examples"""
    print("🧪 Demonstrating Trace Evaluation System")
    print("=" * 50)
    
    # Sample trace 1: Correct answer after 1 reprompt
    trace1 = create_sample_trace(
        "What is 2 + 3?",
        [
            {
                'turn_index': 0,
                'answer': '6',
                'input_tokens': 10,
                'output_tokens': 5
            },
            {
                'turn_index': 1,
                'reprompt': 'Are you sure about 6?',
                'answer': '5',
                'input_tokens': 15,
                'output_tokens': 5,
                'is_reprompt': True
            }
        ]
    )
    
    # Sample trace 2: Wrong answer, no reprompts
    trace2 = create_sample_trace(
        "Is Paris the capital of France?",
        [
            {
                'turn_index': 0,
                'answer': 'No',
                'input_tokens': 12,
                'output_tokens': 3
            }
        ]
    )
    
    # Sample trace 3: Correct answer with multiple reprompts
    trace3 = create_sample_trace(
        "What is the square root of 16?",
        [
            {
                'turn_index': 0,
                'answer': '8',
                'input_tokens': 10,
                'output_tokens': 5
            },
            {
                'turn_index': 1,
                'reprompt': 'Think step by step',
                'answer': '4.5',
                'input_tokens': 20,
                'output_tokens': 8
            },
            {
                'turn_index': 2,
                'reprompt': 'What number times itself equals 16?',
                'answer': '4',
                'input_tokens': 25,
                'output_tokens': 5
            }
        ]
    )
    
    test_cases = [
        (trace1, '5', "Math correction (wrong → right)"),
        (trace2, 'yes', "Boolean QA (wrong answer)"),
        (trace3, '4', "Math with multiple corrections")
    ]
    
    for i, (trace, reference, description) in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {description}")
        print("-" * 30)
        
        result = evaluate_trace(trace, reference)
        
        print(f"Final answer: '{result.get('final_answer', 'N/A')}'")
        print(f"Reference: '{result.get('reference_answer', reference)}'")
        print(f"Exact match: {result['exact_match']}")
        print(f"F1 score: {result['f1']:.3f}")
        print(f"Reprompts used: {result['num_prompts']}")
        print(f"Total tokens: {result.get('total_tokens', 'N/A')}")
        print(f"Accuracy per cost: {result['accuracy_per_cost']:.6f}")
    
    # Batch evaluation example
    print(f"\n📊 Batch Evaluation:")
    print("-" * 30)
    
    traces = [trace1, trace2, trace3]
    references = ['5', 'yes', '4']
    
    batch_results = evaluate_batch(traces, references)
    
    print("Aggregate metrics:")
    for metric, value in batch_results['aggregate_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_evaluation()
