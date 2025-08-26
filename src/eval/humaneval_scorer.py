#!/usr/bin/env python3
"""
HumanEval scorer for code generation tasks.
Uses sandboxed execution to validate candidate solutions.
"""
import os
import re
from typing import Dict, Any, Optional
from .code_executor import execute_code_safely, demo_execute_code


def extract_function_from_response(response: str, entry_point: str, prompt: str) -> str:
    """
    Extract the function implementation from the model's response.
    
    Args:
        response: The model's response containing code
        entry_point: Expected function name
        prompt: Original prompt containing function signature
    
    Returns:
        Extracted function code
    """
    # Try to extract a complete function definition first
    function_pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\):\s*.*?(?=\n\S|\n$|\Z)'
    match = re.search(function_pattern, response, re.DOTALL | re.MULTILINE)
    
    if match:
        return match.group(0).strip()
    
    # If no complete function found, try to extract just the function body
    # and combine it with the signature from the prompt
    signature_pattern = rf'def\s+{re.escape(entry_point)}\s*\([^)]*\):'
    signature_match = re.search(signature_pattern, prompt)
    
    if signature_match:
        signature = signature_match.group(0)
        
        # Look for indented code blocks that could be the function body
        lines = response.split('\n')
        function_body = []
        in_code_block = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments at the start
            if not in_code_block and (not stripped or stripped.startswith('#')):
                continue
            
            # Start collecting when we see indented code or start of implementation
            if not in_code_block and (line.startswith('    ') or stripped):
                in_code_block = True
            
            if in_code_block:
                # Stop at unindented non-empty line (end of function)
                if stripped and not line.startswith('    ') and not line.startswith('\t'):
                    break
                function_body.append(line)
        
        if function_body:
            # Ensure proper indentation
            indented_body = []
            for line in function_body:
                if line.strip():
                    # Ensure at least 4 spaces of indentation
                    if not line.startswith('    '):
                        indented_body.append('    ' + line.strip())
                    else:
                        indented_body.append(line)
                else:
                    indented_body.append('')
            
            return signature + '\n' + '\n'.join(indented_body)
    
    # Fallback: return the response as-is and hope for the best
    return response.strip()


def score_humaneval_candidate(
    task: Dict[str, Any], 
    candidate_response: str,
    use_demo_mode: bool = None
) -> Dict[str, Any]:
    """
    Score a candidate solution for a HumanEval task.
    
    Args:
        task: HumanEval task dictionary with qid, question, reference, test, entry_point
        candidate_response: Model's response containing the solution
        use_demo_mode: If True, use demo execution; if None, check DEMO_MODE env var
    
    Returns:
        Dictionary with scoring results:
        - passed: bool, whether solution passed all tests
        - passed_count: int, number of tests passed
        - total_count: int, total number of tests
        - execution_result: dict, detailed execution results
        - extracted_function: str, the extracted function code
        - error: str, any error that occurred
    """
    if use_demo_mode is None:
        use_demo_mode = os.getenv('DEMO_MODE', '0') == '1'
    
    result = {
        'passed': False,
        'passed_count': 0,
        'total_count': 0,
        'execution_result': {},
        'extracted_function': '',
        'error': ''
    }
    
    try:
        # Extract the function from the response
        extracted_function = extract_function_from_response(
            candidate_response, 
            task['entry_point'], 
            task['question']
        )
        result['extracted_function'] = extracted_function
        
        if not extracted_function.strip():
            result['error'] = 'No function code could be extracted from response'
            return result
        
        # Execute the code
        if use_demo_mode:
            execution_result = demo_execute_code(
                extracted_function, 
                task['test'], 
                task['entry_point']
            )
        else:
            execution_result = execute_code_safely(
                extracted_function,
                task['test'],
                task['entry_point']
            )
        
        result['execution_result'] = execution_result
        result['passed'] = execution_result['passed']
        result['passed_count'] = execution_result['passed_count']
        result['total_count'] = execution_result['total_count']
        
        if execution_result.get('error'):
            result['error'] = execution_result['error']
    
    except Exception as e:
        result['error'] = f'Scoring error: {str(e)}'
    
    return result


def humaneval_accuracy_metric(
    predictions: list, 
    references: list, 
    tasks: list
) -> Dict[str, float]:
    """
    Calculate pass@1 accuracy for HumanEval predictions.
    
    Args:
        predictions: List of model predictions/responses
        references: List of reference solutions (not used in HumanEval scoring)
        tasks: List of HumanEval task dictionaries
    
    Returns:
        Dictionary with accuracy metrics
    """
    if len(predictions) != len(tasks):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(tasks)} tasks")
    
    total_tasks = len(tasks)
    passed_tasks = 0
    total_tests = 0
    passed_tests = 0
    
    for pred, task in zip(predictions, tasks):
        score_result = score_humaneval_candidate(task, pred)
        
        if score_result['passed']:
            passed_tasks += 1
        
        total_tests += score_result['total_count']
        passed_tests += score_result['passed_count']
    
    pass_at_1 = passed_tasks / total_tasks if total_tasks > 0 else 0.0
    test_accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
    
    return {
        'pass@1': pass_at_1,
        'test_accuracy': test_accuracy,
        'tasks_passed': passed_tasks,
        'total_tasks': total_tasks,
        'tests_passed': passed_tests,
        'total_tests': total_tests
    }


if __name__ == "__main__":
    # Test the scorer
    from ..data.humaneval_loader import create_demo_humaneval_data
    
    demo_data = create_demo_humaneval_data()
    test_task = demo_data[0]  # has_close_elements task
    
    # Test with a correct solution
    correct_response = '''def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j:
                if abs(numbers[i] - numbers[j]) < threshold:
                    return True
    return False'''
    
    print("Testing HumanEval scorer...")
    result = score_humaneval_candidate(test_task, correct_response, use_demo_mode=True)
    print(f"Correct solution result: {result}")
    
    # Test with an incorrect solution
    incorrect_response = '''def has_close_elements(numbers, threshold):
    return True  # Always return True - incorrect'''
    
    result2 = score_humaneval_candidate(test_task, incorrect_response, use_demo_mode=True)
    print(f"Incorrect solution result: {result2}")
