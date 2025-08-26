#!/usr/bin/env python3
"""
Sandboxed code executor for HumanEval tasks.
Provides safe execution environment with timeouts and import restrictions.
"""
import os
import sys
import tempfile
import subprocess
import time
import signal
from typing import Dict, Any, Optional
from pathlib import Path


# Denylist of potentially unsafe imports
UNSAFE_IMPORTS = {
    'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 'http',
    'shutil', 'pathlib', 'glob', 'tempfile', 'pickle', 'marshal',
    'importlib', 'ctypes', 'threading', 'multiprocessing', 'asyncio',
    'eval', 'exec', 'compile', 'open', '__import__',
    'file', 'input', 'raw_input'
}


def check_code_safety(code: str) -> bool:
    """
    Basic safety check for suspicious code patterns.
    
    Args:
        code: Python code to check
    
    Returns:
        True if code appears safe, False otherwise
    """
    code_lower = code.lower()
    
    # Check for unsafe imports
    lines = code.split('\n')
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('import ') or line_stripped.startswith('from '):
            for unsafe in UNSAFE_IMPORTS:
                if unsafe in line_stripped:
                    return False
    
    # Check for dangerous builtin usage
    dangerous_patterns = [
        'exec(', 'eval(', 'compile(', '__import__(',
        'open(', 'file(', 'input(', 'raw_input(',
        'getattr(', 'setattr(', 'delattr(',
        'globals(', 'locals(', 'vars('
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code_lower:
            return False
    
    return True


def create_test_module(function_code: str, test_code: str, entry_point: str) -> str:
    """
    Create a complete Python module for testing.
    
    Args:
        function_code: The candidate function implementation
        test_code: The test code to validate the function
        entry_point: The function name being tested
    
    Returns:
        Complete Python module as a string
    """
    # Ensure function code is properly indented (if it's just the body)
    if not function_code.strip().startswith('def '):
        # This is just the function body, need to add the signature
        # We'll get this from parsing or assume it's properly formatted
        function_code = function_code.rstrip()
    
    module_template = f'''
# Generated test module for HumanEval
import sys
import time

# Candidate implementation
{function_code}

# Test code
{test_code}

# Main execution
if __name__ == "__main__":
    try:
        check({entry_point})
        print("PASS: All tests passed")
        sys.exit(0)
    except Exception as e:
        print(f"FAIL: {{e}}")
        sys.exit(1)
'''
    
    return module_template


def execute_code_safely(
    function_code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Execute code in a sandboxed environment with safety checks.
    
    Args:
        function_code: The candidate function implementation
        test_code: Test code to validate the function
        entry_point: Function name being tested
        timeout: Maximum execution time in seconds
    
    Returns:
        Dictionary with execution results:
        - passed: bool, whether all tests passed
        - passed_count: int, number of tests passed (estimated)
        - total_count: int, total number of tests (estimated)
        - stdout: str, standard output
        - stderr: str, standard error
        - runtime_ms: float, execution time in milliseconds
        - error: str, any error that occurred
    """
    start_time = time.time()
    
    result = {
        'passed': False,
        'passed_count': 0,
        'total_count': 0,
        'stdout': '',
        'stderr': '',
        'runtime_ms': 0.0,
        'error': ''
    }
    
    # Basic safety check
    if not check_code_safety(function_code):
        result['error'] = 'Code failed safety check - potentially unsafe patterns detected'
        return result
    
    if not check_code_safety(test_code):
        result['error'] = 'Test code failed safety check - potentially unsafe patterns detected'
        return result
    
    # Create temporary directory for execution
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create the test module
            module_content = create_test_module(function_code, test_code, entry_point)
            module_path = Path(temp_dir) / 'test_module.py'
            
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(module_content)
            
            # Execute in subprocess with timeout
            env = os.environ.copy()
            # Limit Python path to temp directory
            env['PYTHONPATH'] = temp_dir
            
            process = subprocess.Popen(
                [sys.executable, str(module_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=temp_dir,
                env=env,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                
                execution_time = time.time() - start_time
                result['runtime_ms'] = execution_time * 1000
                result['stdout'] = stdout
                result['stderr'] = stderr
                
                # Parse results
                if return_code == 0 and 'PASS' in stdout:
                    result['passed'] = True
                    # Estimate test count from assertions in test code
                    result['total_count'] = test_code.count('assert ')
                    result['passed_count'] = result['total_count']  # All passed if exit code 0
                else:
                    result['passed'] = False
                    result['total_count'] = test_code.count('assert ')
                    result['passed_count'] = 0  # None passed if any failed
                    if stderr:
                        result['error'] = stderr
                    elif 'FAIL' in stdout:
                        result['error'] = stdout
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                result['error'] = f'Code execution timed out after {timeout} seconds'
                result['runtime_ms'] = timeout * 1000
        
        except Exception as e:
            result['error'] = f'Execution error: {str(e)}'
            result['runtime_ms'] = (time.time() - start_time) * 1000
    
    return result


def demo_execute_code(
    function_code: str,
    test_code: str,
    entry_point: str
) -> Dict[str, Any]:
    """
    Demo/mock execution for testing without actual code execution.
    Returns simulated results for demonstration purposes.
    
    Args:
        function_code: The candidate function implementation
        test_code: Test code to validate the function
        entry_point: Function name being tested
    
    Returns:
        Simulated execution results
    """
    # Simple heuristic: if function_code looks reasonable, simulate success
    # This is just for demo mode validation
    
    passed = True
    error = ""
    
    # Basic checks
    if not function_code.strip():
        passed = False
        error = "Empty function code"
    elif 'pass' in function_code and len(function_code.strip()) < 20:
        passed = False
        error = "Function appears to be just a pass statement"
    elif 'TODO' in function_code or 'NotImplemented' in function_code:
        passed = False
        error = "Function not implemented"
    
    # Estimate test count
    total_count = max(1, test_code.count('assert '))
    passed_count = total_count if passed else 0
    
    return {
        'passed': passed,
        'passed_count': passed_count,
        'total_count': total_count,
        'stdout': 'PASS: All tests passed' if passed else 'FAIL: Tests failed',
        'stderr': '',
        'runtime_ms': 50.0,  # Simulated runtime
        'error': error
    }


if __name__ == "__main__":
    # Test the executor
    test_function = '''def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j:
                if abs(numbers[i] - numbers[j]) < threshold:
                    return True
    return False'''
    
    test_code = '''
def check(candidate):
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
'''
    
    print("Testing code executor...")
    result = execute_code_safely(test_function, test_code, "has_close_elements")
    print(f"Result: {result}")
