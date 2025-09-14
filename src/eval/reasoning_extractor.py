#!/usr/bin/env python3
"""
Enhanced Answer Extractor for Full Reasoning Traces

Extracts final answers from full reasoning traces for both math and code problems.
Preserves the complete reasoning while identifying the final answer for accuracy evaluation.
"""

import re
import ast
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class ReasoningExtractor:
    """Extracts answers from reasoning traces while preserving full reasoning."""
    
    def __init__(self):
        # Math answer patterns (in order of priority)
        self.math_patterns = [
            # Final answer indicators
            r"(?:final answer|answer|result|solution)(?:\s*is)?:?\s*([+-]?\d+(?:\.\d+)?)",
            r"(?:therefore|thus|so),?\s*(?:the\s+)?(?:answer\s+is\s*)?([+-]?\d+(?:\.\d+)?)",
            # Answer markers
            r"####\s*([+-]?\d+(?:\.\d+)?)",  # GSM8K style
            r"\$\$\s*([+-]?\d+(?:\.\d+)?)\s*\$\$",  # LaTeX style
            r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",  # Equation end
            # Boxed answers
            r"\\boxed\{([+-]?\d+(?:\.\d+)?)\}",
            # Last number in text (fallback)
            r"([+-]?\d+(?:\.\d+)?)(?:\s*\.?\s*$)",
        ]
        
        # Code function patterns
        self.code_patterns = [
            # Python function definition (complete)
            r"```(?:python)?\s*\n?(def\s+\w+.*?(?:\n(?:\s{4}.*|\s*$))*)\n?```",
            r"(def\s+\w+.*?(?:\n(?:\s{4}.*|\s*$))*)",
            # Fallback for any code block
            r"```(?:python)?\s*\n?(.*?)\n?```",
        ]
    
    def extract_math_answer(self, reasoning_text: str) -> Tuple[Optional[str], str]:
        """
        Extract final numeric answer from math reasoning.
        
        Args:
            reasoning_text: Full reasoning trace from model
            
        Returns:
            (extracted_answer, reasoning_summary)
        """
        if not reasoning_text or not reasoning_text.strip():
            return None, "Empty reasoning trace"
        
        text = reasoning_text.strip()
        
        # Try each pattern in order of priority
        for pattern in self.math_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL))
            if matches:
                # Take the last match for most patterns
                final_match = matches[-1]
                answer = final_match.group(1).strip()
                
                # Clean up answer
                answer = self._normalize_numeric_answer(answer)
                
                if answer:
                    reasoning_summary = self._create_math_reasoning_summary(text, answer)
                    return answer, reasoning_summary
        
        # Fallback: extract last number mentioned
        numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", text)
        if numbers:
            answer = self._normalize_numeric_answer(numbers[-1])
            reasoning_summary = self._create_math_reasoning_summary(text, answer)
            return answer, reasoning_summary
        
        return None, f"No numeric answer found in reasoning: {text[:100]}..."
    
    def extract_code_answer(self, reasoning_text: str, entry_point: str) -> Tuple[Optional[str], str]:
        """
        Extract Python function from code reasoning.
        
        Args:
            reasoning_text: Full reasoning trace from model
            entry_point: Expected function name
            
        Returns:
            (extracted_code, reasoning_summary)
        """
        if not reasoning_text or not reasoning_text.strip():
            return None, "Empty reasoning trace"
        
        text = reasoning_text.strip()
        
        # Try to extract function definition
        for pattern in self.code_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
            if matches:
                # Take the last complete match
                for match in reversed(matches):
                    code = match.group(1).strip()
                    
                    # Validate it's a proper function
                    if self._validate_function(code, entry_point):
                        reasoning_summary = self._create_code_reasoning_summary(text, code)
                        return code, reasoning_summary
        
        # Fallback: look for function definition without code blocks
        func_pattern = rf"def\s+{re.escape(entry_point)}\s*\([^)]*\).*?(?=\ndef|\Z)"
        matches = list(re.finditer(func_pattern, text, re.MULTILINE | re.DOTALL))
        if matches:
            code = matches[-1].group(0).strip()
            if self._validate_function(code, entry_point):
                reasoning_summary = self._create_code_reasoning_summary(text, code)
                return code, reasoning_summary
        
        # Last resort: return any code-like content
        lines = text.split('\n')
        code_lines = [line for line in lines if line.strip().startswith('def ') or 
                     (line.strip() and (line.startswith('    ') or line.startswith('\t')))]
        
        if code_lines:
            code = '\n'.join(code_lines).strip()
            reasoning_summary = self._create_code_reasoning_summary(text, code)
            return code, reasoning_summary
        
        return None, f"No function definition found in reasoning: {text[:100]}..."
    
    def _normalize_numeric_answer(self, answer: str) -> Optional[str]:
        """Normalize numeric answer for consistency."""
        if not answer:
            return None
            
        # Remove common non-numeric suffixes
        answer = re.sub(r'[,.\s]*$', '', answer.strip())
        
        try:
            # Try to parse as number
            if '.' in answer:
                num = float(answer)
                # Keep as float if it has meaningful decimal places
                if num == int(num):
                    return str(int(num))
                else:
                    return f"{num:.10g}"  # Remove trailing zeros
            else:
                num = int(answer)
                return str(num)
        except ValueError:
            return None
    
    def _validate_function(self, code: str, expected_name: str) -> bool:
        """Validate that code contains a proper function definition."""
        if not code or not expected_name:
            return False
            
        try:
            # Check if code parses as valid Python
            ast.parse(code)
            
            # Check if it defines the expected function
            return f"def {expected_name}" in code
        except (SyntaxError, ValueError):
            return False
    
    def _create_math_reasoning_summary(self, full_reasoning: str, extracted_answer: str) -> str:
        """Create a summary of math reasoning process."""
        lines = full_reasoning.split('\n')
        key_lines = []
        
        # Extract key reasoning steps
        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in [
                'calculate', 'compute', 'solve', 'step', 'therefore', 'so',
                'total', 'sum', 'multiply', 'divide', 'add', 'subtract'
            ]):
                key_lines.append(line)
        
        summary = "Key reasoning steps:\n" + '\n'.join(key_lines[:5])  # Top 5 steps
        summary += f"\n\nExtracted answer: {extracted_answer}"
        return summary
    
    def _create_code_reasoning_summary(self, full_reasoning: str, extracted_code: str) -> str:
        """Create a summary of code reasoning process."""
        lines = full_reasoning.split('\n')
        reasoning_lines = []
        
        # Extract reasoning (non-code lines)
        in_code_block = False
        for line in lines:
            if '```' in line:
                in_code_block = not in_code_block
                continue
            if not in_code_block and line.strip() and not line.strip().startswith('def '):
                reasoning_lines.append(line.strip())
        
        if reasoning_lines:
            summary = "Reasoning process:\n" + '\n'.join(reasoning_lines[:3])  # Top 3 reasoning lines
        else:
            summary = "Code solution provided"
        
        summary += f"\n\nExtracted function: {extracted_code.split('(')[0] if '(' in extracted_code else 'function'}"
        return summary
    
    def save_reasoning_trace(self, qid: str, turn: int, reasoning_text: str, 
                           output_dir: Path, dataset_type: str = "math") -> Path:
        """
        Save full reasoning trace to text file.
        
        Args:
            qid: Question/problem ID
            turn: Turn number
            reasoning_text: Full reasoning trace
            output_dir: Output directory for traces
            dataset_type: "math" or "code"
            
        Returns:
            Path to saved trace file
        """
        # Create directory structure
        trace_dir = output_dir / "reasoning_traces" / dataset_type / str(qid)
        trace_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reasoning trace
        trace_file = trace_dir / f"turn_{turn}_reasoning.txt"
        with open(trace_file, 'w', encoding='utf-8') as f:
            f.write(f"Question ID: {qid}\n")
            f.write(f"Turn: {turn}\n")
            f.write(f"Dataset Type: {dataset_type}\n")
            f.write("=" * 50 + "\n\n")
            f.write(reasoning_text)
            f.write("\n\n" + "=" * 50)
            f.write(f"\nTrace saved at: {trace_file}")
        
        return trace_file


def test_extractor():
    """Test the reasoning extractor with sample inputs."""
    extractor = ReasoningExtractor()
    
    # Test math reasoning
    math_reasoning = """
    Let me solve this step by step.
    
    Janet starts with 3 apples.
    She buys 5 more apples.
    
    To find the total: 3 + 5 = 8
    
    Therefore, Janet has 8 apples total.
    """
    
    math_answer, math_summary = extractor.extract_math_answer(math_reasoning)
    print(f"Math Answer: {math_answer}")
    print(f"Math Summary: {math_summary}")
    
    # Test code reasoning  
    code_reasoning = """
    I need to implement a function that checks if any two numbers in a list are closer than a threshold.
    
    My approach:
    1. Compare every pair of numbers
    2. Check if their absolute difference is less than the threshold
    3. Return True if any pair meets the condition
    
    ```python
    def has_close_elements(numbers, threshold):
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i != j:
                    if abs(numbers[i] - numbers[j]) < threshold:
                        return True
        return False
    ```
    """
    
    code_answer, code_summary = extractor.extract_code_answer(code_reasoning, "has_close_elements")
    print(f"\nCode Answer: {code_answer}")
    print(f"Code Summary: {code_summary}")


if __name__ == "__main__":
    test_extractor()