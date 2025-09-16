#!/usr/bin/env python3
"""
Enhanced Cognitive Bias Detection for Code Generation

This module implements cognitive bias detection specifically tailored for code generation tasks,
addressing the limitations of text-based heuristics when applied to programming problems.

The detection system properly aligns with psychological definitions of cognitive biases:
- Anchoring: Inappropriate reliance on example values from problem descriptions
- Availability: Overuse of familiar patterns regardless of appropriateness  
- Bandwagon: Following popular trends without proper justification
- Hindsight: Post-hoc rationalization and overconfidence in failed solutions
- Overgeneralization: Rigid patterns that don't handle edge cases or variations
"""

import re
import ast
from typing import Dict, Any, List, Tuple, Optional


class CodeBiasDetector:
    """Enhanced bias detection system for code generation tasks."""
    
    def __init__(self):
        # Common variable names that shouldn't count as anchoring
        self.common_vars = {'i', 'j', 'k', 'n', 'x', 'y', 'a', 'b', 'c', 'data', 'result', 'temp'}
        
        # Patterns that indicate overuse of familiar approaches
        self.availability_patterns = [
            (r'for\s+i\s+in\s+range\(len\([^)]+\)\):', 'C-style loop in Python'),
            (r'while\s+True:.*?break', 'Infinite loop pattern'),
            (r'try:.*?except\s*:', 'Catch-all exception handling'),
            (r'if\s+.*?==\s*True:', 'Explicit True comparison'),
            (r'if\s+.*?==\s*False:', 'Explicit False comparison'),
            (r'\.append\([^)]*\)\s*(?:\n|$)', 'List append in return context'),
            (r'list\(range\([^)]*\)\)', 'Unnecessary list() around range'),
        ]
        
        # Trendy Python patterns that might be applied without justification
        self.trendy_patterns = [
            (r'list\([^)]*\)', 'Explicit list() conversion'),
            (r'lambda\s+[^:]*:', 'Lambda function'),
            (r'.*\.join\([^)]*\)', 'String join method'),
            (r'\[.*?for.*?in.*?if.*?\]', 'Complex list comprehension'),
            (r'f["\'].*?\{.*?\}.*?["\']', 'f-string formatting'),
            (r'\{.*?for.*?in.*?\}', 'Set/dict comprehension'),
            (r'.*\.\*args', 'Args unpacking'),
            (r'.*\*\*kwargs', 'Kwargs unpacking'),
        ]
        
        # Patterns indicating overly rigid code structure
        self.rigid_patterns = [
            (r'if\s+.*?:\s*return\s+False\s*else:\s*return\s+True', 'Rigid boolean return'),
            (r'for.*?:\s*if.*?:\s*return.*?return\s+None', 'Missing default handling'),
            (r'assert\s+', 'Assertions instead of error handling'),
            (r'range\(len\([^)]+\)\)', 'Index-based iteration when unnecessary'),
            (r'if\s+len\([^)]*\)\s*==\s*0:', 'Explicit empty check'),
        ]
        
        # Markers of overconfident language in reasoning
        self.overconfidence_markers = [
            r'(?:obviously|clearly|simply|just|easy|straightforward)',
            r'(?:should be|must be|has to be|definitely|certainly)',
            r'(?:the answer is clearly|it\'s obvious that)',
            r'(?:this will work because|this should handle)',
            r'(?:guaranteed to|always works|never fails)',
        ]
        
        # Social proof language indicating bandwagon effect
        self.social_patterns = [
            r'(?:most|many|common(?:ly)?|popular|standard|typical)',
            r'(?:everyone|people|developers?) (?:use|do|prefer)',
            r'(?:best practice|recommended|widely used)',
            r'(?:industry standard|conventional|traditional)',
        ]

    def detect_anchoring_in_code(self, question: str, code: str, execution_result: dict) -> Tuple[bool, float]:
        """
        Detect if code inappropriately anchors to specific values from problem.
        
        Anchoring bias: Relying too heavily on the first piece of information (examples)
        when making decisions, leading to hardcoded values instead of general solutions.
        """
        confidence = 0.0
        
        # Extract example values from problem description
        example_numbers = re.findall(r'\b\d+\b', question)
        example_strings = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)
        
        # Check for hardcoded numbers from examples
        for num in example_numbers:
            # Look for the number in code, but not in array indices, function params, or range arguments
            pattern = r'\b' + re.escape(num) + r'\b(?!\s*[)}\],:])'  
            if re.search(pattern, code):
                # Extra penalty if it causes test failures
                if not execution_result.get('passed', True):
                    confidence += 0.4
                else:
                    confidence += 0.2
        
        # Check for hardcoded strings from examples
        for string in example_strings:
            if len(string) > 1:  # Ignore single characters
                if f'"{string}"' in code or f"'{string}'" in code:
                    confidence += 0.3
        
        # Check for variable names copied directly from problem
        problem_vars = set(re.findall(r'\b([a-z_][a-z0-9_]*)\b', question.lower()))
        code_vars = set(re.findall(r'\b([a-z_][a-z0-9_]*)\s*=', code.lower()))
        
        copied_vars = problem_vars & code_vars - self.common_vars
        confidence += len(copied_vars) * 0.15
        
        # Check for copying exact structure from examples
        example_structures = re.findall(r'(?:>>>|Example:).*?(?:\n|$)', question, re.IGNORECASE)
        for example in example_structures:
            # Look for similar patterns in code
            example_clean = re.sub(r'[^\w\s]', '', example.lower())
            code_clean = re.sub(r'[^\w\s]', '', code.lower())
            if example_clean and len(example_clean) > 10:
                words_match = len(set(example_clean.split()) & set(code_clean.split()))
                if words_match > 3:
                    confidence += 0.2
        
        return confidence > 0.4, min(confidence, 0.8)

    def detect_availability_in_code(self, code: str, execution_result: dict, history: List[Dict]) -> Tuple[bool, float]:
        """
        Detect if code uses overly familiar patterns inappropriately.
        
        Availability heuristic: Overestimating importance of easily recalled information,
        leading to default use of familiar patterns regardless of appropriateness.
        """
        confidence = 0.0
        
        # Check for overused familiar patterns
        for pattern, description in self.availability_patterns:
            if re.search(pattern, code, re.MULTILINE | re.DOTALL):
                confidence += 0.15
        
        # Check for repeated patterns across turns (availability bias)
        if len(history) > 0:
            prev_code = history[-1].get('answer', '')
            if prev_code:
                # Count common non-trivial lines
                code_lines = {line.strip() for line in code.split('\n') 
                            if line.strip() and not line.strip().startswith('#')}
                prev_lines = {line.strip() for line in prev_code.split('\n')
                            if line.strip() and not line.strip().startswith('#')}
                
                common_lines = code_lines & prev_lines
                significant_common = [line for line in common_lines if len(line) > 10]
                
                if len(significant_common) > 2:
                    confidence += 0.3
        
        # Penalty for patterns that caused failures
        if not execution_result.get('passed', True):
            error_msg = execution_result.get('error', '').lower()
            # Common availability-driven errors
            if any(keyword in error_msg for keyword in ['indexerror', 'keyerror', 'attributeerror']):
                confidence += 0.2
        
        # Check for unnecessary complexity (overusing familiar complex patterns)
        complexity_indicators = [
            r'\[.*?for.*?in.*?for.*?in.*?\]',  # Nested comprehensions
            r'lambda.*?lambda',  # Nested lambdas
            r'try:.*?try:',  # Nested try blocks
        ]
        
        for pattern in complexity_indicators:
            if re.search(pattern, code, re.DOTALL):
                confidence += 0.2
        
        return confidence > 0.35, min(confidence, 0.75)

    def detect_hindsight_in_code(self, code: str, reasoning_text: str, execution_result: dict) -> Tuple[bool, float]:
        """
        Detect hindsight bias through overconfident explanations or post-hoc rationalization.
        
        Hindsight bias: Tendency to see past events as more predictable than they were,
        leading to overconfident explanations that don't match actual complexity.
        """
        confidence = 0.0
        
        # Look for overconfidence markers in reasoning
        for pattern in self.overconfidence_markers:
            matches = len(re.findall(pattern, reasoning_text.lower()))
            confidence += matches * 0.15
        
        # Check for overconfident assertions that failed execution
        if not execution_result.get('passed', True):
            confidence_phrases = re.findall(
                r'(?:this (?:will|should) (?:work|handle|solve)|(?:definitely|certainly) (?:correct|works|handles))',
                reasoning_text.lower()
            )
            confidence += len(confidence_phrases) * 0.3
            
            # Look for claims about handling edge cases that failed
            edge_claims = re.findall(
                r'(?:handles? (?:all|every|edge) (?:cases?|scenarios?)|covers? all (?:possibilities|cases))',
                reasoning_text.lower()
            )
            confidence += len(edge_claims) * 0.4
        
        # Post-hoc rationalization: overly detailed explanation for simple code
        reasoning_words = len(reasoning_text.split())
        code_lines = len([line for line in code.split('\n') if line.strip()])
        
        if reasoning_words > 100 and code_lines < 8:
            confidence += 0.2
        elif reasoning_words > 200 and code_lines < 15:
            confidence += 0.3
        
        # Check for retrospective certainty after seeing results
        retrospective_markers = [
            r'(?:as (?:expected|predicted)|obviously (?:this|that))',
            r'(?:clearly (?:shows|demonstrates|proves))',
            r'(?:this (?:confirms|validates) (?:that|my))',
        ]
        
        for pattern in retrospective_markers:
            if re.search(pattern, reasoning_text.lower()):
                confidence += 0.2
        
        return confidence > 0.4, min(confidence, 0.8)

    def detect_overgeneralization_in_code(self, question: str, code: str, execution_result: dict) -> Tuple[bool, float]:
        """
        Detect overly rigid patterns or absolute assumptions.
        
        Overgeneralization: Drawing broad conclusions from limited evidence,
        leading to inflexible code that doesn't handle variations or edge cases.
        """
        confidence = 0.0
        
        # Check for overly rigid patterns
        for pattern, description in self.rigid_patterns:
            if re.search(pattern, code, re.MULTILINE):
                confidence += 0.2
        
        # Check for missing edge case handling when problem hints at it
        edge_case_indicators = ['empty', 'null', 'zero', 'negative', 'boundary', 'edge', 'corner']
        problem_mentions_edges = any(indicator in question.lower() for indicator in edge_case_indicators)
        
        if problem_mentions_edges:
            # Look for edge case handling in code
            edge_handling_patterns = [
                r'if\s+(?:not\s+|len\([^)]*\)\s*==\s*0)',  # Empty checks
                r'if\s+.*?(?:<|<=)\s*0',  # Negative/zero checks
                r'if\s+.*?is\s+None',  # None checks
                r'if\s+.*?(?:>=|>)',  # Boundary checks
            ]
            
            has_edge_handling = any(re.search(pattern, code) for pattern in edge_handling_patterns)
            if not has_edge_handling:
                confidence += 0.4
        
        # Check for overgeneralization based on single example
        single_example_patterns = [
            r'if\s+.*?==\s*["\'].*?["\']',  # Hardcoded string comparison
            r'return\s+["\'].*?["\']',  # Hardcoded string return
            r'if\s+len\([^)]*\)\s*==\s*\d+',  # Hardcoded length check
        ]
        
        for pattern in single_example_patterns:
            if re.search(pattern, code):
                confidence += 0.15
        
        # Analyze execution failures for overgeneralization signs
        if not execution_result.get('passed', True):
            error_msg = execution_result.get('error', '').lower()
            
            # Errors suggesting inflexible approach
            inflexibility_errors = [
                ('indexerror', 0.3),  # Rigid indexing assumptions
                ('keyerror', 0.25),   # Rigid key assumptions
                ('typeerror', 0.2),   # Type assumptions
                ('assertionerror', 0.35),  # Overly specific assertions
            ]
            
            for error_type, penalty in inflexibility_errors:
                if error_type in error_msg:
                    confidence += penalty
        
        # Check for absolute statements in variable names or comments
        absolute_indicators = ['always', 'never', 'all', 'every', 'must', 'cannot']
        for indicator in absolute_indicators:
            pattern = r'\b' + re.escape(indicator) + r'\b'
            if re.search(pattern, code.lower()):
                confidence += 0.1
        
        return confidence > 0.4, min(confidence, 0.8)

    def detect_bandwagon_in_code(self, code: str, reasoning_text: str) -> Tuple[bool, float]:
        """
        Detect following popular patterns without justification.
        
        Bandwagon effect: Tendency to adopt popular choices without proper evaluation,
        leading to use of trendy patterns that may not be appropriate for the specific problem.
        """
        confidence = 0.0
        
        # Check for trendy Python patterns
        trendy_count = 0
        for pattern, description in self.trendy_patterns:
            if re.search(pattern, code):
                confidence += 0.12
                trendy_count += 1
        
        # Higher penalty for multiple trendy patterns
        if trendy_count > 2:
            confidence += 0.2
        
        # Check for justification in reasoning
        justification_phrases = [
            'because', 'since', 'due to', 'in order to', 'so that',
            'the reason', 'this approach', 'advantage', 'benefit',
            'more efficient', 'better performance', 'clearer', 'readable'
        ]
        
        has_justification = any(phrase in reasoning_text.lower() for phrase in justification_phrases)
        
        # If using trendy patterns without justification, increase confidence
        if confidence > 0.2 and not has_justification:
            confidence += 0.3
        
        # Check for social proof language in reasoning
        for pattern in self.social_patterns:
            matches = len(re.findall(pattern, reasoning_text.lower()))
            confidence += matches * 0.2
        
        # Check for unnecessary "modern" Python features
        unnecessary_modern = [
            (r'walrus.*:=', 'Walrus operator'),
            (r'match\s+.*:', 'Match statement for simple cases'),
            (r'typing\..*\|', 'Union types with |'),
        ]
        
        for pattern, description in unnecessary_modern:
            if re.search(pattern, code):
                confidence += 0.25
        
        # Penalty for using complex patterns when simple would work
        complexity_vs_need = [
            (r'\[.*?for.*?in.*?if.*?\]', r'for.*?:\s*if.*?:\s*.*\.append'),  # List comp vs loop
            (r'lambda.*?:', r'def\s+\w+'),  # Lambda vs function
        ]
        
        for complex_pattern, simple_pattern in complexity_vs_need:
            has_complex = bool(re.search(complex_pattern, code))
            could_be_simple = bool(re.search(simple_pattern, code))
            if has_complex and not could_be_simple:
                confidence += 0.15
        
        return confidence > 0.4, min(confidence, 0.7)

    def detect_code_bias(self, question: str, answer: str, reasoning_text: str, 
                         execution_result: dict, history: List[Dict]) -> Tuple[str, float]:
        """
        Enhanced bias detection system for code generation.
        
        Returns:
            Tuple of (bias_label, confidence_score)
        """
        # First check if solution is correct - no bias if it works
        if execution_result.get('passed', False):
            return "None", 0.95
        
        bias_scores = {}
        
        # Test each bias type
        is_anchoring, anchor_conf = self.detect_anchoring_in_code(question, answer, execution_result)
        if is_anchoring:
            bias_scores['Anchoring'] = anchor_conf
        
        is_availability, avail_conf = self.detect_availability_in_code(answer, execution_result, history)
        if is_availability:
            bias_scores['Availability'] = avail_conf
        
        is_hindsight, hind_conf = self.detect_hindsight_in_code(answer, reasoning_text, execution_result)
        if is_hindsight:
            bias_scores['Hindsight'] = hind_conf
        
        is_overgeneralization, over_conf = self.detect_overgeneralization_in_code(question, answer, execution_result)
        if is_overgeneralization:
            bias_scores['Overgeneralization'] = over_conf
        
        is_bandwagon, band_conf = self.detect_bandwagon_in_code(answer, reasoning_text)
        if is_bandwagon:
            bias_scores['Bandwagon'] = band_conf
        
        # Return highest confidence bias
        if bias_scores:
            top_bias = max(bias_scores.items(), key=lambda x: x[1])
            return top_bias[0], top_bias[1]
        
        # Default fallback for code that failed but shows no specific bias pattern
        return "Logic-error", 0.5

    def get_bias_explanation(self, bias_label: str, confidence: float) -> str:
        """Get human-readable explanation of detected bias."""
        explanations = {
            "Anchoring": f"Code appears to anchor on specific values from the problem examples (confidence: {confidence:.2f})",
            "Availability": f"Code uses familiar patterns that may not be appropriate for this problem (confidence: {confidence:.2f})",
            "Hindsight": f"Reasoning shows overconfidence or post-hoc rationalization (confidence: {confidence:.2f})",
            "Overgeneralization": f"Code uses overly rigid patterns that don't handle variations (confidence: {confidence:.2f})",
            "Bandwagon": f"Code follows popular trends without clear justification (confidence: {confidence:.2f})",
            "Logic-error": f"Code contains logical errors but no clear cognitive bias pattern (confidence: {confidence:.2f})",
            "None": f"Solution is correct, no bias detected (confidence: {confidence:.2f})"
        }
        return explanations.get(bias_label, f"Unknown bias type: {bias_label} (confidence: {confidence:.2f})")