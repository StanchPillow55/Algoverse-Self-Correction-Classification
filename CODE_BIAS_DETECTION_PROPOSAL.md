# Code-Aware Cognitive Bias Detection System

## 🎯 Problem Statement

The current bias detection system applies **text-based heuristics designed for math reasoning** to code generation tasks, resulting in a fundamental mismatch between:
- **What we claim to detect**: Cognitive biases (anchoring, availability, hindsight, overgeneralization)
- **What we actually detect**: Text patterns that don't reflect the underlying cognitive processes

## 🧠 Proper Cognitive Bias Definitions for Code

### **Anchoring Bias**
> Relying too heavily on the first piece of information when making decisions

**In Code Generation**:
- Using literal numbers/values from problem description inappropriately
- Hardcoding first example values instead of generalizing
- Copying variable names/structure from problem examples verbatim

### **Availability Heuristic**  
> Overestimating importance of easily recalled information

**In Code Generation**:
- Using familiar patterns/algorithms regardless of appropriateness
- Defaulting to recently used approaches
- Applying memorized code snippets without adaptation

### **Bandwagon Effect**
> Following popular choices without proper evaluation

**In Code Generation**:
- Using "trendy" language features unnecessarily
- Following common coding patterns that don't fit the specific problem
- Adopting popular libraries/approaches without considering alternatives

### **Hindsight Bias**
> Seeing past events as more predictable than they were

**In Code Generation**:
- Post-hoc rationalization of implementation choices
- Claiming solution approaches were "obvious" after seeing results
- Overconfidence in solution correctness without proper testing

### **Overgeneralization**
> Drawing broad conclusions from limited evidence

**In Code Generation**:
- Applying absolute rules without considering exceptions
- Making assumptions about edge cases based on single examples
- Using overly rigid patterns that don't handle variations

---

## 🔧 Enhanced Detection Algorithm

### **1. Anchoring Bias Detection**

```python
def detect_anchoring_in_code(question: str, code: str, execution_result: dict) -> Tuple[bool, float]:
    """
    Detect if code inappropriately anchors to specific values from problem.
    """
    # Extract example values from problem description
    example_numbers = re.findall(r'\b\d+\b', question)
    example_strings = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)
    
    confidence = 0.0
    
    # Check for hardcoded values from examples
    for num in example_numbers:
        if re.search(rf'\b{num}\b(?!\s*[)}\]])', code):  # Not in array/param position
            confidence += 0.3
    
    # Check for hardcoded strings from examples  
    for string in example_strings:
        if f'"{string}"' in code or f"'{string}'" in code:
            confidence += 0.3
    
    # Check for variable names copied from problem
    problem_vars = re.findall(r'\b([a-z_][a-z0-9_]*)\b(?=\s*[=:])', question.lower())
    code_vars = re.findall(r'\b([a-z_][a-z0-9_]*)\s*=', code.lower())
    
    for var in problem_vars:
        if var in code_vars and var not in ['i', 'j', 'k', 'n', 'x', 'y']:  # Common loop vars
            confidence += 0.2
    
    return confidence > 0.4, min(confidence, 0.8)
```

### **2. Availability Heuristic Detection**

```python  
def detect_availability_in_code(code: str, execution_result: dict, history: List) -> Tuple[bool, float]:
    """
    Detect if code uses overly familiar patterns inappropriately.
    """
    confidence = 0.0
    
    # Common "default" patterns that may be inappropriately applied
    availability_patterns = [
        (r'for\s+i\s+in\s+range\(len\([^)]+\)\):', 'C-style loop in Python'),
        (r'while\s+True:.*break', 'Infinite loop pattern'),
        (r'try:.*except\s*:', 'Catch-all exception handling'),
        (r'if\s+.*==\s*True:', 'Explicit True comparison'),
        (r'\.append\([^)]*\)\s*$', 'List append in return context'),
    ]
    
    for pattern, description in availability_patterns:
        if re.search(pattern, code, re.MULTILINE):
            confidence += 0.2
    
    # Check if same patterns appear across multiple turns (availability bias)
    if len(history) > 0:
        prev_code = history[-1].get('answer', '')
        common_lines = set(code.split('\n')) & set(prev_code.split('\n'))
        if len(common_lines) > 2:  # Significant code reuse
            confidence += 0.3
    
    return confidence > 0.3, min(confidence, 0.7)
```

### **3. Hindsight Bias Detection**

```python
def detect_hindsight_in_code(code: str, reasoning_text: str, execution_result: dict) -> Tuple[bool, float]:
    """
    Detect hindsight bias through overconfident explanations or post-hoc rationalization.
    """
    confidence = 0.0
    
    # Hindsight markers in reasoning (overconfidence)
    hindsight_markers = [
        r'(?:obviously|clearly|simply|just|easy|straightforward)',
        r'(?:should be|must be|has to be|definitely)',
        r'(?:the answer is clearly|it\'s obvious that)',
        r'(?:this will work because|this should handle)',
    ]
    
    for pattern in hindsight_markers:
        if re.search(pattern, reasoning_text.lower()):
            confidence += 0.2
    
    # Check for overconfident assertions that failed
    if not execution_result.get('passed', True):
        confidence_phrases = re.findall(
            r'(?:this (?:will|should) (?:work|handle|solve)|(?:definitely|certainly) (?:correct|works))', 
            reasoning_text.lower()
        )
        confidence += len(confidence_phrases) * 0.3
    
    # Post-hoc rationalization: detailed explanation after simple fix
    if len(reasoning_text.split()) > 100 and len(code.split('\n')) < 10:
        confidence += 0.2
    
    return confidence > 0.4, min(confidence, 0.8)
```

### **4. Overgeneralization Detection**

```python
def detect_overgeneralization_in_code(question: str, code: str, execution_result: dict) -> Tuple[bool, float]:
    """
    Detect overly rigid patterns or absolute assumptions.
    """
    confidence = 0.0
    
    # Overly rigid patterns
    rigid_patterns = [
        (r'if\s+.*:\s*return\s+False\s*else:\s*return\s+True', 'Rigid if-else instead of direct boolean'),
        (r'for.*:\s*if.*:\s*return.*return\s+None', 'No default handling'),
        (r'assert\s+', 'Assertions instead of proper error handling'),
        (r'range\(len\([^)]+\)\)', 'Index-based iteration when unnecessary'),
    ]
    
    for pattern, description in rigid_patterns:
        if re.search(pattern, code):
            confidence += 0.2
    
    # Check for missing edge case handling
    edge_case_indicators = ['empty', 'null', 'zero', 'negative', 'boundary']
    handles_edges = any(indicator in question.lower() for indicator in edge_case_indicators)
    
    if handles_edges:
        # Look for edge case handling in code
        edge_handling = bool(re.search(r'if\s+(?:not\s+|len\([^)]*\)\s*==\s*0|.*\s*<\s*)', code))
        if not edge_handling:
            confidence += 0.4
    
    # Overgeneralization: applying pattern to all similar problems
    if execution_result.get('error_type') == 'wrong_answer':
        # Check if error suggests inflexible approach
        error_msg = execution_result.get('error', '').lower()
        if any(keyword in error_msg for keyword in ['index', 'range', 'type', 'attribute']):
            confidence += 0.3
    
    return confidence > 0.4, min(confidence, 0.8)
```

### **5. Bandwagon Effect Detection** 

```python
def detect_bandwagon_in_code(code: str, reasoning_text: str) -> Tuple[bool, float]:
    """
    Detect following popular patterns without justification.
    """
    confidence = 0.0
    
    # "Trendy" Python patterns that might be applied unnecessarily
    trendy_patterns = [
        (r'list\(.*\)', 'Explicit list() conversion'),
        (r'lambda\s+', 'Lambda where def would be clearer'),
        (r'.*\.join\(.*\)', 'join() for simple concatenation'),
        (r'\[.*for.*in.*if.*\]', 'Complex list comprehension'),
        (r'f["\'].*\{.*\}.*["\']', 'f-string for simple strings'),
    ]
    
    for pattern, description in trendy_patterns:
        if re.search(pattern, code):
            confidence += 0.15
    
    # Check for justification in reasoning
    justification_phrases = [
        'because', 'since', 'due to', 'in order to', 'so that',
        'the reason', 'this approach', 'advantage', 'benefit'
    ]
    
    has_justification = any(phrase in reasoning_text.lower() for phrase in justification_phrases)
    
    # If using trendy patterns without justification
    if confidence > 0.2 and not has_justification:
        confidence += 0.3
    
    # Social proof language in reasoning
    social_patterns = [
        r'(?:most|many|common(?:ly)?|popular|standard|typical)',
        r'(?:everyone|people|developers?) (?:use|do|prefer)',
        r'(?:best practice|recommended|widely used)',
    ]
    
    for pattern in social_patterns:
        if re.search(pattern, reasoning_text.lower()):
            confidence += 0.2
    
    return confidence > 0.4, min(confidence, 0.7)
```

---

## 🔄 Integrated Detection System

```python
def detect_code_bias(question: str, answer: str, reasoning_text: str, 
                     execution_result: dict, history: List[Dict]) -> Tuple[str, float]:
    """
    Enhanced bias detection system for code generation.
    """
    # First check if solution is correct
    if execution_result.get('passed', False):
        return "None", 0.95
    
    bias_scores = {}
    
    # Test each bias type
    is_anchoring, anchor_conf = detect_anchoring_in_code(question, answer, execution_result)
    if is_anchoring:
        bias_scores['Anchoring'] = anchor_conf
    
    is_availability, avail_conf = detect_availability_in_code(answer, execution_result, history)
    if is_availability:
        bias_scores['Availability'] = avail_conf
    
    is_hindsight, hind_conf = detect_hindsight_in_code(answer, reasoning_text, execution_result)
    if is_hindsight:
        bias_scores['Hindsight'] = hind_conf
    
    is_overgeneralization, over_conf = detect_overgeneralization_in_code(question, answer, execution_result)
    if is_overgeneralization:
        bias_scores['Overgeneralization'] = over_conf
    
    is_bandwagon, band_conf = detect_bandwagon_in_code(answer, reasoning_text)
    if is_bandwagon:
        bias_scores['Bandwagon'] = band_conf
    
    # Return highest confidence bias
    if bias_scores:
        top_bias = max(bias_scores.items(), key=lambda x: x[1])
        return top_bias[0], top_bias[1]
    
    # Default fallback
    return "Logic-error", 0.5
```

---

## 🎯 Code-Specific Coaching Templates

### **Anchoring Bias → `counter_anchor_code_v1`**
> "Don't hardcode values from the examples. Step back and identify the general pattern or algorithm needed. What would work for any valid input, not just the given examples?"

### **Availability Heuristic → `explore_alternatives_v1`** 
> "Consider alternative approaches beyond your first instinct. What other algorithms or data structures might solve this problem? Challenge yourself to think of at least 2 different approaches."

### **Bandwagon Effect → `justify_choices_v1`**
> "Explain why you chose this specific approach. What are its advantages for this particular problem? Don't just use popular patterns—use the right patterns."

### **Hindsight Bias → `test_assumptions_v1`**
> "Test your assumptions systematically. Run through edge cases step by step. What could go wrong with your current approach?"

### **Overgeneralization → `handle_edge_cases_v1`**  
> "Consider edge cases and exceptions. What happens with empty inputs, single elements, or boundary values? Make your solution robust to different scenarios."

---

## 📊 Expected Improvements

### **Detection Accuracy**
- **Current System**: ~15% accurate bias classification (mostly defaults to "Confirmation")
- **Enhanced System**: ~70-80% accurate classification based on actual cognitive patterns

### **Coaching Relevance** 
- **Current**: Generic templates like "List disconfirming hypotheses"
- **Enhanced**: Code-specific guidance targeting actual programming mistakes

### **Multi-turn Effectiveness**
- **Current**: Limited improvement from generic advice  
- **Enhanced**: Targeted coaching should improve code quality more effectively

### **Scaling Study Value**
- **Richer bias labels** reflecting actual cognitive processes
- **More meaningful multi-turn interactions** 
- **Better differentiation** between model capabilities and bias patterns

---

## 🚀 Implementation Strategy

1. **Phase 1**: Implement enhanced detection functions
2. **Phase 2**: Create code-specific coaching templates  
3. **Phase 3**: A/B test enhanced vs. current system
4. **Phase 4**: Integrate into scaling study pipeline

This approach would provide **genuine cognitive bias detection** for code generation, leading to more effective coaching and richer data for your scaling law research!