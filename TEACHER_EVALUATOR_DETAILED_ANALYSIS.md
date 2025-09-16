# Teacher/Evaluator Bot: Step-by-Step Analysis by Dataset

## Overview of the Teacher/Evaluator System

The teacher/evaluator bot operates through **3 core components** that work together to analyze learner responses and provide feedback:

1. **Bias Detection** (`src/agents/teacher.py`) - Identifies cognitive biases in responses
2. **Template Selection** (`src/rts/policy.py`) - Chooses appropriate coaching interventions  
3. **Feedback Generation** (`src/evaluator_feedback.py`) - Converts bias labels to actionable coaching

---

## 🧮 GSM8K / College Math Dataset

### **Step 1: Initial Answer Evaluation**

**Input Processing**:
```python
# Example: "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast..."
qid = "gsm8k_42"
question = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per egg. How much does she make every day?"
reference = "18"  # Ground truth answer
```

**Learner Response Processing**:
```python
# Learner generates full reasoning trace
raw_answer = "Let me solve this step by step..."
full_response = """
Let me solve this step by step.

Janet's ducks lay 16 eggs per day.
She eats 3 eggs for breakfast.
She uses 4 eggs for muffins.
So she uses 3 + 4 = 7 eggs total.
Remaining eggs = 16 - 7 = 9 eggs
She sells each egg for $2.
Total earnings = 9 × $2 = $18

Therefore, Janet makes $18 every day.
"""

# Answer extraction
extracted_answer = reasoning_extractor.extract_math_answer(full_response)
# Result: extracted_answer = "18"
```

**Accuracy Scoring**:
```python
acc0 = gsm8k_em(extracted_answer, reference)  # gsm8k_em("18", "18") = 1
```

### **Step 2: Teacher Bias Detection**

**Bias Analysis Process**:
```python
def detect_bias(question, answer, reference, history):
    # Step 2a: Check if answer is correct
    if answer == reference:  # "18" == "18"
        return "None", 0.95
    
    # Step 2b: Check for anchoring (using numbers from question)
    nums_in_question = re.findall(r"\d+", question)  # ["16", "3", "4", "2"]
    if answer in nums_in_question:  # If answer was "16" or "3"
        return "Anchoring", 0.7
    
    # Step 2c: Check for availability/bandwagon patterns
    if _contains_any(answer, ["everyone", "most people", "commonly"]):
        return "Availability/Bandwagon", 0.7
    
    # Step 2d: Check for hindsight bias patterns
    if _contains_any(answer, ["obvious because", "clearly since"]):
        return "Hindsight", 0.65
    
    # Step 2e: Check for overgeneralization
    if _contains_any(answer, ["always", "never", "all cases"]):
        return "Overgeneralization", 0.65
    
    # Step 2f: Default fallback
    return "Confirmation", 0.6

# Example results:
bias, teacher_conf = detect_bias(question, "18", "18", [])
# Result: ("None", 0.95) - Correct answer, no bias detected
```

### **Step 3: Confidence Combination**

```python
learner_confidence = 0.85  # Model's self-reported confidence
teacher_confidence = 0.95   # Teacher's confidence in bias detection
combined_confidence = combine_confidence(0.85, 0.95, None)
# Result: (0.85 + 0.95) / 2 = 0.90
```

### **Step 4: Coaching Feedback Generation**

```python
coaching_feedback = coaching_from_bias("None")
# Result: "Your reasoning is sound and your answer is correct. Good work applying logical thinking to solve the problem."
```

### **Step 5: Multi-turn Decision (if answer was wrong)**

**Template Selection Process**:
```python
# If answer was wrong, e.g., answer="16", bias="Anchoring", conf=0.75
reprompt, template = select_template("Anchoring", 0.75, False, 0)

# Template selection logic:
confidence_bucket = bucket_conf(0.75)  # "high" (>0.7)
if bias == "Anchoring" and confidence_bucket == "high":
    return True, "counter_anchor_v1"

# Template text: "Ignore any numbers or hints seen earlier. Re-derive from first principles, then compute. Show steps."
```

### **Expected Output Structure**:

```json
{
  "qid": "gsm8k_42",
  "question": "Janet's ducks lay 16 eggs per day...",
  "reference": "18",
  "turns": [
    {
      "answer": "18",
      "raw_answer": "Let me solve this step by step...",
      "response_text": "Full reasoning trace with calculations",
      "reasoning_trace_file": "outputs/reasoning_traces/math/gsm8k_42/turn_0_reasoning.txt",
      "reasoning_summary": "Key reasoning steps:\nCalculate: 16 - 3 - 4 = 9\nTotal: 9 × $2 = $18",
      "self_conf": 0.85,
      "teacher_bias": "None",
      "teacher_conf": 0.95,
      "combined_confidence": 0.90,
      "template": null,
      "accuracy": 1,
      "execution_details": {}
    }
  ],
  "final_accuracy": 1
}
```

---

## 💻 HumanEval / Code Generation Dataset

### **Step 1: Initial Code Generation**

**Input Processing**:
```python
qid = "HumanEval/0"
question = '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    """'''
task = {
    "qid": "HumanEval/0", 
    "entry_point": "has_close_elements",
    "test": "def check(candidate): assert candidate([1.0, 2.0, 3.9], 0.3) == True...",
    "topic": "humaneval"
}
```

**Learner Code Generation**:
```python
full_response = """
I need to implement a function that checks if any two numbers in a list are closer than a threshold.

My approach:
1. Compare every pair of numbers
2. Check if their absolute difference is less than the threshold  
3. Return True if any pair meets the condition

def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j:
                if abs(numbers[i] - numbers[j]) < threshold:
                    return True
    return False
"""

# Code extraction
extracted_code = reasoning_extractor.extract_code_answer(full_response, "has_close_elements")
# Result: Extracted function definition
```

### **Step 2: Code Execution and Scoring**

**HumanEval Execution Process**:
```python
score_result = score_humaneval_candidate(task, extracted_code)

# Detailed execution process:
execution_result = {
    "passed": True,           # All tests passed
    "passed_count": 7,        # Number of test cases passed
    "total_count": 7,         # Total number of test cases
    "stdout": "",            # Program output
    "stderr": "",            # Error output  
    "runtime_ms": 23.4,      # Execution time
    "error": "",             # Any execution errors
    "extracted_function": extracted_code
}

acc0 = int(score_result.get('passed', False))  # 1 or 0
```

### **Step 3: Teacher Bias Detection for Code**

**Code-Specific Bias Analysis**:
```python
# Same detect_bias function, but with code-specific context
question_text = task["question"]
answer_text = extracted_code  # The actual function code

bias, teacher_conf = detect_bias(question_text, answer_text, "", [])

# For HumanEval, bias detection is challenging because:
# 1. "Reference" is empty (no expected code output)
# 2. Answer is code, not text with obvious bias patterns
# 3. Correctness is determined by test execution, not text matching

# Typical results for HumanEval:
# - If tests pass: ("None", 0.95)
# - If tests fail: ("Confirmation", 0.6) - default fallback
```

### **Step 4: Multi-turn Code Improvement (NEW)**

**Template Selection for Code**:
```python
# If code fails tests, e.g., syntax error or logic error
if acc0 == 0:  # Tests failed
    reprompt, template = select_template("Confirmation", 0.6, False, 0)
    # Result: True, "devils_advocate_v1"
    
    # Template text: "List at least one disconfirming hypothesis and test it. If it falsifies your answer, revise."
    
    # Follow-up turn with coaching
    learner.answer(prompt, history + turns, template="devils_advocate_v1", ...)
```

### **Step 5: Multi-turn Code Execution**

**Follow-up Turn Processing**:
```python
# Turn 2: Model attempts to fix the code
raw_answer_1 = "Let me reconsider the implementation..."
full_response_1 = """
Looking at potential issues with my previous code:
1. The nested loop is correct for checking all pairs
2. The condition abs(numbers[i] - numbers[j]) < threshold looks right
3. Wait, let me check edge cases...

def has_close_elements(numbers, threshold):
    if len(numbers) < 2:  # Edge case: less than 2 numbers
        return False
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):  # Avoid duplicate comparisons
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
"""

# Re-extract and re-execute
extracted_code_1 = reasoning_extractor.extract_code_answer(full_response_1, "has_close_elements")
score_result_1 = score_humaneval_candidate(task, extracted_code_1)
acc1 = int(score_result_1.get('passed', False))
```

### **Expected HumanEval Output Structure**:

```json
{
  "qid": "HumanEval/0",
  "question": "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
  "reference": "",
  "turns": [
    {
      "answer": "def has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(len(numbers)):\n            if i != j:\n                if abs(numbers[i] - numbers[j]) < threshold:\n                    return True\n    return False",
      "raw_answer": "I need to implement a function...",
      "response_text": "Full reasoning trace with code explanation",
      "reasoning_trace_file": "outputs/reasoning_traces/code/HumanEval_0/turn_0_reasoning.txt",
      "reasoning_summary": "Reasoning process:\nNeed to check all pairs of numbers\nExtracted function: has_close_elements",
      "self_conf": 0.75,
      "teacher_bias": "None",
      "teacher_conf": 0.95,
      "combined_confidence": 0.85,
      "template": null,
      "accuracy": 1,
      "execution_details": {
        "passed": true,
        "passed_count": 7,
        "total_count": 7,
        "stdout": "",
        "stderr": "",
        "runtime_ms": 23.4,
        "error": "",
        "extracted_function": "def has_close_elements(numbers, threshold):..."
      }
    }
  ],
  "final_accuracy": 1
}
```

---

## 🔧 ToolQA Dataset

### **Step 1: Tool-Augmented Question Processing**

**Input Structure**:
```python
qid = "toolqa_42"
question = "What is the population of the capital city of France?"
reference = "2161000"  # Expected answer
```

**Learner Response with Tool Usage**:
```python
full_response = """
I need to find the population of France's capital city.

First, I'll identify the capital: Paris is the capital of France.
Then I'll look up the current population of Paris.

Using location tool: Paris, France
Using population tool: Paris metropolitan area population is approximately 2.16 million.

Therefore, the population of the capital city of France is 2,161,000.
"""

extracted_answer = reasoning_extractor.extract_math_answer(full_response)
# Result: "2161000"
```

### **Step 2: ToolQA-Specific Bias Detection**

**Enhanced Bias Analysis**:
```python
bias, teacher_conf = detect_bias(question, "2161000", "2161000", [])
# Same algorithm as GSM8K, but with tool-specific context

# Additional ToolQA considerations:
tools_used = extract_tools_used(turn_data)  # ["location", "population"]
tool_accuracy = calculate_tool_accuracy(turn_data)  # Based on correct tool usage
```

### **Expected ToolQA Output Structure**:

```json
{
  "qid": "toolqa_42",
  "question": "What is the population of the capital city of France?",
  "reference": "2161000",
  "turns": [
    {
      "answer": "2161000",
      "response_text": "I need to find the population of France's capital...",
      "teacher_bias": "None",
      "teacher_conf": 0.95,
      "template": null,
      "accuracy": 1,
      "tools_used": ["location", "population"],
      "tool_accuracy": 0.95
    }
  ]
}
```

---

## 🎯 SuperGLUE Dataset

### **Step 1: Multi-task Reasoning Processing**

**Input Examples**:
```python
# BoolQ task
qid = "boolq_156"
question = "Is the following statement true or false: The Great Wall of China is visible from space?"
reference = "False"

# COPA task  
qid = "copa_78"
question = "The woman felt dizzy. What was the CAUSE of this? (A) She stood up quickly (B) She ate breakfast"
reference = "A"
```

**Task Type Detection**:
```python
task_type = extract_task_type(trace, turn)
# Based on question patterns:
# - "true or false" → BoolQ
# - "CAUSE" or "choice" → COPA
# - "entailment" → RTE
```

### **Step 2: SuperGLUE Bias Detection**

**Same Core Algorithm**:
```python
bias, teacher_conf = detect_bias(question, "False", "False", [])
# For correct answers: ("None", 0.95)
# For incorrect answers: Pattern-based detection or "Confirmation" fallback
```

### **Expected SuperGLUE Output Structure**:

```json
{
  "qid": "boolq_156", 
  "question": "Is the following statement true or false: The Great Wall of China is visible from space?",
  "reference": "False",
  "turns": [
    {
      "answer": "False",
      "teacher_bias": "None",
      "teacher_conf": 0.95,
      "template": null,
      "accuracy": 1,
      "task_type": "BoolQ"
    }
  ]
}
```

---

## 📊 Flat JSON Results Output (Chart Format)

### **For GSM8K/Math**:
```json
{
  "columns": ["question", "turn1finalAns", "bias1", "feedback1", "accuracy1", "confidence1", "turn2finalAns", "bias2", "feedback2", "accuracy2", "confidence2"],
  "rows": [
    ["gsm8k_42", "18", "None", "Your reasoning is sound and your answer is correct.", 1, 0.90, "", "", "", "", ""]
  ]
}
```

### **For HumanEval**:
```json
{
  "columns": ["question", "turn1finalAcc", "bias1", "feedback1", "testAccuracy1", "confidence1"],  
  "rows": [
    ["HumanEval/0", 1, "None", "Your reasoning is sound and your answer is correct.", 1.0, 0.85]
  ]
}
```

---

## 🔄 Multi-turn Flow Summary

### **Decision Tree for Each Dataset**:

1. **Turn 0**: Generate initial response
2. **Evaluate**: Score accuracy (dataset-specific method)
3. **Detect Bias**: Apply heuristic bias detection  
4. **If Correct**: Stop, log "None" bias
5. **If Incorrect**: 
   - Select template based on bias + confidence
   - Generate coaching feedback
   - Send template to learner for Turn 1
6. **Repeat**: Until correct or max_turns reached

### **Key Differences by Dataset**:

| Dataset | Evaluation Method | Bias Detection Focus | Multi-turn Behavior |
|---------|------------------|---------------------|---------------------|
| **GSM8K** | Exact match numeric | Number anchoring, reasoning patterns | High benefit |
| **HumanEval** | Code execution | Limited (code context) | Now enabled (experimental) |  
| **ToolQA** | Exact match + tool usage | Tool selection patterns | Moderate benefit |
| **SuperGLUE** | Exact match by task | Task-specific reasoning | Moderate benefit |

This comprehensive teacher/evaluator system provides consistent bias detection and coaching across all datasets while adapting evaluation methods to each domain's specific requirements.