# ToolQA Experiments Guide

This guide provides comprehensive documentation for running ToolQA experiments in the Algoverse Self-Correction pipeline.

## 📋 **Overview**

ToolQA experiments are tool-augmented question answering tasks that test model capabilities across multiple domains including finance (coffee prices), citations (DBLP), location services (Yelp), travel (flights), accommodations (Airbnb), and scheduling (agenda). The enhanced runner addresses the previous 0.0% accuracy issue through improved answer extraction and flexible evaluation.

## 🚀 **Quick Start**

### **Basic Usage**

```bash
# Run on 10 questions with Claude Sonnet
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models claude-3-5-sonnet \
  --max-questions 10

# Run across multiple models
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models gpt-4o-mini claude-3-5-sonnet gpt-4o \
  --max-questions 50

# Disable tools for baseline comparison
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models claude-3-5-sonnet \
  --no-tools
```

### **Available Datasets**

| Dataset | Questions | Description |
|---------|-----------|-------------|
| `toolqa_deterministic_100.json` | 100 | Small validation set |
| `toolqa_deterministic_500.json` | 500 | Medium-scale experiments |
| `toolqa_deterministic_1000.json` | 1000 | Full-scale evaluation |

## 🤖 **Supported Models**

### **OpenAI Models**
- `gpt-4o-mini` (1.8B params) - Fast, cost-effective
- `gpt-4o` (8B params) - Balanced performance
- `gpt-4` (175B params) - Highest capability

### **Anthropic Models**
- `claude-3-haiku` (3B params) - Fast responses
- `claude-3-5-sonnet` (70B params) - **Recommended** for best balance
- `claude-3-opus` (175B params) - Maximum capability

### **Model Selection Guide**
```bash
# Development/Testing - Fast and cheap
--models gpt-4o-mini claude-3-haiku

# Production - Best balance
--models claude-3-5-sonnet gpt-4o

# Research - Full capability comparison
--models claude-3-opus gpt-4 claude-3-5-sonnet gpt-4o gpt-4o-mini claude-3-haiku
```

## 🛠️ **Tool System**

### **Available Tools**

| Tool | Domain | Functions |
|------|--------|-----------|
| **coffee** | Financial data | `get_price_range`, `get_price_on_date`, `get_max_price` |
| **dblp** | Citations | `search_papers`, `get_author_venues`, `count_collaborations` |
| **yelp** | Business data | `search_businesses`, `get_reviews_nearby` |
| **airbnb** | Accommodations | `search_listings`, `get_availability` |
| **agenda** | Scheduling | `search_events`, `get_person_schedule` |
| **flight** | Travel | `search_flights`, `get_route_info` |
| **calculator** | Math | `calculate`, `evaluate_expression` |

### **Tool Integration Process**

1. **Question Analysis**: System determines if question needs tools based on keywords/domain
2. **Tool Selection**: Relevant tools are made available to the model
3. **Model Interaction**: Model makes tool calls with appropriate parameters
4. **Tool Execution**: Tools query datasets and return structured results
5. **Answer Synthesis**: Model combines tool results with follow-up prompt for final answer

## 📊 **Output Structure**

### **Result File Format**

```json
{
  "experiment_info": {
    "dataset": "data/scaling/toolqa_deterministic_100.json",
    "models": ["claude-3-5-sonnet"],
    "max_questions": 100,
    "use_tools": true,
    "timestamp": "2025-09-28T01:22:26.878111",
    "total_questions": 100
  },
  "results": {
    "model_name": [
      {
        "question_id": "hard-coffee-0103",
        "model": "claude-3-5-sonnet",
        "provider": "anthropic",
        "question": "What was the coffee price range from 2000-01-03 to 2020-10-07?",
        "expected_answer": "306.2 USD",
        "model_response": "Detailed response with reasoning...",
        "extracted_answer": "24.01¢/lb to 1069.58¢/lb",
        "is_correct": false,
        "tool_augmented": true,
        "tools_used": ["coffee_get_price_range"],
        "tool_results": [...],
        "response_time": 5.57
      }
    ]
  },
  "summary": {
    "model_name": {
      "total_questions": 100,
      "correct_answers": 67,
      "accuracy": 0.67,
      "tool_usage_count": 89,
      "tool_usage_rate": 0.89,
      "average_response_time": 4.23
    }
  }
}
```

### **Key Result Fields**

| Field | Description |
|-------|-------------|
| `is_correct` | Boolean indicating if extracted answer matches expected |
| `tool_augmented` | Whether question used tools |
| `tools_used` | List of tools called for this question |
| `tool_results` | Detailed tool execution results |
| `extracted_answer` | Final answer extracted from model response |
| `model_response` | Full model response including reasoning |

## 🔧 **Advanced Configuration**

### **Environment Variables**

```bash
# Required API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional Rate Limiting
MAX_CONCURRENCY=2
RPS_LIMIT=2
TPM_LIMIT=120000
MAX_RETRIES=6
```

### **Custom Datasets**

```bash
# Use your own ToolQA-format dataset
python run_toolqa_experiments.py \
  --dataset path/to/your/custom_toolqa.json \
  --models claude-3-5-sonnet
```

**Required dataset format:**
```json
{
  "name": "Custom ToolQA Dataset",
  "samples": [
    {
      "id": "custom_1",
      "question": "Your question here?",
      "answer": "Expected answer",
      "domain": "coffee|dblp|yelp|airbnb|agenda|flight",
      "qid": "unique-identifier"
    }
  ]
}
```

## 📈 **Evaluation Improvements**

### **Enhanced Answer Extraction**

The runner addresses the 0.0% accuracy issue through:

1. **Refusal Pattern Detection**: Identifies when models say "I don't have access" or similar
2. **"FINAL ANSWER:" Format**: Prioritizes explicit final answer formatting  
3. **Multiple Extraction Strategies**: Tries various patterns like "The answer is...", boxed answers
4. **Domain-Aware Extraction**: Handles numeric vs. text answers appropriately

### **Flexible Correctness Matching**

- **Numeric Tolerance**: Allows small differences in numeric answers (0.01 or 0.1% relative)
- **Fuzzy Text Matching**: Handles variations in venue names, locations
- **Substring Matching**: Matches partial answers when appropriate
- **Equivalency Classes**: Recognizes "yes/true/1" as equivalent

### **Example Improvements**

```
❌ Old System:
Expected: "306.2 USD"
Extracted: "I don't have access to real-time coffee price data..."
Result: ❌ Incorrect (0.0% accuracy)

✅ New System:  
Expected: "306.2 USD"
Extracted: "N/A" (refusal detected and handled)
Result: ❌ Incorrect but properly categorized

Expected: "IGARSS"  
Extracted: "IGARSS"
Result: ✅ Correct (proper tool integration)
```

## 🎯 **Best Practices**

### **Model Selection**
- **Development**: Use `gpt-4o-mini` or `claude-3-haiku` for fast iteration
- **Production**: Use `claude-3-5-sonnet` for best balance of cost/performance  
- **Research**: Run full model comparison for scaling analysis

### **Question Limits**
- **Quick test**: 10 questions (~$2)
- **Validation**: 50 questions (~$10)
- **Full evaluation**: 500+ questions (~$50+)

### **Tool Usage**
- **Always compare**: Run with `--no-tools` for baseline
- **Monitor tool success**: Check tool_results for execution errors
- **Domain coverage**: Ensure your questions span available tool domains

## 🚨 **Troubleshooting**

### **Common Issues**

1. **All tools failing**: Check data files in `data/toolqa/` exist
2. **API rate limits**: Reduce concurrency or add delays
3. **Model deprecation warnings**: Update model IDs in script
4. **Low accuracy**: Verify expected answer formats match extraction patterns

### **Debug Mode**

```bash
# Enable detailed logging
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models claude-3-5-sonnet \
  --max-questions 5 2>&1 | tee debug.log
```

### **Accuracy Analysis**

```python
# Analyze results programmatically
import json

with open('results/toolqa_results.json') as f:
    results = json.load(f)

# Check accuracy by domain
domains = {}
for result in results['results']['claude-3-5-sonnet']:
    domain = result.get('domain', 'unknown')
    if domain not in domains:
        domains[domain] = {'correct': 0, 'total': 0}
    domains[domain]['total'] += 1
    if result['is_correct']:
        domains[domain]['correct'] += 1

for domain, stats in domains.items():
    accuracy = stats['correct'] / stats['total']
    print(f"{domain}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
```

## 🔬 **Research Applications**

### **Scaling Studies**
```bash
# Model capability comparison
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_500.json \
  --models gpt-4o-mini claude-3-haiku gpt-4o claude-3-5-sonnet claude-3-opus \
  --output results/scaling_study.json
```

### **Tool Impact Analysis**
```bash
# With tools
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models claude-3-5-sonnet \
  --output results/with_tools.json

# Without tools  
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models claude-3-5-sonnet \
  --no-tools \
  --output results/without_tools.json
```

### **Domain-Specific Analysis**

Create domain-specific subsets:
```python
# Extract coffee-domain questions only
import json

with open('data/scaling/toolqa_deterministic_500.json') as f:
    data = json.load(f)

coffee_questions = [q for q in data['samples'] if q['domain'] == 'coffee']
coffee_dataset = {'name': 'Coffee Domain Only', 'samples': coffee_questions}

with open('data/scaling/toolqa_coffee_only.json', 'w') as f:
    json.dump(coffee_dataset, f, indent=2)
```

## 🔗 **Integration with Main Pipeline**

The ToolQA experiments complement the main scaling study pipeline:

1. **Self-Correction Pipeline**: Focus on multi-turn reasoning improvement
2. **ToolQA Pipeline**: Focus on tool-augmented question answering
3. **Combined Analysis**: Compare model scaling across different task types

Use both for comprehensive model evaluation:

```bash
# Full scaling study
python run_full_scale_study.py --mode production

# ToolQA evaluation  
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_500.json \
  --models claude-3-5-sonnet gpt-4o claude-3-opus

# Compare results across pipelines
```

## 📝 **Citation**

If using ToolQA experiments in research:

```bibtex
@software{algoverse_toolqa,
  title={ToolQA: Enhanced Tool-Augmented Question Answering Evaluation},
  author={Algoverse Research Team},
  year={2024},
  url={https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification}
}
```