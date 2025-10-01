# Algoverse — Self-Correction Scaling Laws Study

This repository provides a comprehensive scaling study pipeline examining self-correction capabilities across model sizes, combining teacher-learner dynamics, bias detection, confidence scoring, and automated statistical analysis to discover scaling laws in AI self-improvement.

## 🔬 **Current Pipeline Overview**

The system executes multi-turn self-correction experiments across:
- **7 Model Sizes**: 1.8B → 175B parameters (GPT-4o-mini → Claude Opus)
- **4 Real Datasets**: GSM8K (1000), HumanEval (164), SuperGLUE (1000), MathBench (1000) 
- **Multi-turn Reasoning**: Up to 3 correction cycles with bias-aware feedback
- **Cost Tracking**: Real-time API cost monitoring and power-law analysis
- **Automated Analysis**: Statistical significance testing and scaling law fitting

## 🚀 **Quick Start**

### **1. Environment Setup**
```bash
# Clone and setup
git clone https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification.git
cd Algoverse-Self-Correction-Classification

# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Install dataset support
pip install datasets  # For HuggingFace dataset downloads
```

### **2. API Keys Configuration** 
Create `.env` file (never committed):
```bash
# Required API Keys
OPENAI_API_KEY=sk-...                   # GPT models
ANTHROPIC_API_KEY=sk-ant-...            # Claude models
HUGGINGFACE_API_KEY=hf_...              # Llama models
REPLICATE_API_TOKEN=r8_...              # Alternative Llama access

# Optional rate limiting
MAX_CONCURRENCY=2
RPS_LIMIT=2
TPM_LIMIT=120000
MAX_RETRIES=6
RETRIES_ENABLED=1

# Demo mode for testing (no API calls)
DEMO_MODE=0  # Set to 1 for demo mode
```

### **3. Demo Mode (No API Keys Needed)**
```bash
export DEMO_MODE=1 PROVIDER=demo

# Quick test
python -m src.main info
python -m src.main run --dataset humaneval --subset subset_20 --max-turns 3 --out runs/demo/test.json --provider demo
```

## 📊 **Full-Scale Study Execution**

### **Current Production Pipeline**
The main scaling study is executed via `run_full_scale_study.py`:

```bash
# Full scaling study (production)
python run_full_scale_study.py --mode production --datasets gsm8k,humaneval,superglue,mathbench --models all

# Quick validation run
python run_full_scale_study.py --mode demo --datasets gsm8k --models gpt-4o-mini

# Cost estimation only
python run_full_scale_study.py --mode estimate --datasets all --models all
```

**Pipeline Configuration** ([`run_full_scale_study.py`](run_full_scale_study.py)):
- **Datasets**: `gsm8k` (1000), `humaneval` (164), `superglue` (1000), `mathbench` (1000)
- **Models**: 7 models from 1.8B to 175B parameters
- **Output**: `full_scale_study_results/`
- **Cost Tracking**: Real-time USD tracking with power-law analysis

## 🔧 **Individual Experiment Commands**

### **Single Model + Dataset Runs**
```bash
# Setup environment
set -a; source .env; set +a
export PROVIDER=openai OPENAI_MODEL=gpt-4o-mini

# GSM8K Math Reasoning
python -m src.main run \
  --dataset data/scaling/gsm8k_sample.csv \
  --max-turns 3 \
  --out runs/gsm8k_gpt4mini.json \
  --provider openai

# HumanEval Code Generation  
python -m src.main run \
  --dataset humaneval \
  --subset full \
  --max-turns 3 \
  --out runs/humaneval_full.json \
  --provider openai

# Custom CSV Dataset
python -m src.main run \
  --dataset path/to/your/dataset.csv \
  --max-turns 3 \
  --out runs/custom_experiment.json \
  --provider openai
```

### **Batch Experiments** 
```bash
# Run multiple models on same dataset
for MODEL in gpt-4o-mini claude-haiku gpt-4o; do
  python -m src.main run \
    --dataset data/scaling/gsm8k_sample.csv \
    --max-turns 3 \
    --out "runs/gsm8k_${MODEL}.json" \
    --provider openai \
    --model "$MODEL"
done
```

## 📂 **Key Pipeline Files**

### **🎯 Main Execution Scripts**
| File | Purpose | Usage |
|------|---------|--------|
| [`run_full_scale_study.py`](run_full_scale_study.py) | **Full scaling study runner** | `python run_full_scale_study.py --mode production` |
| [`run_toolqa_experiments.py`](run_toolqa_experiments.py) | **ToolQA experiments with tool augmentation** | `python run_toolqa_experiments.py --dataset X --models Y` |
| [`src/main.py`](src/main.py) | **Individual experiment runner** | `python -m src.main run --dataset X --provider Y` |

### **📊 Dataset Management** 
| File | Purpose | Details |
|------|---------|---------|
| [`src/data/scaling_datasets.py`](src/data/scaling_datasets.py) | **Auto-download real datasets** | GSM8K, HumanEval, SuperGLUE, MathBench from HuggingFace |
| [`src/data/humaneval_loader.py`](src/data/humaneval_loader.py) | **HumanEval dataset loader** | 164 programming problems with test execution |
| [`src/data/gsm8k_loader.py`](src/data/gsm8k_loader.py) | **GSM8K math dataset loader** | Grade school math word problems |

### **🤖 Teacher/Learner System**
| File | Purpose | Core Functions |
|------|---------|----------------|
| [`src/agents/learner.py`](src/agents/learner.py) | **LearnerBot**: Model interface | `answer()`, confidence scoring, multi-provider support |
| [`src/agents/teacher.py`](src/agents/teacher.py) | **Bias detection & feedback** | `detect_bias()`, `combine_confidence()` |
| [`src/rts/policy.py`](src/rts/policy.py) | **Template selection** | Confidence-aware reprompt selection |
| [`src/evaluator_feedback.py`](src/evaluator_feedback.py) | **Coaching from bias** | Error-aware feedback generation |

### **🔬 Scaling Analysis**
| File | Purpose | Analysis Features |
|------|---------|------------------|
| [`src/scaling/model_registry.py`](src/scaling/model_registry.py) | **Model configs & cost estimation** | 7 models, parameter counts, API costs |
| [`src/scaling/analysis.py`](src/scaling/analysis.py) | **Power-law fitting & statistics** | Δ-improvement analysis, R², confidence intervals |
| [`src/eval/reasoning_extractor.py`](src/eval/reasoning_extractor.py) | **Extract reasoning traces** | Parse multi-turn reasoning, answer extraction |
| [`src/utils/enhanced_trace_formatter.py`](src/utils/enhanced_trace_formatter.py) | **Comprehensive trace logging** | Per-turn traces, bias labels, cost tracking |

### **⚙️ Configuration**
| File | Purpose | Contains |
|------|---------|----------|
| [`configs/scaling_models.json`](configs/scaling_models.json) | **Model & dataset definitions** | 7 models, 4 datasets, experiment phases |
| [`configs/experiments/*.yaml`](configs/experiments/) | **Feature flag configs** | baseline, confidence_only, full_system, etc. |

## 🎭 **Ensemble Voting System (New!)**

Extended implementation featuring **multi-model ensemble voting** for improved accuracy and robustness.

### **Key Features**
- **Multiple Voting Strategies**: Majority voting, confidence-weighted voting, consensus detection, adaptive voting
- **Multi-Provider Support**: Mix OpenAI, Anthropic, and other providers in single ensemble
- **Cost Optimization**: Smart cost control with early stopping and dynamic sizing
- **Comprehensive Analysis**: Detailed ensemble performance metrics and disagreement analysis

### **Quick Ensemble Start**
```bash
# Basic ensemble with demo models
python run_ensemble_experiments.py --dataset gsm8k --subset subset_20 --demo

# Production ensemble with OpenAI models
python run_ensemble_experiments.py \
  --config configs/ensemble_experiments/openai_basic.json \
  --dataset humaneval --subset subset_100

# Batch ensemble experiments
python run_ensemble_experiments.py --batch --dataset gsm8k
```

### **Available Ensemble Configurations**
- **`openai_basic.json`**: 3-model OpenAI ensemble (GPT-4o-mini, GPT-4o, GPT-3.5-turbo)
- **`anthropic_ensemble.json`**: 3-model Anthropic ensemble (Haiku, Sonnet, Opus)
- **`mixed_provider.json`**: Mixed provider ensemble for maximum diversity
- **`demo_ensemble.json`**: Testing configuration (no API calls required)

### **Ensemble Analysis**
```bash
# Analyze ensemble results
python -m src.ensemble.metrics \
  outputs/ensemble_experiments/experiment_id/traces.json \
  outputs/analysis_results
```

**📖 Full ensemble documentation**: [`docs/ENSEMBLE_GUIDE.md`](docs/ENSEMBLE_GUIDE.md)

## 🛠️ **ToolQA Experiments (New!)**

Comprehensive tool-augmented question answering experiments across all supported models with enhanced accuracy evaluation.

### **Key Features**
- **Multi-Provider Support**: All 7 target models (OpenAI, Anthropic, Replicate placeholder)
- **Tool Integration**: Unified tool system for external data access and computation
- **Enhanced Answer Extraction**: Robust extraction from verbose model responses
- **Flexible Evaluation**: Tolerant matching for numeric, text, and fuzzy answers
- **Comprehensive Logging**: Detailed tool usage and accuracy tracking

### **Quick ToolQA Start**
```bash
# Run ToolQA experiments with Claude Sonnet on 100 questions
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models claude-3-5-sonnet \
  --max-questions 100

# Run across all available models
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_500.json \
  --max-questions 50

# Disable tools for baseline comparison
python run_toolqa_experiments.py \
  --dataset data/scaling/toolqa_deterministic_100.json \
  --models gpt-4o-mini \
  --no-tools
```

### **ToolQA Model Support**
| Model | Provider | Tool Support | Status |
|-------|----------|--------------|--------|
| **GPT-4o-mini** | OpenAI | ✅ | Active |
| **GPT-4o** | OpenAI | ✅ | Active |
| **GPT-4** | OpenAI | ✅ | Active |
| **Claude-3-Haiku** | Anthropic | ✅ | Active |
| **Claude-3.5-Sonnet** | Anthropic | ✅ | Active |
| **Claude-3-Opus** | Anthropic | ✅ | Active |
| **Llama-70B** | Replicate | ⏳ | Planned |

### **ToolQA Output Structure**
```
results/comprehensive_toolqa_TIMESTAMP.json
├── experiment_info          # Dataset, models, configuration
├── results                  # Per-model detailed results
│   └── {model_name}/
│       ├── question_id
│       ├── question
│       ├── expected_answer
│       ├── model_response
│       ├── extracted_answer
│       ├── is_correct
│       ├── tool_augmented
│       ├── tools_used[]
│       └── tool_results[]
└── summary                  # Accuracy and tool usage statistics
    └── {model_name}/
        ├── accuracy
        ├── tool_usage_rate
        └── average_response_time
```

**🔧 ToolQA addresses the 0.0 accuracy issue** through:
- Enhanced answer extraction from verbose responses
- Refusal pattern detection and handling
- Flexible correctness matching (numeric tolerance, fuzzy text matching)
- Proper tool result integration with follow-up prompts

**📚 Complete ToolQA documentation**: [`docs/TOOLQA_GUIDE.md`](docs/TOOLQA_GUIDE.md)

## 🔄 **Teacher/Learner Cycle Details**

### **Multi-Turn Self-Correction Process**
```
Turn 0: Initial Answer
├── Learner generates response + confidence
├── Teacher detects bias (overconfidence, error patterns)
├── Save reasoning trace: runs/{RUN_ID}/{dataset_type}/{qid}/turn_0_reasoning.txt
└── Evaluate accuracy

Turn 1: Self-Correction (if needed)  
├── Apply coaching template based on detected bias
├── Learner re-attempts with bias-aware prompt
├── Save reasoning trace: turn_1_reasoning.txt
└── Measure improvement

Turn 2: Final Correction
├── Template selection based on confidence + previous improvement
├── Final answer generation
├── Save final reasoning trace: turn_2_reasoning.txt
└── Compute delta improvement
```

### **Bias Detection Types** ([`src/agents/teacher.py`](src/agents/teacher.py))
- **Overconfidence**: High confidence + wrong answer
- **Underconfidence**: Low confidence + correct answer  
- **Pattern Errors**: Systematic mistakes in reasoning
- **Calculation Errors**: Arithmetic mistakes in math problems

### **Template Selection** ([`src/rts/policy.py`](src/rts/policy.py))
- **Confidence-aware**: Different prompts for confident vs uncertain responses
- **Error-type specific**: Templates tailored to detected error patterns
- **Progressive**: Templates adapt based on improvement trends

## 💰 **Cost Tracking & Analysis**

### **Real-Time Cost Monitoring**
```bash
# Cost estimation before running
python -c "
from src.scaling.model_registry import estimate_experiment_cost
cost = estimate_experiment_cost('gpt-4o', 1000, avg_tokens_per_sample=2000, num_runs=3)
print(f'Estimated cost: ${cost["total_cost_usd"]:.2f}')
"

# View cost breakdown during experiments
tail -f full_scale_study_results/cost_estimate.json
```

### **Power-Law Analysis** ([`src/scaling/analysis.py`](src/scaling/analysis.py))
```python
from src.scaling.analysis import fit_power_law, analyze_scaling_results

# Fit Δ = A × ModelSize^α to improvement data
power_law_fit = fit_power_law(parameter_counts, improvements)
print(f"Scaling exponent α = {power_law_fit.scaling_exponent:.3f}")
print(f"R² = {power_law_fit.r_squared:.3f}")
```

## 📈 **Output Structure**

### **Experiment Results**
```
runs/{RUN_ID}/
├── traces.jsonl                 # Per-turn conversation traces
├── {dataset}_summary.json       # Accuracy, improvement, cost summary  
├── reasoning_traces/            # Full reasoning text files
│   └── {dataset_type}/{qid}/
│       ├── turn_0_reasoning.txt
│       ├── turn_1_reasoning.txt  
│       └── turn_2_reasoning.txt
└── csv_results/                 # Structured analysis data
    ├── analysis_dashboard.txt
    └── {dataset}_reasoning.csv
```

### **Full-Scale Study Results**
```
full_scale_study_results/
├── full_scale_study_results.json    # Complete experiment metadata
├── cost_estimate.json               # Cost breakdown by model/dataset
├── csv_results/
│   └── analysis_dashboard.txt       # Statistical summary
└── reasoning_traces/                # All reasoning traces organized by dataset
    ├── math/{qid}/turn_{N}_reasoning.txt
    └── code/{qid}/turn_{N}_reasoning.txt
```

## 🧪 **Dataset Information**

| Dataset | Size | Type | Metric | Source |
|---------|------|------|--------|---------|
| **GSM8K** | 1000 | Math word problems | Exact match | HuggingFace: `gsm8k` |
| **HumanEval** | 164 | Code generation | Pass@1 execution | HuggingFace: `openai_humaneval` |
| **SuperGLUE** | 1000 | Multi-task reasoning | Exact match | HuggingFace: `aps/super_glue` |
| **MathBench** | 1000 | Advanced mathematics | Exact match | GitHub: `open-compass/MathBench` |

### **Dataset Download**
```bash
# Auto-download all datasets
python -c "
from src.data.scaling_datasets import ScalingDatasetManager
dm = ScalingDatasetManager()
dm.download_dataset('gsm8k')       # Downloads to data/scaling/gsm8k.json
dm.download_dataset('humaneval')   # Downloads to data/scaling/humaneval.json  
dm.download_dataset('superglue')   # Downloads to data/scaling/superglue.json
dm.download_dataset('mathbench')   # Downloads to data/scaling/mathbench.json
"
```

## 🎯 **Model Registry**

| Model | Provider | Size | Cost/1K tokens | Category |
|-------|----------|------|----------------|-----------|
| **GPT-4o-mini** | OpenAI | 1.8B | $0.00015 | Small |
| **Claude Haiku** | Anthropic | 3.0B | $0.00025 | Small |
| **GPT-4o** | OpenAI | 8.0B | $0.0025 | Medium |
| **Claude Sonnet 3.5** | Anthropic | 70B | $0.003 | Medium |
| **Llama-70B** | Replicate | 70B | $0.0007 | Medium |
| **GPT-4** | OpenAI | 175B | $0.03 | Large |
| **Claude Opus** | Anthropic | 175B | $0.015 | Large |

## 🔍 **Example Workflows**

### **Research Workflow: Test Scaling Hypothesis**
```bash
# 1. Validate with small models (cost: ~$5)
python run_full_scale_study.py --mode validation \
  --datasets gsm8k --models gpt-4o-mini,claude-haiku

# 2. Medium-scale test (cost: ~$25)  
python run_full_scale_study.py --mode medium_scale \
  --datasets gsm8k,humaneval --models gpt-4o-mini,claude-haiku,gpt-4o

# 3. Full study (cost: ~$200)
python run_full_scale_study.py --mode production \
  --datasets all --models all

# 4. Analyze results
python -c "
from src.scaling.analysis import analyze_scaling_results
from pathlib import Path
results = analyze_scaling_results(Path('full_scale_study_results'))
print('Scaling Law Results:', results)
"
```

### **Development Workflow: Test New Features**
```bash
# Test in demo mode
export DEMO_MODE=1
python -m src.main run --dataset humaneval --subset subset_20 --max-turns 3 \
  --out runs/dev_test.json --provider demo

# Test with real API (small scale)
export DEMO_MODE=0 PROVIDER=openai OPENAI_MODEL=gpt-4o-mini
python -m src.main run --dataset data/scaling/gsm8k_sample.csv --max-turns 3 \
  --out runs/real_test.json --provider openai
```

## 🏗️ **Architecture Overview**

```
User Request → Dataset Manager → Teacher/Learner Loop → Analysis Pipeline
     ↓              ↓                    ↓                    ↓
   CLI Args    Auto-download      Multi-turn cycles     Statistical analysis
     ↓           HF datasets        Bias detection        Power-law fitting  
   Config         ↓                     ↓                    ↓
     ↓         CSV generation     Confidence scoring    Results export
   Model            ↓                    ↓                    ↓
  Registry     Dataset samples    Template selection     Cost tracking
```

## ⚠️ **Important Notes**

- **API Keys**: Never commit `.env` files. All keys are gitignored.
- **Large Files**: Dataset files (SuperGLUE 364MB) are auto-downloaded and gitignored.
- **Cost Control**: Full study costs ~$200. Start with demo/validation modes.
- **Reproducibility**: Set `RUN_ID`, `DATASET_SPLIT`, `GIT_COMMIT` environment variables for traceability.
- **Safety**: HumanEval code execution is sandboxed but runs locally.

## 🔬 **Research Applications**

This pipeline enables research into:
- **Scaling Laws**: How self-correction improves with model size  
- **Cost-Benefit Analysis**: Optimal model size for given budgets
- **Error Pattern Analysis**: What types of errors larger models fix
- **Multi-Turn Dynamics**: How improvement compounds across correction cycles
- **Cross-Task Generalization**: Whether scaling laws hold across math, code, reasoning

## 📚 **Citation**

If you use this pipeline in research, please cite:
```bibtex
@software{algoverse_scaling_study,
  title={Self-Correction Scaling Laws: A Multi-Model Study},
  author={Algoverse Research Team},
  year={2024},
  url={https://github.com/StanchPillow55/Algoverse-Self-Correction-Classification}
}
```

---

## **Recent Results** (Auto-updated)

**Last Updated**: 2024-09-16  
**Pipeline Version**: v2.1  
**Total Experiments Run**: 47  
**Cost-Benefit Threshold Identified**: ~7B parameters

See [`full_scale_study_results/`](full_scale_study_results/) for latest experimental data.
