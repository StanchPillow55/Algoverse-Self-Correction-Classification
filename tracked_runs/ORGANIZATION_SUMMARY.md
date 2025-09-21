# Experimental Data Organization Summary

This directory contains organized experimental data from the Algoverse Self-Correction Classification study. The data has been systematically organized into individual experiment directories.

## Organization Structure

### Main Directory: `tracked_runs/`
Contains 12 experiment directories, each representing a unique model-dataset-samplesize combination.

### Experiment Directory Structure
Each experiment directory follows the naming convention: `{model}_{dataset}_{sampleSize}`

**Example:** `gpt-4o-mini_gsm8k_1000/`

### Contents of Each Experiment Directory:

1. **Core Trace Files**
   - `fullscale_{model}_{dataset}_{timestamp}_traces.json` - Original full-scale traces
   
2. **CSV Results**
   - `{dataset}_{model}_results_{timestamp}.csv` - Detailed results for each question
   - `{dataset}_{model}_summary_{timestamp}.csv` - Summary statistics and metrics
   - `turn_analysis_{timestamp}.csv` - Multi-turn analysis data

3. **Individual Question Files**
   - `question_001.txt`, `question_002.txt`, etc. - Individual reasoning traces for each question
   - Each file contains:
     - Question ID and text
     - Reference/ground truth answer
     - Multi-turn reasoning responses
     - Bias analysis (Teacher bias, confidence levels)
     - Accuracy information
     - Correction templates used

4. **Enhanced Traces** (if available)
   - `enhanced_traces/` subdirectory containing:
     - `*_accuracy_data.json` - Detailed accuracy analysis
     - `*_full_traces/` - Directory with individual trace files  
     - `*_multi_turn_analysis.json` - Multi-turn behavior analysis
     - `*_summary_metrics.json` - High-level performance metrics

5. **Metadata**
   - `README.md` - Human-readable experiment description
   - `metadata.json` - Machine-readable experiment metadata

## Experiment Configurations

### Models Tested:
- **gpt-4o-mini** - OpenAI GPT-4o mini
- **claude-haiku** - Anthropic Claude 3 Haiku  
- **claude-sonnet** - Anthropic Claude 3 Sonnet

### Datasets Used:
- **gsm8k** - Grade School Math 8K (1000 questions)
- **humaneval** - HumanEval code generation (164 questions)
- **mathbench** - Mathematical reasoning benchmark (100 questions)  
- **superglue** - SuperGLUE NLP benchmark (1000 questions)

### Sample Sizes:
- GSM8K & SuperGLUE: 1000 questions each
- HumanEval: 164 questions (full dataset)
- MathBench: 100 questions

## Total Data Volume

### Individual Question Traces:
- **GSM8K:** 3,000 individual question files (3 models × 1000 questions)
- **HumanEval:** 492 individual question files (3 models × 164 questions)
- **MathBench:** 300 individual question files (3 models × 100 questions)
- **SuperGLUE:** 3,000 individual question files (3 models × 1000 questions)

**Total:** ~6,800 individual question trace files across all experiments

### Key Features Captured:
- Multi-turn reasoning processes
- Bias detection and classification (Confirmation, Anchoring, Availability, etc.)
- Confidence scoring (self-confidence, teacher confidence, combined)
- Self-correction attempts and templates
- Performance accuracy at turn level and final level
- Complete reasoning traces for error analysis

## Usage

Each experiment directory is self-contained and can be analyzed independently. The individual question files provide detailed insight into model reasoning patterns, bias tendencies, and self-correction behaviors.

For quick analysis:
1. Check `README.md` for experiment overview
2. Review `*_summary_metrics.json` for high-level performance
3. Examine individual `question_*.txt` files for detailed reasoning traces
4. Use CSV files for quantitative analysis across questions

## Generated On
September 16, 2025

## Scripts Used
- `organize_experiments.py` - Main organization script
- `extract_individual_traces.py` - Individual trace extraction
- `organize_enhanced_traces.py` - Enhanced traces organization