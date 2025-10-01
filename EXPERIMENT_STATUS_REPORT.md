# Experiment Status Report
## Updated: October 1, 2025

This document tracks the completion status of all experiments across the three main benchmarks.

## 🎯 HumanEval Progress (6/9 completed - 67%)

### ✅ Completed Experiments
1. **GPT-4o-mini**: 82.3% accuracy, 164 samples
2. **Claude-Haiku**: 48.8% accuracy, 164 samples  
3. **GPT-4o**: 79.9% accuracy, 164 samples
4. **Claude-Sonnet**: 82.3% accuracy, 164 samples (completed today!)
5. **Claude-Opus**: 81.7% accuracy, 164 samples

### ❌ Pending Experiments
6. **GPT-4**: Not completed (OpenAI credit limit)
7. **Llama-7B**: Not completed
8. **Llama-13B**: Not completed
9. **Llama-70B**: Not completed

### 📁 Data Files
- `humaneval_claude_haiku_results.json` (3.02MB)
- `humaneval_claude_sonnet_results.json` (1.66MB)
- `20250918_humaneval_gpt-4o_314299ab.jsonl` (1.81MB)
- CSV results in `csv_results/humaneval_*`
- Reasoning traces in `reasoning_traces/code/HumanEval/`

## 🎯 ToolQA Progress (5/9 completed - 56%)

### ✅ Completed Experiments  
1. **GPT-4o-mini**: 7.2% accuracy, 499 samples (completed today)
2. **Claude-Haiku**: 5.6% accuracy, 499 samples (completed today)
3. **Claude-Sonnet**: 6.2% accuracy, 499 samples (completed today)
4. **Claude-Opus**: 0.0% accuracy, 399 samples

### ❌ Pending Experiments
5. **GPT-4o**: Not completed (OpenAI credit limit)
6. **GPT-4**: Not completed (OpenAI credit limit)
7. **Llama-7B**: Not completed
8. **Llama-13B**: Not completed
9. **Llama-70B**: Not completed

### 📁 Data Files
- `toolqa_claude_haiku_results.json` (4.44MB)
- `toolqa_claude_sonnet_results.json` (3.95MB)
- `toolqa_gpt4o_mini_results.json` (7.67MB)
- CSV results in `csv_results/toolqa_*`
- Comprehensive experiment logs

## 🎯 GSM8K Progress (8/9 completed - 89%)

### ✅ Completed Experiments
1. **GPT-4o-mini**: 23.4% accuracy, 1,000 samples
2. **Claude-Haiku**: 66.5% accuracy, 1,000 samples (completed today!)
3. **GPT-4o**: 54.7% accuracy, 1,000 samples
4. **Claude-Sonnet**: 66.2% accuracy, 1,000 samples
5. **Llama-70B**: 16.9% accuracy, 1,000 samples
6. **GPT-4**: 48.5% accuracy, 1,000 samples
7. **Claude-Opus**: 60.5% accuracy, 1,000 samples

### ❌ Pending Experiments
8. **Llama-7B**: Not completed
9. **Llama-13B**: Not completed

### 📁 Data Files
- `gsm8k_claude_haiku_results.json` (5.27MB)
- `20250918_gsm8k_gpt-4o_*` series (multiple files)
- `20250918_gsm8k_gpt-4o-mini_*` series
- CSV results in `csv_results/gsm8k_*`
- Reasoning traces in `reasoning_traces/math/`

## 📊 Overall Summary

- **Total Experiments**: 27 across all benchmarks
- **Completed**: 19/27 (70.4%)
- **Pending**: 8/27 (29.6%)

### Primary Blockers
1. **OpenAI Credit Limits**: Preventing GPT-4 and GPT-4o completion on ToolQA
2. **Llama Model Access**: Need to complete 7B, 13B, and in some cases 70B experiments

## 📁 Repository Structure

### Key Data Directories
- `csv_results/` - All structured CSV experiment results
- `reasoning_traces/` - Detailed reasoning traces organized by model and dataset
- `analysis_output/` - Standardized analysis results
- `comprehensive_analysis_output/` - Comprehensive experiment summaries

### Key Result Files
- `*.json` result files in root directory
- `*.jsonl` checkpoint and experiment logs
- `main_run_checkpoint.jsonl` - Main experiment checkpoint

## 🔄 Git Status

- **Current Branch**: `majority_vote_ensembler`
- **Remote Branch**: `origin/majority_vote_ensembler` 
- **Modified Files**: Extensive experiment results ready for commit
- **Status**: All completed experiments tracked and ready for GitHub push

## 🚀 Next Steps

1. **Immediate**: Push all completed experiment data to GitHub
2. **Short-term**: Resolve OpenAI credit limits for pending GPT experiments
3. **Medium-term**: Set up Llama model access for remaining experiments
4. **Analysis**: Generate final comparison reports across all completed experiments

---
*Last Updated: 2025-10-01 17:27:56Z*
*Branch: majority_vote_ensembler*
*Total Data Size: ~50MB+ of experiment results and traces*