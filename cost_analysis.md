# Scaling Study Cost Analysis

## Dataset Configuration
- **GSM8K**: 200 samples (math reasoning)
- **HumanEval**: 164 samples (code generation) 
- **ToolQA**: 100 samples (QA with tool usage)
- **SuperGLUE**: 150 samples (logical reasoning)
- **MathBench**: 100 samples (college-level math)

**Total Samples**: 714 problems across 5 task types

## Model Portfolio (6 models for cost control)
1. **GPT-4o-mini** ($0.150/$0.600 per 1M tokens)
2. **GPT-4o** ($2.50/$10.00 per 1M tokens)  
3. **Claude Haiku** ($0.25/$1.25 per 1M tokens)
4. **Claude Sonnet** ($3.00/$15.00 per 1M tokens)
5. **Claude Opus** ($15.00/$75.00 per 1M tokens)
6. **Llama-70B** (~$0.70/$2.80 per 1M tokens via API)

## Token Usage Estimation

### Per Sample Token Usage:
- **Initial Prompt**: ~200 tokens
- **Model Response**: ~300 tokens  
- **Self-Correction Prompt**: ~400 tokens
- **Correction Response**: ~350 tokens
- **Total per sample**: ~1,250 tokens
- **Input/Output Split**: ~600 input, ~650 output

### Per Model Totals:
- **714 samples × 1,250 tokens = 892,500 tokens per model**
- **Input tokens**: ~428,400 per model
- **Output tokens**: ~464,100 per model

## Cost Breakdown by Model

| Model | Input Cost | Output Cost | Total per Model | 6 Models Total |
|-------|------------|-------------|-----------------|----------------|
| GPT-4o-mini | $0.064 | $0.278 | $0.34 | $2.04 |
| GPT-4o | $1.07 | $4.64 | $5.71 | $34.26 |
| Claude Haiku | $0.107 | $0.580 | $0.69 | $4.14 |
| Claude Sonnet | $1.29 | $6.96 | $8.25 | $49.50 |
| Claude Opus | $6.43 | $34.81 | $41.24 | $247.44 |
| Llama-70B | $0.30 | $1.30 | $1.60 | $9.60 |

## Total Experiment Cost: **~$347**

## Majority Vote Baseline Cost
- **5 samples per problem** for ensemble baseline
- **714 × 5 = 3,570 baseline samples**
- **Single-turn responses**: ~500 tokens each
- **Additional cost**: ~$200-300 depending on model mix

## **Complete Project Cost: $550-650**

## Cost Optimization Strategies

### Phase 1: Proof of Concept ($150)
- **3 models**: GPT-4o-mini, Claude Haiku, Llama-70B
- **3 datasets**: GSM8K (100), HumanEval (100), ToolQA (50)
- **250 samples total**

### Phase 2: Full Scale ($400)
- **Add remaining models**: GPT-4o, Claude Sonnet, Claude Opus
- **Add remaining datasets**: SuperGLUE, MathBench
- **Complete 714 sample matrix**

## Risk Mitigation
- **Start with cheaper models** to validate pipeline
- **Use subset sampling** for expensive models (Opus)
- **Implement early stopping** if patterns emerge
- **Focus budget on medium-sized models** (best ROI)

## Expected Timeline
- **Week 1**: Infrastructure + Phase 1 ($150)
- **Week 2**: Full experiments + analysis ($400)
- **Buffer**: $100 for debugging/reruns

**Total Budget Needed**: $650 (includes 20% buffer)
