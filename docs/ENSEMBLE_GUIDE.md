# Ensemble Self-Correction System Guide

This guide provides comprehensive instructions for using the ensemble voting implementation of the self-correction classification system.

## Overview

The ensemble system extends the core self-correction framework by using multiple models to solve each problem and combining their responses through sophisticated voting mechanisms. This approach typically improves accuracy and robustness compared to single-model approaches.

## Key Features

- **Multiple Voting Strategies**: Majority voting, confidence-weighted voting, consensus detection, and adaptive voting
- **Heterogeneous Ensembles**: Mix models from different providers (OpenAI, Anthropic, etc.) in single ensemble
- **Multi-Provider Support**: Full support for cross-provider ensembles with independent API management
- **Cost Optimization**: Track and optimize costs across ensemble models and providers
- **Comprehensive Analysis**: Detailed metrics on ensemble performance, disagreement patterns, and consensus strength
- **Dynamic Sizing**: Adjust ensemble size based on problem difficulty and cost constraints

## Quick Start

### 1. Basic Ensemble Experiment

Run a simple ensemble experiment with default OpenAI models:

```bash
python run_ensemble_experiments.py --dataset gsm8k --subset subset_20 --demo
```

### 2. Using Specific Configuration

Run with a specific ensemble configuration:

```bash
python run_ensemble_experiments.py \
  --config configs/ensemble_experiments/anthropic_ensemble.json \
  --dataset humaneval \
  --subset subset_100
```

### 3. Batch Experiments

Run all ensemble configurations in batch mode:

```bash
python run_ensemble_experiments.py \
  --batch \
  --dataset gsm8k \
  --output-dir outputs/batch_ensemble_results
```

## Configuration System

### Configuration Files

Ensemble configurations are stored in `configs/ensemble_experiments/` as JSON files.

#### Basic Structure

```json
{
  "name": "Configuration Name",
  "description": "Description of the ensemble setup",
  "provider": "openai|anthropic|mixed|demo",
  "ensemble_size": 3,
  "ensemble_models": ["model1", "model2", "model3"],
  "voting_strategy": "majority_with_confidence|weighted_confidence|consensus_detection|adaptive",
  "features": {
    "enable_confidence": true,
    "enable_error_awareness": true,
    "enable_multi_turn": true
  },
  "cost_optimization": {
    "enabled": true,
    "max_cost_per_question": 0.10,
    "early_stopping": true
  },
  "experiment_settings": {
    "max_turns": 3,
    "temperature": 0.2,
    "max_tokens": 1024
  }
}
```

#### Available Configurations

1. **`openai_basic.json`**: 3-model OpenAI ensemble (GPT-4o-mini, GPT-4o, GPT-3.5-turbo)
2. **`anthropic_ensemble.json`**: 3-model Anthropic ensemble (Haiku, Sonnet, Opus)
3. **`mixed_provider.json`**: Mixed provider ensemble combining OpenAI and Anthropic
4. **`heterogeneous_ensemble.json`**: True heterogeneous ensemble with 5 models across providers
5. **`demo_ensemble.json`**: Demo configuration for testing without API calls
6. **`demo_heterogeneous.json`**: Demo heterogeneous ensemble for testing

### Creating Custom Configurations

1. Copy an existing configuration file
2. Modify the models, voting strategy, and parameters as needed
3. Save with a descriptive name in `configs/ensemble_experiments/`

## Voting Strategies

### 1. Majority with Confidence (Default)

- Uses simple majority voting
- Breaks ties using confidence scores
- Best for discrete answers (math problems)

### 2. Weighted Confidence

- Weights each response by model confidence
- Good when models have well-calibrated confidence scores
- Useful for mixed-confidence scenarios

### 3. Consensus Detection  

- Uses text similarity to detect consensus
- Best for long-form responses (code generation)
- Handles paraphrased but equivalent answers

### 4. Adaptive Voting

- Automatically chooses strategy based on response characteristics
- Considers diversity, confidence spread, and task type
- Recommended for mixed workloads

## Environment Variables

Control ensemble behavior with environment variables:

```bash
# Ensemble configuration
export ENSEMBLE_SIZE=5                    # Override config ensemble size
export ENSEMBLE_VOTING_STRATEGY=adaptive # Override voting strategy

# Cost control
export ENSEMBLE_MAX_COST_PER_QUESTION=0.20
export ENSEMBLE_EARLY_STOPPING=true

# API keys for multi-provider ensembles
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"

# Demo mode (no API calls)
export DEMO_MODE=1
```

## Analysis and Metrics

### Running Analysis

After an ensemble experiment, analyze the results:

```bash
python -m src.ensemble.metrics \
  outputs/ensemble_experiments/experiment_id/traces.json \
  outputs/analysis_results
```

### Key Metrics

1. **Basic Performance**
   - Ensemble accuracy vs individual model accuracy
   - Success rates by turn number
   - Average turns per question

2. **Voting Analysis**
   - Distribution of voting methods used
   - Consensus ratio statistics
   - Tie-breaking frequency

3. **Disagreement Patterns**
   - When models disagree
   - Success rate during disagreement
   - Correlation between consensus and accuracy

4. **Confidence Calibration**
   - How well ensemble confidence predicts correctness
   - Expected calibration error
   - Confidence distribution analysis

5. **Cost Analysis**
   - Total cost per question
   - Cost efficiency compared to single models
   - Cost vs accuracy trade-offs

## Advanced Usage

### Dynamic Ensemble Sizing

Configure ensembles that adapt size based on problem characteristics:

```json
{
  "dynamic_sizing": {
    "enabled": true,
    "min_size": 3,
    "max_size": 7,
    "sizing_strategy": "confidence_based",
    "early_stopping": {
      "consensus_threshold": 0.8,
      "confidence_threshold": 0.9
    }
  }
}
```

### Heterogeneous Multi-Provider Ensembles

The system now supports **true heterogeneous ensembles** where each model can use a different provider:

```json
{
  "name": "Heterogeneous Ensemble",
  "provider": "mixed",
  "ensemble_size": 5,
  "ensemble_configs": [
    {"provider": "openai", "model": "gpt-4o-mini"},
    {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20241210"},
    {"provider": "openai", "model": "gpt-3.5-turbo"}
  ],
  "voting_strategy": "adaptive"
}
```

**Benefits of Heterogeneous Ensembles:**
- **Provider Diversity**: Reduces single-point-of-failure risk
- **Model Architecture Diversity**: Different training approaches and strengths
- **API Reliability**: Continues working even if one provider has issues
- **Cost Optimization**: Mix expensive and cheaper models strategically

**Usage Examples:**
```bash
# Test heterogeneous ensemble in demo mode
python run_ensemble_experiments.py \
  --config configs/ensemble_experiments/demo_heterogeneous.json \
  --dataset gsm8k --subset subset_20 --demo

# Production heterogeneous ensemble
python run_ensemble_experiments.py \
  --config configs/ensemble_experiments/heterogeneous_ensemble.json \
  --dataset humaneval --subset subset_50
```

### Cost Optimization

Control costs with various strategies:

```json
{
  "cost_optimization": {
    "enabled": true,
    "strategy": "adaptive",
    "max_cost_per_question": 0.15,
    "early_stopping": true,
    "model_pricing_tiers": {
      "cheap": ["gpt-3.5-turbo", "claude-3-haiku"],
      "expensive": ["gpt-4o", "claude-3-opus"]
    }
  }
}
```

## Best Practices

### 1. Model Selection

- **Diversity**: Choose models with different strengths/weaknesses
- **Cost Balance**: Mix expensive and cheaper models strategically  
- **Provider Diversity**: Use multiple providers to reduce single points of failure

### 2. Voting Strategy Selection

- **Math Problems**: Use majority_with_confidence for discrete answers
- **Code Generation**: Use consensus_detection for long-form responses
- **Mixed Workloads**: Use adaptive voting for automatic strategy selection

### 3. Cost Management

- Set realistic cost limits per question
- Use early stopping when high consensus is achieved
- Monitor cost efficiency metrics regularly

### 4. Evaluation

- Always compare ensemble performance to best individual model
- Analyze disagreement patterns to understand when ensemble helps
- Monitor confidence calibration to ensure reliable uncertainty estimates

## Troubleshooting

### Common Issues

1. **High Costs**
   - Reduce ensemble size
   - Use cheaper models
   - Enable early stopping
   - Set stricter cost limits

2. **Poor Consensus**
   - Check model compatibility
   - Adjust similarity thresholds for consensus detection
   - Consider different voting strategies

3. **API Errors**
   - Verify API keys are set correctly
   - Check rate limits and quotas
   - Use demo mode for testing

4. **Low Ensemble Improvement**
   - Ensure model diversity
   - Check if individual models are already near optimal
   - Consider different voting strategies

### Debug Mode

Enable detailed logging for debugging:

```bash
export ENSEMBLE_DEBUG=1
export OPENAI_DEBUG=1
python run_ensemble_experiments.py --config your_config.json
```

## Integration with Main System

The ensemble implementation is designed to be a drop-in replacement for single-model experiments:

```python
from src.ensemble.runner import run_dataset
from src.ensemble.learner import EnsembleLearnerBot

# Use exactly like the main system
results = run_dataset(
    dataset_csv="gsm8k",
    provider="openai", 
    model="gpt-4o-mini",
    config=ensemble_config  # Enables ensemble mode
)
```

## Research Applications

### Comparative Studies

Compare different ensemble strategies:

```bash
# Run batch experiments with different voting strategies
python run_ensemble_experiments.py --batch --dataset gsm8k

# Analyze results across configurations  
python scripts/compare_ensemble_results.py outputs/batch_*/
```

### Scaling Studies

Investigate ensemble size vs performance:

```python
# Create configurations with different ensemble sizes
for size in [3, 5, 7, 9]:
    config = create_scaling_config(size)
    run_ensemble_experiment(config, dataset)
```

### Cost-Benefit Analysis

Analyze cost vs accuracy trade-offs:

```python
from src.ensemble.metrics import EnsembleMetrics

analyzer = EnsembleMetrics()
cost_analysis = analyzer.analyze_cost_efficiency(traces_file)
```

## Contributing

To extend the ensemble system:

1. **New Voting Strategies**: Add methods to `EnsembleLearnerBot`
2. **New Metrics**: Extend `EnsembleMetrics` class
3. **New Configurations**: Add JSON configs for different use cases
4. **Provider Support**: Extend multi-provider capabilities

See the main README for contribution guidelines.

## Citation

If you use the ensemble system in research, please cite:

```bibtex
@software{algoverse_ensemble_2024,
  title={Ensemble Self-Correction for Large Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/algoverse-self-correction-classification}
}
```