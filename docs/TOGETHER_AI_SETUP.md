# Together AI Setup Guide

## Overview
This project now uses Together AI for running Llama models instead of Replicate, providing:
- **4x faster inference** compared to Replicate
- **OpenAI-compatible API** for easier integration
- **Better rate limits** for research experiments
- **Lower costs** (~$0.0009 per 1k tokens for Llama-70B)

## Quick Start

### 1. Get Together AI API Key
1. Sign up at [together.ai](https://together.ai)
2. Navigate to Settings → API Keys
3. Create a new API key

### 2. Add API Key to Environment
Add to your `.env` file:
```bash
TOGETHER_API_KEY=your_api_key_here
```

### 3. Verify Installation
Run the test script:
```bash
python test_together_integration.py
```

You should see all tests passing:
```
✅ PASS: API Connection
✅ PASS: LearnerBot Integration
✅ PASS: Model Configuration
✅ PASS: Cost Estimation
```

## Running Experiments with Llama Models

### Single Model Test
```bash
python run_toolqa_experiments.py --models llama-7b --samples 10
```

### Full Llama Suite
```bash
python run_full_scale_study.py --models llama-7b llama-13b llama-70b --datasets toolqa gsm8k
```

### Available Llama Models
- `llama-7b` - Small model, fastest, ~$0.0002/1k tokens
- `llama-13b` - Medium model, balanced, ~$0.00025/1k tokens  
- `llama-70b` - Large model, best quality, ~$0.0009/1k tokens

## Cost Estimation

For 100 samples with 3 turns each (~6k tokens per sample):
- Llama-7B: ~$0.12
- Llama-13B: ~$0.15
- Llama-70B: ~$0.54
- **Total for all three**: ~$0.81

## Configuration Details

The models are configured in `configs/scaling_models.json`:

```json
{
  "name": "llama-70b",
  "provider": "together",
  "model_id": "meta-llama/Llama-2-70b-chat-hf",
  "size_category": "large",
  "estimated_cost_per_1k_tokens": 0.0009,
  "max_tokens": 4000,
  "temperature": 0.0,
  "available": true,
  "description": "Large Llama model, 70B parameters (via Together AI)"
}
```

## Troubleshooting

### API Key Issues
If you see `TOGETHER_API_KEY not found`:
1. Check your `.env` file exists in the project root
2. Verify the key is correctly formatted
3. Restart your Python session after adding the key

### Rate Limiting
Together AI has generous rate limits, but if you encounter issues:
1. Reduce concurrency: `export MAX_CONCURRENCY=1`
2. Add delay between calls: `export RPS_LIMIT=0.5`

### Model Access
Some models may require approval. Check available models:
```python
from together import Together
client = Together()
models = client.models.list()
llama_models = [m for m in models if 'llama' in m.id.lower()]
```

## Migration from Replicate

If you have existing Replicate experiments:
1. Results format remains compatible
2. Cost tracking works the same way
3. No changes needed to analysis scripts

The main differences:
- Faster execution (4x improvement)
- Lower costs (~30% cheaper)
- More reliable (no async queue delays)

## Advanced Configuration

### Environment Variables
```bash
# Required
TOGETHER_API_KEY=your_key

# Optional overrides
TOGETHER_MODEL=meta-llama/Llama-2-70b-chat-hf
TOGETHER_MAX_TOKENS=2048
TOGETHER_TEMPERATURE=0.2
```

### Custom Rate Limits
For production runs with high volume:
```python
os.environ['TOGETHER_RPM'] = '100'  # Requests per minute
os.environ['TOGETHER_TPM'] = '1000000'  # Tokens per minute
```

## Support

- Together AI Documentation: https://docs.together.ai
- API Status: https://status.together.ai
- Support: support@together.ai

For project-specific issues, check the test script output:
```bash
python test_together_integration.py --verbose
```