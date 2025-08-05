# API Keys Setup Guide

This document explains how to set up API keys for external LLM providers when they become available.

## Current Status

🔄 **Placeholder Mode**: The pipeline currently uses local models and placeholder data for prototyping.

When API keys are provided, the system will automatically switch to live API inference.

## Supported Providers

### OpenAI
- **Models**: GPT-4, text-embedding-ada-002, text-embedding-3-small
- **Use Cases**: Text generation, embeddings
- **Environment Variable**: `OPENAI_API_KEY`

### Anthropic (Claude)
- **Models**: claude-3-sonnet-20240229
- **Use Cases**: Text generation
- **Environment Variable**: `ANTHROPIC_API_KEY`

### Local Models (Available Now)
- **sentence-transformers/all-MiniLM-L6-v2**: Lightweight embedding model
- **sentence-transformers/all-mpnet-base-v2**: Higher quality embedding model
- **sentence-transformers/distilbert-base-nli-mean-tokens**: Alternative embedding model

## Quick Setup Instructions

### Step 1: Copy Environment File
```bash
cp .env.example .env
```

### Step 2: Add Your API Keys
Edit `.env` file:
```bash
# API Keys (replace with actual keys when available)
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here
```

### Step 3: Verify Setup
```bash
python -m src.main model-status
```

This will show you which APIs are available and which models the system will use.

## API Key Priority

The system follows this priority order:

1. **OpenAI APIs** (if key available)
   - Text generation: GPT-4
   - Embeddings: text-embedding-ada-002

2. **Anthropic APIs** (if key available)
   - Text generation: Claude-3-Sonnet

3. **Local Models** (fallback)
   - Embeddings: sentence-transformers models
   - Text generation: Placeholder responses

## Automatic Switching

The pipeline automatically detects available API keys and switches between:

- **Local mode**: Uses HuggingFace models and placeholder responses
- **API mode**: Uses live API calls for generation and embeddings

No code changes are required - just add the API keys to your `.env` file.

## Testing API Integration

Once you have API keys:

1. **Check model status**:
   ```bash
   python -m src.main model-status
   ```

2. **Generate synthetic data** (will use API if available):
   ```bash
   python -m src.main generate-data --samples-per-error 5
   ```

3. **Verify the generated data** includes real API responses instead of placeholders

## Troubleshooting

### Common Issues

1. **API Key Not Detected**
   - Ensure `.env` file is in the project root
   - Check for typos in environment variable names
   - Restart the application after adding keys

2. **Rate Limiting**
   - OpenAI and Anthropic have rate limits
   - Reduce `--samples-per-error` when generating data
   - Add delays between API calls if needed

3. **Invalid API Key**
   - Verify the key is correct and has appropriate permissions
   - Check the API key hasn't expired

### Getting Help

Run the model status command to see current configuration:
```bash
python -m src.main model-status
```

This will show:
- Which APIs are available
- Which models are being used
- Current configuration status

## Future Enhancements

When API keys are available, the following features will be enhanced:

- **Higher Quality Synthetic Data**: Real LLM-generated examples instead of placeholders
- **Better Embeddings**: API-based embeddings for improved classification
- **Advanced Error Generation**: More sophisticated error pattern generation
- **Real-time Classification**: Live inference for new text samples

## Security Notes

- Never commit API keys to version control
- Use environment variables or secure key management systems
- Rotate API keys regularly
- Monitor API usage and costs
