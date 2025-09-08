#!/bin/bash
#
# Setup Script for Scaling Study
# Prepares the environment for multi-model self-correction experiments
#

set -euo pipefail

echo "🚀 Setting up Scaling Study Infrastructure"
echo "=========================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/scaling
mkdir -p outputs/scaling_experiments
mkdir -p configs
mkdir -p logs

# Install additional dependencies
echo "📦 Installing dependencies..."
pip install anthropic replicate pandas scikit-learn matplotlib seaborn

# Download datasets
echo "📥 Downloading datasets..."
python -c "
from src.data.scaling_datasets import ScalingDatasetManager
manager = ScalingDatasetManager()
results = manager.download_all_datasets()
print('Dataset download results:', results)
"

# Initialize model configurations
echo "⚙️  Initializing model configurations..."
python -c "
from src.utils.scaling_model_manager import ScalingModelManager
manager = ScalingModelManager()
manager.save_config()
print('Model configurations saved')
"

# Test model availability
echo "🔍 Checking model availability..."
python -c "
from src.utils.scaling_model_manager import ScalingModelManager
manager = ScalingModelManager()
available = manager.get_available_models()
print(f'Available models: {[m.name for m in available]}')
"

# Estimate costs
echo "💰 Estimating experiment costs..."
python scripts/run_scaling_experiment.py --estimate-costs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add API keys to .env file:"
echo "   - OPENAI_API_KEY=your-key"
echo "   - ANTHROPIC_API_KEY=your-key"
echo "   - REPLICATE_API_TOKEN=your-token"
echo ""
echo "2. Run a small test experiment:"
echo "   python scripts/run_scaling_experiment.py --models gpt-4o-mini --datasets toolqa --sample-sizes 10"
echo ""
echo "3. Run full scaling experiment:"
echo "   python scripts/run_scaling_experiment.py --config configs/scaling_experiment.yaml"
echo ""
echo "4. Monitor progress:"
echo "   tail -f logs/scaling_experiment.log"
