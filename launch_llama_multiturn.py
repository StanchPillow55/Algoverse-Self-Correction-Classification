#!/usr/bin/env python3
"""
Launch Script for Llama Multi-Turn Experiment

Ensures output structure matches existing OpenAI/Anthropic runs:
- For 500 questions: 500 consolidated .txt files (one per question, all turns)
- Chart-ready accuracy JSON data
- Compatible traces.json format
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from src.utils.consolidated_trace_formatter import ConsolidatedTraceFormatter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_environment():
    """Set up environment variables for Llama experiment."""
    print("🔧 Setting up Llama experiment environment...")
    
    # Verify API keys
    replicate_key = os.getenv('REPLICATE_API_TOKEN')
    if not replicate_key:
        print("❌ REPLICATE_API_TOKEN not found in .env file")
        print("Please add: REPLICATE_API_TOKEN=r8_...")
        return False
    
    # Set conservative rate limits for Llama
    os.environ.update({
        'MAX_CONCURRENCY': '1',
        'RPS_LIMIT': '0.5',
        'TPM_LIMIT': '30000',
        'MAX_RETRIES': '10',
        'RETRIES_ENABLED': '1',
        'PROVIDER': 'replicate',
        'REPLICATE_MODEL': 'meta/llama-2-70b-chat'
    })
    
    print("✅ Environment configured for Llama-70B")
    return True

def estimate_costs(dataset_size: int):
    """Estimate costs for Llama experiment."""
    # Replicate Llama-70B costs approximately $0.0007 per 1K tokens
    # Multi-turn experiment with 3 turns: ~6K tokens per question average
    tokens_per_question = 6000
    total_tokens = dataset_size * tokens_per_question
    cost_per_1k_tokens = 0.0007
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    print(f"📊 Cost Estimation:")
    print(f"   Dataset size: {dataset_size} questions")
    print(f"   Estimated tokens: {total_tokens:,}")
    print(f"   Estimated cost: ${estimated_cost:.2f}")
    print(f"   Cost per question: ${estimated_cost/dataset_size:.4f}")
    
    return estimated_cost

def run_llama_experiment(dataset: str, subset: str, output_dir: str):
    """Run the Llama multi-turn experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"llama70b_{dataset}_{timestamp}"
    output_path = f"{output_dir}/{experiment_id}_traces.json"
    
    print(f"🚀 Starting Llama multi-turn experiment...")
    print(f"   Experiment ID: {experiment_id}")
    print(f"   Dataset: {dataset}")
    print(f"   Subset: {subset}")
    print(f"   Output: {output_path}")
    
    # Build the command
    cmd_parts = [
        "python", "-m", "src.main", "run",
        f"--dataset={dataset}",
        f"--subset={subset}",
        "--max-turns=3",
        f"--out={output_path}",
        "--provider=replicate",
        "--model=llama-70b",
        "--config=configs/llama_multiturn_config.yaml",
        "--checkpoint-every=10",
        "--resume"
    ]
    
    cmd = " ".join(cmd_parts)
    print(f"🔥 Command: {cmd}")
    
    # Execute the experiment
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            return output_path
        else:
            print("❌ Experiment failed:")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return None

def format_results(traces_file: str, experiment_id: str):
    """Format results with consolidated structure."""
    if not Path(traces_file).exists():
        print(f"❌ Traces file not found: {traces_file}")
        return
    
    print("📋 Formatting results with consolidated structure...")
    
    formatter = ConsolidatedTraceFormatter(output_dir="runs")
    outputs = formatter.format_llama_experiment_traces(traces_file, experiment_id)
    
    if outputs:
        print("✅ Results formatted successfully:")
        for output_type, path in outputs.items():
            print(f"   {output_type}: {path}")
        
        # Verify structure
        verify_output_structure(outputs)
    else:
        print("❌ Failed to format results")

def verify_output_structure(outputs: dict):
    """Verify the output structure matches requirements."""
    print("🔍 Verifying output structure...")
    
    # Check for reasoning traces directory
    reasoning_dir = outputs.get("reasoning_traces_dir")
    if reasoning_dir and Path(reasoning_dir).exists():
        # Count .txt files
        txt_files = list(Path(reasoning_dir).rglob("*.txt"))
        print(f"   ✅ Reasoning traces: {len(txt_files)} .txt files found")
        
        # Verify structure: dataset_type/question_id/consolidated_reasoning.txt
        if txt_files:
            sample_file = txt_files[0]
            parts = sample_file.parts
            if len(parts) >= 3 and parts[-1] == "consolidated_reasoning.txt":
                print(f"   ✅ File structure: {parts[-3]}/{parts[-2]}/{parts[-1]}")
            else:
                print(f"   ⚠️  Unexpected file structure: {sample_file}")
    
    # Check for chart data
    chart_data = outputs.get("accuracy_chart_data")
    if chart_data and Path(chart_data).exists():
        with open(chart_data, 'r') as f:
            data = json.load(f)
            total_questions = data.get("experiment_metadata", {}).get("total_questions", 0)
            print(f"   ✅ Chart data: {total_questions} questions processed")
    
    print("✅ Output structure verification complete")

def main():
    parser = argparse.ArgumentParser(description="Launch Llama Multi-Turn Experiment")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "humaneval", "superglue", "mathbench"],
                       help="Dataset to use")
    parser.add_argument("--subset", default="subset_100", choices=["subset_20", "subset_100", "subset_500", "full"],
                       help="Dataset subset size")
    parser.add_argument("--output-dir", default="runs", help="Output directory")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate costs, don't run")
    
    args = parser.parse_args()
    
    print("🦙 Llama Multi-Turn Experiment Launcher")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Estimate costs
    subset_sizes = {"subset_20": 20, "subset_100": 100, "subset_500": 500, "full": 1000}
    dataset_size = subset_sizes.get(args.subset, 100)
    
    estimated_cost = estimate_costs(dataset_size)
    
    if args.estimate_only:
        print("💰 Cost estimation complete. Use --no-estimate-only to run experiment.")
        return 0
    
    # Confirm execution
    response = input(f"\\n🤔 Continue with experiment? (estimated cost: ${estimated_cost:.2f}) [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Experiment cancelled")
        return 0
    
    # Run experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"llama70b_{args.dataset}_{timestamp}"
    
    traces_file = run_llama_experiment(args.dataset, args.subset, args.output_dir)
    
    if traces_file:
        # Format results with consolidated structure
        format_results(traces_file, experiment_id)
        
        print("\\n🎉 Llama multi-turn experiment complete!")
        print("\\n📂 Output Structure:")
        print(f"   runs/{experiment_id}/")
        print("   ├── reasoning_traces/[dataset_type]/[question_id]/consolidated_reasoning.txt")
        print("   ├── accuracy_chart_data.json")
        print("   ├── traces.json") 
        print("   └── summary.json")
        
    else:
        print("❌ Experiment failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())