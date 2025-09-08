#!/usr/bin/env python3
"""
Analyze Scaling Laws

Analyzes scaling laws in self-correction experiments.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.scaling_analyzer import ScalingAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Analyze scaling laws in experiments")
    parser.add_argument("--results-dir", default="outputs/scaling_experiments",
                       help="Directory containing experiment results")
    parser.add_argument("--output", default="outputs/scaling_analysis.json",
                       help="Output file for analysis results")
    
    args = parser.parse_args()
    
    print("🔬 Analyzing Scaling Laws")
    print("=" * 30)
    
    # Initialize analyzer
    analyzer = ScalingAnalyzer()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = analyzer.load_experiment_results(args.results_dir)
    
    if not results:
        print("❌ No results found to analyze")
        return 1
    
    print(f"✓ Loaded {len(results)} experiment results")
    
    # Analyze scaling
    print("Analyzing scaling patterns...")
    analysis = analyzer.analyze_scaling(results)
    
    # Print summary
    print("\n📊 Scaling Analysis Summary")
    print("=" * 40)
    
    summary = analysis.get("summary", {})
    print(f"Total experiments: {summary.get('total_experiments', 0)}")
    print(f"Models tested: {summary.get('models_tested', 0)}")
    print(f"Datasets tested: {summary.get('datasets_tested', 0)}")
    print(f"Average improvement: {summary.get('avg_improvement', 0):.3f}")
    print(f"Max improvement: {summary.get('max_improvement', 0):.3f}")
    print(f"Min improvement: {summary.get('min_improvement', 0):.3f}")
    
    # Print scaling laws
    scaling_laws = analysis.get("scaling_laws", {})
    if scaling_laws:
        print("\n🔬 Scaling Laws")
        print("-" * 20)
        
        for law_name, law_data in scaling_laws.items():
            print(f"\n{law_name}:")
            print(f"  Equation: {law_data['equation']}")
            print(f"  R² = {law_data['r_squared']:.3f}")
            print(f"  P-value = {law_data['p_value']:.3f}")
            if 'interpretation' in law_data:
                print(f"  Interpretation: {law_data['interpretation']}")
    
    # Print cost analysis
    cost_analysis = analysis.get("cost_analysis", {})
    if cost_analysis:
        print("\n💰 Cost Analysis")
        print("-" * 20)
        print(f"Total cost: ${cost_analysis.get('total_cost', 0):.2f}")
        print(f"Avg cost per experiment: ${cost_analysis.get('avg_cost_per_experiment', 0):.2f}")
        print(f"Most cost-efficient model: {cost_analysis.get('most_cost_efficient_model', 'N/A')}")
        
        efficiency_range = cost_analysis.get('cost_efficiency_range', {})
        if efficiency_range:
            print(f"Cost efficiency range: {efficiency_range.get('min', 0):.3f} - {efficiency_range.get('max', 0):.3f}")
    
    # Print recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        print("\n💡 Recommendations")
        print("-" * 20)
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Save analysis
    print(f"\nSaving analysis to: {args.output}")
    analyzer.save_analysis(analysis, args.output)
    
    print("✅ Analysis complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
