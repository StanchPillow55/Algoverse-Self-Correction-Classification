#!/usr/bin/env python3
"""
Run scaling analysis on experiment results
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Run scaling analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run scaling analysis on experiment results')
    parser.add_argument('--results-dir', default='outputs/scaling_experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output-file', default='outputs/scaling_analysis.json',
                       help='Output file for analysis results')
    parser.add_argument('--create-plots', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    print("📊 Running Scaling Analysis")
    print("=" * 30)
    
    try:
        from src.utils.scaling_analyzer import ScalingAnalyzer
        
        # Initialize analyzer
        analyzer = ScalingAnalyzer()
        print("✓ Scaling analyzer initialized")
        
        # Load experiment results
        results = analyzer.load_experiment_results(args.results_dir)
        print(f"✓ Loaded {len(results)} experiment results")
        
        if not results:
            print("⚠️  No results found to analyze")
            print(f"   Check that results exist in: {args.results_dir}")
            return
        
        # Run analysis
        analysis = analyzer.analyze_scaling(results)
        print("✓ Scaling analysis completed")
        
        # Save analysis
        analyzer.save_analysis(analysis, args.output_file)
        print(f"✓ Analysis saved to: {args.output_file}")
        
        # Print summary
        print("\n📈 Analysis Summary")
        print("-" * 20)
        
        summary = analysis.get("summary", {})
        print(f"Total experiments: {summary.get('total_experiments', 0)}")
        print(f"Models tested: {summary.get('models_tested', 0)}")
        print(f"Datasets tested: {summary.get('datasets_tested', 0)}")
        print(f"Average improvement: {summary.get('avg_improvement', 0):.3f}")
        print(f"Max improvement: {summary.get('max_improvement', 0):.3f}")
        
        # Print scaling laws
        scaling_laws = analysis.get("scaling_laws", {})
        if "model_size_vs_improvement" in scaling_laws:
            scaling = scaling_laws["model_size_vs_improvement"]
            print(f"\n🔬 Scaling Law: {scaling['equation']}")
            print(f"   R² = {scaling['r_squared']:.3f}")
            print(f"   Exponent = {scaling['exponent']:.3f}")
            print(f"   Interpretation: {scaling['interpretation']}")
        
        # Print cost analysis
        cost_analysis = analysis.get("cost_analysis", {})
        if cost_analysis:
            print(f"\n💰 Cost Analysis")
            print(f"   Total cost: ${cost_analysis.get('total_cost', 0):.2f}")
            print(f"   Most efficient model: {cost_analysis.get('most_cost_efficient_model', 'N/A')}")
        
        # Print recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Create plots if requested
        if args.create_plots:
            print(f"\n🎨 Creating visualization plots...")
            try:
                from scripts.visualize_scaling_laws import create_scaling_plots
                create_scaling_plots(args.output_file, "outputs/figures")
                print("✓ Plots created successfully")
            except Exception as e:
                print(f"⚠️  Error creating plots: {e}")
        
        print(f"\n✅ Scaling analysis complete!")
        
    except Exception as e:
        print(f"❌ Error running scaling analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
