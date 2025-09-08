#!/usr/bin/env python3
"""
Analyze Scaling Study Results

Analyzes results from scaling experiments and generates reports.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.result_aggregator import ResultAggregator

def main():
    parser = argparse.ArgumentParser(description="Analyze scaling study results")
    parser.add_argument("--results-dir", default="outputs/scaling_experiments",
                       help="Directory containing experiment results")
    parser.add_argument("--pattern", default="*.json",
                       help="Pattern to match result files")
    parser.add_argument("--output", default="scaling_analysis_report.json",
                       help="Output filename for analysis report")
    parser.add_argument("--print-summary", action="store_true",
                       help="Print summary to console")
    
    args = parser.parse_args()
    
    print("🔍 Analyzing Scaling Study Results")
    print("=" * 40)
    
    # Initialize aggregator
    aggregator = ResultAggregator(args.results_dir)
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    count = aggregator.load_results(args.pattern)
    
    if count == 0:
        print("❌ No results found to analyze")
        return 1
    
    print(f"✓ Loaded {count} experiment results")
    
    # Create DataFrame
    df = aggregator.create_dataframe()
    print(f"✓ Created DataFrame with {len(df)} rows")
    
    # Generate report
    print("Generating analysis report...")
    report = aggregator.generate_report()
    
    # Save report
    output_path = aggregator.save_report(args.output)
    print(f"✓ Report saved to: {output_path}")
    
    # Print summary if requested
    if args.print_summary:
        aggregator.print_summary()
    
    # Print key insights
    print("\n🎯 Key Insights:")
    recommendations = report.get("recommendations", [])
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
