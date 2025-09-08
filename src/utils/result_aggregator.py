"""
Result Aggregation System for Scaling Study

Aggregates results from multiple experiments and calculates scaling metrics.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    model_name: str
    dataset_name: str
    sample_size: int
    initial_accuracy: float
    final_accuracy: float
    improvement: float
    cost: float
    tokens: int
    latency: float
    metadata: Dict[str, Any]

class ResultAggregator:
    """Aggregates and analyzes results from scaling experiments."""
    
    def __init__(self, results_dir: str = "outputs/scaling_experiments"):
        """Initialize result aggregator."""
        self.results_dir = Path(results_dir)
        self.results: List[ExperimentResult] = []
        self.df: Optional[pd.DataFrame] = None
    
    def load_results(self, pattern: str = "*.json") -> int:
        """Load results from JSON files in the results directory."""
        result_files = list(self.results_dir.glob(pattern))
        
        if not result_files:
            logger.warning(f"No result files found in {self.results_dir}")
            return 0
        
        loaded_count = 0
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract experiment metadata from filename
                filename = file_path.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    model_name = parts[0]
                    dataset_name = parts[1]
                    sample_size = int(parts[2])
                else:
                    # Fallback: try to extract from JSON data
                    model_name = data.get('metadata', {}).get('model', 'unknown')
                    dataset_name = data.get('metadata', {}).get('dataset', 'unknown')
                    sample_size = data.get('metadata', {}).get('sample_size', 0)
                
                # Calculate metrics
                traces = data.get('traces', [])
                if not traces:
                    continue
                
                # Calculate initial and final accuracy
                initial_correct = sum(1 for t in traces if t.get('turns', [{}])[0].get('accuracy', 0) == 1)
                final_correct = sum(1 for t in traces if t.get('final_accuracy', 0) == 1)
                
                initial_accuracy = initial_correct / len(traces)
                final_accuracy = final_correct / len(traces)
                improvement = final_accuracy - initial_accuracy
                
                # Estimate cost and tokens (rough approximation)
                total_tokens = sum(len(t.get('turns', [])) * 100 for t in traces)  # Rough estimate
                cost = total_tokens * 0.001  # Rough cost estimate
                
                # Create result object
                result = ExperimentResult(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    sample_size=sample_size,
                    initial_accuracy=initial_accuracy,
                    final_accuracy=final_accuracy,
                    improvement=improvement,
                    cost=cost,
                    tokens=total_tokens,
                    latency=0.0,  # Not tracked in current system
                    metadata={
                        "file": str(file_path),
                        "total_samples": len(traces),
                        "timestamp": data.get('timestamp', 0)
                    }
                )
                
                self.results.append(result)
                loaded_count += 1
                
            except Exception as e:
                logger.error(f"Failed to load result file {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {loaded_count} experiment results")
        return loaded_count
    
    def create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from results."""
        if not self.results:
            logger.warning("No results to create DataFrame")
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'model_name': result.model_name,
                'dataset_name': result.dataset_name,
                'sample_size': result.sample_size,
                'initial_accuracy': result.initial_accuracy,
                'final_accuracy': result.final_accuracy,
                'improvement': result.improvement,
                'cost': result.cost,
                'tokens': result.tokens,
                'latency': result.latency,
                'cost_efficiency': result.improvement / result.cost if result.cost > 0 else 0
            })
        
        self.df = pd.DataFrame(data)
        return self.df
    
    def calculate_scaling_metrics(self) -> Dict[str, Any]:
        """Calculate scaling metrics from aggregated results."""
        if self.df is None or self.df.empty:
            logger.warning("No data to calculate scaling metrics")
            return {}
        
        metrics = {
            "summary": {
                "total_experiments": len(self.df),
                "total_cost": self.df['cost'].sum(),
                "total_tokens": self.df['tokens'].sum(),
                "avg_improvement": self.df['improvement'].mean(),
                "max_improvement": self.df['improvement'].max(),
                "min_improvement": self.df['improvement'].min()
            },
            "by_model": {},
            "by_dataset": {},
            "by_sample_size": {},
            "scaling_analysis": {}
        }
        
        # Analysis by model
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            metrics["by_model"][model] = {
                "experiments": len(model_data),
                "avg_improvement": model_data['improvement'].mean(),
                "avg_cost": model_data['cost'].mean(),
                "avg_cost_efficiency": model_data['cost_efficiency'].mean(),
                "total_cost": model_data['cost'].sum()
            }
        
        # Analysis by dataset
        for dataset in self.df['dataset_name'].unique():
            dataset_data = self.df[self.df['dataset_name'] == dataset]
            metrics["by_dataset"][dataset] = {
                "experiments": len(dataset_data),
                "avg_improvement": dataset_data['improvement'].mean(),
                "avg_cost": dataset_data['cost'].mean(),
                "avg_cost_efficiency": dataset_data['cost_efficiency'].mean()
            }
        
        # Analysis by sample size
        for size in self.df['sample_size'].unique():
            size_data = self.df[self.df['sample_size'] == size]
            metrics["by_sample_size"][size] = {
                "experiments": len(size_data),
                "avg_improvement": size_data['improvement'].mean(),
                "avg_cost": size_data['cost'].mean()
            }
        
        # Scaling analysis
        if len(self.df) > 1:
            # Correlation between sample size and improvement
            size_improvement_corr = self.df['sample_size'].corr(self.df['improvement'])
            
            # Correlation between cost and improvement
            cost_improvement_corr = self.df['cost'].corr(self.df['improvement'])
            
            metrics["scaling_analysis"] = {
                "size_improvement_correlation": size_improvement_corr,
                "cost_improvement_correlation": cost_improvement_corr,
                "improvement_std": self.df['improvement'].std(),
                "cost_std": self.df['cost'].std()
            }
        
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Create DataFrame if not exists
        if self.df is None:
            self.create_dataframe()
        
        # Calculate metrics
        metrics = self.calculate_scaling_metrics()
        
        # Add detailed analysis
        report = {
            "metadata": {
                "total_experiments": len(self.results),
                "results_dir": str(self.results_dir),
                "generated_at": pd.Timestamp.now().isoformat()
            },
            "metrics": metrics,
            "recommendations": self._generate_recommendations(metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if not metrics:
            return ["No data available for recommendations"]
        
        # Cost efficiency recommendations
        by_model = metrics.get("by_model", {})
        if by_model:
            best_model = max(by_model.items(), key=lambda x: x[1].get("avg_cost_efficiency", 0))
            recommendations.append(f"Most cost-efficient model: {best_model[0]} (${best_model[1]['avg_cost_efficiency']:.2f} improvement per dollar)")
        
        # Improvement recommendations
        summary = metrics.get("summary", {})
        avg_improvement = summary.get("avg_improvement", 0)
        
        if avg_improvement > 0.1:
            recommendations.append("Self-correction shows significant improvement (>10%)")
        elif avg_improvement > 0.05:
            recommendations.append("Self-correction shows moderate improvement (5-10%)")
        else:
            recommendations.append("Self-correction shows minimal improvement (<5%)")
        
        # Scaling recommendations
        scaling = metrics.get("scaling_analysis", {})
        size_corr = scaling.get("size_improvement_correlation", 0)
        
        if size_corr > 0.5:
            recommendations.append("Strong positive correlation between sample size and improvement")
        elif size_corr > 0.2:
            recommendations.append("Moderate positive correlation between sample size and improvement")
        else:
            recommendations.append("Weak correlation between sample size and improvement")
        
        return recommendations
    
    def save_report(self, filename: str = "scaling_analysis_report.json") -> Path:
        """Save analysis report to file."""
        report = self.generate_report()
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print summary to console."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n📊 Scaling Study Results Summary")
        print("=" * 50)
        
        # Basic stats
        total_experiments = len(self.results)
        total_cost = sum(r.cost for r in self.results)
        avg_improvement = sum(r.improvement for r in self.results) / total_experiments
        
        print(f"Total experiments: {total_experiments}")
        print(f"Total cost: ${total_cost:.2f}")
        print(f"Average improvement: {avg_improvement:.3f}")
        
        # By model
        print("\nBy Model:")
        by_model = {}
        for result in self.results:
            if result.model_name not in by_model:
                by_model[result.model_name] = []
            by_model[result.model_name].append(result)
        
        for model, results in by_model.items():
            avg_imp = sum(r.improvement for r in results) / len(results)
            total_cost = sum(r.cost for r in results)
            print(f"  {model:15} | {len(results):2} exps | {avg_imp:.3f} avg imp | ${total_cost:.2f}")
        
        # By dataset
        print("\nBy Dataset:")
        by_dataset = {}
        for result in self.results:
            if result.dataset_name not in by_dataset:
                by_dataset[result.dataset_name] = []
            by_dataset[result.dataset_name].append(result)
        
        for dataset, results in by_dataset.items():
            avg_imp = sum(r.improvement for r in results) / len(results)
            total_cost = sum(r.cost for r in results)
            print(f"  {dataset:15} | {len(results):2} exps | {avg_imp:.3f} avg imp | ${total_cost:.2f}")

def main():
    """Test the result aggregator."""
    aggregator = ResultAggregator()
    
    # Load results
    count = aggregator.load_results()
    print(f"Loaded {count} experiment results")
    
    if count > 0:
        # Create DataFrame
        df = aggregator.create_dataframe()
        print(f"Created DataFrame with {len(df)} rows")
        
        # Generate report
        report = aggregator.generate_report()
        print("Generated analysis report")
        
        # Print summary
        aggregator.print_summary()
        
        # Save report
        output_path = aggregator.save_report()
        print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()
