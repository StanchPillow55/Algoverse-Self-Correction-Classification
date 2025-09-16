"""
Cost and bias analysis for the Teacher-Learner RTS scaling study.

This module analyzes API costs, token usage, and cognitive bias patterns
across different models and experiments.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class CostBiasAnalyzer:
    """Analyzes costs and biases in scaling study results."""
    
    def __init__(self, results_dir: str = "outputs"):
        """Initialize the cost and bias analyzer."""
        self.results_dir = Path(results_dir)
        self.cost_data = []
        self.bias_data = []
        
    def load_experiment_data(self, experiment_dirs: List[str]) -> Dict:
        """Load experiment data for analysis."""
        for exp_dir in experiment_dirs:
            exp_path = self.results_dir / exp_dir
            if exp_path.exists():
                self._load_single_experiment(exp_path)
        
        return {
            'cost_data': self.cost_data,
            'bias_data': self.bias_data
        }
    
    def _load_single_experiment(self, exp_path: Path):
        """Load data from a single experiment."""
        exp_name = exp_path.name
        
        # Load traces for bias analysis
        for trace_file in exp_path.glob("**/traces.json"):
            with open(trace_file, 'r') as f:
                traces = json.load(f)
                if 'items' in traces:
                    for item in traces['items']:
                        self._extract_bias_data(item, exp_name)
        
        # Load cost data from debug files
        for debug_file in exp_path.glob("**/openai_debug.jsonl"):
            self._load_cost_data(debug_file, exp_name, "openai")
        
        for debug_file in exp_path.glob("**/anthropic_debug.jsonl"):
            self._load_cost_data(debug_file, exp_name, "anthropic")
    
    def _extract_bias_data(self, item: Dict, exp_name: str):
        """Extract bias data from trace item."""
        sample_id = item.get('id', 'unknown')
        turns = item.get('turns', [])
        
        for turn in turns:
            # Extract bias information if available
            bias_info = turn.get('evaluator_feedback', {})
            if isinstance(bias_info, dict):
                bias_label = bias_info.get('bias_label', 'None')
                confidence = turn.get('confidence', 0.0)
                
                self.bias_data.append({
                    'experiment': exp_name,
                    'sample_id': sample_id,
                    'turn': turn.get('turn_index', 0),
                    'bias_label': bias_label,
                    'confidence': confidence,
                    'is_correct': item.get('final', {}).get('correct', False)
                })
    
    def _load_cost_data(self, debug_file: Path, exp_name: str, provider: str):
        """Load cost data from debug files."""
        try:
            with open(debug_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Extract cost information if available
                        cost_info = {
                            'experiment': exp_name,
                            'provider': provider,
                            'model': data.get('model', 'unknown'),
                            'input_tokens': data.get('input_tokens', 0),
                            'output_tokens': data.get('output_tokens', 0),
                            'total_cost': data.get('total_cost', 0.0)
                        }
                        self.cost_data.append(cost_info)
        except Exception as e:
            print(f"Error loading cost data from {debug_file}: {e}")
    
    def analyze_costs(self) -> Dict:
        """Analyze API costs and token usage."""
        if not self.cost_data:
            return {'error': 'No cost data available'}
        
        df = pd.DataFrame(self.cost_data)
        
        # Calculate cost metrics
        total_cost = df['total_cost'].sum()
        total_tokens = (df['input_tokens'] + df['output_tokens']).sum()
        avg_cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
        
        # Cost by provider
        cost_by_provider = df.groupby('provider').agg({
            'total_cost': 'sum',
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        }).to_dict('index')
        
        # Cost by model
        cost_by_model = df.groupby('model').agg({
            'total_cost': 'sum',
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        }).to_dict('index')
        
        # Cost by experiment
        cost_by_experiment = df.groupby('experiment').agg({
            'total_cost': 'sum',
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        }).to_dict('index')
        
        return {
            'total_cost': total_cost,
            'total_tokens': total_tokens,
            'avg_cost_per_token': avg_cost_per_token,
            'cost_by_provider': cost_by_provider,
            'cost_by_model': cost_by_model,
            'cost_by_experiment': cost_by_experiment,
            'summary': {
                'total_api_calls': len(df),
                'unique_models': df['model'].nunique(),
                'unique_providers': df['provider'].nunique()
            }
        }
    
    def analyze_biases(self) -> Dict:
        """Analyze cognitive bias patterns."""
        if not self.bias_data:
            return {'error': 'No bias data available'}
        
        df = pd.DataFrame(self.bias_data)
        
        # Bias distribution
        bias_counts = df['bias_label'].value_counts().to_dict()
        bias_percentages = (df['bias_label'].value_counts(normalize=True) * 100).to_dict()
        
        # Bias by experiment
        bias_by_experiment = df.groupby(['experiment', 'bias_label']).size().unstack(fill_value=0).to_dict('index')
        
        # Bias by confidence level
        df['confidence_level'] = pd.cut(df['confidence'], 
                                      bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                      labels=['Low', 'Medium', 'High', 'Very High'])
        bias_by_confidence = df.groupby(['confidence_level', 'bias_label']).size().unstack(fill_value=0).to_dict('index')
        
        # Accuracy by bias type
        accuracy_by_bias = df.groupby('bias_label')['is_correct'].agg(['mean', 'count']).to_dict('index')
        
        # Turn analysis
        turn_analysis = df.groupby('turn').agg({
            'bias_label': lambda x: x.mode().iloc[0] if len(x) > 0 else 'None',
            'confidence': 'mean',
            'is_correct': 'mean'
        }).to_dict('index')
        
        return {
            'bias_distribution': bias_counts,
            'bias_percentages': bias_percentages,
            'bias_by_experiment': bias_by_experiment,
            'bias_by_confidence': bias_by_confidence,
            'accuracy_by_bias': accuracy_by_bias,
            'turn_analysis': turn_analysis,
            'summary': {
                'total_observations': len(df),
                'unique_biases': df['bias_label'].nunique(),
                'most_common_bias': df['bias_label'].mode().iloc[0] if len(df) > 0 else 'None'
            }
        }
    
    def create_cost_plots(self, output_dir: str = "outputs/cost_analysis") -> List[str]:
        """Create cost analysis plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.cost_data:
            return []
        
        df = pd.DataFrame(self.cost_data)
        plot_files = []
        
        # 1. Cost by provider
        plt.figure(figsize=(10, 6))
        cost_by_provider = df.groupby('provider')['total_cost'].sum()
        cost_by_provider.plot(kind='bar', color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Total Cost by Provider')
        plt.xlabel('Provider')
        plt.ylabel('Total Cost ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_file = output_path / "cost_by_provider.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 2. Token usage by model
        plt.figure(figsize=(12, 6))
        token_usage = df.groupby('model')[['input_tokens', 'output_tokens']].sum()
        token_usage.plot(kind='bar', stacked=True, color=['lightblue', 'darkblue'])
        plt.title('Token Usage by Model')
        plt.xlabel('Model')
        plt.ylabel('Tokens')
        plt.xticks(rotation=45)
        plt.legend(['Input Tokens', 'Output Tokens'])
        plt.tight_layout()
        
        plot_file = output_path / "token_usage_by_model.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 3. Cost efficiency (cost per accuracy)
        if self.bias_data:
            bias_df = pd.DataFrame(self.bias_data)
            accuracy_by_model = bias_df.groupby('experiment')['is_correct'].mean()
            cost_by_experiment = df.groupby('experiment')['total_cost'].sum()
            
            plt.figure(figsize=(10, 6))
            plt.scatter(cost_by_experiment.values, accuracy_by_model.values, s=100, alpha=0.7)
            for i, exp in enumerate(cost_by_experiment.index):
                plt.annotate(exp, (cost_by_experiment.iloc[i], accuracy_by_model.iloc[i]))
            plt.xlabel('Total Cost ($)')
            plt.ylabel('Accuracy')
            plt.title('Cost vs Accuracy')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_file = output_path / "cost_vs_accuracy.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files.append(str(plot_file))
        
        return plot_files
    
    def create_bias_plots(self, output_dir: str = "outputs/bias_analysis") -> List[str]:
        """Create bias analysis plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.bias_data:
            return []
        
        df = pd.DataFrame(self.bias_data)
        plot_files = []
        
        # 1. Bias distribution
        plt.figure(figsize=(10, 6))
        bias_counts = df['bias_label'].value_counts()
        bias_counts.plot(kind='bar', color='lightcoral')
        plt.title('Distribution of Cognitive Biases')
        plt.xlabel('Bias Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_file = output_path / "bias_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 2. Bias by confidence level
        plt.figure(figsize=(12, 6))
        df['confidence_level'] = pd.cut(df['confidence'], 
                                      bins=[0, 0.3, 0.6, 0.8, 1.0], 
                                      labels=['Low', 'Medium', 'High', 'Very High'])
        bias_by_conf = pd.crosstab(df['confidence_level'], df['bias_label'])
        bias_by_conf.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Bias Distribution by Confidence Level')
        plt.xlabel('Confidence Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Bias Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plot_file = output_path / "bias_by_confidence.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 3. Accuracy by bias type
        plt.figure(figsize=(10, 6))
        accuracy_by_bias = df.groupby('bias_label')['is_correct'].mean()
        accuracy_by_bias.plot(kind='bar', color='lightgreen')
        plt.title('Accuracy by Bias Type')
        plt.xlabel('Bias Type')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_file = output_path / "accuracy_by_bias.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        return plot_files
    
    def generate_analysis_report(self, output_file: str = "outputs/cost_bias_analysis/report.json") -> str:
        """Generate comprehensive cost and bias analysis report."""
        cost_analysis = self.analyze_costs()
        bias_analysis = self.analyze_biases()
        
        cost_plots = self.create_cost_plots()
        bias_plots = self.create_bias_plots()
        
        report = {
            'cost_analysis': cost_analysis,
            'bias_analysis': bias_analysis,
            'cost_plots': cost_plots,
            'bias_plots': bias_plots,
            'timestamp': pd.Timestamp.now().isoformat(),
            'description': 'Cost and bias analysis for Teacher-Learner RTS experiments'
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(output_path)

def main():
    """Main function to run cost and bias analysis."""
    analyzer = CostBiasAnalyzer()
    
    # Load experiment data
    experiment_dirs = [
        "phase1_validation",
        "phase2_medium_scale", 
        "phase3_full_scale_v2"
    ]
    
    print("Loading experiment data...")
    analyzer.load_experiment_data(experiment_dirs)
    
    print("Analyzing costs and biases...")
    report_file = analyzer.generate_analysis_report()
    
    print(f"Cost and bias analysis complete! Report saved to: {report_file}")
    
    # Print summary
    cost_analysis = analyzer.analyze_costs()
    bias_analysis = analyzer.analyze_biases()
    
    if 'error' not in cost_analysis:
        print(f"\nCost Summary:")
        print(f"Total cost: ${cost_analysis['total_cost']:.4f}")
        print(f"Total tokens: {cost_analysis['total_tokens']:,}")
        print(f"Avg cost per token: ${cost_analysis['avg_cost_per_token']:.6f}")
    
    if 'error' not in bias_analysis:
        print(f"\nBias Summary:")
        print(f"Total observations: {bias_analysis['summary']['total_observations']}")
        print(f"Most common bias: {bias_analysis['summary']['most_common_bias']}")
        print(f"Bias distribution: {bias_analysis['bias_distribution']}")

if __name__ == "__main__":
    main()

