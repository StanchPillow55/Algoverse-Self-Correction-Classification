"""
Power-law analysis for scaling study results.

This module implements power-law fitting and scaling analysis for the
Teacher-Learner RTS experiments across different model sizes and datasets.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

class PowerLawAnalyzer:
    """Analyzes power-law relationships in scaling study results."""
    
    def __init__(self, results_dir: str = "outputs"):
        """Initialize the power-law analyzer."""
        self.results_dir = Path(results_dir)
        self.results = {}
        
    def load_experiment_results(self, experiment_dirs: List[str]) -> Dict:
        """Load results from multiple experiment directories."""
        for exp_dir in experiment_dirs:
            exp_path = self.results_dir / exp_dir
            if exp_path.exists():
                self._load_single_experiment(exp_path)
        
        return self.results
    
    def _load_single_experiment(self, exp_path: Path):
        """Load results from a single experiment directory."""
        exp_name = exp_path.name
        
        # Load experiment summary
        summary_file = exp_path / "experiment_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                self.results[exp_name] = {
                    'summary': summary,
                    'models': {}
                }
        
        # Load individual model results
        for model_file in exp_path.glob("*_scaling_result.json"):
            model_name = model_file.stem.replace("_scaling_result", "")
            with open(model_file, 'r') as f:
                model_data = json.load(f)
                if exp_name in self.results:
                    self.results[exp_name]['models'][model_name] = model_data
    
    def extract_scaling_data(self) -> pd.DataFrame:
        """Extract scaling data for power-law analysis."""
        scaling_data = []
        
        for exp_name, exp_data in self.results.items():
            for model_name, model_data in exp_data.get('models', {}).items():
                # Extract model size (approximate based on model name)
                model_size = self._get_model_size(model_name)
                
                # Extract accuracy metrics
                summary = model_data.get('summary', {})
                accuracy = summary.get('final_accuracy_mean', 0.0)
                items = summary.get('items', 0)
                
                scaling_data.append({
                    'experiment': exp_name,
                    'model': model_name,
                    'model_size': model_size,
                    'accuracy': accuracy,
                    'samples': items,
                    'log_model_size': np.log10(model_size) if model_size > 0 else 0,
                    'log_accuracy': np.log10(accuracy) if accuracy > 0 else -10
                })
        
        return pd.DataFrame(scaling_data)
    
    def _get_model_size(self, model_name: str) -> int:
        """Get approximate model size in parameters."""
        size_mapping = {
            'gpt-4o-mini': 1e9,  # ~1B parameters
            'gpt-4o': 1.8e12,    # ~1.8T parameters
            'claude-haiku': 1e9,  # ~1B parameters
            'claude-sonnet': 1e12, # ~1T parameters
            'llama-7b': 7e9,     # 7B parameters
            'llama-13b': 13e9,   # 13B parameters
            'llama-70b': 70e9,   # 70B parameters
        }
        return size_mapping.get(model_name, 1e9)  # Default to 1B
    
    def fit_power_law(self, df: pd.DataFrame, x_col: str = 'model_size', 
                     y_col: str = 'accuracy') -> Dict:
        """Fit power-law relationship: y = a * x^b."""
        # Filter out zero values
        valid_data = df[(df[x_col] > 0) & (df[y_col] > 0)].copy()
        
        if len(valid_data) < 3:
            return {'error': 'Insufficient data for power-law fitting'}
        
        x = valid_data[x_col].values
        y = valid_data[y_col].values
        
        # Power-law function: y = a * x^b
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        try:
            # Fit power law
            popt, pcov = curve_fit(power_law, x, y, maxfev=10000)
            a, b = popt
            
            # Calculate R-squared
            y_pred = power_law(x, a, b)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate standard errors
            param_errors = np.sqrt(np.diag(pcov))
            
            return {
                'a': a,
                'b': b,
                'r_squared': r_squared,
                'a_error': param_errors[0],
                'b_error': param_errors[1],
                'n_points': len(valid_data),
                'equation': f"y = {a:.2e} * x^{b:.3f}",
                'valid': r_squared > 0.85  # Threshold for valid power law
            }
        except Exception as e:
            return {'error': f'Power-law fitting failed: {str(e)}'}
    
    def analyze_scaling_laws(self) -> Dict:
        """Analyze scaling laws across different experiments and models."""
        df = self.extract_scaling_data()
        
        if df.empty:
            return {'error': 'No scaling data found'}
        
        results = {
            'overall_scaling': self.fit_power_law(df),
            'by_experiment': {},
            'by_model_family': {},
            'summary_statistics': self._calculate_summary_stats(df)
        }
        
        # Analyze by experiment
        for exp in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp]
            results['by_experiment'][exp] = self.fit_power_law(exp_df)
        
        # Analyze by model family
        df['model_family'] = df['model'].str.split('-').str[0]
        for family in df['model_family'].unique():
            family_df = df[df['model_family'] == family]
            results['by_model_family'][family] = self.fit_power_law(family_df)
        
        return results
    
    def _calculate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the scaling data."""
        return {
            'total_experiments': df['experiment'].nunique(),
            'total_models': df['model'].nunique(),
            'total_samples': df['samples'].sum(),
            'accuracy_range': [df['accuracy'].min(), df['accuracy'].max()],
            'model_size_range': [df['model_size'].min(), df['model_size'].max()],
            'mean_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std()
        }
    
    def create_scaling_plots(self, output_dir: str = "outputs/scaling_analysis") -> List[str]:
        """Create visualization plots for scaling analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = self.extract_scaling_data()
        if df.empty:
            return []
        
        plot_files = []
        
        # 1. Overall scaling plot
        plt.figure(figsize=(10, 8))
        # Filter out zero values for log plot
        valid_data = df[(df['model_size'] > 0) & (df['accuracy'] > 0)]
        if not valid_data.empty:
            plt.loglog(valid_data['model_size'], valid_data['accuracy'], 'o', alpha=0.7, label='Data points')
        else:
            plt.scatter(df['model_size'], df['accuracy'], alpha=0.7, label='Data points')
        
        # Fit and plot power law
        power_law_result = self.fit_power_law(df)
        if 'error' not in power_law_result:
            x_fit = np.logspace(np.log10(df['model_size'].min()), 
                               np.log10(df['model_size'].max()), 100)
            y_fit = power_law_result['a'] * np.power(x_fit, power_law_result['b'])
            plt.loglog(x_fit, y_fit, 'r-', 
                      label=f"Power law: {power_law_result['equation']} (R²={power_law_result['r_squared']:.3f})")
        
        plt.xlabel('Model Size (Parameters)')
        plt.ylabel('Accuracy')
        plt.title('Scaling Law: Accuracy vs Model Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = output_path / "scaling_law_overall.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 2. By experiment comparison
        plt.figure(figsize=(12, 8))
        for exp in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp]
            valid_exp_data = exp_df[(exp_df['model_size'] > 0) & (exp_df['accuracy'] > 0)]
            if not valid_exp_data.empty:
                plt.loglog(valid_exp_data['model_size'], valid_exp_data['accuracy'], 'o', 
                          label=f'{exp}', alpha=0.7)
            else:
                plt.scatter(exp_df['model_size'], exp_df['accuracy'], 
                          label=f'{exp}', alpha=0.7)
        
        plt.xlabel('Model Size (Parameters)')
        plt.ylabel('Accuracy')
        plt.title('Scaling Laws by Experiment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = output_path / "scaling_law_by_experiment.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        # 3. Model family comparison
        plt.figure(figsize=(12, 8))
        df['model_family'] = df['model'].str.split('-').str[0]
        colors = plt.cm.Set1(np.linspace(0, 1, df['model_family'].nunique()))
        
        for i, family in enumerate(df['model_family'].unique()):
            family_df = df[df['model_family'] == family]
            valid_family_data = family_df[(family_df['model_size'] > 0) & (family_df['accuracy'] > 0)]
            if not valid_family_data.empty:
                plt.loglog(valid_family_data['model_size'], valid_family_data['accuracy'], 'o', 
                          color=colors[i], label=f'{family}', alpha=0.7)
            else:
                plt.scatter(family_df['model_size'], family_df['accuracy'], 
                          color=colors[i], label=f'{family}', alpha=0.7)
        
        plt.xlabel('Model Size (Parameters)')
        plt.ylabel('Accuracy')
        plt.title('Scaling Laws by Model Family')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = output_path / "scaling_law_by_model_family.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(plot_file))
        
        return plot_files
    
    def generate_scaling_report(self, output_file: str = "outputs/scaling_analysis/scaling_report.json") -> str:
        """Generate comprehensive scaling analysis report."""
        analysis_results = self.analyze_scaling_laws()
        plot_files = self.create_scaling_plots()
        
        report = {
            'analysis_results': analysis_results,
            'plot_files': plot_files,
            'timestamp': pd.Timestamp.now().isoformat(),
            'description': 'Power-law scaling analysis for Teacher-Learner RTS experiments'
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(output_path)

def main():
    """Main function to run power-law analysis."""
    analyzer = PowerLawAnalyzer()
    
    # Load experiment results
    experiment_dirs = [
        "phase1_validation",
        "phase2_medium_scale", 
        "phase3_full_scale_v2"
    ]
    
    print("Loading experiment results...")
    analyzer.load_experiment_results(experiment_dirs)
    
    print("Analyzing scaling laws...")
    report_file = analyzer.generate_scaling_report()
    
    print(f"Scaling analysis complete! Report saved to: {report_file}")
    
    # Print summary
    df = analyzer.extract_scaling_data()
    print(f"\nSummary:")
    print(f"Total experiments: {df['experiment'].nunique()}")
    print(f"Total models: {df['model'].nunique()}")
    print(f"Total samples: {df['samples'].sum()}")
    print(f"Accuracy range: {df['accuracy'].min():.3f} - {df['accuracy'].max():.3f}")

if __name__ == "__main__":
    main()
