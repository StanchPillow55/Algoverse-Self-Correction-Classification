#!/usr/bin/env python3
"""
Comprehensive Analysis of All Experimental Results

This script analyzes experimental results from multiple sources:
- CSV results directory (GPT-4o, GPT-4o-mini results in csv_results/)
- Traditional runs directory (metrics.json, config.json format)
- Legacy JSONL format files

Supports all models, datasets, and configurations with proper accuracy calculation.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ComprehensiveResultsAnalyzer:
    def __init__(self, 
                 runs_dir: str = "/Users/bradleyharaguchi/Algoverse-Self-Correction-Classification/runs",
                 csv_results_dir: str = "/Users/bradleyharaguchi/Algoverse-Self-Correction-Classification/csv_results"):
        self.runs_dir = Path(runs_dir)
        self.csv_results_dir = Path(csv_results_dir)
        self.results_df = pd.DataFrame()
        self.output_dir = Path("comprehensive_analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_csv_results(self):
        """Analyze results from CSV files in csv_results directory."""
        print("🔍 Analyzing CSV results directory...")
        results = []
        
        if not self.csv_results_dir.exists():
            print(f"❌ CSV results directory not found: {self.csv_results_dir}")
            return []
            
        # Find all summary CSV files
        summary_files = list(self.csv_results_dir.glob("*summary*.csv"))
        print(f"📊 Found {len(summary_files)} summary CSV files")
        
        for csv_file in summary_files:
            try:
                print(f"  Processing: {csv_file.name}")
                
                # Parse filename to extract experiment info
                exp_info = self.parse_csv_filename(csv_file.name)
                if not exp_info['valid']:
                    print(f"    ⚠️ Could not parse filename: {csv_file.name}")
                    continue
                
                # Load the CSV data
                df = pd.read_csv(csv_file)
                if df.empty:
                    print(f"    ⚠️ Empty CSV file: {csv_file.name}")
                    continue
                
                # Calculate metrics from the CSV data
                metrics = self.calculate_csv_metrics(df)
                
                # Combine experiment info with metrics
                result = {**exp_info, **metrics}
                results.append(result)
                
                print(f"    ✅ Loaded {len(df)} samples, accuracy: {metrics.get('accuracy_mean', 'N/A'):.3f}")
                
            except Exception as e:
                print(f"    ❌ Error processing {csv_file.name}: {e}")
                continue
        
        print(f"📊 Successfully loaded {len(results)} experiments from CSV results")
        return results
    
    def parse_csv_filename(self, filename: str) -> Dict:
        """Parse CSV filename to extract experiment parameters."""
        # Pattern: dataset_model_type_timestamp.csv
        # Example: mathbench_sample_500.csv_gpt-4o_summary_20250921_104217.csv
        
        filename_clean = filename.replace('.csv', '')
        parts = filename_clean.split('_')
        
        result = {'valid': False, 'filename': filename}
        
        # Try to extract model name
        model_keywords = ['gpt-4o-mini', 'gpt-4o', 'claude', 'llama']
        model = None
        for keyword in model_keywords:
            if keyword in filename_clean.lower():
                model = keyword
                break
        
        if not model:
            return result
            
        # Try to extract dataset name and sample size
        dataset = None
        sample_size = None
        
        # Look for dataset patterns
        if 'mathbench' in filename_clean.lower():
            dataset = 'mathbench'
        elif 'superglue' in filename_clean.lower():
            dataset = 'superglue'
        elif 'gsm8k' in filename_clean.lower():
            dataset = 'gsm8k'
        elif 'humaneval' in filename_clean.lower():
            dataset = 'humaneval'
        
        # Extract sample size
        size_match = re.search(r'(\d+)', filename_clean)
        if size_match:
            sample_size = int(size_match.group(1))
        
        # Extract timestamp if available
        timestamp_match = re.search(r'(\d{8}_\d{6})', filename_clean)
        timestamp = timestamp_match.group(1) if timestamp_match else 'unknown'
        
        return {
            'valid': True,
            'filename': filename,
            'dataset': dataset or 'unknown',
            'model': model,
            'sample_size': sample_size,
            'timestamp': timestamp,
            'max_turns': 3,  # Default based on current experiments
            'temperature': 0.2,  # Default based on current experiments
            'seed': 42,  # Default
            'source': 'csv_results'
        }
    
    def calculate_csv_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate accuracy metrics from CSV data."""
        metrics = {
            'accuracy_mean': 0,
            'accuracy_ci_lower': 0,
            'accuracy_ci_upper': 0,
            'n_samples': len(df)
        }
        
        try:
            # Try to find accuracy column
            accuracy_col = None
            for col in ['final_accuracy', 'accuracy', 'correct', 'final_acc']:
                if col in df.columns:
                    accuracy_col = col
                    break
            
            if accuracy_col is None:
                print(f"    ⚠️ No accuracy column found in CSV")
                return metrics
            
            # Calculate accuracy statistics
            accuracy_values = pd.to_numeric(df[accuracy_col], errors='coerce').dropna()
            
            if len(accuracy_values) == 0:
                print(f"    ⚠️ No valid accuracy values found")
                return metrics
                
            mean_acc = accuracy_values.mean()
            std_acc = accuracy_values.std()
            n = len(accuracy_values)
            
            # Calculate 95% confidence interval
            if n > 1:
                se = std_acc / np.sqrt(n)
                ci_margin = 1.96 * se
                ci_lower = max(0, mean_acc - ci_margin)
                ci_upper = min(1, mean_acc + ci_margin)
            else:
                ci_lower = mean_acc
                ci_upper = mean_acc
            
            metrics = {
                'accuracy_mean': float(mean_acc),
                'accuracy_ci_lower': float(ci_lower),
                'accuracy_ci_upper': float(ci_upper),
                'n_samples': int(n)
            }
            
            return metrics
            
        except Exception as e:
            print(f"    ❌ Error calculating metrics: {e}")
            return metrics
    
    def analyze_runs_directory(self):
        """Analyze traditional runs directory format."""
        print("🔍 Analyzing traditional runs directory...")
        results = []
        
        if not self.runs_dir.exists():
            print(f"❌ Runs directory not found: {self.runs_dir}")
            return []
        
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith('.'):
                continue
                
            try:
                # Extract experiment info from directory name
                exp_info = self.extract_experiment_info(run_dir.name)
                if not exp_info["valid"]:
                    continue
                    
                # Load metrics and config
                metrics = self.load_metrics(run_dir)
                config = self.load_config(run_dir)
                
                if not metrics:
                    continue
                    
                # Combine information
                result = {
                    **exp_info,
                    "accuracy_mean": metrics.get("accuracy_mean", 0),
                    "accuracy_ci_lower": metrics.get("ci95", [0, 0])[0],
                    "accuracy_ci_upper": metrics.get("ci95", [0, 0])[1], 
                    "n_samples": metrics.get("n", 0),
                    "config": config,
                    "source": "runs_directory"
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"  ❌ Error processing {run_dir.name}: {e}")
                continue
                
        print(f"📊 Loaded {len(results)} runs from runs directory")
        return results
    
    def extract_experiment_info(self, run_name: str) -> Dict:
        """Extract experiment parameters from run directory name"""
        # Pattern: TIMESTAMP__DATASET__SPLIT__MODEL__SEED__TEMP__MAXTURNS
        pattern = r"(\d{8}T\d{6}Z)__([^_]+)(?:_(\w+))?__(\w+)__([^_]+)__seed(\d+)__t([\d.]+)__mt(\d+)"
        match = re.match(pattern, run_name)
        
        if not match:
            return {"valid": False, "run_name": run_name}
            
        timestamp, dataset, split, model, _, seed, temp, max_turns = match.groups()
        
        # Parse dataset and sample size
        if 'sample' in dataset:
            if any(char.isdigit() for char in dataset):
                sample_size = int(''.join(filter(str.isdigit, dataset)))
            else:
                sample_size = None
        elif dataset.endswith(('1k', '1000')):
            sample_size = 1000
        elif dataset.endswith(('100')):
            sample_size = 100
        elif dataset.endswith(('20')):
            sample_size = 20
        elif dataset.endswith(('10')):
            sample_size = 10
        elif dataset.endswith(('5')):
            sample_size = 5
        else:
            sample_size = None
            
        # Clean dataset name
        base_dataset = dataset.replace('_sample', '').replace('_1k', '').replace('_100', '').replace('_20', '').replace('_10', '').replace('_5', '')
        
        return {
            "valid": True,
            "timestamp": timestamp,
            "dataset": base_dataset,
            "split": split or "dev",
            "model": model,
            "seed": int(seed),
            "temperature": float(temp),
            "max_turns": int(max_turns),
            "sample_size": sample_size,
            "run_name": run_name,
            "source": "runs_directory"
        }
    
    def load_metrics(self, run_dir: Path) -> Dict:
        """Load metrics.json for a run"""
        metrics_file = run_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
        return {}
    
    def load_config(self, run_dir: Path) -> Dict:
        """Load config.json for a run"""
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                return json.load(f)
        return {}
    
    def combine_and_standardize_data(self, csv_results: List[Dict], runs_results: List[Dict]):
        """Combine CSV and runs directory results, standardize format"""
        print("🔄 Combining and standardizing all experimental data...")
        
        all_results = csv_results + runs_results
        
        if not all_results:
            print("❌ No experimental data found!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        
        # Standardize model names
        model_mapping = {
            'gpt-4o-mini': 'GPT-4o-mini',
            'gpt-4o': 'GPT-4o', 
            'gpt-4': 'GPT-4',
            'claude-haiku': 'Claude-3-Haiku',
            'claude-3-haiku-20240307': 'Claude-3-Haiku',
            'claude-3-5-sonnet-20241210': 'Claude-3.5-Sonnet',
            'claude-3-opus-20240229': 'Claude-3-Opus',
            'meta-llama-2-70b-chat': 'Llama-2-70B',
            'meta/llama-2-70b-chat': 'Llama-2-70B',
        }
        
        df['model_clean'] = df['model'].map(model_mapping).fillna(df['model'])
        
        # Standardize dataset names
        dataset_mapping = {
            'gsm8k': 'GSM8K',
            'mathbench': 'MathBench', 
            'superglue': 'SuperGLUE',
            'toolqa': 'ToolQA',
            'humaneval': 'HumanEval'
        }
        
        df['dataset_clean'] = df['dataset'].map(dataset_mapping).fillna(df['dataset'])
        
        # Create experiment condition labels
        df['condition'] = df['max_turns'].map({
            1: 'Baseline',
            2: 'Self-Correction (2-turn)',
            3: 'Self-Correction (3-turn)'
        }).fillna('Multi-turn')
        
        # Filter out invalid results
        df = df[
            (df['accuracy_mean'] >= 0) & 
            (df['accuracy_mean'] <= 1) &
            (df['n_samples'] > 0)
        ].copy()
        
        # Sort by dataset, model, and sample size
        df = df.sort_values(['dataset_clean', 'model_clean', 'sample_size', 'max_turns'])
        
        self.standardized_df = df
        
        # Save standardized data
        output_file = self.output_dir / "comprehensive_standardized_results.csv"
        df.to_csv(output_file, index=False)
        print(f"💾 Saved standardized results to {output_file}")
        
        return df
    
    def generate_summary_statistics(self):
        """Generate summary statistics and data overview"""
        if self.standardized_df.empty:
            return
            
        print("📈 Generating comprehensive summary statistics...")
        
        # Group by dataset and model for summary
        summary_table = []
        
        for dataset in sorted(self.standardized_df['dataset_clean'].unique()):
            dataset_df = self.standardized_df[self.standardized_df['dataset_clean'] == dataset]
            
            for model in sorted(dataset_df['model_clean'].unique()):
                model_df = dataset_df[dataset_df['model_clean'] == model]
                
                for max_turns in sorted(model_df['max_turns'].unique()):
                    condition_df = model_df[model_df['max_turns'] == max_turns]
                    
                    if len(condition_df) > 0:
                        row = condition_df.iloc[-1]  # Take most recent run
                        summary_table.append({
                            'Dataset': dataset,
                            'Model': model,
                            'Max Turns': max_turns,
                            'Sample Size': row['sample_size'],
                            'Accuracy': f"{row['accuracy_mean']:.3f}",
                            'CI Lower': f"{row['accuracy_ci_lower']:.3f}",
                            'CI Upper': f"{row['accuracy_ci_upper']:.3f}",
                            'N Samples': row['n_samples'],
                            'Source': row['source']
                        })
        
        summary_df = pd.DataFrame(summary_table)
        
        # Save summary table
        summary_file = self.output_dir / "experiment_summary_table.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Print summary
        print(f"\n📋 COMPREHENSIVE EXPERIMENTAL RESULTS SUMMARY")
        print(f"=" * 60)
        print(f"Total experiments analyzed: {len(self.standardized_df)}")
        print(f"Unique models: {self.standardized_df['model_clean'].nunique()}")
        print(f"Unique datasets: {self.standardized_df['dataset_clean'].nunique()}")
        print(f"Data sources: {', '.join(self.standardized_df['source'].unique())}")
        print(f"\n📊 Results by Dataset and Model:")
        print(f"-" * 60)
        
        # Print detailed results
        for _, row in summary_df.iterrows():
            print(f"{row['Dataset']:12} | {row['Model']:12} | {row['Max Turns']}T | "
                  f"n={row['N Samples']:4} | Acc: {row['Accuracy']} "
                  f"[{row['CI Lower']}-{row['CI Upper']}] | {row['Source']}")
        
        print(f"\n💾 Detailed summary saved to: {summary_file}")
        return summary_df
    
    def create_analysis_dashboard(self):
        """Create comprehensive analysis dashboard"""
        dashboard_path = self.output_dir / "comprehensive_analysis_dashboard.txt"
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write("📊 COMPREHENSIVE EXPERIMENTAL RESULTS DASHBOARD\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📁 Analysis Output Directory: {self.output_dir}\n\n")
            
            f.write("📋 Generated Files:\n")
            f.write("  • comprehensive_standardized_results.csv - All experiments combined\n")
            f.write("  • experiment_summary_table.csv - Summary by dataset/model\n")
            f.write("  • comprehensive_analysis_dashboard.txt - This dashboard\n\n")
            
            f.write("🔍 Data Sources Analyzed:\n")
            f.write(f"  • CSV Results Directory: {self.csv_results_dir}\n")
            f.write(f"  • Traditional Runs Directory: {self.runs_dir}\n\n")
            
            if not self.standardized_df.empty:
                f.write("📈 Summary Statistics:\n")
                f.write(f"  • Total experiments: {len(self.standardized_df)}\n")
                f.write(f"  • Unique models: {self.standardized_df['model_clean'].nunique()}\n")
                f.write(f"  • Unique datasets: {self.standardized_df['dataset_clean'].nunique()}\n")
                f.write(f"  • Sample sizes: {sorted(self.standardized_df['sample_size'].dropna().unique())}\n")
                f.write(f"  • Max turns: {sorted(self.standardized_df['max_turns'].unique())}\n\n")
                
                f.write("🤖 Models Analyzed:\n")
                for model in sorted(self.standardized_df['model_clean'].unique()):
                    count = len(self.standardized_df[self.standardized_df['model_clean'] == model])
                    f.write(f"  • {model}: {count} experiments\n")
                f.write("\n")
                
                f.write("📚 Datasets Analyzed:\n")
                for dataset in sorted(self.standardized_df['dataset_clean'].unique()):
                    count = len(self.standardized_df[self.standardized_df['dataset_clean'] == dataset])
                    f.write(f"  • {dataset}: {count} experiments\n")
            
            f.write(f"\n" + "=" * 60)
            f.write(f"\nDashboard generated: {datetime.now()}")
        
        print(f"📄 Comprehensive dashboard created: {dashboard_path}")
        return dashboard_path
    
    def run_full_analysis(self):
        """Run complete comprehensive analysis pipeline"""
        print("🚀 Starting comprehensive experimental results analysis...")
        
        # Step 1: Analyze CSV results
        csv_results = self.analyze_csv_results()
        
        # Step 2: Analyze runs directory
        runs_results = self.analyze_runs_directory()
        
        # Step 3: Combine and standardize
        standardized_data = self.combine_and_standardize_data(csv_results, runs_results)
        
        if standardized_data.empty:
            print("❌ No data to analyze!")
            return None
            
        # Step 4: Generate summary statistics
        summary = self.generate_summary_statistics()
        
        # Step 5: Create dashboard
        dashboard = self.create_analysis_dashboard()
        
        print(f"\n✅ Comprehensive analysis complete!")
        print(f"📁 Results saved to {self.output_dir}/")
        print(f"📄 Dashboard: {dashboard}")
        
        return standardized_data, summary


if __name__ == "__main__":
    analyzer = ComprehensiveResultsAnalyzer()
    results, summary = analyzer.run_full_analysis()
    
    if results is not None:
        print(f"\n📋 Analysis Complete:")
        print(f"  • {len(results)} total experiments analyzed")
        print(f"  • Results include both CSV and runs directory formats")
        print(f"  • GPT-4o results now properly parsed and analyzed")
        print(f"  • Ready for further analysis and visualization")
    else:
        print("❌ Analysis failed - no valid experimental data found")