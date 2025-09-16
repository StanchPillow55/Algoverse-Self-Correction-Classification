#!/usr/bin/env python3
"""
Comprehensive Analysis of Experimental Results

This script analyzes David's experimental results across:
- Multiple models: GPT-4o, GPT-4o-mini, Claude variants, Llama-2-70b, Llama-3-70b
- Multiple datasets: GSM8K, MathBench, SuperGLUE, ToolQA, HumanEval  
- Multiple sample sizes: 5, 10, 20, 100, 1000
- Multiple turn configurations: mt1 (baseline), mt2, mt3 (self-correction)

Analysis includes:
1. Data extraction and standardization
2. Scaling law fitting (power-law curves)
3. Statistical significance testing  
4. Cost-benefit analysis
5. Publication-ready visualizations
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExperimentalResultsAnalyzer:
    def __init__(self, runs_dir: str = "/Users/bradleyharaguchi/Algoverse-Self-Correction-Classification/runs"):
        self.runs_dir = Path(runs_dir)
        self.results_df = pd.DataFrame()
        self.branch_results_df = pd.DataFrame() 
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_experiment_info(self, run_name: str) -> Dict:
        """Extract experiment parameters from run directory name"""
        # Pattern: TIMESTAMP__DATASET__SPLIT__MODEL__SEED__TEMP__MAXTURNS
        pattern = r"(\d{8}T\d{6}Z)__([^_]+)(?:_(\w+))?__(\w+)__([^_]+)__seed(\d+)__t([\d.]+)__mt(\d+)"
        match = re.match(pattern, run_name)
        
        if not match:
            # Handle some edge cases in naming
            return {"valid": False, "run_name": run_name}
            
        timestamp, dataset, split, model, _, seed, temp, max_turns = match.groups()
        
        # Parse dataset and sample size
        dataset_parts = dataset.split('_')
        if 'sample' in dataset:
            if any(char.isdigit() for char in dataset):
                sample_size = int(''.join(filter(str.isdigit, dataset)))
            else:
                sample_size = None  # Will need to check traces.json
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
            "run_name": run_name
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
    
    def analyze_local_runs(self):
        """Analyze runs in local directory"""
        print("🔍 Analyzing local experimental runs...")
        results = []
        
        for run_dir in self.runs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith('.'):
                continue
                
            exp_info = self.extract_experiment_info(run_dir.name)
            if not exp_info["valid"]:
                continue
                
            metrics = self.load_metrics(run_dir)
            config = self.load_config(run_dir)
            
            if not metrics:
                continue
                
            # Extract key metrics
            result = {
                **exp_info,
                "accuracy_mean": metrics.get("accuracy_mean", 0),
                "accuracy_ci_lower": metrics.get("ci95", [0, 0])[0],
                "accuracy_ci_upper": metrics.get("ci95", [0, 0])[1], 
                "n_samples": metrics.get("n", 0),
                "config": config
            }
            
            results.append(result)
            
        self.results_df = pd.DataFrame(results)
        print(f"📊 Loaded {len(self.results_df)} local experimental runs")
        return self.results_df
    
    def analyze_branch_runs(self):
        """Analyze runs from David's branch using git"""
        print("🌿 Analyzing runs from David's Phase 3 branch...")
        
        # First, make sure we have the latest refs
        subprocess.run(['git', 'fetch', 'origin'], cwd=self.runs_dir.parent, capture_output=True)
        
        # Check available branches
        branches_result = subprocess.run(['git', 'branch', '-r'], 
                                       cwd=self.runs_dir.parent, capture_output=True, text=True)
        print(f"Available remote branches: {branches_result.stdout.strip()}")
        
        # Try multiple potential branch names for Phase 3
        potential_branches = [
            'origin/merge/phase2_new_src',
            'origin/phase3', 
            'origin/main',
            'origin/master'
        ]
        
        branch_runs = []
        
        for branch_name in potential_branches:
            print(f"\nTrying branch: {branch_name}")
            
            # List all files in the branch
            result = subprocess.run([
                "git", "ls-tree", "-r", "--name-only", branch_name
            ], cwd=self.runs_dir.parent, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Could not access branch {branch_name}: {result.stderr}")
                continue
                
            files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Look for runs directories and experiment files
            run_files = [f for f in files if 'runs/' in f and (f.endswith('.json') or f.endswith('.jsonl'))]
            config_files = [f for f in files if '/config' in f.lower() and (f.endswith('.json') or f.endswith('.jsonl'))]
            result_files = [f for f in files if any(keyword in f.lower() for keyword in ['result', 'metric', 'output']) and f.endswith('.json')]
            
            all_experiment_files = list(set(run_files + config_files + result_files))
            
            print(f"Found {len(all_experiment_files)} potential experiment files in {branch_name}")
            
            if len(all_experiment_files) > 0:
                print(f"Sample files: {all_experiment_files[:5]}")
                
                for file_path in all_experiment_files:
                    try:
                        # Extract file content from branch
                        cmd = ['git', 'show', f'{branch_name}:{file_path}']
                        file_result = subprocess.run(cmd, cwd=self.runs_dir.parent, capture_output=True, text=True)
                        
                        if file_result.returncode == 0 and file_result.stdout.strip():
                            content = file_result.stdout.strip()
                            
                            # Try to parse the content
                            try:
                                if file_path.endswith('.jsonl'):
                                    # Handle JSONL files
                                    for line_num, line in enumerate(content.split('\n')):
                                        if line.strip():
                                            try:
                                                data = json.loads(line)
                                                run_info = self.parse_branch_data(data, branch_name, file_path, line_num)
                                                if run_info:
                                                    branch_runs.append(run_info)
                                            except json.JSONDecodeError:
                                                continue
                                else:
                                    # Handle single JSON files
                                    data = json.loads(content)
                                    run_info = self.parse_branch_data(data, branch_name, file_path)
                                    if run_info:
                                        branch_runs.append(run_info)
                                        
                            except json.JSONDecodeError:
                                # Skip non-JSON files
                                continue
                                
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
            
            if branch_runs:
                print(f"Successfully extracted {len(branch_runs)} runs from {branch_name}")
                break  # Stop trying other branches if we found data
        
        self.branch_results_df = pd.DataFrame(branch_runs)
        print(f"📊 Total loaded {len(self.branch_results_df)} runs from Phase 3 branch")
        return self.branch_results_df
    
    def parse_branch_data(self, data, branch_name, file_path, line_num=None):
        """Parse data from branch files to extract run information"""
        try:
            # Try to extract experiment info from the data or file path
            run_info = {
                'source': f'branch:{branch_name}',
                'file_path': file_path,
                'model': None,
                'dataset': None,
                'max_turns': None,
                'n_samples': None,
                'accuracy_mean': None,
                'accuracy_ci_lower': None,
                'accuracy_ci_upper': None,
                'config': data,
                'valid': False
            }
            
            # Extract from data structure
            if isinstance(data, dict):
                # Look for common keys in experiment configs/results
                run_info['model'] = data.get('model', data.get('model_name', data.get('engine')))
                run_info['dataset'] = data.get('dataset', data.get('dataset_name', data.get('task')))
                run_info['max_turns'] = data.get('max_turns', data.get('turns', data.get('num_turns')))
                run_info['n_samples'] = data.get('n_samples', data.get('num_samples', data.get('n')))
                
                # Look for accuracy metrics
                if 'accuracy_mean' in data:
                    run_info['accuracy_mean'] = data['accuracy_mean']
                elif 'accuracy' in data:
                    run_info['accuracy_mean'] = data['accuracy']
                elif 'acc' in data:
                    run_info['accuracy_mean'] = data['acc']
                    
                # Look for confidence intervals
                if 'ci95' in data and isinstance(data['ci95'], list) and len(data['ci95']) >= 2:
                    run_info['accuracy_ci_lower'] = data['ci95'][0]
                    run_info['accuracy_ci_upper'] = data['ci95'][1]
            
            # Try to extract from file path if data doesn't have info
            if not run_info['model'] or not run_info['dataset']:
                # Parse filename for experiment details
                path_parts = file_path.lower().split('/')
                filename = path_parts[-1]
                
                # Look for model names
                model_keywords = ['gpt', 'claude', 'llama', 'sonnet', 'haiku', 'opus']
                for keyword in model_keywords:
                    if keyword in filename:
                        if not run_info['model']:
                            run_info['model'] = keyword
                
                # Look for dataset names
                dataset_keywords = ['gsm8k', 'mathbench', 'superglue', 'humaneval', 'toolqa']
                for keyword in dataset_keywords:
                    if keyword in filename:
                        if not run_info['dataset']:
                            run_info['dataset'] = keyword
                            
                # Look for phase information
                phase_keywords = ['phase1', 'phase2', 'phase3']
                for keyword in phase_keywords:
                    if keyword in filename:
                        # Infer max_turns from phase
                        if keyword == 'phase1' and not run_info['max_turns']:
                            run_info['max_turns'] = 1
                        elif keyword == 'phase2' and not run_info['max_turns']:
                            run_info['max_turns'] = 2
                        elif keyword == 'phase3' and not run_info['max_turns']:
                            run_info['max_turns'] = 3
            
            # Mark as valid if we have minimum required info
            if run_info['model'] and run_info['dataset']:
                run_info['valid'] = True
                
                # Set defaults
                if run_info['max_turns'] is None:
                    run_info['max_turns'] = 1
                if run_info['n_samples'] is None:
                    run_info['n_samples'] = 0
                if run_info['accuracy_mean'] is None:
                    run_info['accuracy_mean'] = 0
                if run_info['accuracy_ci_lower'] is None:
                    run_info['accuracy_ci_lower'] = 0
                if run_info['accuracy_ci_upper'] is None:
                    run_info['accuracy_ci_upper'] = 0
            
            return run_info if run_info['valid'] else None
            
        except Exception as e:
            print(f"Error parsing branch data: {e}")
            return None
    
    def combine_and_standardize_data(self):
        """Combine local and branch data, standardize format"""
        print("🔄 Combining and standardizing all experimental data...")
        
        # Mark local data source
        if not self.results_df.empty:
            self.results_df["source"] = "local"
            
        # Combine datasets
        if not self.branch_results_df.empty:
            all_results = pd.concat([self.results_df, self.branch_results_df], ignore_index=True)
        else:
            all_results = self.results_df.copy()
            
        if all_results.empty:
            print("❌ No experimental data found!")
            return pd.DataFrame()
        
        # Extract actual model names from config column
        def extract_actual_model(row):
            config_str = str(row['config'])
            try:
                # Try to parse the config string as a dict
                if "'model':" in config_str:
                    # Extract model name using regex
                    import re
                    model_match = re.search(r"'model':\s*'([^']+)'", config_str)
                    if model_match:
                        return model_match.group(1)
                elif '"model":' in config_str:
                    model_match = re.search(r'"model":\s*"([^"]+)"', config_str)
                    if model_match:
                        return model_match.group(1)
                        
                # Fall back to the original model column
                return row['model']
            except:
                return row['model']
        
        all_results['actual_model'] = all_results.apply(extract_actual_model, axis=1)
        
        # Standardize model names
        model_mapping = {
            'gpt-4o-mini': 'GPT-4o-mini',
            'gpt-4o': 'GPT-4o', 
            'gpt-4': 'GPT-4',
            'claude-haiku': 'Claude-3-Haiku',
            'claude-3-haiku-20240307': 'Claude-3-Haiku',
            'claude-3-5-sonnet-20241022': 'Claude-3-Sonnet',
            'claude-3-5-sonnet-20241022': 'Claude-3.5-Sonnet',
            'claude-3-opus-20240229': 'Claude-3-Opus',
            'meta-llama-2-70b-chat': 'Llama-2-70B',
            'meta/llama-2-70b-chat': 'Llama-2-70B',
            'meta-meta-llama-3-70b': 'Llama-3-70B',
            'meta-llama-llama-2-70b-chat': 'Llama-2-70B',
            'demo': 'Demo Model'
        }
        
        all_results['model_clean'] = all_results['actual_model'].map(model_mapping).fillna(all_results['actual_model'])
        
        # Clean up temporary dataset names first
        def clean_dataset_name(dataset_name):
            dataset_str = str(dataset_name).lower()
            
            # Remove temporary prefixes
            if dataset_str.startswith('tmp') and len(dataset_str) > 10:
                return 'temp_test'  # Mark as temporary test
            
            # Clean up common patterns
            dataset_str = dataset_str.replace('_sample', '').replace('_1k', '').replace('_100', '').replace('_20', '').replace('_10', '').replace('_5', '')
            
            return dataset_str
        
        all_results['dataset_clean_temp'] = all_results['dataset'].apply(clean_dataset_name)
        
        # Standardize dataset names
        dataset_mapping = {
            'gsm8k': 'GSM8K',
            'mathbench': 'MathBench', 
            'superglue': 'SuperGLUE',
            'toolqa': 'ToolQA',
            'humaneval': 'HumanEval',
            'math': 'MATH',
            'math20': 'MATH-20',
            'gsm8k16': 'GSM8K-16',
            'test': 'GSM8K-Test',  # Based on context, test files seem to be GSM8K variants
            'temp_test': 'Temporary Test'
        }
        
        all_results['dataset_clean'] = all_results['dataset_clean_temp'].map(dataset_mapping).fillna(all_results['dataset_clean_temp'])
        
        # Create experiment condition labels
        all_results['condition'] = all_results['max_turns'].map({
            1: 'Baseline',
            2: 'Self-Correction (2-turn)',
            3: 'Self-Correction (3-turn)'
        })
        
        # Filter out invalid results and demo/temporary runs
        all_results = all_results[
            (all_results['accuracy_mean'] >= 0) & 
            (all_results['accuracy_mean'] <= 1) &
            (all_results['n_samples'] > 0) &
            (all_results['model_clean'] != 'Demo Model') &  # Remove demo runs
            (all_results['dataset_clean'] != 'Temporary Test')  # Remove temporary test runs
        ].copy()
        
        self.standardized_df = all_results
        
        # Save standardized data
        output_file = self.output_dir / "standardized_results.csv"
        all_results.to_csv(output_file, index=False)
        print(f"💾 Saved standardized results to {output_file}")
        
        return all_results
    
    def generate_summary_statistics(self):
        """Generate summary statistics and data overview"""
        if self.standardized_df.empty:
            return
            
        print("📈 Generating summary statistics...")
        
        summary_stats = {
            'total_experiments': int(len(self.standardized_df)),
            'unique_models': int(self.standardized_df['model_clean'].nunique()),
            'unique_datasets': int(self.standardized_df['dataset_clean'].nunique()),
            'sample_sizes': [int(x) for x in sorted(self.standardized_df['n_samples'].dropna().unique())],
            'max_turns': [int(x) for x in sorted(self.standardized_df['max_turns'].unique())],
            'models_tested': list(self.standardized_df['model_clean'].unique()),
            'datasets_tested': list(self.standardized_df['dataset_clean'].unique())
        }
        
        # Print overview
        print(f"  📊 Total experiments: {summary_stats['total_experiments']}")
        print(f"  🤖 Models tested: {summary_stats['unique_models']} ({', '.join(summary_stats['models_tested'])})")
        print(f"  📚 Datasets tested: {summary_stats['unique_datasets']} ({', '.join(summary_stats['datasets_tested'])})")
        print(f"  📈 Sample sizes: {summary_stats['sample_sizes']}")
        print(f"  🔄 Max turns: {summary_stats['max_turns']}")
        
        # Save summary
        with open(self.output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
            
        return summary_stats
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting comprehensive experimental results analysis...")
        
        # Step 1: Load all data
        self.analyze_local_runs()
        self.analyze_branch_runs() 
        
        # Step 2: Combine and standardize
        standardized_data = self.combine_and_standardize_data()
        
        if standardized_data.empty:
            print("❌ No data to analyze!")
            return None
            
        # Step 3: Generate summary statistics
        self.generate_summary_statistics()
        
        print("✅ Data extraction and standardization complete!")
        print(f"📁 Results saved to {self.output_dir}/")
        
        return standardized_data

if __name__ == "__main__":
    analyzer = ExperimentalResultsAnalyzer()
    results = analyzer.run_full_analysis()
    
    if results is not None:
        print(f"\n📋 Analysis Summary:")
        print(f"  • {len(results)} total experimental runs analyzed")
        print(f"  • Data standardized and saved to analysis_output/")
        print(f"  • Ready for scaling analysis and visualization")
    else:
        print("❌ Analysis failed - no valid experimental data found")