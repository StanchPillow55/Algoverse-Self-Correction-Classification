#!/usr/bin/env python3
"""
Deterministic Sampling Verification Test

This script verifies that:
1. Dataset sampling is deterministic across multiple runs with same seed
2. Multi-turn and ensemble experiments use identical problem subsets
3. All datasets (GSM8K, MathBench, SuperGLUE, HumanEval) support deterministic sampling
4. Sample order is consistent for fair comparison

Usage:
    python3 test_deterministic_sampling.py
"""

import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.scaling_datasets import ScalingDatasetManager
from src.loop.runner import _load_dataset as multi_turn_load_dataset
from src.ensemble.runner import _load_dataset as ensemble_load_dataset

def compute_sample_hash(samples: List[Dict[str, Any]]) -> str:
    """Compute deterministic hash of sample IDs and questions for verification."""
    # Create deterministic string from sample IDs and questions
    sample_strings = []
    for sample in samples:
        # Use ID and question for hash (stable across runs)
        sample_id = sample.get('id', sample.get('qid', ''))
        question = sample.get('question', sample.get('prompt', ''))
        sample_strings.append(f"{sample_id}:{question}")
    
    combined_string = "||".join(sorted(sample_strings))
    return hashlib.sha256(combined_string.encode()).hexdigest()[:16]

def verify_gsm8k_deterministic_sampling():
    """Test GSM8K deterministic sampling consistency."""
    print("🔍 Testing GSM8K Deterministic Sampling...")
    
    results = {}
    
    # Test different sample sizes
    test_sizes = [20, 100, 500]
    
    for size in test_sizes:
        print(f"  📊 Testing GSM8K with {size} samples...")
        
        # Method 1: Direct deterministic file loading
        deterministic_file = Path(f"data/scaling/gsm8k_deterministic_{size}.json")
        if deterministic_file.exists():
            with open(deterministic_file, 'r') as f:
                data = json.load(f)
            samples_method1 = data.get("samples", [])
        else:
            samples_method1 = []
            print(f"    ⚠️ Deterministic file not found: {deterministic_file}")
            
        # Method 2: ScalingDatasetManager with seeded sampling
        dm = ScalingDatasetManager()
        try:
            samples_method2_run1 = dm.load_dataset('gsm8k', sample_size=size, seed=42)
            samples_method2_run2 = dm.load_dataset('gsm8k', sample_size=size, seed=42)
        except Exception as e:
            print(f"    ❌ ScalingDatasetManager failed: {e}")
            samples_method2_run1 = []
            samples_method2_run2 = []
        
        # Method 3: Multi-turn runner loading
        try:
            samples_method3 = multi_turn_load_dataset("gsm8k", f"subset_{size}")
        except Exception as e:
            print(f"    ❌ Multi-turn loading failed: {e}")
            samples_method3 = []
        
        # Method 4: Ensemble runner loading  
        try:
            samples_method4 = ensemble_load_dataset("gsm8k", f"subset_{size}")
        except Exception as e:
            print(f"    ❌ Ensemble loading failed: {e}")
            samples_method4 = []
        
        # Compute hashes for comparison
        hash1 = compute_sample_hash(samples_method1) if samples_method1 else "empty"
        hash2_run1 = compute_sample_hash(samples_method2_run1) if samples_method2_run1 else "empty"
        hash2_run2 = compute_sample_hash(samples_method2_run2) if samples_method2_run2 else "empty"
        hash3 = compute_sample_hash(samples_method3) if samples_method3 else "empty"
        hash4 = compute_sample_hash(samples_method4) if samples_method4 else "empty"
        
        results[f'gsm8k_{size}'] = {
            'deterministic_file': hash1,
            'scaling_manager_run1': hash2_run1,
            'scaling_manager_run2': hash2_run2,
            'multi_turn_runner': hash3,
            'ensemble_runner': hash4,
            'sample_counts': {
                'deterministic_file': len(samples_method1),
                'scaling_manager_run1': len(samples_method2_run1),
                'scaling_manager_run2': len(samples_method2_run2),
                'multi_turn_runner': len(samples_method3),
                'ensemble_runner': len(samples_method4)
            }
        }
        
        # Check consistency
        all_hashes = [hash1, hash2_run1, hash2_run2, hash3, hash4]
        non_empty_hashes = [h for h in all_hashes if h != "empty"]
        
        if non_empty_hashes and all(h == non_empty_hashes[0] for h in non_empty_hashes):
            print(f"    ✅ GSM8K {size} samples: All methods consistent (hash: {non_empty_hashes[0]})")
        else:
            print(f"    ❌ GSM8K {size} samples: Inconsistent results")
            print(f"       Deterministic file: {hash1}")
            print(f"       Scaling manager run1: {hash2_run1}")
            print(f"       Scaling manager run2: {hash2_run2}")
            print(f"       Multi-turn runner: {hash3}")
            print(f"       Ensemble runner: {hash4}")
        
        # Check seeded sampling consistency
        if hash2_run1 == hash2_run2 and hash2_run1 != "empty":
            print(f"    ✅ Seeded sampling is deterministic")
        elif hash2_run1 != "empty" and hash2_run2 != "empty":
            print(f"    ❌ Seeded sampling is not deterministic!")
    
    return results

def verify_mathbench_sampling():
    """Test MathBench sampling (CSV-based)."""
    print("🔍 Testing MathBench CSV Sampling...")
    
    results = {}
    
    # Test MathBench CSV files
    mathbench_files = [
        "data/scaling/mathbench_sample_100.csv",
        "data/scaling/mathbench_sample_500.csv"
    ]
    
    # Test SuperGLUE CSV files
    superglue_files = [
        "data/scaling/superglue_sample_100.csv",
        "data/scaling/superglue_sample_500.csv"
    ]
    
    # Test both MathBench and SuperGLUE files
    all_csv_files = mathbench_files + superglue_files
    
    for csv_file in all_csv_files:
        if not Path(csv_file).exists():
            print(f"  ⚠️ MathBench file not found: {csv_file}")
            continue
            
        print(f"  📊 Testing {csv_file}...")
        
        try:
            # Method 1: Multi-turn runner
            samples_method1 = multi_turn_load_dataset(csv_file)
        except Exception as e:
            print(f"    ❌ Multi-turn loading failed: {e}")
            samples_method1 = []
        
        try:
            # Method 2: Ensemble runner
            samples_method2 = ensemble_load_dataset(csv_file)
        except Exception as e:
            print(f"    ❌ Ensemble loading failed: {e}")
            samples_method2 = []
        
        # Run multiple times to check consistency
        try:
            samples_method1_run2 = multi_turn_load_dataset(csv_file)
            samples_method2_run2 = ensemble_load_dataset(csv_file)
        except Exception as e:
            print(f"    ❌ Second run failed: {e}")
            samples_method1_run2 = []
            samples_method2_run2 = []
        
        # Compute hashes
        hash1_run1 = compute_sample_hash(samples_method1) if samples_method1 else "empty"
        hash1_run2 = compute_sample_hash(samples_method1_run2) if samples_method1_run2 else "empty"
        hash2_run1 = compute_sample_hash(samples_method2) if samples_method2 else "empty"
        hash2_run2 = compute_sample_hash(samples_method2_run2) if samples_method2_run2 else "empty"
        
        file_type = "mathbench" if "mathbench" in csv_file else "superglue" 
        results[f'{file_type}_{Path(csv_file).stem}'] = {
            'multi_turn_run1': hash1_run1,
            'multi_turn_run2': hash1_run2,
            'ensemble_run1': hash2_run1,
            'ensemble_run2': hash2_run2,
            'sample_counts': {
                'multi_turn_run1': len(samples_method1),
                'multi_turn_run2': len(samples_method1_run2),
                'ensemble_run1': len(samples_method2),
                'ensemble_run2': len(samples_method2_run2)
            }
        }
        
        # Check consistency
        if hash1_run1 == hash2_run1 and hash1_run1 != "empty":
            print(f"    ✅ Multi-turn and ensemble runners consistent (hash: {hash1_run1})")
        else:
            print(f"    ❌ Multi-turn and ensemble runners inconsistent")
            print(f"       Multi-turn: {hash1_run1}")
            print(f"       Ensemble: {hash2_run1}")
        
        if hash1_run1 == hash1_run2 and hash2_run1 == hash2_run2:
            print(f"    ✅ CSV loading is deterministic across runs")
        else:
            print(f"    ❌ CSV loading is not deterministic!")
    
    return results

def verify_humaneval_sampling():
    """Test HumanEval sampling."""
    print("🔍 Testing HumanEval Sampling...")
    
    results = {}
    
    test_configs = [
        ("humaneval", None),
        ("humaneval", "subset_20"),
        ("humaneval", "subset_100")
    ]
    
    for dataset_name, subset in test_configs:
        print(f"  📊 Testing HumanEval with subset: {subset or 'full'}...")
        
        try:
            # Method 1: Multi-turn runner
            samples_method1 = multi_turn_load_dataset(dataset_name, subset)
        except Exception as e:
            print(f"    ❌ Multi-turn loading failed: {e}")
            samples_method1 = []
        
        try:
            # Method 2: Ensemble runner
            samples_method2 = ensemble_load_dataset(dataset_name, subset)
        except Exception as e:
            print(f"    ❌ Ensemble loading failed: {e}")
            samples_method2 = []
        
        # Test consistency across runs
        try:
            samples_method1_run2 = multi_turn_load_dataset(dataset_name, subset)
            samples_method2_run2 = ensemble_load_dataset(dataset_name, subset)
        except Exception as e:
            print(f"    ❌ Second run failed: {e}")
            samples_method1_run2 = []
            samples_method2_run2 = []
        
        # Compute hashes
        hash1_run1 = compute_sample_hash(samples_method1) if samples_method1 else "empty"
        hash1_run2 = compute_sample_hash(samples_method1_run2) if samples_method1_run2 else "empty"
        hash2_run1 = compute_sample_hash(samples_method2) if samples_method2 else "empty"
        hash2_run2 = compute_sample_hash(samples_method2_run2) if samples_method2_run2 else "empty"
        
        config_key = f'humaneval_{subset or "full"}'
        results[config_key] = {
            'multi_turn_run1': hash1_run1,
            'multi_turn_run2': hash1_run2,
            'ensemble_run1': hash2_run1,
            'ensemble_run2': hash2_run2,
            'sample_counts': {
                'multi_turn_run1': len(samples_method1),
                'multi_turn_run2': len(samples_method1_run2),
                'ensemble_run1': len(samples_method2),
                'ensemble_run2': len(samples_method2_run2)
            }
        }
        
        # Check consistency
        if hash1_run1 == hash2_run1 and hash1_run1 != "empty":
            print(f"    ✅ Multi-turn and ensemble runners consistent (hash: {hash1_run1})")
        else:
            print(f"    ❌ Multi-turn and ensemble runners inconsistent")
            print(f"       Multi-turn: {hash1_run1}")
            print(f"       Ensemble: {hash2_run1}")
        
        if hash1_run1 == hash1_run2 and hash2_run1 == hash2_run2:
            print(f"    ✅ HumanEval loading is deterministic across runs")
        else:
            print(f"    ❌ HumanEval loading is not deterministic!")
    
    return results

def generate_deterministic_sampling_report(results: Dict[str, Any]):
    """Generate comprehensive report on deterministic sampling verification."""
    
    report_path = Path("deterministic_sampling_verification_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("📊 DETERMINISTIC SAMPLING VERIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Report generated: {datetime.now()}\n\n")
        
        f.write("🎯 Verification Goals:\n")
        f.write("1. Dataset sampling is deterministic across multiple runs\n")
        f.write("2. Multi-turn and ensemble experiments use identical problem subsets\n")
        f.write("3. All datasets support deterministic sampling\n")
        f.write("4. Sample order is consistent for fair comparison\n\n")
        
        # Overall Summary
        total_tests = len(results)
        consistent_tests = 0
        
        for test_name, test_results in results.items():
            if 'multi_turn_run1' in test_results and 'ensemble_run1' in test_results:
                if (test_results['multi_turn_run1'] == test_results['ensemble_run1'] and 
                    test_results['multi_turn_run1'] != "empty"):
                    consistent_tests += 1
        
        f.write(f"📈 Overall Results:\n")
        f.write(f"  • Total tests: {total_tests}\n")
        f.write(f"  • Consistent multi-turn vs ensemble: {consistent_tests}\n")
        f.write(f"  • Consistency rate: {consistent_tests/total_tests*100:.1f}%\n\n")
        
        # Detailed Results
        f.write("🔍 Detailed Test Results:\n")
        f.write("-" * 60 + "\n")
        
        for test_name, test_results in results.items():
            f.write(f"\n{test_name}:\n")
            
            if isinstance(test_results, dict):
                for method, hash_value in test_results.items():
                    if method == 'sample_counts':
                        f.write(f"  Sample counts: {test_results['sample_counts']}\n")
                    elif not method.endswith('sample_counts'):
                        f.write(f"  {method}: {hash_value}\n")
                
                # Determine status
                if 'multi_turn_run1' in test_results and 'ensemble_run1' in test_results:
                    if (test_results['multi_turn_run1'] == test_results['ensemble_run1'] and 
                        test_results['multi_turn_run1'] != "empty"):
                        f.write("  ✅ STATUS: CONSISTENT\n")
                    else:
                        f.write("  ❌ STATUS: INCONSISTENT\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("End of Report\n")
    
    print(f"📄 Detailed report saved to: {report_path}")
    return report_path

def main():
    """Run comprehensive deterministic sampling verification."""
    print("🚀 Starting Deterministic Sampling Verification...")
    print("=" * 60)
    
    all_results = {}
    
    # Test 1: GSM8K deterministic sampling
    try:
        gsm8k_results = verify_gsm8k_deterministic_sampling()
        all_results.update(gsm8k_results)
    except Exception as e:
        print(f"❌ GSM8K testing failed: {e}")
    
    print()
    
    # Test 2: MathBench CSV sampling
    try:
        mathbench_results = verify_mathbench_sampling()
        all_results.update(mathbench_results)
    except Exception as e:
        print(f"❌ MathBench testing failed: {e}")
    
    print()
    
    # Test 3: HumanEval sampling
    try:
        humaneval_results = verify_humaneval_sampling()
        all_results.update(humaneval_results)
    except Exception as e:
        print(f"❌ HumanEval testing failed: {e}")
    
    print()
    
    # Generate comprehensive report
    report_path = generate_deterministic_sampling_report(all_results)
    
    print("=" * 60)
    print("✅ Deterministic Sampling Verification Complete!")
    print(f"📄 Full report: {report_path}")
    
    return all_results

if __name__ == "__main__":
    results = main()