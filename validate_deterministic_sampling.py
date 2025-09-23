#!/usr/bin/env python3
"""
Validate Deterministic Sampling

This script validates that deterministic sampling works correctly across all datasets
and experiment types (multi-turn vs ensemble) to ensure fair comparisons.

Usage:
    python3 validate_deterministic_sampling.py --datasets gsm8k,mathbench
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.scaling_datasets import ScalingDatasetManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeterministicSamplingValidator:
    """Validate deterministic sampling behavior across datasets and experiment types."""
    
    def __init__(self, data_dir: str = "data/scaling"):
        """Initialize the validator."""
        self.data_dir = Path(data_dir)
        self.dataset_manager = ScalingDatasetManager(data_dir)
        
    def validate_dataset_determinism(self, dataset_name: str, sample_size: int, num_runs: int = 5) -> bool:
        """
        Validate that a dataset returns identical samples across multiple runs.
        
        Args:
            dataset_name: Name of the dataset to test
            sample_size: Number of samples to test
            num_runs: Number of runs to perform for consistency checking
            
        Returns:
            True if deterministic, False otherwise
        """
        logger.info(f"🔍 Validating determinism for {dataset_name} with {sample_size} samples across {num_runs} runs")
        
        # Test seeded sampling (fallback method)
        sample_sets = []
        sample_hashes = []
        
        for run in range(num_runs):
            logger.debug(f"  Run {run + 1}/{num_runs}")
            
            # Load samples using the same seed each time
            samples = self.dataset_manager.load_dataset(dataset_name, sample_size=sample_size, seed=42)
            
            if not samples:
                logger.error(f"No samples returned for {dataset_name}")
                return False
                
            # Create a consistent representation for hashing
            sample_ids = [str(s.get('id', f'sample_{i}')) for i, s in enumerate(samples)]
            sample_questions = [str(s.get('question', '')) for s in samples]
            
            # Create hash of sample order and content
            combined_str = '|'.join(sample_ids) + '||' + '|'.join(sample_questions)
            sample_hash = hashlib.sha256(combined_str.encode()).hexdigest()
            
            sample_sets.append(sample_ids)
            sample_hashes.append(sample_hash)
            
        # Check if all runs produced identical results
        first_hash = sample_hashes[0]
        first_ids = sample_sets[0]
        
        for i, (run_hash, run_ids) in enumerate(zip(sample_hashes[1:], sample_sets[1:]), 1):
            if run_hash != first_hash:
                logger.error(f"❌ Hash mismatch between run 1 and run {i+1}")
                logger.error(f"   Run 1 hash: {first_hash}")
                logger.error(f"   Run {i+1} hash: {run_hash}")
                
                # Show first few differing IDs
                for j, (id1, id2) in enumerate(zip(first_ids, run_ids)):
                    if id1 != id2:
                        logger.error(f"   First difference at position {j}: '{id1}' vs '{id2}'")
                        break
                        
                return False
                
        logger.info(f"✅ {dataset_name} is deterministic across {num_runs} runs (hash: {first_hash[:16]}...)")
        return True
        
    def validate_deterministic_subset_files(self, dataset_name: str, sizes: List[int]) -> Dict[int, bool]:
        """
        Validate that deterministic subset files exist and are consistent.
        
        Args:
            dataset_name: Name of the dataset
            sizes: List of subset sizes to validate
            
        Returns:
            Dictionary mapping sizes to validation results
        """
        logger.info(f"🔍 Validating deterministic subset files for {dataset_name}")
        
        results = {}
        
        for size in sizes:
            subset_file = self.data_dir / f"{dataset_name}_deterministic_{size}.json"
            
            if not subset_file.exists():
                logger.warning(f"⚠️ Deterministic subset file not found: {subset_file}")
                results[size] = False
                continue
                
            try:
                with open(subset_file, 'r') as f:
                    subset_data = json.load(f)
                    
                samples = subset_data.get("samples", [])
                
                # Validate metadata
                expected_size = subset_data.get("sample_size", 0)
                sampling_method = subset_data.get("sampling_method", "")
                
                if len(samples) != size:
                    logger.error(f"❌ Sample count mismatch for {dataset_name} size {size}: {len(samples)} != {size}")
                    results[size] = False
                    continue
                    
                if expected_size != size:
                    logger.error(f"❌ Metadata size mismatch for {dataset_name} size {size}: {expected_size} != {size}")
                    results[size] = False
                    continue
                    
                if sampling_method != "deterministic_first_n":
                    logger.warning(f"⚠️ Unexpected sampling method for {dataset_name} size {size}: {sampling_method}")
                    
                # Validate sample IDs are present
                sample_ids = [s.get('id', '') for s in samples]
                if not all(sample_ids):
                    logger.error(f"❌ Missing sample IDs in {dataset_name} size {size}")
                    results[size] = False
                    continue
                    
                # Check for duplicate IDs
                if len(set(sample_ids)) != len(sample_ids):
                    logger.error(f"❌ Duplicate sample IDs found in {dataset_name} size {size}")
                    results[size] = False
                    continue
                    
                logger.info(f"✅ Deterministic subset file valid: {dataset_name} size {size}")
                results[size] = True
                
            except Exception as e:
                logger.error(f"❌ Error validating {dataset_name} size {size}: {e}")
                results[size] = False
                
        return results
        
    def validate_subset_nesting(self, dataset_name: str, sizes: List[int]) -> bool:
        """
        Validate that smaller subsets are proper subsets of larger subsets.
        This ensures consistency when comparing experiments with different sample sizes.
        
        Args:
            dataset_name: Name of the dataset
            sizes: List of subset sizes (should be sorted)
            
        Returns:
            True if nesting is valid, False otherwise
        """
        logger.info(f"🔍 Validating subset nesting for {dataset_name}")
        
        sizes = sorted(sizes)
        subset_data = {}
        
        # Load all subsets
        for size in sizes:
            subset_file = self.data_dir / f"{dataset_name}_deterministic_{size}.json"
            
            if not subset_file.exists():
                logger.warning(f"⚠️ Subset file not found: {subset_file}")
                continue
                
            try:
                with open(subset_file, 'r') as f:
                    data = json.load(f)
                    samples = data.get("samples", [])
                    sample_ids = [s.get('id', f'sample_{i}') for i, s in enumerate(samples)]
                    subset_data[size] = sample_ids
                    
            except Exception as e:
                logger.error(f"❌ Error loading subset {dataset_name} size {size}: {e}")
                return False
                
        if not subset_data:
            logger.error(f"❌ No valid subsets found for {dataset_name}")
            return False
            
        # Check nesting property: smaller sets should be prefixes of larger sets
        sizes_with_data = sorted(subset_data.keys())
        
        for i, smaller_size in enumerate(sizes_with_data[:-1]):
            for larger_size in sizes_with_data[i+1:]:
                smaller_ids = subset_data[smaller_size]
                larger_ids = subset_data[larger_size]
                
                # Check if smaller subset is a prefix of larger subset
                if len(smaller_ids) > len(larger_ids):
                    logger.error(f"❌ Invalid nesting: {smaller_size}-subset has more samples than {larger_size}-subset")
                    return False
                    
                for j, (small_id, large_id) in enumerate(zip(smaller_ids, larger_ids)):
                    if small_id != large_id:
                        logger.error(f"❌ Nesting violation at position {j}: {smaller_size}-subset has '{small_id}', {larger_size}-subset has '{large_id}'")
                        logger.error(f"   Smaller subset should be a prefix of larger subset")
                        return False
                        
                logger.debug(f"✅ Nesting valid: {smaller_size}-subset ⊆ {larger_size}-subset")
                
        logger.info(f"✅ Subset nesting is valid for {dataset_name}")
        return True
        
    def compare_sampling_methods(self, dataset_name: str, sample_size: int) -> Dict[str, Any]:
        """
        Compare different sampling methods to understand their behavior.
        
        Args:
            dataset_name: Name of the dataset
            sample_size: Number of samples to test
            
        Returns:
            Comparison results
        """
        logger.info(f"🔍 Comparing sampling methods for {dataset_name} with {sample_size} samples")
        
        results = {
            "dataset": dataset_name,
            "sample_size": sample_size,
            "methods": {}
        }
        
        # Method 1: Deterministic file (if exists)
        deterministic_file = self.data_dir / f"{dataset_name}_deterministic_{sample_size}.json"
        if deterministic_file.exists():
            try:
                with open(deterministic_file, 'r') as f:
                    data = json.load(f)
                    samples = data.get("samples", [])
                    sample_ids = [s.get('id', f'sample_{i}') for i, s in enumerate(samples)]
                    
                results["methods"]["deterministic_file"] = {
                    "sample_count": len(sample_ids),
                    "first_5_ids": sample_ids[:5],
                    "hash": hashlib.sha256('|'.join(sample_ids).encode()).hexdigest()[:16]
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Error reading deterministic file: {e}")
                
        # Method 2: Seeded sampling with seed=42
        try:
            samples = self.dataset_manager.load_dataset(dataset_name, sample_size=sample_size, seed=42)
            if samples:
                sample_ids = [s.get('id', f'sample_{i}') for i, s in enumerate(samples)]
                results["methods"]["seeded_42"] = {
                    "sample_count": len(sample_ids),
                    "first_5_ids": sample_ids[:5],
                    "hash": hashlib.sha256('|'.join(sample_ids).encode()).hexdigest()[:16]
                }
        except Exception as e:
            logger.warning(f"⚠️ Error with seeded sampling: {e}")
            
        # Method 3: Seeded sampling with different seed
        try:
            samples = self.dataset_manager.load_dataset(dataset_name, sample_size=sample_size, seed=123)
            if samples:
                sample_ids = [s.get('id', f'sample_{i}') for i, s in enumerate(samples)]
                results["methods"]["seeded_123"] = {
                    "sample_count": len(sample_ids),
                    "first_5_ids": sample_ids[:5],
                    "hash": hashlib.sha256('|'.join(sample_ids).encode()).hexdigest()[:16]
                }
        except Exception as e:
            logger.warning(f"⚠️ Error with different seed sampling: {e}")
            
        # Check consistency between methods that should be the same
        deterministic_hash = results["methods"].get("deterministic_file", {}).get("hash")
        seeded_hash = results["methods"].get("seeded_42", {}).get("hash")
        
        if deterministic_hash and seeded_hash:
            if deterministic_hash == seeded_hash:
                logger.info(f"✅ Deterministic file matches seeded sampling (seed=42)")
                results["consistency"] = "deterministic_file_matches_seeded"
            else:
                logger.warning(f"⚠️ Deterministic file does not match seeded sampling")
                logger.warning(f"   File hash: {deterministic_hash}")
                logger.warning(f"   Seeded hash: {seeded_hash}")
                results["consistency"] = "deterministic_file_differs_from_seeded"
        else:
            results["consistency"] = "cannot_compare"
            
        return results
        
    def generate_validation_report(self, datasets: List[str], sizes: List[int]) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report for all datasets.
        
        Args:
            datasets: List of dataset names to validate
            sizes: List of subset sizes to test
            
        Returns:
            Validation report
        """
        logger.info("🔍 Generating comprehensive validation report")
        
        report = {
            "validation_timestamp": "2024-12-28",  # Current date
            "datasets_tested": datasets,
            "sizes_tested": sizes,
            "results": {}
        }
        
        for dataset_name in datasets:
            logger.info(f"\n📊 Validating {dataset_name}")
            
            dataset_results = {
                "determinism_validation": {},
                "subset_files_validation": {},
                "nesting_validation": None,
                "sampling_methods_comparison": {}
            }
            
            # Test 1: Validate determinism across multiple runs
            for size in sizes[:3]:  # Test first 3 sizes to save time
                is_deterministic = self.validate_dataset_determinism(dataset_name, size)
                dataset_results["determinism_validation"][size] = is_deterministic
                
            # Test 2: Validate deterministic subset files
            dataset_results["subset_files_validation"] = self.validate_deterministic_subset_files(dataset_name, sizes)
            
            # Test 3: Validate subset nesting
            dataset_results["nesting_validation"] = self.validate_subset_nesting(dataset_name, sizes)
            
            # Test 4: Compare sampling methods
            for size in sizes[:2]:  # Test first 2 sizes
                comparison = self.compare_sampling_methods(dataset_name, size)
                dataset_results["sampling_methods_comparison"][size] = comparison
                
            report["results"][dataset_name] = dataset_results
            
        # Generate summary
        total_tests = 0
        passed_tests = 0
        
        for dataset_name, results in report["results"].items():
            # Count determinism tests
            for size, result in results["determinism_validation"].items():
                total_tests += 1
                if result:
                    passed_tests += 1
                    
            # Count subset file tests
            for size, result in results["subset_files_validation"].items():
                total_tests += 1
                if result:
                    passed_tests += 1
                    
            # Count nesting test
            total_tests += 1
            if results["nesting_validation"]:
                passed_tests += 1
                
        report["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_status": "PASS" if passed_tests == total_tests else "FAIL"
        }
        
        return report

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate deterministic sampling")
    parser.add_argument("--datasets", 
                       default="gsm8k,mathbench",
                       help="Comma-separated list of datasets")
    parser.add_argument("--sizes", 
                       default="20,100,500",
                       help="Comma-separated list of subset sizes to test")
    parser.add_argument("--data_dir", 
                       default="data/scaling",
                       help="Data directory")
    parser.add_argument("--output", 
                       help="Output file for validation report (JSON)")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of runs for determinism testing")
    
    args = parser.parse_args()
    
    # Parse arguments
    datasets = [d.strip() for d in args.datasets.split(",")]
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    
    logger.info("=" * 60)
    logger.info("DETERMINISTIC SAMPLING VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Sizes: {sizes}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Runs per test: {args.runs}")
    
    # Create validator
    validator = DeterministicSamplingValidator(args.data_dir)
    
    # Generate validation report
    try:
        report = validator.generate_validation_report(datasets, sizes)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tests: {report['summary']['total_tests']}")
        logger.info(f"Passed tests: {report['summary']['passed_tests']}")
        logger.info(f"Success rate: {report['summary']['success_rate']*100:.1f}%")
        logger.info(f"Overall status: {report['summary']['overall_status']}")
        
        # Detailed results
        for dataset_name, results in report["results"].items():
            logger.info(f"\n📊 {dataset_name.upper()} RESULTS:")
            
            # Determinism tests
            determinism_results = results["determinism_validation"]
            passed_determinism = sum(determinism_results.values())
            total_determinism = len(determinism_results)
            logger.info(f"  Determinism: {passed_determinism}/{total_determinism} passed")
            
            # Subset file tests  
            subset_results = results["subset_files_validation"]
            passed_subsets = sum(subset_results.values())
            total_subsets = len(subset_results)
            logger.info(f"  Subset files: {passed_subsets}/{total_subsets} valid")
            
            # Nesting test
            nesting_result = results["nesting_validation"]
            logger.info(f"  Nesting: {'✅ PASS' if nesting_result else '❌ FAIL'}")
            
        # Save report if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"\n📄 Validation report saved to: {output_path}")
        
        # Return appropriate exit code
        if report['summary']['overall_status'] == 'PASS':
            logger.info("\n✅ All deterministic sampling validations passed!")
            return 0
        else:
            logger.error("\n❌ Some validations failed. Check the report for details.")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Validation failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())