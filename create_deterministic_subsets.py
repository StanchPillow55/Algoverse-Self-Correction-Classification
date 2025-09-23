#!/usr/bin/env python3
"""
Create Deterministic Dataset Subsets

This script creates deterministic subsets of datasets to ensure consistent sampling 
across multi-turn and ensemble experiments for fair comparison.

Usage:
    python3 create_deterministic_subsets.py --datasets gsm8k,mathbench --sizes 100,500
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.scaling_datasets import ScalingDatasetManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeterministicSubsetGenerator:
    """Generate deterministic subsets for consistent experimental conditions."""
    
    def __init__(self, output_dir: str = "data/scaling"):
        """Initialize the generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_manager = ScalingDatasetManager()
        
        # Common subset sizes for experiments
        self.default_sizes = [20, 50, 100, 500, 1000]
        
    def create_deterministic_subsets(self, dataset_name: str, sizes: List[int] = None) -> Dict[str, str]:
        """
        Create deterministic subsets for a dataset using FIRST N sampling (not random).
        This ensures identical subsets across different experiment runs.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'gsm8k', 'mathbench')
            sizes: List of subset sizes to create
            
        Returns:
            Dictionary mapping subset names to file paths
        """
        if sizes is None:
            sizes = self.default_sizes
            
        logger.info(f"Creating deterministic subsets for {dataset_name}")
        logger.info(f"Subset sizes: {sizes}")
        
        # First, ensure the full dataset exists
        if not self.dataset_manager.download_dataset(dataset_name, force=False):
            logger.error(f"Failed to download dataset: {dataset_name}")
            return {}
            
        # Get dataset info
        dataset_info = self.dataset_manager.get_dataset_info(dataset_name)
        total_samples = dataset_info.get('total_samples', 0)
        
        logger.info(f"Total samples in {dataset_name}: {total_samples}")
        
        # Load full dataset
        dataset_path = self.output_dir / f"{dataset_name}.json"
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return {}
            
        with open(dataset_path, 'r') as f:
            full_data = json.load(f)
            
        samples = full_data.get("samples", [])
        if not samples:
            logger.error(f"No samples found in dataset: {dataset_name}")
            return {}
            
        # Create deterministic subsets
        created_files = {}
        
        for size in sizes:
            if size > len(samples):
                logger.warning(f"Requested size {size} exceeds dataset size {len(samples)} for {dataset_name}")
                continue
                
            # Create deterministic subset using FIRST N samples
            subset_path = self.output_dir / f"{dataset_name}_deterministic_{size}.json"
            
            subset_data = {
                "name": full_data.get("name", dataset_name),
                "description": full_data.get("description", f"Deterministic subset of {dataset_name}"),
                "original_dataset": dataset_name,
                "sampling_method": "deterministic_first_n",
                "sample_size": size,
                "total_original_size": len(samples),
                "created_by": "create_deterministic_subsets.py",
                "samples": samples[:size]  # Take first N samples deterministically
            }
            
            # Save subset
            with open(subset_path, 'w') as f:
                json.dump(subset_data, f, indent=2)
                
            created_files[f"subset_{size}"] = str(subset_path)
            logger.info(f"✅ Created deterministic subset: {size} samples -> {subset_path}")
            
            # Verify first few samples to ensure determinism
            sample_ids = [s.get('id', f'sample_{i}') for i, s in enumerate(samples[:min(5, size)])]
            logger.debug(f"First 5 sample IDs in {size}-sample subset: {sample_ids}")
            
        return created_files
    
    def verify_deterministic_consistency(self, dataset_name: str, size: int) -> bool:
        """
        Verify that deterministic subsets are consistent by loading the same subset twice.
        
        Args:
            dataset_name: Name of the dataset
            size: Size of the subset to verify
            
        Returns:
            True if consistent, False otherwise
        """
        logger.info(f"Verifying deterministic consistency for {dataset_name} subset size {size}")
        
        subset_path = self.output_dir / f"{dataset_name}_deterministic_{size}.json"
        
        if not subset_path.exists():
            logger.error(f"Subset file not found: {subset_path}")
            return False
            
        # Load the subset twice and compare
        with open(subset_path, 'r') as f:
            data1 = json.load(f)
            
        with open(subset_path, 'r') as f:
            data2 = json.load(f)
            
        samples1 = data1.get("samples", [])
        samples2 = data2.get("samples", [])
        
        if len(samples1) != len(samples2):
            logger.error(f"Sample count mismatch: {len(samples1)} vs {len(samples2)}")
            return False
            
        # Compare first few samples
        for i in range(min(10, len(samples1))):
            id1 = samples1[i].get('id', f'sample_{i}')
            id2 = samples2[i].get('id', f'sample_{i}')
            
            if id1 != id2:
                logger.error(f"Sample ID mismatch at position {i}: {id1} vs {id2}")
                return False
                
        logger.info(f"✅ Deterministic consistency verified for {dataset_name} subset size {size}")
        return True
    
    def create_sample_mapping_file(self, dataset_name: str, sizes: List[int]) -> str:
        """
        Create a mapping file that shows which samples are included in each subset size.
        This is useful for verifying consistency across experiments.
        
        Args:
            dataset_name: Name of the dataset
            sizes: List of subset sizes
            
        Returns:
            Path to the mapping file
        """
        logger.info(f"Creating sample mapping file for {dataset_name}")
        
        mapping = {
            "dataset": dataset_name,
            "created_by": "create_deterministic_subsets.py",
            "subsets": {}
        }
        
        for size in sizes:
            subset_path = self.output_dir / f"{dataset_name}_deterministic_{size}.json"
            
            if not subset_path.exists():
                logger.warning(f"Subset file not found: {subset_path}")
                continue
                
            with open(subset_path, 'r') as f:
                subset_data = json.load(f)
                
            samples = subset_data.get("samples", [])
            sample_ids = [s.get('id', f'sample_{i}') for i, s in enumerate(samples)]
            
            mapping["subsets"][str(size)] = {
                "size": len(sample_ids),
                "sample_ids": sample_ids[:20],  # First 20 IDs for verification
                "total_samples": len(sample_ids)
            }
            
        # Save mapping file
        mapping_path = self.output_dir / f"{dataset_name}_deterministic_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
            
        logger.info(f"✅ Sample mapping saved to: {mapping_path}")
        return str(mapping_path)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create deterministic dataset subsets")
    parser.add_argument("--datasets", 
                       default="gsm8k,mathbench",
                       help="Comma-separated list of datasets")
    parser.add_argument("--sizes", 
                       default="20,50,100,500,1000",
                       help="Comma-separated list of subset sizes")
    parser.add_argument("--output_dir", 
                       default="data/scaling",
                       help="Output directory for subsets")
    parser.add_argument("--verify", action="store_true",
                       help="Verify deterministic consistency after creation")
    
    args = parser.parse_args()
    
    # Parse arguments
    datasets = [d.strip() for d in args.datasets.split(",")]
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    
    logger.info("=" * 60)
    logger.info("DETERMINISTIC SUBSET GENERATOR")
    logger.info("=" * 60)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Subset sizes: {sizes}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create generator
    generator = DeterministicSubsetGenerator(args.output_dir)
    
    total_created = 0
    all_results = {}
    
    # Process each dataset
    for dataset_name in datasets:
        logger.info(f"\n📊 Processing dataset: {dataset_name}")
        
        try:
            # Create deterministic subsets
            results = generator.create_deterministic_subsets(dataset_name, sizes)
            
            if results:
                total_created += len(results)
                all_results[dataset_name] = results
                
                logger.info(f"✅ Created {len(results)} subsets for {dataset_name}")
                
                # Create sample mapping for verification
                mapping_file = generator.create_sample_mapping_file(dataset_name, sizes)
                
                # Verify consistency if requested
                if args.verify:
                    logger.info(f"🔍 Verifying consistency for {dataset_name}...")
                    for size in sizes:
                        if f"subset_{size}" in results:
                            is_consistent = generator.verify_deterministic_consistency(dataset_name, size)
                            if not is_consistent:
                                logger.error(f"❌ Consistency check failed for {dataset_name} size {size}")
                            else:
                                logger.info(f"✅ Verified consistency for {dataset_name} size {size}")
            else:
                logger.error(f"❌ Failed to create subsets for {dataset_name}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {dataset_name}: {e}")
            continue
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total subsets created: {total_created}")
    logger.info(f"Datasets processed: {len(all_results)}")
    
    for dataset_name, results in all_results.items():
        logger.info(f"  {dataset_name}: {len(results)} subsets")
        for subset_name, file_path in results.items():
            logger.info(f"    {subset_name}: {Path(file_path).name}")
    
    if total_created > 0:
        logger.info("\n✅ All deterministic subsets created successfully!")
        logger.info("These subsets ensure consistent sampling across multi-turn and ensemble experiments.")
        logger.info("Sample selection is deterministic based on original dataset order (first N samples).")
        return 0
    else:
        logger.error("❌ No subsets were created successfully.")
        return 1

if __name__ == "__main__":
    sys.exit(main())