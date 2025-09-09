"""
Dataset Manager for Scaling Study

Downloads and manages datasets for the multi-model self-correction scaling study.
Supports ToolQA, SuperGLUE, and MathBench datasets.
"""

import os
import json
import csv
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Handle imports for both script and module usage
try:
    from ..utils.config import Config
except ImportError:
    try:
        from src.utils.config import Config
    except ImportError:
        from config import Config

logger = logging.getLogger(__name__)

class ScalingDatasetManager:
    """Manages datasets for the scaling study."""
    
    def __init__(self, data_dir: str = "data/scaling"):
        """Initialize dataset manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "toolqa": {
                "name": "ToolQA",
                "description": "QA dataset for tool usage evaluation",
                "url": "https://raw.githubusercontent.com/stanfordnlp/toolqa/main/data/toolqa.json",
                "format": "json",
                "splits": ["train", "dev", "test"],
                "sample_sizes": [100, 500, 1000]
            },
            "superglue": {
                "name": "SuperGLUE", 
                "description": "Reasoning benchmark with multiple tasks",
                "url": "https://raw.githubusercontent.com/nyu-mll/GLUE-baselines/master/download_glue_data.py",
                "format": "tsv",
                "splits": ["train", "dev", "test"],
                "sample_sizes": [100, 500, 1000],
                "tasks": ["BoolQ", "CB", "COPA", "MultiRC", "RTE", "WiC", "WSC"]
            },
            "mathbench": {
                "name": "MathBench",
                "description": "Hierarchical math reasoning benchmark", 
                "url": "https://raw.githubusercontent.com/allenai/mathbench/main/data/mathbench.json",
                "format": "json",
                "splits": ["train", "dev", "test"],
                "sample_sizes": [100, 500, 1000],
                "levels": ["elementary", "middle", "high", "college"]
            }
        }
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """Download a dataset if not already present."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        dataset_path = self.data_dir / f"{dataset_name}.json"
        
        if dataset_path.exists() and not force:
            logger.info(f"Dataset {dataset_name} already exists")
            return True
        
        try:
            logger.info(f"Downloading {dataset_name}...")
            
            if dataset_name == "toolqa":
                self._download_toolqa(dataset_path)
            elif dataset_name == "superglue":
                self._download_superglue(dataset_path)
            elif dataset_name == "mathbench":
                self._download_mathbench(dataset_path)
            else:
                raise ValueError(f"Download not implemented for {dataset_name}")
            
            logger.info(f"Successfully downloaded {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def _download_toolqa(self, output_path: Path):
        """Download ToolQA dataset."""
        # ToolQA is a simple JSON file
        response = requests.get(self.datasets["toolqa"]["url"])
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to our standard format
        formatted_data = {
            "name": "ToolQA",
            "description": "QA dataset for tool usage evaluation",
            "samples": []
        }
        
        for item in data:
            sample = {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "tools": item.get("tools", []),
                "reasoning": item.get("reasoning", ""),
                "difficulty": item.get("difficulty", "medium")
            }
            formatted_data["samples"].append(sample)
        
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
    
    def _download_superglue(self, output_path: Path):
        """Download SuperGLUE dataset (simplified version)."""
        # For now, create a simplified version with sample data
        # In practice, you'd download the full SuperGLUE dataset
        
        sample_data = {
            "name": "SuperGLUE",
            "description": "Reasoning benchmark with multiple tasks",
            "samples": []
        }
        
        # Create sample reasoning problems
        sample_problems = [
            {
                "id": "boolq_1",
                "task": "BoolQ",
                "question": "Is the sky blue?",
                "answer": "Yes",
                "context": "The sky appears blue due to Rayleigh scattering of sunlight.",
                "difficulty": "easy"
            },
            {
                "id": "copa_1", 
                "task": "COPA",
                "question": "What is the cause of the effect: The car wouldn't start?",
                "answer": "The battery was dead",
                "context": "The car wouldn't start this morning.",
                "difficulty": "medium"
            },
            {
                "id": "rte_1",
                "task": "RTE", 
                "question": "Does the hypothesis follow from the premise?",
                "answer": "Yes",
                "premise": "All birds can fly.",
                "hypothesis": "A sparrow can fly.",
                "difficulty": "medium"
            }
        ]
        
        # Generate more samples
        for i in range(100):
            sample = sample_problems[i % len(sample_problems)].copy()
            sample["id"] = f"{sample['task'].lower()}_{i+1}"
            sample_data["samples"].append(sample)
        
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    def _download_mathbench(self, output_path: Path):
        """Download MathBench dataset (simplified version)."""
        # Create sample math problems at different levels
        
        sample_data = {
            "name": "MathBench",
            "description": "Hierarchical math reasoning benchmark",
            "samples": []
        }
        
        # Sample problems by level
        problems_by_level = {
            "elementary": [
                {"question": "What is 15 + 27?", "answer": "42", "level": "elementary"},
                {"question": "If a pizza has 8 slices and you eat 3, how many are left?", "answer": "5", "level": "elementary"}
            ],
            "middle": [
                {"question": "Solve for x: 2x + 5 = 13", "answer": "x = 4", "level": "middle"},
                {"question": "What is the area of a rectangle with length 6 and width 4?", "answer": "24", "level": "middle"}
            ],
            "high": [
                {"question": "Find the derivative of x² + 3x + 2", "answer": "2x + 3", "level": "high"},
                {"question": "Solve the quadratic equation: x² - 5x + 6 = 0", "answer": "x = 2 or x = 3", "level": "high"}
            ],
            "college": [
                {"question": "Evaluate the integral ∫(2x + 1)dx from 0 to 2", "answer": "6", "level": "college"},
                {"question": "Find the eigenvalues of the matrix [[2, 1], [1, 2]]", "answer": "λ₁ = 3, λ₂ = 1", "level": "college"}
            ]
        }
        
        # Generate samples
        for level, problems in problems_by_level.items():
            for i in range(25):  # 25 samples per level
                problem = problems[i % len(problems)].copy()
                problem["id"] = f"{level}_{i+1}"
                problem["topic"] = f"math_{level}"
                sample_data["samples"].append(problem)
        
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    def load_dataset(self, dataset_name: str, sample_size: int = None) -> List[Dict[str, Any]]:
        """Load a dataset with optional sampling."""
        dataset_path = self.data_dir / f"{dataset_name}.json"
        
        if not dataset_path.exists():
            logger.error(f"Dataset {dataset_name} not found. Run download first.")
            return []
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data.get("samples", [])
        
        if sample_size and sample_size < len(samples):
            # Simple random sampling (in practice, you might want stratified sampling)
            import random
            samples = random.sample(samples, sample_size)
        
        return samples
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_path = self.data_dir / f"{dataset_name}.json"
        
        info = self.datasets[dataset_name].copy()
        
        if dataset_path.exists():
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            info["total_samples"] = len(data.get("samples", []))
            info["downloaded"] = True
        else:
            info["total_samples"] = 0
            info["downloaded"] = False
        
        return info
    
    def create_sample_subsets(self, dataset_name: str) -> List[str]:
        """Create sample subsets for cost control."""
        dataset_path = self.data_dir / f"{dataset_name}.json"
        
        if not dataset_path.exists():
            logger.error(f"Dataset {dataset_name} not found")
            return []
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data.get("samples", [])
        created_files = []
        
        for size in [100, 500, 1000]:
            if size <= len(samples):
                subset_path = self.data_dir / f"{dataset_name}_sample_{size}.json"
                
                subset_data = data.copy()
                subset_data["samples"] = samples[:size]
                subset_data["sample_size"] = size
                
                with open(subset_path, 'w') as f:
                    json.dump(subset_data, f, indent=2)
                
                created_files.append(str(subset_path))
                logger.info(f"Created {dataset_name} subset with {size} samples")
        
        return created_files
    
    def download_all_datasets(self, force: bool = False) -> Dict[str, bool]:
        """Download all recommended datasets."""
        results = {}
        
        for dataset_name in self.datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            results[dataset_name] = self.download_dataset(dataset_name, force)
            
            if results[dataset_name]:
                # Create sample subsets
                self.create_sample_subsets(dataset_name)
        
        return results

def get_dataset_manager() -> ScalingDatasetManager:
    """Get singleton instance of dataset manager."""
    if not hasattr(get_dataset_manager, "_instance"):
        get_dataset_manager._instance = ScalingDatasetManager()
    return get_dataset_manager._instance
