"""
Dataset Manager for Scaling Study - ENHANCED VERSION

Downloads and manages REAL datasets for the multi-model self-correction scaling study.
Supports GSM8K, HumanEval, SuperGLUE, and Hendrycks MATH datasets from Hugging Face.
"""

import os
import json
import csv
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Hugging Face datasets integration
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None
    logging.warning("datasets library not installed. Install with: pip install datasets")

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
        
        # REAL Dataset configurations using Hugging Face
        self.datasets = {
            "gsm8k": {
                "name": "GSM8K",
                "description": "Grade School Math 8K - Mathematical word problems",
                "hf_name": "gsm8k",
                "hf_config": "main",
                "format": "hf",
                "splits": ["train", "test"],
                "sample_sizes": [100, 500, 1000],
                "expected_sizes": {"train": 7473, "test": 1319}
            },
            "humaneval": {
                "name": "HumanEval",
                "description": "Code generation benchmark with 164 programming problems",
                "hf_name": "openai_humaneval",
                "hf_config": None,
                "format": "hf",
                "splits": ["test"],
                "sample_sizes": [100, 164],  # 164 is full dataset
                "expected_sizes": {"test": 164}
            },
            "superglue": {
                "name": "SuperGLUE", 
                "description": "Reasoning benchmark with multiple tasks (146K+ samples)",
                "hf_name": "aps/super_glue",
                "hf_config": None,  # Will be set per task
                "format": "hf_multi",
                "splits": ["train", "validation", "test"],
                "sample_sizes": [100, 500, 1000, 5000],
                "tasks": ["boolq", "cb", "copa", "multirc", "record", "rte", "wic", "wsc"],
                "expected_sizes": {
                    "boolq": 9427, "cb": 250, "copa": 400, "multirc": 27243,
                    "record": 100730, "rte": 2490, "wic": 5428, "wsc": 554
                }
            },
            "mathbench": {
                "name": "MathBench",
                "description": "Open-Compass MathBench: Multi-level mathematics evaluation (3,709 problems)", 
                "format": "github_release",
                "github_repo": "open-compass/MathBench",
                "release_tag": "v0.1.0",
                "splits": ["test"],  # MathBench is primarily a test benchmark
                "sample_sizes": [100, 500, 1000, 3709],  # 3709 is full dataset
                "expected_sizes": {"total": 3709},
                "levels": ["arithmetic", "primary", "middle", "high", "college"],
                "languages": ["en", "zh"]  # Bilingual support
            },
            "math": {
                "name": "Hendrycks MATH",
                "description": "Competition-level mathematics problems", 
                "hf_name": "EleutherAI/hendrycks_math",
                "hf_config": None,  # Will be set per subject
                "format": "hf_multi",
                "splits": ["train", "test"],
                "sample_sizes": [100, 500, 1000],
                "subjects": ["algebra", "counting_and_probability", "geometry", 
                            "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
                "expected_sizes": {"total_train": 7500, "total_test": 5000}  # Approximate
            }
        }
    
    def download_dataset(self, dataset_name: str, force: bool = False) -> bool:
        """Download a REAL dataset using Hugging Face datasets."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if load_dataset is None:
            logger.error("datasets library not installed. Install with: pip install datasets")
            return False
        
        dataset_config = self.datasets[dataset_name]
        dataset_path = self.data_dir / f"{dataset_name}.json"
        
        if dataset_path.exists() and not force:
            logger.info(f"Dataset {dataset_name} already exists")
            return True
        
        try:
            logger.info(f"Downloading {dataset_name} from Hugging Face...")
            
            if dataset_name == "gsm8k":
                return self._download_gsm8k(dataset_path)
            elif dataset_name == "humaneval":
                return self._download_humaneval(dataset_path)
            elif dataset_name == "superglue":
                return self._download_superglue_real(dataset_path)
            elif dataset_name == "mathbench":
                return self._download_mathbench_real(dataset_path)
            elif dataset_name == "math":
                return self._download_math_real(dataset_path)
            else:
                raise ValueError(f"Download not implemented for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def _download_gsm8k(self, output_path: Path) -> bool:
        """Download GSM8K dataset from Hugging Face."""
        try:
            # Load full GSM8K dataset
            train_data = load_dataset("gsm8k", "main", split="train")
            test_data = load_dataset("gsm8k", "main", split="test")
            
            # Convert to our standard format
            formatted_data = {
                "name": "GSM8K",
                "description": "Grade School Math 8K - Mathematical word problems",
                "samples": []
            }
            
            # Process train + test data
            all_data = list(train_data) + list(test_data)
            
            for idx, item in enumerate(all_data):
                sample = {
                    "id": str(idx),
                    "question": item["question"],
                    "answer": item["answer"],
                    "split": "train" if idx < len(train_data) else "test",
                    "difficulty": "grade_school",
                    "topic": "math_word_problem"
                }
                formatted_data["samples"].append(sample)
            
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            
            logger.info(f"Downloaded GSM8K: {len(formatted_data['samples'])} samples")
            return True
            
        except Exception as e:
            logger.error(f"GSM8K download failed: {e}")
            return False
    
    def _download_humaneval(self, output_path: Path) -> bool:
        """Download HumanEval dataset from Hugging Face."""
        try:
            # Load HumanEval dataset  
            dataset = load_dataset("openai_humaneval", split="test")
            
            # Convert to our standard format
            formatted_data = {
                "name": "HumanEval",
                "description": "Code generation benchmark with 164 programming problems",
                "samples": []
            }
            
            for idx, item in enumerate(dataset):
                sample = {
                    "id": item["task_id"],
                    "question": item["prompt"],
                    "answer": item["canonical_solution"],
                    "entry_point": item["entry_point"],
                    "test_code": item["test"],
                    "split": "test",
                    "difficulty": "coding",
                    "topic": "programming"
                }
                formatted_data["samples"].append(sample)
            
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            
            logger.info(f"Downloaded HumanEval: {len(formatted_data['samples'])} samples")
            return True
            
        except Exception as e:
            logger.error(f"HumanEval download failed: {e}")
            return False
    
    def _download_superglue_real(self, output_path: Path) -> bool:
        """Download REAL SuperGLUE dataset from Hugging Face."""
        try:
            formatted_data = {
                "name": "SuperGLUE",
                "description": "Reasoning benchmark with multiple tasks",
                "samples": []
            }
            
            # Download all SuperGLUE tasks
            tasks = self.datasets["superglue"]["tasks"]
            
            for task in tasks:
                try:
                    # Load train split for each task
                    task_data = load_dataset("aps/super_glue", task, split="train")
                    
                    for idx, item in enumerate(task_data):
                        sample = {
                            "id": f"{task}_{idx}",
                            "task": task,
                            "question": self._extract_question(item, task),
                            "answer": self._extract_answer(item, task),
                            "context": self._extract_context(item, task),
                            "split": "train",
                            "difficulty": "reasoning",
                            "topic": f"superglue_{task}"
                        }
                        formatted_data["samples"].append(sample)
                        
                except Exception as e:
                    logger.warning(f"Failed to download SuperGLUE task {task}: {e}")
                    continue
            
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            
            logger.info(f"Downloaded SuperGLUE: {len(formatted_data['samples'])} samples across {len(tasks)} tasks")
            return True
            
        except Exception as e:
            logger.error(f"SuperGLUE download failed: {e}")
            return False
    
    def _extract_question(self, item: dict, task: str) -> str:
        """Extract question from SuperGLUE item based on task type."""
        if task == "boolq":
            return f"Context: {item['passage']}\n\nQuestion: {item['question']}"
        elif task == "cb":
            return f"Premise: {item['premise']}\n\nHypothesis: {item['hypothesis']}\n\nDoes the premise entail the hypothesis?"
        elif task == "copa":
            return f"Premise: {item['premise']}\n\nWhat is the {item['question']}?\nChoice 1: {item['choice1']}\nChoice 2: {item['choice2']}"
        elif task == "multirc":
            return f"Paragraph: {item.get('paragraph', '')}\n\nQuestion: {item.get('question', '')}\n\nAnswer: {item.get('answer', '')}"
        elif task == "record":
            return f"Passage: {item.get('passage', '')}\n\nQuery: {item.get('query', '')}"
        elif task == "rte":
            return f"Premise: {item.get('premise', '')}\n\nHypothesis: {item.get('hypothesis', '')}\n\nDoes the premise entail the hypothesis?"
        elif task == "wic":
            return f"Sentence 1: {item.get('sentence1', '')}\nSentence 2: {item.get('sentence2', '')}\n\nIs the word '{item.get('word', '')}' used with the same meaning in both sentences?"
        elif task == "wsc":
            return f"Text: {item.get('text', '')}\n\nWhat does the pronoun '{item.get('target', {}).get('span1_text', '')}' refer to?"
        else:
            return str(item.get('question', item.get('text', str(item))))
    
    def _extract_answer(self, item: dict, task: str) -> str:
        """Extract answer from SuperGLUE item based on task type."""
        label = item.get('label', 0)
        
        if task in ["boolq", "cb", "rte", "wic", "wsc"]:
            return "Yes" if label == 1 else "No" if label == 0 else "Maybe" if label == 2 else str(label)
        elif task == "copa":
            return item.get('choice1' if label == 0 else 'choice2', str(label))
        elif task in ["multirc", "record"]:
            return str(label)
        else:
            return str(label)
    
    def _extract_context(self, item: dict, task: str) -> str:
        """Extract context from SuperGLUE item based on task type."""
        if task == "boolq":
            return item.get('passage', '')
        elif task in ["cb", "rte"]:
            return f"Premise: {item.get('premise', '')}"
        elif task == "copa":
            return item.get('premise', '')
        elif task == "multirc":
            return item.get('paragraph', '')
        elif task == "record":
            return item.get('passage', '')
        elif task == "wic":
            return f"Word: {item.get('word', '')}"
        elif task == "wsc":
            return item.get('text', '')
        else:
            return str(item)
    
    def _download_mathbench_real(self, output_path: Path) -> bool:
        """Download REAL MathBench dataset from GitHub releases."""
        try:
            # MathBench release URL
            release_url = "https://github.com/open-compass/MathBench/releases/download/v0.1.0/mathbench_v1.zip"
            
            logger.info("Downloading MathBench from GitHub releases...")
            
            # Download the zip file
            response = requests.get(release_url)
            response.raise_for_status()
            
            # Extract to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = Path(temp_dir) / "mathbench_v1.zip"
                
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find all JSON files in the extracted directory
                mathbench_dir = Path(temp_dir) / "mathbench_v1"
                json_files = list(mathbench_dir.rglob("*.json"))
                
                if not json_files:
                    logger.error("No JSON files found in MathBench release")
                    return False
                
                # Combine all problems from different files
                formatted_data = {
                    "name": "MathBench",
                    "description": "Open-Compass MathBench: Multi-level mathematics evaluation",
                    "samples": []
                }
                
                sample_id = 0
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                        
                        # Extract level and language from filename
                        filename = json_file.name
                        level = "unknown"
                        lang = "en"
                        
                        if "primary" in filename:
                            level = "primary"
                        elif "middle" in filename:
                            level = "middle"
                        elif "high" in filename:
                            level = "high"
                        elif "college" in filename:
                            level = "college"
                        elif "arithmetic" in filename:
                            level = "arithmetic"
                        
                        if "zh" in filename or "chinese" in filename.lower():
                            lang = "zh"
                        
                        # Process each problem in the file
                        for item in file_data:
                            sample = {
                                "id": f"mathbench_{sample_id}",
                                "question": item.get("problem", item.get("question", str(item))),
                                "answer": item.get("answer", item.get("solution", "")),
                                "level": level,
                                "language": lang,
                                "split": "test",
                                "difficulty": level,
                                "topic": f"mathbench_{level}"
                            }
                            formatted_data["samples"].append(sample)
                            sample_id += 1
                            
                    except Exception as e:
                        logger.warning(f"Error processing {json_file}: {e}")
                        continue
                
                # Save the combined dataset
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Downloaded MathBench: {len(formatted_data['samples'])} samples")
                return True
                
        except Exception as e:
            logger.error(f"MathBench download failed: {e}")
            return False
    
    def _download_math_real(self, output_path: Path) -> bool:
        """Download REAL Hendrycks MATH dataset from Hugging Face."""
        try:
            formatted_data = {
                "name": "Hendrycks MATH",
                "description": "Competition-level mathematics problems",
                "samples": []
            }
            
            # Download all MATH subjects
            subjects = self.datasets["math"]["subjects"]
            
            sample_id = 0
            for subject in subjects:
                try:
                    # Load train and test splits for each subject
                    for split in ["train", "test"]:
                        try:
                            subject_data = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
                            
                            for item in subject_data:
                                sample = {
                                    "id": f"math_{subject}_{sample_id}",
                                    "question": item["problem"],
                                    "answer": item["solution"],
                                    "subject": subject,
                                    "split": split,
                                    "difficulty": "competition",
                                    "topic": f"math_{subject}"
                                }
                                formatted_data["samples"].append(sample)
                                sample_id += 1
                                
                        except Exception as e:
                            logger.warning(f"Failed to download MATH {subject} {split}: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Failed to download MATH subject {subject}: {e}")
                    continue
            
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f, indent=2)
            
            logger.info(f"Downloaded Hendrycks MATH: {len(formatted_data['samples'])} samples across {len(subjects)} subjects")
            return True
            
        except Exception as e:
            logger.error(f"Hendrycks MATH download failed: {e}")
            return False
    
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
