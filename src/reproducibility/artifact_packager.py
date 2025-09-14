"""
Reproducibility infrastructure for the Teacher-Learner RTS scaling study.

This module creates reproducible experiment packages with all necessary
artifacts, configurations, and documentation.
"""

import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml

class ArtifactPackager:
    """Packages experiments for reproducibility."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the artifact packager."""
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / "reproducible_artifacts"
        self.output_dir.mkdir(exist_ok=True)
    
    def create_experiment_package(self, experiment_name: str, 
                                experiment_dirs: List[str],
                                include_models: bool = False) -> str:
        """Create a complete experiment package."""
        package_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)
        
        print(f"Creating experiment package: {package_name}")
        
        # 1. Copy experiment results
        self._copy_experiment_results(package_dir, experiment_dirs)
        
        # 2. Copy source code
        self._copy_source_code(package_dir)
        
        # 3. Copy configurations
        self._copy_configurations(package_dir)
        
        # 4. Copy datasets
        self._copy_datasets(package_dir)
        
        # 5. Create environment specification
        self._create_environment_spec(package_dir)
        
        # 6. Create reproduction instructions
        self._create_reproduction_instructions(package_dir, experiment_name)
        
        # 7. Create experiment metadata
        self._create_experiment_metadata(package_dir, experiment_name, experiment_dirs)
        
        # 8. Create zip archive
        zip_path = self._create_zip_archive(package_dir)
        
        print(f"Experiment package created: {zip_path}")
        return str(zip_path)
    
    def _copy_experiment_results(self, package_dir: Path, experiment_dirs: List[str]):
        """Copy experiment results to package."""
        results_dir = package_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        for exp_dir in experiment_dirs:
            src_dir = self.project_root / "outputs" / exp_dir
            if src_dir.exists():
                dst_dir = results_dir / exp_dir
                shutil.copytree(src_dir, dst_dir)
                print(f"  Copied results: {exp_dir}")
    
    def _copy_source_code(self, package_dir: Path):
        """Copy source code to package."""
        src_dir = package_dir / "src"
        shutil.copytree(self.project_root / "src", src_dir)
        
        # Copy key scripts
        scripts_to_copy = [
            "scripts/run_scaling_simple.py",
            "scripts/run_humaneval.py",
            "scripts/run_gsm8k.py"
        ]
        
        for script in scripts_to_copy:
            src_script = self.project_root / script
            if src_script.exists():
                dst_script = package_dir / script
                dst_script.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_script, dst_script)
        
        print("  Copied source code")
    
    def _copy_configurations(self, package_dir: Path):
        """Copy configuration files to package."""
        config_dir = package_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        config_files = [
            "configs/scaling_models.json",
            "configs/humaneval_models.json",
            "configs/gsm8k_models.json"
        ]
        
        for config_file in config_files:
            src_config = self.project_root / config_file
            if src_config.exists():
                shutil.copy2(src_config, config_dir / Path(config_file).name)
        
        print("  Copied configurations")
    
    def _copy_datasets(self, package_dir: Path):
        """Copy dataset files to package."""
        data_dir = package_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Copy scaling datasets
        scaling_data_dir = data_dir / "scaling"
        scaling_data_dir.mkdir(exist_ok=True)
        
        scaling_datasets = [
            "data/scaling/toolqa_sample.csv",
            "data/scaling/gsm8k_sample_100_realistic.csv",
            "data/scaling/mathbench_sample_100_realistic.csv",
            "data/scaling/superglue_sample_500_realistic.csv"
        ]
        
        for dataset in scaling_datasets:
            src_dataset = self.project_root / dataset
            if src_dataset.exists():
                shutil.copy2(src_dataset, scaling_data_dir / Path(dataset).name)
        
        print("  Copied datasets")
    
    def _create_environment_spec(self, package_dir: Path):
        """Create environment specification files."""
        # Copy requirements.txt
        if (self.project_root / "requirements.txt").exists():
            shutil.copy2(self.project_root / "requirements.txt", package_dir / "requirements.txt")
        
        # Create environment.yml for conda
        env_spec = {
            'name': 'teacher-learner-rts',
            'channels': ['conda-forge', 'pytorch'],
            'dependencies': [
                'python=3.9',
                'pip',
                'numpy',
                'pandas',
                'scipy',
                'matplotlib',
                'seaborn',
                'scikit-learn',
                'jupyter',
                'tqdm',
                'requests',
                'python-dotenv',
                'openai',
                'anthropic',
                'huggingface-hub',
                'transformers',
                'torch',
                'accelerate',
                'pip'
            ]
        }
        
        with open(package_dir / "environment.yml", 'w') as f:
            yaml.dump(env_spec, f, default_flow_style=False)
        
        print("  Created environment specifications")
    
    def _create_reproduction_instructions(self, package_dir: Path, experiment_name: str):
        """Create reproduction instructions."""
        instructions = f"""# Reproduction Instructions for {experiment_name}

## Environment Setup

### Option 1: Using Conda
```bash
conda env create -f environment.yml
conda activate teacher-learner-rts
pip install -r requirements.txt
```

### Option 2: Using pip
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with the following API keys:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
REPLICATE_API_TOKEN=your_replicate_token_here
DEMO_MODE=false
```

## Running Experiments

### Phase 1: Validation
```bash
python scripts/run_scaling_simple.py --dataset data/scaling/toolqa_sample.csv --phase 1 --output-dir outputs/phase1_validation --max-turns 2
```

### Phase 2: Medium Scale
```bash
python scripts/run_scaling_simple.py --dataset data/scaling/toolqa_sample.csv --phase 2 --output-dir outputs/phase2_medium_scale --max-turns 3
```

### Phase 3: Full Scale
```bash
python scripts/run_scaling_simple.py --dataset data/scaling/toolqa_sample.csv --phase 3 --output-dir outputs/phase3_full_scale --max-turns 3
```

## Analysis

### Power-Law Analysis
```bash
python -m src.analysis.power_law_analysis
```

### Enhanced Trace Formatting
The enhanced trace formatter will automatically run after each experiment, creating:
- Full reasoning traces as .txt files
- CSV outputs for final answers and multi-turn accuracy
- Summary metrics

## Results Structure

- `results/`: Contains all experiment results
- `outputs/enhanced_traces/`: Contains formatted traces and CSV outputs
- `outputs/scaling_analysis/`: Contains power-law analysis results

## Notes

- All experiments use the same random seed for reproducibility
- API rate limits may affect execution time
- Some models may require specific hardware (e.g., GPU for HuggingFace models)
"""
        
        with open(package_dir / "REPRODUCTION_INSTRUCTIONS.md", 'w') as f:
            f.write(instructions)
        
        print("  Created reproduction instructions")
    
    def _create_experiment_metadata(self, package_dir: Path, experiment_name: str, experiment_dirs: List[str]):
        """Create experiment metadata file."""
        metadata = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'experiment_directories': experiment_dirs,
            'project_version': '1.0.0',
            'description': 'Teacher-Learner RTS Scaling Study',
            'datasets': [
                'ToolQA (100 samples)',
                'GSM8K (100 samples)', 
                'MathBench (100 samples)',
                'SuperGLUE (500 samples)'
            ],
            'models': [
                'gpt-4o-mini',
                'gpt-4o', 
                'claude-haiku',
                'claude-sonnet',
                'llama-7b',
                'llama-13b',
                'llama-70b'
            ],
            'phases': {
                'phase1': 'Validation (2 models × 1 dataset × 100 samples)',
                'phase2': 'Medium Scale (4 models × 2 datasets × 500 samples)',
                'phase3': 'Full Scale (7 models × 4 datasets × 1000 samples)'
            }
        }
        
        with open(package_dir / "experiment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("  Created experiment metadata")
    
    def _create_zip_archive(self, package_dir: Path) -> Path:
        """Create zip archive of the package."""
        zip_path = package_dir.with_suffix('.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir.parent)
                    zipf.write(file_path, arcname)
        
        # Remove the directory after zipping
        shutil.rmtree(package_dir)
        
        return zip_path

def main():
    """Main function to create experiment packages."""
    packager = ArtifactPackager()
    
    # Create packages for different experiment sets
    packages = [
        {
            'name': 'phase1_validation',
            'dirs': ['phase1_validation']
        },
        {
            'name': 'phase2_medium_scale', 
            'dirs': ['phase2_medium_scale']
        },
        {
            'name': 'phase3_full_scale',
            'dirs': ['phase3_full_scale_v2']
        },
        {
            'name': 'complete_study',
            'dirs': ['phase1_validation', 'phase2_medium_scale', 'phase3_full_scale_v2']
        }
    ]
    
    created_packages = []
    
    for package in packages:
        try:
            zip_path = packager.create_experiment_package(
                package['name'],
                package['dirs']
            )
            created_packages.append(zip_path)
        except Exception as e:
            print(f"Error creating package {package['name']}: {e}")
    
    print(f"\nCreated {len(created_packages)} experiment packages:")
    for package in created_packages:
        print(f"  - {package}")

if __name__ == "__main__":
    main()

