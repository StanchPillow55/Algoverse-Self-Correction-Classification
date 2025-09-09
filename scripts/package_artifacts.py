#!/usr/bin/env python3
"""
Package artifacts for reproducible research
"""

import sys
import json
import shutil
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_anonymized_package(output_dir: str = "outputs/artifacts") -> str:
    """Create an anonymized package for research submission."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped package name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"scaling_study_artifacts_{timestamp}"
    package_dir = output_path / package_name
    
    print(f"📦 Creating anonymized package: {package_name}")
    print("=" * 50)
    
    # Create package directory
    package_dir.mkdir(exist_ok=True)
    
    # Copy essential files
    essential_files = [
        "src/",
        "scripts/",
        "configs/",
        "data/scaling/",
        "requirements.txt",
        "README.md"
    ]
    
    print("📁 Copying essential files...")
    for file_path in essential_files:
        src = project_root / file_path
        dst = package_dir / file_path
        
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  ✓ {file_path}")
        elif src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  ✓ {file_path}/")
    
    # Create anonymized configuration
    create_anonymized_config(package_dir)
    
    # Create reproducibility script
    create_reproducibility_script(package_dir)
    
    # Create artifact manifest
    create_artifact_manifest(package_dir)
    
    # Create zip archive
    zip_path = output_path / f"{package_name}.zip"
    create_zip_archive(package_dir, zip_path)
    
    print(f"\n✅ Package created: {zip_path}")
    print(f"   Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    return str(zip_path)

def create_anonymized_config(package_dir: Path):
    """Create anonymized configuration file."""
    
    # Load original config
    config_path = project_root / "configs" / "scaling_models.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Anonymize model names and providers
    anonymized_config = {
        "models": [],
        "datasets": config.get("datasets", {}),
        "phases": config.get("phases", {})
    }
    
    # Anonymize models
    model_mapping = {}
    for i, model in enumerate(config.get("models", [])):
        original_name = model.get("name", f"model_{i}")
        anonymized_name = f"model_{i+1}"
        
        model_mapping[original_name] = anonymized_name
        
        anonymized_model = {
            **model,
            "name": anonymized_name,
            "provider": "anonymized",
            "model_id": "anonymized",
            "description": f"Model {i+1} - {model.get('size_category', 'unknown')} category"
        }
        
        anonymized_config["models"].append(anonymized_model)
    
    # Save anonymized config
    config_file = package_dir / "configs" / "anonymized_models.json"
    with open(config_file, 'w') as f:
        json.dump(anonymized_config, f, indent=2)
    
    # Save mapping for internal use
    mapping_file = package_dir / "model_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(model_mapping, f, indent=2)
    
    print("  ✓ Anonymized configuration created")

def create_reproducibility_script(package_dir: Path):
    """Create script to reproduce experiments."""
    
    script_content = '''#!/bin/bash
# Reproducibility Script for Scaling Study
# This script reproduces the scaling study experiments

set -e

echo "🔬 Reproducing Scaling Study Experiments"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set up environment
echo "🔧 Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create output directories
mkdir -p outputs/scaling_experiments
mkdir -p outputs/figures
mkdir -p outputs/cost_tracking

# Run Phase 1 experiments (validation)
echo "🚀 Running Phase 1 experiments..."
python scripts/run_scaling_simple.py --phase 1 --dataset data/scaling/toolqa_sample_100.csv --output-dir outputs/scaling_experiments/phase1

# Run Phase 2 experiments (medium scale)
echo "🚀 Running Phase 2 experiments..."
python scripts/run_scaling_simple.py --phase 2 --dataset data/scaling/superglue_sample_500.csv --output-dir outputs/scaling_experiments/phase2

# Run Phase 3 experiments (full scale)
echo "🚀 Running Phase 3 experiments..."
python scripts/run_scaling_simple.py --phase 3 --dataset data/scaling/mathbench_sample_1000.csv --output-dir outputs/scaling_experiments/phase3

# Run scaling analysis
echo "📊 Running scaling analysis..."
python scripts/run_scaling_analysis.py --results-dir outputs/scaling_experiments --create-plots

# Generate model documentation
echo "📚 Generating model documentation..."
python scripts/document_models.py --format json

echo "✅ Reproduction complete!"
echo "Results available in outputs/scaling_experiments/"
echo "Analysis available in outputs/scaling_analysis.json"
echo "Figures available in outputs/figures/"
'''
    
    script_file = package_dir / "reproduce_experiments.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_file.chmod(0o755)
    
    print("  ✓ Reproducibility script created")

def create_artifact_manifest(package_dir: Path):
    """Create manifest of all artifacts."""
    
    manifest = {
        "package_info": {
            "name": "scaling_study_artifacts",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "description": "Artifacts for scaling study of self-correction in large language models"
        },
        "contents": {
            "source_code": {
                "description": "Complete source code for the scaling study",
                "location": "src/",
                "files": list_files_recursive(package_dir / "src")
            },
            "scripts": {
                "description": "Analysis and experiment scripts",
                "location": "scripts/",
                "files": list_files_recursive(package_dir / "scripts")
            },
            "configurations": {
                "description": "Model and experiment configurations",
                "location": "configs/",
                "files": list_files_recursive(package_dir / "configs")
            },
            "datasets": {
                "description": "Sample datasets for experiments",
                "location": "data/scaling/",
                "files": list_files_recursive(package_dir / "data" / "scaling")
            },
            "documentation": {
                "description": "README and documentation",
                "location": ".",
                "files": ["README.md", "reproduce_experiments.sh"]
            }
        },
        "reproduction_instructions": {
            "1": "Extract the package",
            "2": "Install dependencies: pip install -r requirements.txt",
            "3": "Set up API keys in .env file",
            "4": "Run: bash reproduce_experiments.sh",
            "5": "Check outputs/ directory for results"
        },
        "requirements": {
            "python": ">=3.8",
            "dependencies": "See requirements.txt",
            "api_keys": "OpenAI, Anthropic, and/or Replicate API keys required"
        }
    }
    
    manifest_file = package_dir / "ARTIFACT_MANIFEST.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("  ✓ Artifact manifest created")

def list_files_recursive(directory: Path) -> List[str]:
    """List all files in a directory recursively."""
    if not directory.exists():
        return []
    
    files = []
    for item in directory.rglob("*"):
        if item.is_file():
            files.append(str(item.relative_to(directory)))
    
    return sorted(files)

def create_zip_archive(source_dir: Path, zip_path: Path):
    """Create zip archive of the package."""
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir.parent)
                zipf.write(file_path, arcname)
    
    print("  ✓ Zip archive created")

def create_tar_archive(source_dir: Path, tar_path: Path):
    """Create tar.gz archive of the package."""
    
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(source_dir, arcname=source_dir.name)
    
    print("  ✓ Tar.gz archive created")

def main():
    """Main function to package artifacts."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Package artifacts for reproducible research')
    parser.add_argument('--output-dir', default='outputs/artifacts',
                       help='Output directory for packages')
    parser.add_argument('--format', choices=['zip', 'tar', 'both'], default='zip',
                       help='Archive format')
    
    args = parser.parse_args()
    
    try:
        # Create anonymized package
        zip_path = create_anonymized_package(args.output_dir)
        
        if args.format in ['tar', 'both']:
            # Create tar.gz version
            source_dir = Path(zip_path).with_suffix('')
            tar_path = Path(zip_path).with_suffix('.tar.gz')
            create_tar_archive(source_dir, tar_path)
            print(f"✅ Tar archive created: {tar_path}")
        
        print(f"\n🎉 Artifact packaging complete!")
        print(f"   Package: {zip_path}")
        print(f"   Ready for submission or sharing")
        
    except Exception as e:
        print(f"❌ Error packaging artifacts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
