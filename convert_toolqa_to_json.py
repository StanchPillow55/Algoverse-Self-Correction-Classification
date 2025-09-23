#!/usr/bin/env python3
"""
Convert ToolQA CSV datasets to JSON format expected by the experiment runner.

This script reads the CSV files in the datasets/ directory and converts them
to the JSON format expected by the ScalingDatasetManager.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def convert_csv_to_json(csv_file: Path, output_dir: Path):
    """Convert a ToolQA CSV file to the expected JSON format."""
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert each row to the expected format
    samples = []
    for _, row in df.iterrows():
        sample = {
            "question": row["question"],
            "reference": str(row["reference"]),  # Ensure reference is string
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "qid": row["qid"]
        }
        samples.append(sample)
    
    # Create the JSON structure
    dataset_name = csv_file.stem  # e.g., "toolqa_deterministic_100"
    sample_size = dataset_name.split("_")[-1]  # e.g., "100"
    
    # Create the JSON structure to match ScalingDatasetManager expectations
    json_data = {
        "name": f"ToolQA {sample_size}",
        "description": "ToolQA deterministic subset - Multi-domain reasoning benchmark",
        "samples": []
    }
    
    # Convert samples to the expected format
    for i, sample in enumerate(samples):
        formatted_sample = {
            "id": f"toolqa_{i}",
            "question": sample["question"],
            "answer": sample["reference"],  # Convert reference to answer
            "domain": sample["domain"],
            "difficulty": sample["difficulty"],
            "qid": sample["qid"],
            "split": "test",
            "topic": f"toolqa_{sample['domain']}"
        }
        json_data["samples"].append(formatted_sample)
    
    # Write the JSON file
    json_file = output_dir / f"{dataset_name}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {csv_file.name} -> {json_file.name} ({len(samples)} samples)")
    
    return json_file


def main():
    """Convert all ToolQA CSV files to JSON format."""
    
    # Set up paths
    csv_dir = Path("datasets")
    output_dir = Path("data/scaling")
    
    # Find all ToolQA CSV files
    csv_files = list(csv_dir.glob("toolqa_deterministic_*.csv"))
    
    if not csv_files:
        print("No ToolQA CSV files found in datasets/ directory")
        return
    
    print(f"Found {len(csv_files)} ToolQA CSV files to convert:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # Convert each file
    converted_files = []
    for csv_file in csv_files:
        try:
            json_file = convert_csv_to_json(csv_file, output_dir)
            converted_files.append(json_file)
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")
    
    print(f"\n✅ Successfully converted {len(converted_files)} files:")
    for json_file in converted_files:
        print(f"  - {json_file.name}")
    
    # Create a mapping file for reference
    mapping_data = {
        "toolqa_datasets": {
            "created": datetime.now().isoformat(),
            "source": "Converted from ToolQA deterministic CSV files",
            "files": [f.name for f in converted_files],
            "total_samples": sum(
                len(pd.read_csv(csv_dir / f"toolqa_deterministic_{f.stem.split('_')[-1]}.csv"))
                for f in converted_files
            )
        }
    }
    
    mapping_file = output_dir / "toolqa_deterministic_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"\n📋 Created mapping file: {mapping_file.name}")


if __name__ == "__main__":
    main()