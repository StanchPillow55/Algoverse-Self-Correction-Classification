#!/usr/bin/env python3
"""
Convert ToolQA JSON datasets to CSV format for main pipeline compatibility
"""

import json
import csv
import sys
from pathlib import Path

def convert_toolqa_json_to_csv(json_path: str, output_csv: str = None):
    """Convert ToolQA JSON to CSV format compatible with main pipeline"""
    
    if not output_csv:
        json_file = Path(json_path)
        output_csv = json_file.with_suffix('.csv')
    
    print(f"Converting {json_path} to {output_csv}")
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    samples = data.get('samples', data if isinstance(data, list) else [])
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['qid', 'question', 'ground_truth', 'topic']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for sample in samples:
            writer.writerow({
                'qid': sample.get('qid', sample.get('id', 'unknown')),
                'question': sample.get('question', ''),
                'ground_truth': sample.get('answer', sample.get('reference', '')),
                'topic': sample.get('topic', sample.get('domain', 'toolqa'))
            })
    
    print(f"✅ Converted {len(samples)} samples to {output_csv}")
    return output_csv

if __name__ == "__main__":
    # Convert the deterministic ToolQA datasets
    datasets = [
        "data/scaling/toolqa_deterministic_500.json",
        "data/scaling/toolqa_deterministic_100.json", 
        "data/scaling/toolqa_deterministic_1000.json"
    ]
    
    for dataset in datasets:
        if Path(dataset).exists():
            convert_toolqa_json_to_csv(dataset)
        else:
            print(f"⚠️ Dataset not found: {dataset}")