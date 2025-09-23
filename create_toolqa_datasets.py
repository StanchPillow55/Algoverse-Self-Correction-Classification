#!/usr/bin/env python3
"""
Script to create deterministic ToolQA CSV datasets from the official ToolQA repository.

This script:
1. Combines all ToolQA questions from different domains and difficulties
2. Creates deterministic subsets (100, 500, 1000 samples) 
3. Generates CSV files compatible with the experiment runner
4. Creates JSON mapping files and metadata
"""

import json
import pandas as pd
import random
from pathlib import Path
import re

def load_toolqa_data(toolqa_dir: Path):
    """Load all ToolQA questions from JSONL files."""
    questions = []
    
    # Process easy questions
    easy_dir = toolqa_dir / "data" / "questions" / "easy"
    for jsonl_file in easy_dir.glob("*.jsonl"):
        domain = jsonl_file.stem.replace("-easy", "")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                questions.append({
                    'qid': data['qid'],
                    'domain': domain,
                    'difficulty': 'easy',
                    'question': data['question'],
                    'reference': data['answer']
                })
    
    # Process hard questions  
    hard_dir = toolqa_dir / "data" / "questions" / "hard"
    for jsonl_file in hard_dir.glob("*.jsonl"):
        domain = jsonl_file.stem.replace("-hard", "")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                questions.append({
                    'qid': data['qid'],
                    'domain': domain,
                    'difficulty': 'hard',
                    'question': data['question'],
                    'reference': data['answer']
                })
    
    return questions

def create_deterministic_subsets(questions, seed=42):
    """Create deterministic subsets with reproducible shuffling."""
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle questions deterministically
    shuffled_questions = questions.copy()
    random.shuffle(shuffled_questions)
    
    # Create subsets
    subsets = {
        'toolqa_deterministic_100': shuffled_questions[:100],
        'toolqa_deterministic_500': shuffled_questions[:500],
        'toolqa_deterministic_1000': shuffled_questions[:1000]
    }
    
    return subsets, shuffled_questions

def save_csv_datasets(subsets, output_dir: Path):
    """Save subsets as CSV files compatible with the experiment runner."""
    output_dir.mkdir(exist_ok=True)
    
    saved_files = []
    
    for name, questions in subsets.items():
        df = pd.DataFrame(questions)
        
        # Reorder columns to match expected format
        df = df[['question', 'reference', 'domain', 'difficulty', 'qid']]
        
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        saved_files.append(csv_path)
        
        print(f"✓ Created {csv_path} with {len(df)} questions")
        
        # Show domain distribution
        domain_counts = df['domain'].value_counts()
        print(f"  Domain distribution: {dict(domain_counts)}")
        
        # Show difficulty distribution  
        difficulty_counts = df['difficulty'].value_counts()
        print(f"  Difficulty distribution: {dict(difficulty_counts)}")
        print()
    
    return saved_files

def save_json_datasets(subsets, output_dir: Path):
    """Save subsets as JSON files for completeness."""
    json_files = []
    
    for name, questions in subsets.items():
        json_path = output_dir / f"{name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        json_files.append(json_path)
        print(f"✓ Created {json_path}")
    
    return json_files

def create_mapping_file(subsets, all_questions, output_dir: Path):
    """Create a mapping file describing the subsets."""
    mapping = {
        'metadata': {
            'creation_date': pd.Timestamp.now().isoformat(),
            'total_questions_available': len(all_questions),
            'seed_used': 42,
            'source': 'https://github.com/night-chen/ToolQA',
            'description': 'Deterministic subsets of ToolQA dataset for reproducible experiments'
        },
        'subsets': {}
    }
    
    for name, questions in subsets.items():
        df = pd.DataFrame(questions)
        domain_counts = df['domain'].value_counts().to_dict()
        difficulty_counts = df['difficulty'].value_counts().to_dict()
        
        mapping['subsets'][name] = {
            'size': len(questions),
            'domains': domain_counts,
            'difficulties': difficulty_counts,
            'question_ids': [q['qid'] for q in questions[:10]]  # First 10 for reference
        }
    
    # Overall statistics
    all_df = pd.DataFrame(all_questions)
    mapping['overall_statistics'] = {
        'total_questions': len(all_questions),
        'domains': all_df['domain'].value_counts().to_dict(),
        'difficulties': all_df['difficulty'].value_counts().to_dict(),
    }
    
    mapping_path = output_dir / 'toolqa_deterministic_mapping.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"✓ Created {mapping_path}")
    return mapping_path

def main():
    """Main function to create ToolQA datasets."""
    
    # Paths
    base_dir = Path(__file__).parent
    toolqa_dir = base_dir / "ToolQA"
    output_dir = base_dir / "datasets"
    
    print("Creating ToolQA Deterministic Datasets")
    print("=" * 50)
    
    # Check if ToolQA directory exists
    if not toolqa_dir.exists():
        print(f"Error: ToolQA directory not found at {toolqa_dir}")
        print("Please run: git clone https://github.com/night-chen/ToolQA.git")
        return
    
    # Load all ToolQA data
    print("Loading ToolQA data...")
    all_questions = load_toolqa_data(toolqa_dir)
    print(f"Loaded {len(all_questions)} total questions")
    
    # Show domain and difficulty distribution
    df = pd.DataFrame(all_questions)
    print(f"\nOverall domain distribution:")
    for domain, count in df['domain'].value_counts().items():
        print(f"  {domain}: {count}")
    
    print(f"\nOverall difficulty distribution:")
    for difficulty, count in df['difficulty'].value_counts().items():
        print(f"  {difficulty}: {count}")
    
    # Create deterministic subsets
    print(f"\nCreating deterministic subsets (seed=42)...")
    subsets, shuffled_questions = create_deterministic_subsets(all_questions)
    
    # Save CSV datasets
    print(f"\nSaving CSV datasets to {output_dir}/...")
    output_dir.mkdir(exist_ok=True)
    csv_files = save_csv_datasets(subsets, output_dir)
    
    # Save JSON datasets  
    print(f"Saving JSON datasets...")
    json_files = save_json_datasets(subsets, output_dir)
    
    # Create mapping file
    print(f"\nCreating mapping file...")
    mapping_file = create_mapping_file(subsets, all_questions, output_dir)
    
    print("\n" + "=" * 50)
    print("✅ ToolQA dataset creation completed successfully!")
    print(f"\nFiles created:")
    for file_path in csv_files + json_files + [mapping_file]:
        print(f"  - {file_path}")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Total questions available: {len(all_questions)}")
    print(f"  - Subsets created: {', '.join(subsets.keys())}")
    print(f"  - Domains: {len(df['domain'].unique())} ({', '.join(sorted(df['domain'].unique()))})")
    print(f"  - Difficulties: {', '.join(sorted(df['difficulty'].unique()))}")
    
    # Sample a few questions for verification
    print(f"\n🔍 Sample questions from 100-sample subset:")
    sample_df = pd.DataFrame(subsets['toolqa_deterministic_100'])
    for i in range(3):
        q = sample_df.iloc[i]
        print(f"  {i+1}. [{q['domain']}-{q['difficulty']}] {q['question']}")
        print(f"     Answer: {q['reference']}")
    
    print(f"\n🎯 Ready for experiments! Use these CSV files with the experiment runner.")

if __name__ == "__main__":
    main()