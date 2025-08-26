#!/usr/bin/env python3
"""
Download SWEBench dataset and convert to CSV format.
"""
import pandas as pd
from datasets import load_dataset
from pathlib import Path

def download_swebench():
    """Download SWEBench dataset and convert to CSV"""
    print("Loading SWEBench dataset from Hugging Face...")
    
    # Load the dataset (this will download it if not cached)
    dataset = load_dataset("princeton-nlp/SWE-bench")
    
    print(f"Dataset loaded. Available splits: {list(dataset.keys())}")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset['test'])  # Use test split for evaluation
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Rename columns to match your expected format
    # SWEBench has 'instance_id', 'repo', 'base_commit', 'test_patch', 'test_file', 'problem_statement'
    # We'll use 'problem_statement' as the question and create a simple answer column
    
    # Create the CSV with expected structure
    output_df = pd.DataFrame({
        'qid': df['instance_id'],
        'question': df['problem_statement'],
        'reference': df['test_patch']  # Use test_patch as reference answer
    })
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Save full dataset
    output_df.to_csv("data/swebench_full.csv", index=False)
    print(f"Saved full dataset to data/swebench_full.csv ({len(output_df)} examples)")
    
    # Create smaller subsets for testing
    subset_20 = output_df.head(20)
    subset_100 = output_df.head(100)
    
    subset_20.to_csv("data/swebench_20.csv", index=False)
    subset_100.to_csv("data/swebench_100.csv", index=False)
    
    print(f"Created subsets: swebench_20.csv ({len(subset_20)} examples), swebench_100.csv ({len(subset_100)} examples)")
    
    # Show sample data
    print("\nSample data:")
    print(output_df.head(2).to_string())
    
    return output_df

if __name__ == "__main__":
    download_swebench()