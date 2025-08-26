#!/usr/bin/env python3
"""
Download a simpler code generation dataset suitable for the pipeline.
"""
import pandas as pd
import requests
from pathlib import Path

def download_human_eval():
    """Download HumanEval dataset - much simpler than SWEBench"""
    print("Downloading HumanEval dataset...")
    
    # HumanEval is a simpler code generation benchmark
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    
    try:
        # Download and parse
        import gzip
        import json
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the gzipped JSONL
        data = []
        for line in gzip.decompress(response.content).decode('utf-8').split('\n'):
            if line.strip():
                item = json.loads(line)
                data.append({
                    'qid': f"human_eval_{item['task_id']}",
                    'question': item['prompt'],
                    'reference': item['canonical_solution']
                })
        
        print(f"Downloaded {len(data)} examples from HumanEval")
        return data
        
    except Exception as e:
        print(f"Error downloading HumanEval: {e}")
        return None

def download_code_alpaca():
    """Download Code Alpaca dataset - another simple option"""
    print("Downloading Code Alpaca dataset...")
    
    url = "https://raw.githubusercontent.com/sahil280114/CodeAlpaca/master/data/code_alpaca_20k.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Convert to our format
        converted = []
        for i, item in enumerate(data[:100]):  # Take first 100 for testing
            converted.append({
                'qid': f"code_alpaca_{i}",
                'question': item['instruction'],
                'reference': item['output']
            })
        
        print(f"Downloaded {len(converted)} examples from Code Alpaca")
        return converted
        
    except Exception as e:
        print(f"Error downloading Code Alpaca: {e}")
        return None

def create_synthetic_code_dataset():
    """Create a synthetic code dataset for testing"""
    print("Creating synthetic code dataset...")
    
    synthetic_data = [
        {
            'qid': 'synthetic_1',
            'question': 'Write a function to calculate the factorial of a number.',
            'reference': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)'
        },
        {
            'qid': 'synthetic_2',
            'question': 'Create a function to check if a string is a palindrome.',
            'reference': 'def is_palindrome(s):\n    return s == s[::-1]'
        },
        {
            'qid': 'synthetic_3',
            'question': 'Write a function to find the maximum of three numbers.',
            'reference': 'def max_of_three(a, b, c):\n    return max(a, b, c)'
        },
        {
            'qid': 'synthetic_4',
            'question': 'Create a function to count vowels in a string.',
            'reference': 'def count_vowels(s):\n    vowels = "aeiouAEIOU"\n    return sum(1 for char in s if char in vowels)'
        },
        {
            'qid': 'synthetic_5',
            'question': 'Write a function to reverse a list.',
            'reference': 'def reverse_list(lst):\n    return lst[::-1]'
        }
    ]
    
    print(f"Created {len(synthetic_data)} synthetic examples")
    return synthetic_data

def main():
    """Main function to download/create code datasets"""
    Path("data").mkdir(exist_ok=True)
    
    # Try different datasets in order of preference
    datasets = [
        ("HumanEval", download_human_eval),
        ("Code Alpaca", download_code_alpaca),
        ("Synthetic", create_synthetic_code_dataset)
    ]
    
    for name, downloader in datasets:
        print(f"\nTrying {name}...")
        data = downloader()
        if data:
            # Save the dataset
            df = pd.DataFrame(data)
            df.to_csv(f"data/{name.lower().replace(' ', '_')}_code.csv", index=False)
            print(f"✅ Saved {name} dataset to data/{name.lower().replace(' ', '_')}_code.csv")
            
            # Create smaller subsets
            df.head(20).to_csv(f"data/{name.lower().replace(' ', '_')}_code_20.csv", index=False)
            df.head(100).to_csv(f"data/{name.lower().replace(' ', '_')}_code_100.csv", index=False)
            print(f"✅ Created subsets: {name.lower().replace(' ', '_')}_code_20.csv, {name.lower().replace(' ', '_')}_code_100.csv")
            
            # Show sample
            print(f"\nSample from {name}:")
            print(f"Question: {data[0]['question'][:100]}...")
            print(f"Reference: {data[0]['reference'][:100]}...")
            break
        else:
            print(f"❌ Failed to download {name}")
    else:
        print("❌ All dataset downloads failed, using synthetic data only")
        data = create_synthetic_code_dataset()
        df = pd.DataFrame(data)
        df.to_csv("data/synthetic_code.csv", index=False)

if __name__ == "__main__":
    main()