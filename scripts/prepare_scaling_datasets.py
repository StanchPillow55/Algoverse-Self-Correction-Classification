#!/usr/bin/env python3
"""
Prepare Datasets for Scaling Study

Creates sample datasets for the scaling study using your existing data
and generates additional samples for ToolQA, SuperGLUE, and MathBench.
"""

import os
import json
import csv
import random
from pathlib import Path

def create_toolqa_dataset(output_path, num_samples=100):
    """Create a ToolQA-style dataset for tool usage evaluation."""
    
    # Sample tool usage questions
    questions = [
        {
            "id": "toolqa_1",
            "question": "What is the current weather in New York?",
            "answer": "I need to use a weather API to get current conditions.",
            "tools": ["weather_api"],
            "reasoning": "This requires real-time data that I don't have access to.",
            "difficulty": "easy"
        },
        {
            "id": "toolqa_2", 
            "question": "Calculate the square root of 144",
            "answer": "12",
            "tools": ["calculator"],
            "reasoning": "I can calculate this mathematically.",
            "difficulty": "easy"
        },
        {
            "id": "toolqa_3",
            "question": "What is the latest news about AI?",
            "answer": "I need to search recent news sources for current information.",
            "tools": ["news_api", "web_search"],
            "reasoning": "This requires access to current news data.",
            "difficulty": "medium"
        },
        {
            "id": "toolqa_4",
            "question": "Translate 'Hello' to Spanish",
            "answer": "Hola",
            "tools": ["translation_api"],
            "reasoning": "I can provide a basic translation.",
            "difficulty": "easy"
        },
        {
            "id": "toolqa_5",
            "question": "What is the stock price of Apple?",
            "answer": "I need to check current stock market data.",
            "tools": ["stock_api"],
            "reasoning": "Stock prices change constantly and require real-time data.",
            "difficulty": "medium"
        }
    ]
    
    # Generate more samples
    all_samples = []
    for i in range(num_samples):
        base_question = questions[i % len(questions)]
        sample = base_question.copy()
        sample["id"] = f"toolqa_{i+1}"
        all_samples.append(sample)
    
    # Save as JSON
    dataset = {
        "name": "ToolQA",
        "description": "QA dataset for tool usage evaluation",
        "samples": all_samples
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created ToolQA dataset with {len(all_samples)} samples")

def create_superglue_dataset(output_path, num_samples=100):
    """Create a SuperGLUE-style reasoning dataset."""
    
    # Sample reasoning problems
    problems = [
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
        },
        {
            "id": "wic_1",
            "task": "WiC",
            "question": "Does 'bank' mean the same thing in both sentences?",
            "answer": "No",
            "sentence1": "I went to the bank to deposit money.",
            "sentence2": "The river bank was muddy.",
            "difficulty": "medium"
        },
        {
            "id": "wsc_1",
            "task": "WSC",
            "question": "What does 'it' refer to in: The trophy doesn't fit in the brown suitcase because it's too big.",
            "answer": "trophy",
            "context": "The trophy doesn't fit in the brown suitcase because it's too big.",
            "difficulty": "hard"
        }
    ]
    
    # Generate more samples
    all_samples = []
    for i in range(num_samples):
        base_problem = problems[i % len(problems)]
        sample = base_problem.copy()
        sample["id"] = f"{base_problem['task'].lower()}_{i+1}"
        all_samples.append(sample)
    
    # Save as JSON
    dataset = {
        "name": "SuperGLUE",
        "description": "Reasoning benchmark with multiple tasks",
        "samples": all_samples
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created SuperGLUE dataset with {len(all_samples)} samples")

def create_mathbench_dataset(output_path, num_samples=100):
    """Create a MathBench-style hierarchical math dataset."""
    
    # Sample math problems by level
    problems_by_level = {
        "elementary": [
            {"question": "What is 15 + 27?", "answer": "42", "level": "elementary", "topic": "arithmetic"},
            {"question": "If a pizza has 8 slices and you eat 3, how many are left?", "answer": "5", "level": "elementary", "topic": "word_problems"},
            {"question": "What is 6 × 7?", "answer": "42", "level": "elementary", "topic": "multiplication"},
            {"question": "What is 100 ÷ 4?", "answer": "25", "level": "elementary", "topic": "division"}
        ],
        "middle": [
            {"question": "Solve for x: 2x + 5 = 13", "answer": "x = 4", "level": "middle", "topic": "algebra"},
            {"question": "What is the area of a rectangle with length 6 and width 4?", "answer": "24", "level": "middle", "topic": "geometry"},
            {"question": "What is 3² + 4²?", "answer": "25", "level": "middle", "topic": "exponents"},
            {"question": "What is 20% of 150?", "answer": "30", "level": "middle", "topic": "percentages"}
        ],
        "high": [
            {"question": "Find the derivative of x² + 3x + 2", "answer": "2x + 3", "level": "high", "topic": "calculus"},
            {"question": "Solve the quadratic equation: x² - 5x + 6 = 0", "answer": "x = 2 or x = 3", "level": "high", "topic": "algebra"},
            {"question": "What is sin(30°)?", "answer": "0.5", "level": "high", "topic": "trigonometry"},
            {"question": "What is log₂(8)?", "answer": "3", "level": "high", "topic": "logarithms"}
        ],
        "college": [
            {"question": "Evaluate the integral ∫(2x + 1)dx from 0 to 2", "answer": "6", "level": "college", "topic": "calculus"},
            {"question": "Find the eigenvalues of the matrix [[2, 1], [1, 2]]", "answer": "λ₁ = 3, λ₂ = 1", "level": "college", "topic": "linear_algebra"},
            {"question": "What is the limit as x approaches 0 of (sin x)/x?", "answer": "1", "level": "college", "topic": "calculus"},
            {"question": "Solve the differential equation dy/dx = 2y", "answer": "y = Ce^(2x)", "level": "college", "topic": "differential_equations"}
        ]
    }
    
    # Generate samples
    all_samples = []
    samples_per_level = num_samples // 4  # Distribute evenly across levels
    
    for level, problems in problems_by_level.items():
        for i in range(samples_per_level):
            base_problem = problems[i % len(problems)]
            sample = base_problem.copy()
            sample["id"] = f"{level}_{i+1}"
            all_samples.append(sample)
    
    # Save as JSON
    dataset = {
        "name": "MathBench",
        "description": "Hierarchical math reasoning benchmark",
        "samples": all_samples
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created MathBench dataset with {len(all_samples)} samples")

def create_sample_csvs(output_dir, num_samples=100):
    """Create CSV versions of the datasets for your existing pipeline."""
    
    # ToolQA CSV
    toolqa_data = []
    for i in range(num_samples):
        toolqa_data.append({
            "qid": f"toolqa_{i+1}",
            "question": f"Sample tool question {i+1}",
            "ground_truth": f"Sample answer {i+1}",
            "topic": "tools"
        })
    
    toolqa_csv = output_dir / "toolqa_sample.csv"
    with open(toolqa_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "question", "ground_truth", "topic"])
        writer.writeheader()
        writer.writerows(toolqa_data)
    
    print(f"Created ToolQA CSV with {len(toolqa_data)} samples")
    
    # SuperGLUE CSV
    superglue_data = []
    for i in range(num_samples):
        superglue_data.append({
            "qid": f"superglue_{i+1}",
            "question": f"Sample reasoning question {i+1}",
            "ground_truth": f"Sample reasoning answer {i+1}",
            "topic": "reasoning"
        })
    
    superglue_csv = output_dir / "superglue_sample.csv"
    with open(superglue_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "question", "ground_truth", "topic"])
        writer.writeheader()
        writer.writerows(superglue_data)
    
    print(f"Created SuperGLUE CSV with {len(superglue_data)} samples")
    
    # MathBench CSV
    mathbench_data = []
    for i in range(num_samples):
        mathbench_data.append({
            "qid": f"mathbench_{i+1}",
            "question": f"Sample math problem {i+1}",
            "ground_truth": f"Sample math answer {i+1}",
            "topic": "math"
        })
    
    mathbench_csv = output_dir / "mathbench_sample.csv"
    with open(mathbench_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "question", "ground_truth", "topic"])
        writer.writeheader()
        writer.writerows(mathbench_data)
    
    print(f"Created MathBench CSV with {len(mathbench_data)} samples")
    
    return [toolqa_csv, superglue_csv, mathbench_csv]

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for scaling study")
    parser.add_argument("--output-dir", default="data/scaling",
                       help="Output directory for datasets")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples per dataset")
    parser.add_argument("--create-csvs", action="store_true",
                       help="Also create CSV versions for existing pipeline")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Preparing Scaling Study Datasets")
    print("=" * 40)
    print(f"Output directory: {output_dir}")
    print(f"Samples per dataset: {args.num_samples}")
    print()
    
    # Create JSON datasets
    create_toolqa_dataset(output_dir / "toolqa.json", args.num_samples)
    create_superglue_dataset(output_dir / "superglue.json", args.num_samples)
    create_mathbench_dataset(output_dir / "mathbench.json", args.num_samples)
    
    # Create CSV versions if requested
    if args.create_csvs:
        print()
        print("Creating CSV versions for existing pipeline...")
        csv_files = create_sample_csvs(output_dir, args.num_samples)
        print(f"CSV files created: {[str(f) for f in csv_files]}")
    
    print()
    print("✅ Dataset preparation complete!")
    print()
    print("Next steps:")
    print("1. Set up API keys in .env file")
    print("2. Run Phase 1 validation:")
    print("   python scripts/run_scaling_simple.py --dataset data/scaling/toolqa_sample.csv --phase 1")
    print("3. Run Phase 2 medium scale:")
    print("   python scripts/run_scaling_simple.py --dataset data/scaling/superglue_sample.csv --phase 2")

if __name__ == "__main__":
    import argparse
    main()
