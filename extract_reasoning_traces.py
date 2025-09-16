#!/usr/bin/env python3
"""
Extract reasoning traces from aggregated JSON files and convert to simple .txt format
for easy access by research partners.

This script processes the "good" runs and creates individual .txt files per problem
with the format:
- problem_1.txt
- turn 1
- [Full reasoning trace]
- Feedback1
- turn2
- [Full reasoning trace]
- Feedback2
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

# Define the good runs mapping
GOOD_RUNS = {
    "llama-3-70b": {
        "gsm8k": "20250914T110424Z",
        "superglue": "20250914T033113Z", 
        "toolqa": "20250913T105258Z",
        "mathbench": "20250914T045203Z"
    },
    "gpt-4": {
        "gsm8k": "20250914T071025Z",
        "superglue": "20250914T071025Z",
        "toolqa": "20250913T110235Z", 
        "mathbench": "20250910T133420Z"
    },
    "gpt-4o": {
        "gsm8k": "20250913T145036Z",
        "superglue": "20250913T121131Z",
        "toolqa": "20250913T083152Z",
        "mathbench": "20250910T132536Z"
    },
    "gpt-4o-mini": {
        "gsm8k": "20250913T142510Z",
        "superglue": "20250913T120434Z",
        "toolqa": "20250913T081723Z",
        "mathbench": "20250910T131758Z"
    },
    "claude-haiku": {
        "gsm8k": "20250913T213057Z",
        "superglue": "20250913T122909Z",
        "toolqa": "20250913T082455Z",
        "mathbench": "20250910T135124Z"
    },
    "claude-sonnet": {
        "gsm8k": "20250913T215757Z",
        "superglue": "20250913T124632Z",
        "toolqa": "20250913T084811Z",
        "mathbench": "20250910T142409Z"
    }
}

def find_json_file(model: str, dataset: str) -> str:
    """Find the aggregated JSON file for a given model and dataset."""
    # Check outputs directory first
    outputs_dir = Path("outputs")
    
    # Try different possible locations
    possible_paths = [
        outputs_dir / "phase3_full" / f"{model}_scaling_result.json",
        outputs_dir / "phase3_llama_only" / f"{model}_scaling_result.json", 
        outputs_dir / "phase3_custom" / f"{model}_scaling_result.json",
        outputs_dir / "phase2_medium" / f"{model}_scaling_result.json",
        outputs_dir / "phase1_validation" / f"{model}_scaling_result.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

def extract_reasoning_from_runs(model: str, dataset: str, timestamp: str) -> List[Dict]:
    """Extract reasoning traces from raw runs directory."""
    runs_dir = Path("runs")
    
    # Find the run directory
    run_pattern = f"{timestamp}__*__dev__*{model}*"
    run_dirs = list(runs_dir.glob(f"{timestamp}__*"))
    
    if not run_dirs:
        print(f"Warning: No run directory found for {model} {dataset} {timestamp}")
        return []
    
    run_dir = run_dirs[0]
    traces_file = run_dir / "traces.json"
    
    if not traces_file.exists():
        print(f"Warning: No traces.json found in {run_dir}")
        return []
    
    try:
        with open(traces_file, 'r') as f:
            traces_data = json.load(f)
        
        # Extract items from traces
        items = traces_data.get("items", [])
        processed_traces = []
        
        for item in items:
            qid = item.get("id", "unknown")
            turns = item.get("turns", [])
            
            trace = {
                "qid": qid,
                "question": "",  # Will be filled from prompts if available
                "reference": "",  # Will be filled from prompts if available
                "turns": []
            }
            
            for turn in turns:
                turn_data = {
                    "turn_index": turn.get("turn_index", 0),
                    "answer": turn.get("normalized_answer", ""),
                    "self_conf": turn.get("confidence", 0.0),
                    "teacher_bias": turn.get("evaluator_feedback", {}).get("bias_label", ""),
                    "teacher_conf": turn.get("evaluator_feedback", {}).get("confidence", 0.0),
                    "template": turn.get("prompt_id", ""),
                    "accuracy": 0,  # Will be calculated
                    "execution_details": turn.get("exec_result", {})
                }
                trace["turns"].append(turn_data)
            
            processed_traces.append(trace)
        
        return processed_traces
        
    except Exception as e:
        print(f"Error processing {traces_file}: {e}")
        return []

def get_reasoning_trace_content(run_dir: Path, qid: str, turn_index: int) -> str:
    """Get the actual reasoning trace content from cot.txt files."""
    # Try different possible paths for the cot.txt file
    possible_paths = [
        run_dir / "gsm8k" / qid / f"turn_{turn_index}" / "cot.txt",
        run_dir / "gsm8k" / f"{qid}" / f"turn_{turn_index}" / "cot.txt",
        run_dir / qid / f"turn_{turn_index}" / "cot.txt"
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    content = f.read().strip()
                    if content and content != "ERROR_EMPTY_RESPONSE":
                        return content
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
    
    return ""

def format_reasoning_trace(trace: Dict, run_dir: Path = None) -> str:
    """Format a single reasoning trace into the desired .txt format."""
    output = []
    
    # Add question
    if trace.get("question"):
        output.append(f"Question: {trace['question']}")
        output.append("")
    
    # Add reference answer if available
    if trace.get("reference"):
        output.append(f"Reference Answer: {trace['reference']}")
        output.append("")
    
    # Process each turn
    for i, turn in enumerate(trace.get("turns", []), 1):
        output.append(f"Turn {i}")
        output.append("-" * 20)
        
        # Get the actual reasoning trace content
        if run_dir:
            # Try to get turn_index, fallback to i-1 if not available
            turn_index = turn.get("turn_index", i-1)
            reasoning_content = get_reasoning_trace_content(run_dir, trace["qid"], turn_index)
            if reasoning_content:
                output.append(reasoning_content)
            else:
                output.append(f"Final Answer: {turn.get('answer', 'N/A')}")
        else:
            output.append(f"Final Answer: {turn.get('answer', 'N/A')}")
        
        output.append("")
        
        # Add feedback information
        if turn.get("teacher_bias"):
            output.append(f"Teacher Bias: {turn['teacher_bias']}")
        if turn.get("teacher_conf"):
            output.append(f"Teacher Confidence: {turn['teacher_conf']}")
        if turn.get("template"):
            output.append(f"Template Used: {turn['template']}")
        if turn.get("accuracy") is not None:
            output.append(f"Accuracy: {turn['accuracy']}")
        
        output.append("")
    
    return "\n".join(output)

def process_model_dataset(model: str, dataset: str, timestamp: str, output_dir: Path):
    """Process reasoning traces for a specific model-dataset combination."""
    print(f"Processing {model} on {dataset}...")
    
    # Try to find aggregated JSON file first
    json_file = find_json_file(model, dataset)
    traces = []
    
    if json_file:
        print(f"  Found aggregated JSON: {json_file}")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            traces = data.get("traces", [])
        except Exception as e:
            print(f"  Error reading JSON file: {e}")
    
    # If no aggregated JSON, try to extract from raw runs
    if not traces:
        print(f"  Extracting from raw runs...")
        traces = extract_reasoning_from_runs(model, dataset, timestamp)
    
    if not traces:
        print(f"  Warning: No traces found for {model} on {dataset}")
        return
    
    # Create output directory for this model-dataset combination
    model_output_dir = output_dir / f"{model}_{dataset}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the run directory for getting actual reasoning content
    runs_dir = Path("runs")
    run_dirs = list(runs_dir.glob(f"{timestamp}__*"))
    run_dir = run_dirs[0] if run_dirs else None
    
    # Process each trace
    for i, trace in enumerate(traces, 1):
        qid = trace.get("qid", f"problem_{i}")
        
        # Format the reasoning trace
        formatted_trace = format_reasoning_trace(trace, run_dir)
        
        # Write to file
        output_file = model_output_dir / f"problem_{qid}.txt"
        with open(output_file, 'w') as f:
            f.write(formatted_trace)
    
    print(f"  Created {len(traces)} reasoning trace files in {model_output_dir}")

def main():
    """Main function to process all good runs."""
    output_dir = Path("reasoning_traces")
    output_dir.mkdir(exist_ok=True)
    
    print("Extracting reasoning traces for all good runs...")
    print("=" * 60)
    
    total_processed = 0
    
    for model, datasets in GOOD_RUNS.items():
        print(f"\nProcessing {model}:")
        print("-" * 40)
        
        for dataset, timestamp in datasets.items():
            try:
                process_model_dataset(model, dataset, timestamp, output_dir)
                total_processed += 1
            except Exception as e:
                print(f"  Error processing {model} {dataset}: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"Processing complete! Created reasoning traces for {total_processed} model-dataset combinations.")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create a summary file
    summary_file = output_dir / "README.md"
    with open(summary_file, 'w') as f:
        f.write("# Reasoning Traces for Good Runs\n\n")
        f.write("This directory contains reasoning traces for the 'good' experimental runs.\n\n")
        f.write("## Structure\n\n")
        f.write("Each model-dataset combination has its own directory:\n")
        f.write("- `{model}_{dataset}/` - Contains individual problem files\n")
        f.write("- `problem_{qid}.txt` - Individual reasoning trace for each problem\n\n")
        f.write("## Format\n\n")
        f.write("Each problem file contains:\n")
        f.write("- Question text\n")
        f.write("- Reference answer (if available)\n")
        f.write("- Turn-by-turn reasoning traces\n")
        f.write("- Teacher feedback and bias information\n")
        f.write("- Template usage information\n\n")
        f.write("## Models and Datasets Processed\n\n")
        for model, datasets in GOOD_RUNS.items():
            f.write(f"### {model}\n")
            for dataset in datasets.keys():
                f.write(f"- {dataset}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
