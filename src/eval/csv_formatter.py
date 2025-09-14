#!/usr/bin/env python3
"""
CSV Formatter for Reasoning Traces and Accuracy Results

Formats experimental results into CSV files for analysis, preserving both
reasoning traces and accuracy metrics across multiple turns.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class ReasoningCSVFormatter:
    """Formats reasoning traces and accuracy results into CSV files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_experiment_results(self, traces: List[Dict[str, Any]], 
                                 experiment_config: Dict[str, Any],
                                 output_filename: str = None) -> Path:
        """
        Format complete experiment results into CSV.
        
        Args:
            traces: List of problem traces with turns
            experiment_config: Configuration used for experiment
            output_filename: Custom filename for CSV output
            
        Returns:
            Path to generated CSV file
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = experiment_config.get('dataset_name', 'unknown')
            model = experiment_config.get('model', 'unknown')
            output_filename = f"{dataset_name}_{model}_results_{timestamp}.csv"
        
        csv_path = self.output_dir / output_filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                'problem_id', 'dataset', 'model', 'provider', 'temperature',
                'turn', 'max_turns', 'template', 'reasoning_trace_file',
                'extracted_answer', 'reference_answer', 'accuracy', 
                'self_confidence', 'teacher_bias', 'teacher_confidence',
                'combined_confidence', 'reasoning_summary', 'execution_details',
                'final_accuracy', 'total_turns', 'experiment_config'
            ]
            writer.writerow(header)
            
            # Write data rows
            for trace in traces:
                problem_id = trace.get('qid', '')
                reference = trace.get('reference', '')
                final_accuracy = trace.get('final_accuracy', 0)
                total_turns = len(trace.get('turns', []))
                
                for turn_idx, turn in enumerate(trace.get('turns', [])):
                    row = [
                        problem_id,
                        experiment_config.get('dataset_name', ''),
                        experiment_config.get('model', ''),
                        experiment_config.get('provider', ''),
                        experiment_config.get('temperature', ''),
                        turn_idx,
                        experiment_config.get('max_turns', ''),
                        turn.get('template', ''),
                        turn.get('reasoning_trace_file', ''),  # Path to .txt file
                        turn.get('answer', ''),
                        reference,
                        turn.get('accuracy', 0),
                        turn.get('self_conf', 0),
                        turn.get('teacher_bias', ''),
                        turn.get('teacher_conf', 0),
                        turn.get('combined_confidence', 0),
                        turn.get('reasoning_summary', ''),
                        json.dumps(turn.get('execution_details', {})),
                        final_accuracy,
                        total_turns,
                        json.dumps(experiment_config)
                    ]
                    writer.writerow(row)
        
        print(f"✅ Results formatted to CSV: {csv_path}")
        return csv_path
    
    def format_summary_results(self, traces: List[Dict[str, Any]], 
                              experiment_config: Dict[str, Any],
                              output_filename: str = None) -> Path:
        """
        Format summary results (one row per problem) into CSV.
        
        Args:
            traces: List of problem traces
            experiment_config: Configuration used for experiment  
            output_filename: Custom filename for summary CSV
            
        Returns:
            Path to generated summary CSV file
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = experiment_config.get('dataset_name', 'unknown')
            model = experiment_config.get('model', 'unknown')
            output_filename = f"{dataset_name}_{model}_summary_{timestamp}.csv"
        
        csv_path = self.output_dir / output_filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                'problem_id', 'dataset', 'model', 'provider', 'temperature',
                'question', 'reference_answer', 'final_answer', 'final_accuracy',
                'total_turns', 'initial_accuracy', 'improvement',
                'reasoning_trace_files', 'templates_used', 'biases_detected',
                'final_confidence', 'experiment_config'
            ]
            writer.writerow(header)
            
            # Write summary rows
            for trace in traces:
                turns = trace.get('turns', [])
                if not turns:
                    continue
                    
                problem_id = trace.get('qid', '')
                question = trace.get('question', '')
                reference = trace.get('reference', '')
                final_accuracy = trace.get('final_accuracy', 0)
                total_turns = len(turns)
                initial_accuracy = turns[0].get('accuracy', 0) if turns else 0
                improvement = final_accuracy - initial_accuracy
                
                # Collect trace files and templates
                trace_files = [turn.get('reasoning_trace_file', '') for turn in turns]
                templates = [turn.get('template', '') for turn in turns if turn.get('template')]
                biases = [turn.get('teacher_bias', '') for turn in turns if turn.get('teacher_bias')]
                final_confidence = turns[-1].get('combined_confidence', 0) if turns else 0
                
                row = [
                    problem_id,
                    experiment_config.get('dataset_name', ''),
                    experiment_config.get('model', ''),
                    experiment_config.get('provider', ''),
                    experiment_config.get('temperature', ''),
                    question,
                    reference,
                    turns[-1].get('answer', '') if turns else '',
                    final_accuracy,
                    total_turns,
                    initial_accuracy,
                    improvement,
                    '|'.join(filter(None, trace_files)),
                    '|'.join(filter(None, templates)),
                    '|'.join(filter(None, biases)),
                    final_confidence,
                    json.dumps(experiment_config)
                ]
                writer.writerow(row)
        
        print(f"✅ Summary formatted to CSV: {csv_path}")
        return csv_path
    
    def format_turn_analysis(self, traces: List[Dict[str, Any]],
                           output_filename: str = None) -> Path:
        """
        Format turn-by-turn analysis for multi-turn accuracy insights.
        
        Args:
            traces: List of problem traces
            output_filename: Custom filename for turn analysis CSV
            
        Returns:
            Path to generated turn analysis CSV file
        """
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"turn_analysis_{timestamp}.csv"
        
        csv_path = self.output_dir / output_filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                'turn', 'total_problems', 'correct_problems', 'accuracy_rate',
                'improvement_from_prev', 'cumulative_improvement',
                'avg_confidence', 'most_common_bias', 'most_common_template'
            ]
            writer.writerow(header)
            
            # Calculate turn-by-turn statistics
            max_turns = max(len(trace.get('turns', [])) for trace in traces)
            
            for turn_idx in range(max_turns):
                turn_data = []
                confidences = []
                biases = []
                templates = []
                
                for trace in traces:
                    turns = trace.get('turns', [])
                    if len(turns) > turn_idx:
                        turn = turns[turn_idx]
                        turn_data.append(turn)
                        
                        if turn.get('combined_confidence'):
                            confidences.append(turn['combined_confidence'])
                        if turn.get('teacher_bias'):
                            biases.append(turn['teacher_bias'])
                        if turn.get('template'):
                            templates.append(turn['template'])
                
                if not turn_data:
                    continue
                
                total_problems = len(turn_data)
                correct_problems = sum(1 for turn in turn_data if turn.get('accuracy', 0) == 1)
                accuracy_rate = correct_problems / total_problems if total_problems > 0 else 0
                
                # Calculate improvement metrics
                improvement_from_prev = 0
                cumulative_improvement = 0
                if turn_idx > 0:
                    prev_correct = sum(1 for trace in traces 
                                     if len(trace.get('turns', [])) > turn_idx - 1 
                                     and trace['turns'][turn_idx - 1].get('accuracy', 0) == 1)
                    prev_total = sum(1 for trace in traces 
                                   if len(trace.get('turns', [])) > turn_idx - 1)
                    prev_accuracy = prev_correct / prev_total if prev_total > 0 else 0
                    improvement_from_prev = accuracy_rate - prev_accuracy
                    
                    # Cumulative improvement from turn 0
                    initial_correct = sum(1 for trace in traces 
                                        if len(trace.get('turns', [])) > 0 
                                        and trace['turns'][0].get('accuracy', 0) == 1)
                    initial_total = sum(1 for trace in traces 
                                      if len(trace.get('turns', [])) > 0)
                    initial_accuracy = initial_correct / initial_total if initial_total > 0 else 0
                    cumulative_improvement = accuracy_rate - initial_accuracy
                
                # Calculate aggregates
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                most_common_bias = max(set(biases), key=biases.count) if biases else ''
                most_common_template = max(set(templates), key=templates.count) if templates else ''
                
                row = [
                    turn_idx,
                    total_problems,
                    correct_problems,
                    round(accuracy_rate, 4),
                    round(improvement_from_prev, 4),
                    round(cumulative_improvement, 4),
                    round(avg_confidence, 3),
                    most_common_bias,
                    most_common_template
                ]
                writer.writerow(row)
        
        print(f"✅ Turn analysis formatted to CSV: {csv_path}")
        return csv_path


def create_analysis_dashboard(csv_files: List[Path], output_dir: Path) -> Path:
    """
    Create a simple analysis dashboard with key metrics.
    
    Args:
        csv_files: List of generated CSV file paths
        output_dir: Output directory for dashboard
        
    Returns:
        Path to dashboard file
    """
    dashboard_path = output_dir / "analysis_dashboard.txt"
    
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write("📊 REASONING TRACES ANALYSIS DASHBOARD\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("📁 Generated Files:\n")
        for csv_file in csv_files:
            f.write(f"  • {csv_file.name}\n")
        f.write("\n")
        
        f.write("📋 Analysis Instructions:\n")
        f.write("1. Results CSV: Complete turn-by-turn data with reasoning traces\n")
        f.write("2. Summary CSV: One row per problem with final results\n") 
        f.write("3. Turn Analysis CSV: Turn-by-turn accuracy progression\n\n")
        
        f.write("📂 Reasoning Traces Location:\n")
        f.write("  • Full reasoning traces saved in: reasoning_traces/\n")
        f.write("  • Math problems: reasoning_traces/math/{problem_id}/turn_{n}_reasoning.txt\n")
        f.write("  • Code problems: reasoning_traces/code/{problem_id}/turn_{n}_reasoning.txt\n\n")
        
        f.write("🔍 Key Metrics to Analyze:\n")
        f.write("  • Multi-turn accuracy improvement\n")
        f.write("  • Template effectiveness by bias type\n")
        f.write("  • Confidence calibration\n")
        f.write("  • Reasoning quality vs accuracy correlation\n")
        
        f.write("\n" + "=" * 50)
        f.write(f"\nDashboard generated: {datetime.now()}")
    
    return dashboard_path


if __name__ == "__main__":
    # Test the CSV formatter
    test_traces = [
        {
            "qid": "test_001",
            "question": "What is 3 + 5?",
            "reference": "8",
            "final_accuracy": 1,
            "turns": [
                {
                    "answer": "8",
                    "accuracy": 1,
                    "self_conf": 0.95,
                    "teacher_bias": "None",
                    "teacher_conf": 0.9,
                    "template": None,
                    "reasoning_trace_file": "reasoning_traces/math/test_001/turn_0_reasoning.txt",
                    "reasoning_summary": "Simple addition problem"
                }
            ]
        }
    ]
    
    test_config = {
        "dataset_name": "gsm8k_sample",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "temperature": 0.2,
        "max_turns": 3
    }
    
    formatter = ReasoningCSVFormatter(Path("outputs/test_csv"))
    
    results_csv = formatter.format_experiment_results(test_traces, test_config)
    summary_csv = formatter.format_summary_results(test_traces, test_config)
    turn_csv = formatter.format_turn_analysis(test_traces)
    
    dashboard = create_analysis_dashboard([results_csv, summary_csv, turn_csv], Path("outputs/test_csv"))
    
    print(f"Test complete. Dashboard: {dashboard}")