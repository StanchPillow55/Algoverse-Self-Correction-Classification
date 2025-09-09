"""
Trace Formatting System for Scaling Study

Separates full traces from accuracy traces and formats them for easy analysis.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TraceFormatter:
    """Formats traces for easy analysis and visualization."""
    
    def __init__(self, output_dir: str = "outputs/traces_formatted"):
        """Initialize trace formatter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_experiment_traces(self, traces_file: str, experiment_id: str) -> Dict[str, str]:
        """Format traces from a single experiment."""
        traces_path = Path(traces_file)
        if not traces_path.exists():
            logger.error(f"Traces file not found: {traces_file}")
            return {}
        
        # Load traces
        traces = self._load_traces(traces_path)
        if not traces:
            logger.error(f"No traces found in {traces_file}")
            return {}
        
        # Create formatted outputs
        outputs = {}
        
        # 1. Full traces (detailed)
        full_traces_file = self.output_dir / f"{experiment_id}_full_traces.json"
        self._save_full_traces(traces, full_traces_file)
        outputs["full_traces"] = str(full_traces_file)
        
        # 2. Accuracy traces (multi-turn summary)
        accuracy_traces_file = self.output_dir / f"{experiment_id}_accuracy_traces.json"
        self._save_accuracy_traces(traces, accuracy_traces_file)
        outputs["accuracy_traces"] = str(accuracy_traces_file)
        
        # 3. CSV summary for easy analysis
        csv_file = self.output_dir / f"{experiment_id}_summary.csv"
        self._save_csv_summary(traces, csv_file)
        outputs["csv_summary"] = str(csv_file)
        
        # 4. Multi-turn accuracy breakdown
        multi_turn_file = self.output_dir / f"{experiment_id}_multi_turn_accuracy.json"
        self._save_multi_turn_accuracy(traces, multi_turn_file)
        outputs["multi_turn_accuracy"] = str(multi_turn_file)
        
        logger.info(f"Formatted traces for {experiment_id}: {len(outputs)} files created")
        return outputs
    
    def _load_traces(self, traces_path: Path) -> List[Dict[str, Any]]:
        """Load traces from JSON or JSONL file."""
        traces = []
        
        # Try to load as JSON first (regardless of extension)
        try:
            with open(traces_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    traces = data
                elif isinstance(data, dict) and 'traces' in data:
                    traces = data['traces']
                elif isinstance(data, dict) and 'summary' in data:
                    # Handle summary format - extract traces if available
                    traces = data.get('traces', [])
        except json.JSONDecodeError:
            # If JSON fails, try JSONL
            if traces_path.suffix == '.jsonl':
                with open(traces_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                traces.append(json.loads(line))
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
        
        return traces
    
    def _save_full_traces(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save full detailed traces."""
        full_traces = {
            "experiment_metadata": {
                "total_samples": len(traces),
                "format_version": "1.0",
                "description": "Full detailed traces with all turn information"
            },
            "traces": traces
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_traces, f, indent=2)
    
    def _save_accuracy_traces(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save accuracy-focused traces with multi-turn summary."""
        accuracy_traces = {
            "experiment_metadata": {
                "total_samples": len(traces),
                "format_version": "1.0",
                "description": "Accuracy traces with multi-turn summary"
            },
            "samples": []
        }
        
        for trace in traces:
            sample_id = trace.get('problem_id', 'unknown')
            turns = trace.get('turns', [])
            
            # Extract accuracy per turn (handle different trace formats)
            turn_accuracies = []
            for i, turn in enumerate(turns):
                # Handle different field names for accuracy
                accuracy = 0
                if 'is_correct' in turn:
                    accuracy = 1 if turn['is_correct'] else 0
                elif 'accuracy' in turn:
                    accuracy = turn['accuracy']
                
                # Handle different field names for confidence
                confidence = 0.0
                if 'model_reported_confidence' in turn:
                    confidence = turn['model_reported_confidence']
                elif 'confidence' in turn:
                    confidence = turn['confidence']
                
                turn_accuracies.append({
                    "turn": i,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "answer": turn.get('response_text', turn.get('answer', '')),
                    "feedback": turn.get('evaluator_feedback', turn.get('feedback', '')),
                    "bias_label": turn.get('evaluator_bias_label', ''),
                    "is_final": turn.get('response_is_final', False)
                })
            
            # Calculate summary metrics
            initial_accuracy = turn_accuracies[0]['accuracy'] if turn_accuracies else 0
            final_accuracy = trace.get('final_correct', 0)
            if final_accuracy == 0 and turn_accuracies:
                final_accuracy = turn_accuracies[-1]['accuracy']
            improvement = final_accuracy - initial_accuracy
            
            sample_data = {
                "sample_id": sample_id,
                "initial_accuracy": initial_accuracy,
                "final_accuracy": final_accuracy,
                "improvement": improvement,
                "total_turns": len(turns),
                "turn_accuracies": turn_accuracies,
                "metadata": {
                    "question": trace.get('original_problem_text', trace.get('question', '')),
                    "reference": trace.get('reference', ''),
                    "model": trace.get('model_name', trace.get('model', 'unknown')),
                    "dataset": trace.get('dataset', 'unknown'),
                    "run_id": trace.get('run_id', 'unknown')
                }
            }
            
            accuracy_traces["samples"].append(sample_data)
        
        with open(output_file, 'w') as f:
            json.dump(accuracy_traces, f, indent=2)
    
    def _save_csv_summary(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save CSV summary for easy analysis."""
        rows = []
        
        for trace in traces:
            sample_id = trace.get('problem_id', 'unknown')
            turns = trace.get('turns', [])
            
            # Basic info
            row = {
                'sample_id': sample_id,
                'model': trace.get('model', 'unknown'),
                'dataset': trace.get('dataset', 'unknown'),
                'total_turns': len(turns),
                'question': trace.get('question', '')[:100] + '...' if len(trace.get('question', '')) > 100 else trace.get('question', ''),
                'reference': trace.get('reference', '')
            }
            
            # Per-turn accuracy
            for i, turn in enumerate(turns):
                row[f'turn_{i}_accuracy'] = turn.get('accuracy', 0)
                row[f'turn_{i}_confidence'] = turn.get('confidence', 0.0)
                row[f'turn_{i}_answer'] = str(turn.get('answer', ''))[:50] + '...' if len(str(turn.get('answer', ''))) > 50 else str(turn.get('answer', ''))
            
            # Summary metrics
            if turns:
                row['initial_accuracy'] = turns[0].get('accuracy', 0)
                row['final_accuracy'] = turns[-1].get('accuracy', 0)
                row['improvement'] = row['final_accuracy'] - row['initial_accuracy']
            else:
                row['initial_accuracy'] = 0
                row['final_accuracy'] = 0
                row['improvement'] = 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def _save_multi_turn_accuracy(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save multi-turn accuracy breakdown."""
        multi_turn_data = {
            "experiment_metadata": {
                "total_samples": len(traces),
                "format_version": "1.0",
                "description": "Multi-turn accuracy breakdown for scaling analysis"
            },
            "accuracy_by_turn": {},
            "improvement_analysis": {},
            "model_performance": {}
        }
        
        # Calculate accuracy by turn
        max_turns = max(len(trace.get('turns', [])) for trace in traces)
        
        for turn_num in range(max_turns):
            turn_accuracies = []
            for trace in traces:
                turns = trace.get('turns', [])
                if turn_num < len(turns):
                    turn_accuracies.append(turns[turn_num].get('accuracy', 0))
            
            if turn_accuracies:
                multi_turn_data["accuracy_by_turn"][f"turn_{turn_num}"] = {
                    "accuracy": sum(turn_accuracies) / len(turn_accuracies),
                    "count": len(turn_accuracies),
                    "correct": sum(turn_accuracies),
                    "total": len(turn_accuracies)
                }
        
        # Calculate improvement analysis
        improvements = []
        for trace in traces:
            turns = trace.get('turns', [])
            if len(turns) >= 2:
                initial = turns[0].get('accuracy', 0)
                final = turns[-1].get('accuracy', 0)
                improvements.append(final - initial)
        
        if improvements:
            multi_turn_data["improvement_analysis"] = {
                "mean_improvement": sum(improvements) / len(improvements),
                "max_improvement": max(improvements),
                "min_improvement": min(improvements),
                "positive_improvements": sum(1 for imp in improvements if imp > 0),
                "total_samples": len(improvements)
            }
        
        # Calculate model performance
        model_stats = {}
        for trace in traces:
            model = trace.get('model', 'unknown')
            if model not in model_stats:
                model_stats[model] = {
                    "samples": 0,
                    "initial_accuracy": 0,
                    "final_accuracy": 0,
                    "improvements": []
                }
            
            turns = trace.get('turns', [])
            if turns:
                initial = turns[0].get('accuracy', 0)
                final = turns[-1].get('accuracy', 0)
                improvement = final - initial
                
                model_stats[model]["samples"] += 1
                model_stats[model]["initial_accuracy"] += initial
                model_stats[model]["final_accuracy"] += final
                model_stats[model]["improvements"].append(improvement)
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["samples"] > 0:
                multi_turn_data["model_performance"][model] = {
                    "samples": stats["samples"],
                    "avg_initial_accuracy": stats["initial_accuracy"] / stats["samples"],
                    "avg_final_accuracy": stats["final_accuracy"] / stats["samples"],
                    "avg_improvement": sum(stats["improvements"]) / len(stats["improvements"]),
                    "improvement_rate": sum(1 for imp in stats["improvements"] if imp > 0) / len(stats["improvements"])
                }
        
        with open(output_file, 'w') as f:
            json.dump(multi_turn_data, f, indent=2)
    
    def format_all_experiments(self, experiments_dir: str) -> Dict[str, Any]:
        """Format traces from all experiments in a directory."""
        experiments_path = Path(experiments_dir)
        if not experiments_path.exists():
            logger.error(f"Experiments directory not found: {experiments_dir}")
            return {}
        
        results = {}
        
        # Find all trace files
        trace_files = list(experiments_path.glob("**/traces.json")) + list(experiments_path.glob("**/traces.jsonl"))
        
        for trace_file in trace_files:
            # Extract experiment ID from path
            experiment_id = trace_file.parent.name
            if trace_file.parent.parent.name != experiments_path.name:
                experiment_id = f"{trace_file.parent.parent.name}_{experiment_id}"
            
            logger.info(f"Formatting traces for experiment: {experiment_id}")
            formatted = self.format_experiment_traces(str(trace_file), experiment_id)
            if formatted:
                results[experiment_id] = formatted
        
        # Create combined summary
        self._create_combined_summary(results)
        
        return results
    
    def _create_combined_summary(self, results: Dict[str, Any]):
        """Create a combined summary of all experiments."""
        summary = {
            "experiments": list(results.keys()),
            "total_experiments": len(results),
            "formatted_files": {}
        }
        
        for exp_id, files in results.items():
            summary["formatted_files"][exp_id] = {
                "full_traces": files.get("full_traces", ""),
                "accuracy_traces": files.get("accuracy_traces", ""),
                "csv_summary": files.get("csv_summary", ""),
                "multi_turn_accuracy": files.get("multi_turn_accuracy", "")
            }
        
        summary_file = self.output_dir / "experiments_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Combined summary saved to: {summary_file}")

def main():
    """Test the trace formatter."""
    formatter = TraceFormatter()
    
    # Test with existing traces
    test_traces = "runs/dev_run/traces.jsonl"
    if Path(test_traces).exists():
        print("🧪 Testing Trace Formatter")
        print("=" * 40)
        
        results = formatter.format_experiment_traces(test_traces, "test_experiment")
        
        print("✅ Formatted files created:")
        for file_type, file_path in results.items():
            print(f"  {file_type}: {file_path}")
        
        print(f"\n📁 Output directory: {formatter.output_dir}")
    else:
        print("❌ No test traces found")

if __name__ == "__main__":
    main()
