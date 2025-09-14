"""
Enhanced Trace Formatter for Scaling Study

Properly separates full traces (.txt) from accuracy data (.json) for easy analysis.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedTraceFormatter:
    """Enhanced formatter that properly separates full traces from accuracy data."""
    
    def __init__(self, output_dir: str = "outputs/enhanced_traces"):
        """Initialize enhanced trace formatter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_experiment_traces(self, traces_file: str, experiment_id: str) -> Dict[str, str]:
        """Format traces with proper separation of full traces and accuracy data."""
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
        
        # 1. Full traces as .txt files (one per sample)
        full_traces_dir = self.output_dir / f"{experiment_id}_full_traces"
        full_traces_dir.mkdir(exist_ok=True)
        self._save_full_traces_txt(traces, full_traces_dir)
        outputs["full_traces_dir"] = str(full_traces_dir)
        
        # 2. Accuracy data as .json files
        accuracy_file = self.output_dir / f"{experiment_id}_accuracy_data.json"
        self._save_accuracy_data_json(traces, accuracy_file)
        outputs["accuracy_data"] = str(accuracy_file)
        
        # 3. Summary metrics as .json
        summary_file = self.output_dir / f"{experiment_id}_summary_metrics.json"
        self._save_summary_metrics(traces, summary_file)
        outputs["summary_metrics"] = str(summary_file)
        
        # 4. Multi-turn analysis as .json
        multi_turn_file = self.output_dir / f"{experiment_id}_multi_turn_analysis.json"
        self._save_multi_turn_analysis(traces, multi_turn_file)
        outputs["multi_turn_analysis"] = str(multi_turn_file)
        
        logger.info(f"Enhanced formatting complete for {experiment_id}: {len(outputs)} outputs created")
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
                elif isinstance(data, dict) and 'items' in data:
                    traces = data['items']
                elif isinstance(data, dict) and 'summary' in data:
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
                                continue
        
        return traces
    
    def _save_full_traces_txt(self, traces: List[Dict[str, Any]], output_dir: Path):
        """Save full traces as individual .txt files for each sample."""
        for i, trace in enumerate(traces):
            sample_id = trace.get('id', f'sample_{i}')
            filename = f"{sample_id}_full_trace.txt"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(f"FULL TRACE FOR SAMPLE: {sample_id}\n")
                f.write("=" * 50 + "\n\n")
                
                # Basic info
                f.write(f"Sample ID: {sample_id}\n")
                f.write(f"Final Prediction: {trace.get('final', {}).get('predicted', 'N/A')}\n")
                f.write(f"Final Correct: {trace.get('final', {}).get('correct', 'N/A')}\n")
                f.write(f"Total Turns: {len(trace.get('turns', []))}\n\n")
                
                # Turn-by-turn details
                turns = trace.get('turns', [])
                for turn_idx, turn in enumerate(turns):
                    f.write(f"TURN {turn_idx + 1}:\n")
                    f.write("-" * 20 + "\n")
                    
                    # Prompt ID and reference
                    f.write(f"Prompt ID: {turn.get('prompt_id', 'N/A')}\n")
                    f.write(f"Prompt Text Ref: {turn.get('prompt_text_ref', 'N/A')}\n")
                    f.write(f"Learner Output Ref: {turn.get('learner_output_ref', 'N/A')}\n")
                    
                    # Confidence
                    confidence = turn.get('confidence', 'N/A')
                    f.write(f"Model Confidence: {confidence}\n")
                    
                    # Normalized answer
                    normalized_answer = turn.get('normalized_answer', 'N/A')
                    f.write(f"Normalized Answer: {normalized_answer}\n")
                    
                    # Execution result
                    exec_result = turn.get('exec_result', 'N/A')
                    f.write(f"Execution Result: {exec_result}\n")
                    
                    # Evaluator feedback
                    feedback = turn.get('evaluator_feedback', {})
                    if feedback:
                        f.write(f"Evaluator Feedback: {feedback}\n")
                    
                    f.write("\n" + "=" * 50 + "\n\n")
    
    def _save_accuracy_data_json(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save accuracy data as structured JSON."""
        accuracy_data = {
            "experiment_metadata": {
                "total_samples": len(traces),
                "format_version": "2.0",
                "description": "Accuracy data for scaling analysis"
            },
            "samples": []
        }
        
        for trace in traces:
            sample_id = trace.get('id', 'unknown')
            turns = trace.get('turns', [])
            
            # Extract accuracy per turn
            turn_accuracies = []
            for i, turn in enumerate(turns):
                # Handle different accuracy formats
                accuracy = 0
                if 'is_correct' in turn:
                    accuracy = 1 if turn['is_correct'] else 0
                elif 'accuracy' in turn:
                    accuracy = turn['accuracy']
                
                # Handle different confidence formats
                confidence = 0.0
                if 'model_reported_confidence' in turn:
                    confidence = turn['model_reported_confidence']
                elif 'confidence' in turn:
                    confidence = turn['confidence']
                
                turn_accuracies.append({
                    "turn": i,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "is_final": turn.get('response_is_final', False)
                })
            
            # Calculate summary metrics
            initial_accuracy = turn_accuracies[0]['accuracy'] if turn_accuracies else 0
            final_accuracy = trace.get('final', {}).get('correct', 0)
            if isinstance(final_accuracy, bool):
                final_accuracy = 1 if final_accuracy else 0
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
                    "question_type": self._classify_question_type(trace),
                    "model": trace.get('model_name', trace.get('model', 'unknown')),
                    "dataset": trace.get('dataset', 'unknown')
                }
            }
            
            accuracy_data["samples"].append(sample_data)
        
        with open(output_file, 'w') as f:
            json.dump(accuracy_data, f, indent=2)
    
    def _save_summary_metrics(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save summary metrics as JSON."""
        total_samples = len(traces)
        correct_samples = sum(1 for trace in traces if trace.get('final_accuracy', trace.get('final_correct', 0)) == 1)
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        # Calculate per-turn accuracy
        max_turns = max(len(trace.get('turns', [])) for trace in traces)
        turn_accuracies = {}
        
        for turn_num in range(max_turns):
            turn_correct = 0
            turn_total = 0
            for trace in traces:
                turns = trace.get('turns', [])
                if turn_num < len(turns):
                    turn_total += 1
                    if turns[turn_num].get('is_correct', turns[turn_num].get('accuracy', 0)) == 1:
                        turn_correct += 1
            
            if turn_total > 0:
                turn_accuracies[f"turn_{turn_num}"] = {
                    "accuracy": turn_correct / turn_total,
                    "correct": turn_correct,
                    "total": turn_total
                }
        
        summary = {
            "overall_metrics": {
                "total_samples": total_samples,
                "correct_samples": correct_samples,
                "overall_accuracy": overall_accuracy,
                "max_turns": max_turns
            },
            "turn_accuracies": turn_accuracies,
            "improvement_analysis": self._calculate_improvement_analysis(traces)
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_multi_turn_analysis(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save multi-turn analysis for scaling study."""
        analysis = {
            "scaling_analysis": {
                "total_samples": len(traces),
                "format_version": "2.0",
                "description": "Multi-turn analysis for scaling laws"
            },
            "turn_analysis": {},
            "model_performance": {},
            "task_type_analysis": {}
        }
        
        # Analyze by turn
        max_turns = max(len(trace.get('turns', [])) for trace in traces)
        for turn_num in range(max_turns):
            turn_data = []
            for trace in traces:
                turns = trace.get('turns', [])
                if turn_num < len(turns):
                    turn_data.append({
                        "accuracy": turns[turn_num].get('is_correct', turns[turn_num].get('accuracy', 0)),
                        "confidence": turns[turn_num].get('model_reported_confidence', turns[turn_num].get('confidence', 0)),
                        "model": trace.get('model_name', trace.get('model', 'unknown')),
                        "task_type": self._classify_question_type(trace)
                    })
            
            if turn_data:
                analysis["turn_analysis"][f"turn_{turn_num}"] = {
                    "accuracy_mean": sum(t['accuracy'] for t in turn_data) / len(turn_data),
                    "confidence_mean": sum(t['confidence'] for t in turn_data) / len(turn_data),
                    "sample_count": len(turn_data)
                }
        
        # Analyze by model
        model_stats = {}
        for trace in traces:
            model = trace.get('model_name', trace.get('model', 'unknown'))
            if model not in model_stats:
                model_stats[model] = []
            
            turns = trace.get('turns', [])
            if turns:
                initial_acc = turns[0].get('is_correct', turns[0].get('accuracy', 0))
                final_acc = trace.get('final_accuracy', trace.get('final_correct', 0))
                if final_acc == 0:
                    final_acc = turns[-1].get('is_correct', turns[-1].get('accuracy', 0))
                
                model_stats[model].append({
                    "initial_accuracy": initial_acc,
                    "final_accuracy": final_acc,
                    "improvement": final_acc - initial_acc,
                    "total_turns": len(turns)
                })
        
        for model, stats in model_stats.items():
            if stats:
                analysis["model_performance"][model] = {
                    "avg_initial_accuracy": sum(s['initial_accuracy'] for s in stats) / len(stats),
                    "avg_final_accuracy": sum(s['final_accuracy'] for s in stats) / len(stats),
                    "avg_improvement": sum(s['improvement'] for s in stats) / len(stats),
                    "sample_count": len(stats)
                }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def _classify_question_type(self, trace: Dict[str, Any]) -> str:
        """Classify the type of question based on content."""
        question = trace.get('question', trace.get('original_problem_text', ''))
        if 'def ' in question or 'function' in question.lower():
            return 'code_generation'
        elif any(word in question.lower() for word in ['calculate', 'solve', 'math', 'number', 'count']):
            return 'mathematical_reasoning'
        elif any(word in question.lower() for word in ['explain', 'why', 'how', 'what']):
            return 'reasoning'
        else:
            return 'other'
    
    def _calculate_improvement_analysis(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate improvement analysis across all samples."""
        improvements = []
        for trace in traces:
            turns = trace.get('turns', [])
            if len(turns) >= 2:
                initial = turns[0].get('is_correct', turns[0].get('accuracy', 0))
                final = trace.get('final_accuracy', trace.get('final_correct', 0))
                if final == 0 and turns:
                    final = turns[-1].get('is_correct', turns[-1].get('accuracy', 0))
                improvements.append(final - initial)
        
        if improvements:
            return {
                "mean_improvement": sum(improvements) / len(improvements),
                "max_improvement": max(improvements),
                "min_improvement": min(improvements),
                "positive_improvements": sum(1 for imp in improvements if imp > 0),
                "total_samples": len(improvements)
            }
        else:
            return {"note": "No multi-turn samples found"}

def main():
    """Test the enhanced trace formatter."""
    formatter = EnhancedTraceFormatter()
    
    # Test with existing traces
    test_traces = "runs/dev_run/traces_clean.jsonl"
    if Path(test_traces).exists():
        print("🧪 Testing Enhanced Trace Formatter")
        print("=" * 50)
        
        results = formatter.format_experiment_traces(test_traces, "test_enhanced")
        
        print("✅ Enhanced formatting complete:")
        for output_type, output_path in results.items():
            print(f"  {output_type}: {output_path}")
        
        print(f"\n📁 Output directory: {formatter.output_dir}")
    else:
        print("❌ No test traces found")

if __name__ == "__main__":
    main()
