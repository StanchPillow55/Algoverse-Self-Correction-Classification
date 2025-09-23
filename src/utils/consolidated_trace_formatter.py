"""
Consolidated Trace Formatter for Llama Multi-Turn Experiments

Creates ONE .txt file per question containing ALL turns, matching the desired output structure.
Also generates chart-ready accuracy JSON data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConsolidatedTraceFormatter:
    """Formatter that creates consolidated traces with one file per question."""
    
    def __init__(self, output_dir: str = "runs"):
        """Initialize consolidated trace formatter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def format_llama_experiment_traces(self, traces_file: str, experiment_id: str) -> Dict[str, str]:
        """Format Llama traces with consolidated structure matching OpenAI/Anthropic format."""
        traces_path = Path(traces_file)
        if not traces_path.exists():
            logger.error(f"Traces file not found: {traces_file}")
            return {}
        
        # Load traces
        traces = self._load_traces(traces_path)
        if not traces:
            logger.error(f"No traces found in {traces_file}")
            return {}
        
        # Create output directories
        experiment_dir = self.output_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Create reasoning traces directory with consolidated files
        reasoning_traces_dir = experiment_dir / "reasoning_traces"
        reasoning_traces_dir.mkdir(exist_ok=True)
        
        outputs = {}
        
        # 1. Create consolidated reasoning traces (ONE .txt per question)
        self._save_consolidated_reasoning_traces(traces, reasoning_traces_dir)
        outputs["reasoning_traces_dir"] = str(reasoning_traces_dir)
        
        # 2. Create chart-ready accuracy data
        accuracy_chart_file = experiment_dir / "accuracy_chart_data.json"
        self._save_accuracy_chart_data(traces, accuracy_chart_file)
        outputs["accuracy_chart_data"] = str(accuracy_chart_file)
        
        # 3. Create main traces.json file (compatible with existing format)
        main_traces_file = experiment_dir / "traces.json"
        self._save_main_traces_json(traces, main_traces_file)
        outputs["main_traces"] = str(main_traces_file)
        
        # 4. Create summary metrics
        summary_file = experiment_dir / "summary.json"
        self._save_experiment_summary(traces, summary_file, experiment_id)
        outputs["summary"] = str(summary_file)
        
        logger.info(f"Consolidated formatting complete for {experiment_id}: {len(outputs)} outputs created")
        return outputs
    
    def _load_traces(self, traces_path: Path) -> List[Dict[str, Any]]:
        """Load traces from JSON or JSONL file."""
        traces = []
        
        try:
            with open(traces_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    traces = data
                elif isinstance(data, dict) and 'traces' in data:
                    traces = data['traces']
                elif isinstance(data, dict) and 'items' in data:
                    traces = data['items']
        except json.JSONDecodeError:
            # Try JSONL format
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
    
    def _save_consolidated_reasoning_traces(self, traces: List[Dict[str, Any]], output_dir: Path):
        """Save consolidated reasoning traces - ONE .txt file per question with ALL turns."""
        for i, trace in enumerate(traces):
            # Extract question ID - try different formats
            qid = trace.get('qid', trace.get('id', trace.get('question_id', f'q_{i}')))
            
            # Determine dataset type for directory structure
            dataset_type = self._get_dataset_type(trace)
            dataset_dir = output_dir / dataset_type
            dataset_dir.mkdir(exist_ok=True)
            
            # Create question directory
            question_dir = dataset_dir / str(qid)
            question_dir.mkdir(exist_ok=True)
            
            # Create consolidated trace file (ALL turns in ONE file)
            consolidated_file = question_dir / "consolidated_reasoning.txt"
            
            with open(consolidated_file, 'w') as f:
                f.write(f"CONSOLIDATED REASONING TRACE\\n")
                f.write(f"Question ID: {qid}\\n")
                f.write(f"Dataset Type: {dataset_type}\\n")
                f.write("=" * 50 + "\\n\\n")
                
                # Write question
                question = trace.get('question', trace.get('problem', 'N/A'))
                f.write(f"QUESTION:\\n{question}\\n\\n")
                
                # Write reference answer if available
                reference = trace.get('reference', trace.get('answer', ''))
                if reference:
                    f.write(f"REFERENCE ANSWER:\\n{reference}\\n\\n")
                
                # Write all turns in sequence
                turns = trace.get('turns', [])
                for turn_idx, turn in enumerate(turns):
                    f.write(f"TURN {turn_idx}:\\n")
                    f.write("-" * 30 + "\\n")
                    
                    # Extract response text
                    response = turn.get('response_text', turn.get('raw_answer', turn.get('answer', 'N/A')))
                    f.write(f"MODEL RESPONSE:\\n{response}\\n\\n")
                    
                    # Extract final answer
                    final_answer = turn.get('answer', turn.get('extracted_answer', 'N/A'))
                    f.write(f"EXTRACTED ANSWER: {final_answer}\\n")
                    
                    # Confidence and accuracy
                    confidence = turn.get('self_conf', turn.get('confidence', 'N/A'))
                    accuracy = turn.get('accuracy', turn.get('is_correct', 'N/A'))
                    f.write(f"CONFIDENCE: {confidence}\\n")
                    f.write(f"ACCURACY: {accuracy}\\n")
                    
                    # Teacher feedback if available
                    if 'teacher_bias' in turn:
                        f.write(f"TEACHER BIAS: {turn['teacher_bias']}\\n")
                    if 'template' in turn:
                        f.write(f"FEEDBACK TEMPLATE: {turn['template']}\\n")
                    
                    f.write("\\n" + "=" * 50 + "\\n\\n")
                
                # Final results
                final_accuracy = trace.get('final_accuracy', 'N/A')
                f.write(f"FINAL ACCURACY: {final_accuracy}\\n")
                
                improvement = final_accuracy - turns[0].get('accuracy', 0) if isinstance(final_accuracy, (int, float)) and turns else 'N/A'
                f.write(f"IMPROVEMENT: {improvement}\\n")
    
    def _save_accuracy_chart_data(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save accuracy data in chart-ready JSON format."""
        chart_data = {
            "experiment_metadata": {
                "total_questions": len(traces),
                "model": "llama-70b",
                "dataset": self._get_dataset_name(traces),
                "timestamp": str(pd.Timestamp.now()),
                "format": "chart_ready"
            },
            "turn_accuracy": {
                "labels": [],
                "data": []
            },
            "question_results": [],
            "summary_metrics": {}
        }
        
        # Calculate turn-by-turn accuracy
        max_turns = max(len(trace.get('turns', [])) for trace in traces)
        turn_labels = [f"Turn {i}" for i in range(max_turns)]
        chart_data["turn_accuracy"]["labels"] = turn_labels
        
        turn_accuracies = []
        for turn_idx in range(max_turns):
            correct_count = 0
            total_count = 0
            
            for trace in traces:
                turns = trace.get('turns', [])
                if turn_idx < len(turns):
                    accuracy = turns[turn_idx].get('accuracy', 0)
                    if accuracy == 1:
                        correct_count += 1
                    total_count += 1
            
            turn_accuracy = correct_count / total_count if total_count > 0 else 0
            turn_accuracies.append(round(turn_accuracy * 100, 2))  # Convert to percentage
        
        chart_data["turn_accuracy"]["data"] = turn_accuracies
        
        # Individual question results for detailed analysis
        for i, trace in enumerate(traces):
            qid = trace.get('qid', trace.get('id', f'q_{i}'))
            turns = trace.get('turns', [])
            
            question_result = {
                "question_id": qid,
                "initial_accuracy": turns[0].get('accuracy', 0) if turns else 0,
                "final_accuracy": trace.get('final_accuracy', 0),
                "turn_accuracies": [turn.get('accuracy', 0) for turn in turns],
                "turn_confidences": [turn.get('self_conf', 0) for turn in turns]
            }
            chart_data["question_results"].append(question_result)
        
        # Summary metrics for dashboard
        initial_correct = sum(1 for trace in traces if trace.get('turns', [{}])[0].get('accuracy', 0) == 1)
        final_correct = sum(1 for trace in traces if trace.get('final_accuracy', 0) == 1)
        improved = sum(1 for trace in traces if trace.get('final_accuracy', 0) > trace.get('turns', [{}])[0].get('accuracy', 0))
        
        chart_data["summary_metrics"] = {
            "initial_accuracy": round(initial_correct / len(traces) * 100, 2),
            "final_accuracy": round(final_correct / len(traces) * 100, 2),
            "improvement_rate": round(improved / len(traces) * 100, 2),
            "questions_improved": improved,
            "total_questions": len(traces)
        }
        
        with open(output_file, 'w') as f:
            json.dump(chart_data, f, indent=2)
    
    def _save_main_traces_json(self, traces: List[Dict[str, Any]], output_file: Path):
        """Save main traces.json file compatible with existing format."""
        # Calculate summary
        total_items = len(traces)
        final_correct = sum(1 for trace in traces if trace.get('final_accuracy', 0) == 1)
        final_accuracy_mean = final_correct / total_items if total_items > 0 else 0
        
        traces_data = {
            "summary": {
                "items": total_items,
                "final_accuracy_mean": final_accuracy_mean
            },
            "traces": traces
        }
        
        with open(output_file, 'w') as f:
            json.dump(traces_data, f, indent=2)
    
    def _save_experiment_summary(self, traces: List[Dict[str, Any]], output_file: Path, experiment_id: str):
        """Save experiment summary metrics."""
        total_items = len(traces)
        if total_items == 0:
            return
        
        # Calculate metrics
        initial_correct = sum(1 for trace in traces if trace.get('turns', [{}])[0].get('accuracy', 0) == 1)
        final_correct = sum(1 for trace in traces if trace.get('final_accuracy', 0) == 1)
        
        summary = {
            "experiment_id": experiment_id,
            "model": "llama-70b",
            "dataset": self._get_dataset_name(traces),
            "timestamp": str(pd.Timestamp.now()),
            "metrics": {
                "total_questions": total_items,
                "initial_accuracy": initial_correct / total_items,
                "final_accuracy": final_correct / total_items,
                "improvement": (final_correct - initial_correct) / total_items,
                "questions_improved": sum(1 for trace in traces if trace.get('final_accuracy', 0) > trace.get('turns', [{}])[0].get('accuracy', 0))
            },
            "output_structure": {
                "reasoning_traces": "reasoning_traces/[dataset_type]/[question_id]/consolidated_reasoning.txt",
                "accuracy_data": "accuracy_chart_data.json",
                "main_traces": "traces.json"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _get_dataset_type(self, trace: Dict[str, Any]) -> str:
        """Determine dataset type from trace."""
        dataset = trace.get('dataset', trace.get('dataset_name', '')).lower()
        if 'gsm8k' in dataset or 'math' in dataset:
            return 'math'
        elif 'humaneval' in dataset or 'code' in dataset:
            return 'code'
        elif 'superglue' in dataset:
            return 'reasoning'
        else:
            return 'other'
    
    def _get_dataset_name(self, traces: List[Dict[str, Any]]) -> str:
        """Extract dataset name from traces."""
        if traces:
            return traces[0].get('dataset', traces[0].get('dataset_name', 'unknown'))
        return 'unknown'


if __name__ == "__main__":
    # Test the formatter
    formatter = ConsolidatedTraceFormatter()
    print("Consolidated Trace Formatter ready for Llama experiments")