"""
CSV Output Formatter for Scaling Study

Creates CSV files with final answers and multi-turn accuracy metrics for easy analysis.
"""

import csv
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CSVOutputFormatter:
    """Formatter that creates CSV files with final answers and accuracy metrics."""
    
    def __init__(self, output_dir: str = "outputs/csv_results"):
        """Initialize CSV output formatter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_final_answers_csv(self, traces_file: str, experiment_id: str) -> str:
        """Create CSV with final answers and basic metrics."""
        traces_path = Path(traces_file)
        if not traces_path.exists():
            logger.error(f"Traces file not found: {traces_file}")
            return ""
        
        # Load traces
        traces = self._load_traces(traces_path)
        if not traces:
            logger.error(f"No traces found in {traces_file}")
            return ""
        
        # Create CSV data
        csv_data = []
        for i, trace in enumerate(traces):
            sample_id = trace.get('qid', f'sample_{i}')
            question = trace.get('question', trace.get('original_problem_text', ''))
            reference = trace.get('reference', trace.get('ground_truth', ''))
            
            # Get final answer
            turns = trace.get('turns', [])
            final_answer = ""
            if turns:
                final_answer = turns[-1].get('answer', turns[-1].get('response_text', ''))
            
            # Get accuracy metrics
            initial_accuracy = 0
            final_accuracy = 0
            if turns:
                initial_accuracy = 1 if turns[0].get('is_correct', False) else 0
                final_accuracy = 1 if turns[-1].get('is_correct', False) else 0
            
            # Get confidence metrics
            initial_confidence = 0.0
            final_confidence = 0.0
            if turns:
                initial_confidence = turns[0].get('model_reported_confidence', 0.0)
                final_confidence = turns[-1].get('model_reported_confidence', 0.0)
            
            improvement = final_accuracy - initial_accuracy
            total_turns = len(turns)
            
            csv_data.append({
                'sample_id': sample_id,
                'question': question,
                'reference_answer': reference,
                'final_answer': final_answer,
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'improvement': improvement,
                'initial_confidence': initial_confidence,
                'final_confidence': final_confidence,
                'total_turns': total_turns,
                'model': trace.get('model_name', trace.get('model', 'unknown')),
                'dataset': trace.get('dataset', 'unknown'),
                'question_type': self._classify_question_type(trace)
            })
        
        # Save CSV
        output_file = self.output_dir / f"{experiment_id}_final_answers.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Created final answers CSV: {output_file}")
        return str(output_file)
    
    def create_multi_turn_accuracy_csv(self, traces_file: str, experiment_id: str) -> str:
        """Create CSV with multi-turn accuracy analysis."""
        traces_path = Path(traces_file)
        if not traces_path.exists():
            logger.error(f"Traces file not found: {traces_file}")
            return ""
        
        # Load traces
        traces = self._load_traces(traces_path)
        if not traces:
            logger.error(f"No traces found in {traces_file}")
            return ""
        
        # Create multi-turn data
        csv_data = []
        for i, trace in enumerate(traces):
            sample_id = trace.get('qid', f'sample_{i}')
            turns = trace.get('turns', [])
            
            for turn_idx, turn in enumerate(turns):
                accuracy = 1 if turn.get('is_correct', False) else 0
                confidence = turn.get('model_reported_confidence', 0.0)
                is_final = turn.get('response_is_final', False)
                
                csv_data.append({
                    'sample_id': sample_id,
                    'turn': turn_idx + 1,
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'is_final': is_final,
                    'model': trace.get('model_name', trace.get('model', 'unknown')),
                    'dataset': trace.get('dataset', 'unknown'),
                    'question_type': self._classify_question_type(trace)
                })
        
        # Save CSV
        output_file = self.output_dir / f"{experiment_id}_multi_turn_accuracy.csv"
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Created multi-turn accuracy CSV: {output_file}")
        return str(output_file)
    
    def create_summary_metrics_csv(self, traces_file: str, experiment_id: str) -> str:
        """Create CSV with summary metrics for scaling analysis."""
        traces_path = Path(traces_file)
        if not traces_path.exists():
            logger.error(f"Traces file not found: {traces_file}")
            return ""
        
        # Load traces
        traces = self._load_traces(traces_path)
        if not traces:
            logger.error(f"No traces found in {traces_file}")
            return ""
        
        # Calculate summary metrics
        total_samples = len(traces)
        correct_samples = sum(1 for trace in traces if trace.get('final_accuracy', trace.get('final_correct', 0)) == 1)
        overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0
        
        # Calculate per-turn accuracy
        max_turns = max(len(trace.get('turns', [])) for trace in traces) if traces else 0
        turn_metrics = []
        
        for turn_num in range(max_turns):
            turn_correct = 0
            turn_total = 0
            turn_confidence_sum = 0.0
            
            for trace in traces:
                turns = trace.get('turns', [])
                if turn_num < len(turns):
                    turn_total += 1
                    if turns[turn_num].get('is_correct', False):
                        turn_correct += 1
                    turn_confidence_sum += turns[turn_num].get('model_reported_confidence', 0.0)
            
            if turn_total > 0:
                turn_metrics.append({
                    'turn': turn_num + 1,
                    'accuracy': turn_correct / turn_total,
                    'correct': turn_correct,
                    'total': turn_total,
                    'avg_confidence': turn_confidence_sum / turn_total
                })
        
        # Create summary data
        summary_data = {
            'experiment_id': experiment_id,
            'total_samples': total_samples,
            'correct_samples': correct_samples,
            'overall_accuracy': overall_accuracy,
            'max_turns': max_turns
        }
        
        # Save summary CSV
        summary_file = self.output_dir / f"{experiment_id}_summary_metrics.csv"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data.keys())
            writer.writeheader()
            writer.writerow(summary_data)
        
        # Save turn metrics CSV
        turn_file = self.output_dir / f"{experiment_id}_turn_metrics.csv"
        if turn_metrics:
            df = pd.DataFrame(turn_metrics)
            df.to_csv(turn_file, index=False)
        
        logger.info(f"Created summary metrics CSV: {summary_file}")
        logger.info(f"Created turn metrics CSV: {turn_file}")
        
        return str(summary_file)
    
    def create_model_comparison_csv(self, traces_files: List[str], experiment_ids: List[str]) -> str:
        """Create CSV comparing performance across models."""
        all_data = []
        
        for traces_file, exp_id in zip(traces_files, experiment_ids):
            traces_path = Path(traces_file)
            if not traces_path.exists():
                continue
            
            traces = self._load_traces(traces_path)
            if not traces:
                continue
            
            # Calculate metrics for this experiment
            total_samples = len(traces)
            correct_samples = sum(1 for trace in traces if trace.get('final_accuracy', trace.get('final_correct', 0)) == 1)
            overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0
            
            # Calculate improvement
            improvements = []
            for trace in traces:
                turns = trace.get('turns', [])
                if len(turns) >= 2:
                    initial = 1 if turns[0].get('is_correct', False) else 0
                    final = 1 if turns[-1].get('is_correct', False) else 0
                    improvements.append(final - initial)
            
            avg_improvement = sum(improvements) / len(improvements) if improvements else 0
            
            # Get model info
            model = traces[0].get('model_name', traces[0].get('model', 'unknown')) if traces else 'unknown'
            dataset = traces[0].get('dataset', 'unknown') if traces else 'unknown'
            
            all_data.append({
                'experiment_id': exp_id,
                'model': model,
                'dataset': dataset,
                'total_samples': total_samples,
                'correct_samples': correct_samples,
                'overall_accuracy': overall_accuracy,
                'avg_improvement': avg_improvement,
                'max_turns': max(len(trace.get('turns', [])) for trace in traces) if traces else 0
            })
        
        # Save comparison CSV
        output_file = self.output_dir / "model_comparison.csv"
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Created model comparison CSV: {output_file}")
        return str(output_file)
    
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
                elif isinstance(data, dict) and 'summary' in data:
                    traces = data.get('traces', [])
        except json.JSONDecodeError:
            # Try JSONL
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

def main():
    """Test the CSV output formatter."""
    formatter = CSVOutputFormatter()
    
    # Test with existing traces
    test_traces = "runs/dev_run/traces.jsonl"
    if Path(test_traces).exists():
        print("🧪 Testing CSV Output Formatter")
        print("=" * 50)
        
        # Create all CSV outputs
        final_answers = formatter.create_final_answers_csv(test_traces, "test_csv")
        multi_turn = formatter.create_multi_turn_accuracy_csv(test_traces, "test_csv")
        summary = formatter.create_summary_metrics_csv(test_traces, "test_csv")
        
        print("✅ CSV outputs created:")
        print(f"  Final answers: {final_answers}")
        print(f"  Multi-turn accuracy: {multi_turn}")
        print(f"  Summary metrics: {summary}")
        
        print(f"\n📁 Output directory: {formatter.output_dir}")
    else:
        print("❌ No test traces found")

if __name__ == "__main__":
    main()


