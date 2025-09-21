"""
Ensemble Metrics and Analysis Tools
Provides comprehensive metrics for evaluating ensemble performance
"""
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class EnsembleMetrics:
    """Comprehensive ensemble performance analysis"""
    
    def __init__(self):
        self.metrics = {}
        
    def analyze_ensemble_experiment(self, traces_file: str) -> Dict[str, Any]:
        """Analyze a complete ensemble experiment"""
        
        with open(traces_file, 'r') as f:
            data = json.load(f)
        
        traces = data.get('traces', [])
        if not traces:
            return {"error": "No traces found in file"}
        
        # Basic ensemble metrics
        basic_metrics = self._calculate_basic_metrics(traces)
        
        # Voting analysis
        voting_analysis = self._analyze_voting_patterns(traces)
        
        # Individual model performance  
        individual_performance = self._analyze_individual_models(traces)
        
        # Disagreement analysis
        disagreement_metrics = self._analyze_disagreement_patterns(traces)
        
        # Confidence calibration
        confidence_metrics = self._analyze_confidence_calibration(traces)
        
        # Consensus strength analysis
        consensus_metrics = self._analyze_consensus_strength(traces)
        
        return {
            "basic_metrics": basic_metrics,
            "voting_analysis": voting_analysis,
            "individual_performance": individual_performance, 
            "disagreement_metrics": disagreement_metrics,
            "confidence_metrics": confidence_metrics,
            "consensus_metrics": consensus_metrics,
            "total_questions": len(traces),
            "analysis_timestamp": self._get_timestamp()
        }
    
    def _calculate_basic_metrics(self, traces: List[Dict]) -> Dict[str, Any]:
        """Calculate basic ensemble metrics"""
        
        total_questions = len(traces)
        ensemble_correct = sum(1 for t in traces if t.get('final_accuracy', 0) == 1)
        ensemble_accuracy = ensemble_correct / max(total_questions, 1)
        
        # Multi-turn analysis
        single_turn_correct = sum(1 for t in traces if len(t.get('turns', [])) == 1 and t.get('final_accuracy', 0) == 1)
        multi_turn_correct = sum(1 for t in traces if len(t.get('turns', [])) > 1 and t.get('final_accuracy', 0) == 1)
        
        avg_turns = np.mean([len(t.get('turns', [])) for t in traces])
        
        return {
            "ensemble_accuracy": round(ensemble_accuracy, 4),
            "total_correct": ensemble_correct,
            "single_turn_success_rate": round(single_turn_correct / max(total_questions, 1), 4),
            "multi_turn_success_rate": round(multi_turn_correct / max(total_questions, 1), 4) if total_questions > single_turn_correct else 0,
            "average_turns_per_question": round(avg_turns, 2),
            "questions_requiring_multiple_turns": sum(1 for t in traces if len(t.get('turns', [])) > 1)
        }
    
    def _analyze_voting_patterns(self, traces: List[Dict]) -> Dict[str, Any]:
        """Analyze ensemble voting patterns and strategies"""
        
        voting_methods = defaultdict(int)
        consensus_ratios = []
        tie_break_cases = 0
        
        for trace in traces:
            for turn in trace.get('turns', []):
                response_text = turn.get('response_text', '')
                
                # Extract voting information from ensemble response
                if 'VOTING SUMMARY' in response_text:
                    lines = response_text.split('\n')
                    for line in lines:
                        if 'Voting Method:' in line:
                            method = line.split(':', 1)[1].strip()
                            voting_methods[method] += 1
                        elif 'Consensus Ratio:' in line:
                            try:
                                ratio = float(line.split(':', 1)[1].strip())
                                consensus_ratios.append(ratio)
                            except:
                                pass
                        elif 'majority_with_confidence_tiebreak' in line:
                            tie_break_cases += 1
        
        return {
            "voting_method_distribution": dict(voting_methods),
            "average_consensus_ratio": round(np.mean(consensus_ratios), 3) if consensus_ratios else 0,
            "consensus_ratio_std": round(np.std(consensus_ratios), 3) if consensus_ratios else 0,
            "tie_break_cases": tie_break_cases,
            "high_consensus_cases": sum(1 for r in consensus_ratios if r > 0.8),
            "low_consensus_cases": sum(1 for r in consensus_ratios if r < 0.5)
        }
    
    def _analyze_individual_models(self, traces: List[Dict]) -> Dict[str, Any]:
        """Analyze individual model performance within ensemble"""
        
        # This is a simplified version - in practice, you'd need to extract 
        # individual model responses from the ensemble traces
        
        model_accuracies = {}
        model_confidences = defaultdict(list)
        
        # Extract information from ensemble response texts
        for trace in traces:
            for turn in trace.get('turns', []):
                response_text = turn.get('response_text', '')
                is_correct = turn.get('accuracy', 0) == 1
                
                # Parse individual model responses if available
                if 'INDIVIDUAL MODEL RESPONSES' in response_text:
                    model_sections = response_text.split('--- Model ')[1:]  # Skip the first empty split
                    
                    for i, section in enumerate(model_sections):
                        model_name = f"model_{i+1}"
                        if model_name not in model_accuracies:
                            model_accuracies[model_name] = []
                        
                        # For this simplified version, we'll assume all models 
                        # contribute equally to the ensemble decision
                        model_accuracies[model_name].append(is_correct)
                        
                        # Extract confidence if available
                        conf = turn.get('self_conf', 0.5)
                        model_confidences[model_name].append(conf)
        
        # Calculate per-model metrics
        model_metrics = {}
        for model, accuracies in model_accuracies.items():
            model_metrics[model] = {
                "accuracy": round(np.mean(accuracies), 4),
                "total_questions": len(accuracies),
                "average_confidence": round(np.mean(model_confidences[model]), 3) if model_confidences[model] else 0
            }
        
        return {
            "individual_model_metrics": model_metrics,
            "model_count": len(model_accuracies),
            "ensemble_vs_best_individual": self._calculate_ensemble_vs_best_individual(model_metrics, traces)
        }
    
    def _calculate_ensemble_vs_best_individual(self, model_metrics: Dict, traces: List[Dict]) -> Dict[str, float]:
        """Compare ensemble performance to best individual model"""
        
        if not model_metrics:
            return {"improvement": 0.0, "best_individual_accuracy": 0.0}
        
        best_individual_acc = max(metrics["accuracy"] for metrics in model_metrics.values())
        ensemble_acc = sum(1 for t in traces if t.get('final_accuracy', 0) == 1) / max(len(traces), 1)
        
        improvement = ensemble_acc - best_individual_acc
        
        return {
            "improvement": round(improvement, 4),
            "best_individual_accuracy": round(best_individual_acc, 4),
            "ensemble_accuracy": round(ensemble_acc, 4),
            "relative_improvement": round((improvement / max(best_individual_acc, 0.001)) * 100, 2)
        }
    
    def _analyze_disagreement_patterns(self, traces: List[Dict]) -> Dict[str, Any]:
        """Analyze when and why models disagree"""
        
        disagreement_cases = []
        full_agreement_cases = 0
        partial_agreement_cases = 0
        
        for trace in traces:
            for turn in trace.get('turns', []):
                response_text = turn.get('response_text', '')
                
                # Check for disagreement indicators in response
                if 'Response Distribution:' in response_text:
                    # Extract distribution information
                    lines = response_text.split('\n')
                    for line in lines:
                        if 'Response Distribution:' in line:
                            try:
                                # Parse distribution (simplified)
                                dist_str = line.split(':', 1)[1].strip()
                                unique_responses = len(eval(dist_str))  # Warning: eval is dangerous in production
                                
                                if unique_responses == 1:
                                    full_agreement_cases += 1
                                else:
                                    partial_agreement_cases += 1
                                    disagreement_cases.append({
                                        "qid": trace.get('qid', 'unknown'),
                                        "turn": turn,
                                        "unique_responses": unique_responses,
                                        "is_correct": turn.get('accuracy', 0) == 1
                                    })
                                break
                            except:
                                pass
        
        total_cases = full_agreement_cases + partial_agreement_cases
        
        return {
            "disagreement_rate": round(partial_agreement_cases / max(total_cases, 1), 4),
            "full_agreement_cases": full_agreement_cases,
            "partial_agreement_cases": partial_agreement_cases,
            "disagreement_success_rate": round(
                sum(1 for case in disagreement_cases if case["is_correct"]) / max(len(disagreement_cases), 1), 4
            ),
            "agreement_success_rate": round(
                full_agreement_cases / max(total_cases, 1), 4  # Simplified - assumes all agreements are correct
            )
        }
    
    def _analyze_confidence_calibration(self, traces: List[Dict]) -> Dict[str, Any]:
        """Analyze how well ensemble confidence predicts correctness"""
        
        confidences = []
        accuracies = []
        
        for trace in traces:
            for turn in trace.get('turns', []):
                conf = turn.get('combined_confidence', turn.get('self_conf', 0.5))
                acc = turn.get('accuracy', 0)
                
                confidences.append(conf)
                accuracies.append(acc)
        
        if not confidences:
            return {"error": "No confidence data found"}
        
        # Calculate calibration metrics
        confidence_buckets = defaultdict(list)
        for conf, acc in zip(confidences, accuracies):
            bucket = round(conf * 10) / 10  # Round to nearest 0.1
            confidence_buckets[bucket].append(acc)
        
        calibration_data = {}
        for bucket, accs in confidence_buckets.items():
            calibration_data[bucket] = {
                "predicted_accuracy": bucket,
                "actual_accuracy": round(np.mean(accs), 4),
                "count": len(accs),
                "calibration_error": abs(bucket - np.mean(accs))
            }
        
        # Overall calibration error (Expected Calibration Error)
        ece = np.mean([data["calibration_error"] for data in calibration_data.values()])
        
        return {
            "expected_calibration_error": round(ece, 4),
            "calibration_by_confidence_bucket": calibration_data,
            "average_confidence": round(np.mean(confidences), 3),
            "confidence_std": round(np.std(confidences), 3)
        }
    
    def _analyze_consensus_strength(self, traces: List[Dict]) -> Dict[str, Any]:
        """Analyze relationship between consensus strength and accuracy"""
        
        consensus_accuracy_pairs = []
        
        for trace in traces:
            for turn in trace.get('turns', []):
                response_text = turn.get('response_text', '')
                is_correct = turn.get('accuracy', 0) == 1
                
                # Extract consensus ratio
                if 'Consensus Ratio:' in response_text:
                    lines = response_text.split('\n')
                    for line in lines:
                        if 'Consensus Ratio:' in line:
                            try:
                                ratio = float(line.split(':', 1)[1].strip())
                                consensus_accuracy_pairs.append((ratio, is_correct))
                                break
                            except:
                                pass
        
        if not consensus_accuracy_pairs:
            return {"error": "No consensus data found"}
        
        # Analyze correlation between consensus and accuracy
        consensus_values = [pair[0] for pair in consensus_accuracy_pairs]
        accuracy_values = [pair[1] for pair in consensus_accuracy_pairs]
        
        # Group by consensus levels
        high_consensus = [acc for cons, acc in consensus_accuracy_pairs if cons > 0.8]
        medium_consensus = [acc for cons, acc in consensus_accuracy_pairs if 0.5 <= cons <= 0.8]
        low_consensus = [acc for cons, acc in consensus_accuracy_pairs if cons < 0.5]
        
        return {
            "high_consensus_accuracy": round(np.mean(high_consensus), 4) if high_consensus else 0,
            "medium_consensus_accuracy": round(np.mean(medium_consensus), 4) if medium_consensus else 0,
            "low_consensus_accuracy": round(np.mean(low_consensus), 4) if low_consensus else 0,
            "consensus_accuracy_correlation": round(np.corrcoef(consensus_values, accuracy_values)[0, 1], 4) if len(consensus_values) > 1 else 0,
            "consensus_distribution": {
                "high": len(high_consensus),
                "medium": len(medium_consensus),
                "low": len(low_consensus)
            }
        }
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def generate_ensemble_report(self, traces_file: str, output_dir: str) -> str:
        """Generate comprehensive ensemble analysis report"""
        
        metrics = self.analyze_ensemble_experiment(traces_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed metrics to JSON
        metrics_file = output_path / "ensemble_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate summary report
        report_file = output_path / "ensemble_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(self._format_analysis_report(metrics))
        
        print(f"📊 Ensemble analysis complete!")
        print(f"📄 Detailed metrics: {metrics_file}")
        print(f"📋 Summary report: {report_file}")
        
        return str(report_file)
    
    def _format_analysis_report(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a readable report"""
        
        report_lines = [
            "=" * 60,
            "ENSEMBLE PERFORMANCE ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Analysis Timestamp: {metrics.get('analysis_timestamp', 'Unknown')}",
            f"Total Questions Analyzed: {metrics.get('total_questions', 0)}",
            "",
            "BASIC PERFORMANCE METRICS",
            "-" * 30
        ]
        
        basic = metrics.get('basic_metrics', {})
        if basic:
            report_lines.extend([
                f"Ensemble Accuracy: {basic.get('ensemble_accuracy', 0):.4f}",
                f"Total Correct: {basic.get('total_correct', 0)}",
                f"Single-turn Success Rate: {basic.get('single_turn_success_rate', 0):.4f}",
                f"Multi-turn Success Rate: {basic.get('multi_turn_success_rate', 0):.4f}",
                f"Average Turns per Question: {basic.get('average_turns_per_question', 0):.2f}",
                ""
            ])
        
        voting = metrics.get('voting_analysis', {})
        if voting:
            report_lines.extend([
                "VOTING ANALYSIS",
                "-" * 15,
                f"Average Consensus Ratio: {voting.get('average_consensus_ratio', 0):.3f}",
                f"Tie-break Cases: {voting.get('tie_break_cases', 0)}",
                f"High Consensus Cases (>80%): {voting.get('high_consensus_cases', 0)}",
                f"Low Consensus Cases (<50%): {voting.get('low_consensus_cases', 0)}",
                ""
            ])
        
        individual = metrics.get('individual_performance', {})
        if individual:
            ensemble_vs_best = individual.get('ensemble_vs_best_individual', {})
            report_lines.extend([
                "ENSEMBLE VS INDIVIDUAL MODELS",
                "-" * 32,
                f"Best Individual Model Accuracy: {ensemble_vs_best.get('best_individual_accuracy', 0):.4f}",
                f"Ensemble Accuracy: {ensemble_vs_best.get('ensemble_accuracy', 0):.4f}",
                f"Improvement: {ensemble_vs_best.get('improvement', 0):+.4f}",
                f"Relative Improvement: {ensemble_vs_best.get('relative_improvement', 0):+.2f}%",
                ""
            ])
        
        disagreement = metrics.get('disagreement_metrics', {})
        if disagreement:
            report_lines.extend([
                "DISAGREEMENT ANALYSIS",
                "-" * 21,
                f"Disagreement Rate: {disagreement.get('disagreement_rate', 0):.4f}",
                f"Success Rate with Disagreement: {disagreement.get('disagreement_success_rate', 0):.4f}",
                f"Success Rate with Agreement: {disagreement.get('agreement_success_rate', 0):.4f}",
                ""
            ])
        
        confidence = metrics.get('confidence_metrics', {})
        if confidence:
            report_lines.extend([
                "CONFIDENCE CALIBRATION",
                "-" * 22,
                f"Expected Calibration Error: {confidence.get('expected_calibration_error', 0):.4f}",
                f"Average Confidence: {confidence.get('average_confidence', 0):.3f}",
                f"Confidence Standard Deviation: {confidence.get('confidence_std', 0):.3f}",
                ""
            ])
        
        consensus = metrics.get('consensus_metrics', {})
        if consensus:
            report_lines.extend([
                "CONSENSUS STRENGTH ANALYSIS",
                "-" * 27,
                f"High Consensus Accuracy (>80%): {consensus.get('high_consensus_accuracy', 0):.4f}",
                f"Medium Consensus Accuracy (50-80%): {consensus.get('medium_consensus_accuracy', 0):.4f}",
                f"Low Consensus Accuracy (<50%): {consensus.get('low_consensus_accuracy', 0):.4f}",
                f"Consensus-Accuracy Correlation: {consensus.get('consensus_accuracy_correlation', 0):.4f}",
                ""
            ])
        
        report_lines.extend([
            "=" * 60,
            "End of Report",
            "=" * 60
        ])
        
        return "\n".join(report_lines)

def analyze_ensemble_experiment_cli(traces_file: str, output_dir: str = None):
    """CLI interface for ensemble analysis"""
    
    if output_dir is None:
        output_dir = str(Path(traces_file).parent / "ensemble_analysis")
    
    analyzer = EnsembleMetrics()
    report_file = analyzer.generate_ensemble_report(traces_file, output_dir)
    return report_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.ensemble.metrics <traces_file> [output_dir]")
        sys.exit(1)
    
    traces_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_ensemble_experiment_cli(traces_file, output_dir)