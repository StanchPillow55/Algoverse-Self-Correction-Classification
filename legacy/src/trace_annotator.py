"""
Gold-Label Annotation System for Failure Modes and ΔAccuracy
"""
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class FailureMode(Enum):
    """Enumeration of failure modes"""
    ANCHORED = "anchored"                    # Stuck on wrong answer despite evidence
    OVERCORRECTED = "overcorrected"          # Changed correct answer to wrong one  
    CORRECTED = "corrected"                  # Changed wrong answer to correct one
    UNCHANGED_CORRECT = "unchanged_correct"  # Kept correct answer unchanged
    WAVERING = "wavering"                    # Oscillating between answers
    PERFECTIONISM = "perfectionism"          # Unnecessarily revising correct answer

class AccuracyDelta(Enum):
    """Change in accuracy"""
    IMPROVED = +1     # Wrong → Correct
    UNCHANGED = 0     # Correct → Correct OR Wrong → Wrong
    DEGRADED = -1     # Correct → Wrong

@dataclass
class TraceAnnotation:
    """Annotation for a single trace transition"""
    failure_mode: FailureMode
    delta_accuracy: AccuracyDelta
    confidence_change: float
    reasoning: str
    
class TraceAnnotatorEngine:
    """
    Engine for annotating multi-pass traces with failure modes and accuracy deltas.
    
    This implements the gold-labeling logic for training data generation.
    """
    
    def __init__(self):
        self.accuracy_cache = {}  # Cache for answer correctness evaluation
    
    def annotate_trace(self, trace_data: Dict) -> Dict:
        """
        Annotate a complete multi-pass trace.
        
        Args:
            trace_data: Multi-pass trace with turns
            
        Returns:
            Annotated trace with failure modes and deltas
        """
        annotated_trace = trace_data.copy()
        annotated_trace["annotations"] = []
        
        turns = trace_data.get("turns", [])
        reference_answer = trace_data.get("reference_answer", "")
        
        # Annotate each transition between turns
        for i in range(len(turns) - 1):
            current_turn = turns[i]
            next_turn = turns[i + 1]
            
            annotation = self._annotate_turn_transition(
                current_turn, next_turn, reference_answer, i
            )
            annotated_trace["annotations"].append(annotation)
        
        # Add overall trace statistics
        annotated_trace["trace_stats"] = self._compute_trace_statistics(
            turns, reference_answer
        )
        
        return annotated_trace
    
    def _annotate_turn_transition(self, 
                                 current_turn: Dict, 
                                 next_turn: Dict,
                                 reference_answer: str,
                                 turn_index: int) -> Dict:
        """
        Annotate the transition between two consecutive turns.
        
        Args:
            current_turn: Current turn data
            next_turn: Next turn data  
            reference_answer: Ground truth answer
            turn_index: Index of current turn
            
        Returns:
            Annotation dictionary
        """
        current_answer = current_turn.get("answer", "")
        next_answer = next_turn.get("answer", "")
        prompt_id = next_turn.get("prompt_id", "")
        
        # Evaluate correctness
        current_correct = self._is_correct(current_answer, reference_answer)
        next_correct = self._is_correct(next_answer, reference_answer)
        
        # Determine failure mode and accuracy delta
        failure_mode = self._classify_failure_mode(
            current_answer, next_answer, current_correct, next_correct
        )
        
        delta_accuracy = self._compute_accuracy_delta(current_correct, next_correct)
        
        # Calculate confidence change
        current_conf = current_turn.get("confidence", 0.5)
        next_conf = next_turn.get("confidence", 0.5)
        confidence_change = next_conf - current_conf
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            failure_mode, current_answer, next_answer, 
            current_correct, next_correct, prompt_id
        )
        
        return {
            "turn_transition": f"{turn_index}->{turn_index+1}",
            "failure_mode": failure_mode.value,
            "delta_accuracy": delta_accuracy.value,
            "confidence_change": round(confidence_change, 3),
            "reasoning": reasoning,
            "prompt_used": prompt_id,
            "answer_change": {
                "from": current_answer,
                "to": next_answer,
                "from_correct": current_correct,
                "to_correct": next_correct
            }
        }
    
    def _classify_failure_mode(self, 
                              current_answer: str, 
                              next_answer: str,
                              current_correct: bool, 
                              next_correct: bool) -> FailureMode:
        """Classify the type of failure mode in this transition"""
        
        # No change in answer
        if current_answer.strip() == next_answer.strip():
            if current_correct:
                return FailureMode.UNCHANGED_CORRECT
            else:
                return FailureMode.ANCHORED
        
        # Answer changed
        if not current_correct and next_correct:
            return FailureMode.CORRECTED
        elif current_correct and not next_correct:
            return FailureMode.OVERCORRECTED
        elif current_correct and next_correct:
            return FailureMode.PERFECTIONISM  # Unnecessary revision
        else:
            # Both wrong - could be wavering or anchoring
            return FailureMode.WAVERING
    
    def _compute_accuracy_delta(self, 
                               current_correct: bool, 
                               next_correct: bool) -> AccuracyDelta:
        """Compute the change in accuracy"""
        if not current_correct and next_correct:
            return AccuracyDelta.IMPROVED
        elif current_correct and not next_correct:
            return AccuracyDelta.DEGRADED
        else:
            return AccuracyDelta.UNCHANGED
    
    def _is_correct(self, answer: str, reference: str) -> bool:
        """
        Evaluate if an answer is correct relative to reference.
        
        This is a simplified implementation - in practice would need
        more sophisticated evaluation depending on task type.
        """
        # Cache results
        cache_key = (answer.strip().lower(), reference.strip().lower())
        if cache_key in self.accuracy_cache:
            return self.accuracy_cache[cache_key]
        
        # Simple exact match (could be enhanced with fuzzy matching, etc.)
        is_correct = self._fuzzy_match(answer, reference)
        self.accuracy_cache[cache_key] = is_correct
        return is_correct
    
    def _fuzzy_match(self, answer: str, reference: str) -> bool:
        """Fuzzy matching for answer correctness"""
        answer_clean = answer.strip().lower()
        reference_clean = reference.strip().lower()
        
        # Exact match
        if answer_clean == reference_clean:
            return True
        
        # Extract numbers for math problems
        import re
        answer_nums = re.findall(r'\d+(?:\.\d+)?', answer)
        ref_nums = re.findall(r'\d+(?:\.\d+)?', reference)
        
        if answer_nums and ref_nums:
            # For math problems, match on final numeric answer
            try:
                return float(answer_nums[-1]) == float(ref_nums[-1])
            except (ValueError, IndexError):
                pass
        
        # Substring containment
        return reference_clean in answer_clean or answer_clean in reference_clean
    
    def _generate_reasoning(self, 
                           failure_mode: FailureMode,
                           current_answer: str,
                           next_answer: str,
                           current_correct: bool,
                           next_correct: bool,
                           prompt_id: str) -> str:
        """Generate human-readable reasoning for the annotation"""
        
        templates = {
            FailureMode.ANCHORED: (
                f"Model stuck on incorrect answer '{current_answer}' despite {prompt_id} prompt. "
                f"Shows anchoring bias - unable to revise wrong answer."
            ),
            FailureMode.OVERCORRECTED: (
                f"Model changed from correct answer '{current_answer}' to incorrect '{next_answer}' "
                f"after {prompt_id} prompt. Shows overcorrection tendency."
            ),
            FailureMode.CORRECTED: (
                f"Model successfully corrected from wrong answer '{current_answer}' to correct '{next_answer}' "
                f"using {prompt_id} prompt. Good self-correction."
            ),
            FailureMode.UNCHANGED_CORRECT: (
                f"Model maintained correct answer '{current_answer}' despite {prompt_id} prompt. "
                f"Shows appropriate confidence in correct response."
            ),
            FailureMode.WAVERING: (
                f"Model changed from '{current_answer}' to '{next_answer}' (both incorrect) "
                f"after {prompt_id} prompt. Shows answer wavering without improvement."
            ),
            FailureMode.PERFECTIONISM: (
                f"Model unnecessarily revised correct answer '{current_answer}' to '{next_answer}' "
                f"after {prompt_id} prompt. Shows perfectionism bias."
            )
        }
        
        return templates.get(failure_mode, "Unknown failure mode pattern.")
    
    def _compute_trace_statistics(self, turns: List[Dict], reference_answer: str) -> Dict:
        """Compute overall statistics for the trace"""
        if not turns:
            return {}
        
        first_answer = turns[0].get("answer", "")
        last_answer = turns[-1].get("answer", "")
        
        initial_correct = self._is_correct(first_answer, reference_answer)
        final_correct = self._is_correct(last_answer, reference_answer)
        
        # Count failure modes
        failure_counts = {mode.value: 0 for mode in FailureMode}
        total_confidence_change = 0
        
        for i in range(len(turns) - 1):
            current_turn = turns[i]
            next_turn = turns[i + 1]
            
            current_correct = self._is_correct(current_turn.get("answer", ""), reference_answer)
            next_correct = self._is_correct(next_turn.get("answer", ""), reference_answer)
            
            failure_mode = self._classify_failure_mode(
                current_turn.get("answer", ""),
                next_turn.get("answer", ""),
                current_correct,
                next_correct
            )
            failure_counts[failure_mode.value] += 1
            
            conf_change = next_turn.get("confidence", 0.5) - current_turn.get("confidence", 0.5)
            total_confidence_change += conf_change
        
        return {
            "total_turns": len(turns),
            "initial_correct": initial_correct,
            "final_correct": final_correct,
            "net_accuracy_change": int(final_correct) - int(initial_correct),
            "failure_mode_counts": failure_counts,
            "avg_confidence_change": round(total_confidence_change / max(1, len(turns)-1), 3),
            "initial_confidence": turns[0].get("confidence", 0.5),
            "final_confidence": turns[-1].get("confidence", 0.5)
        }

# Example usage functions
def create_sample_annotations() -> List[Dict]:
    """Create sample annotations for different failure patterns"""
    
    samples = [
        {
            "trace_snippet": {
                "initial_correct": False,
                "previous_answer": "London",
                "current_answer": "Paris",
                "reprompt_id": "neutral_verification"
            },
            "expected_annotation": {
                "failure_mode": "corrected",
                "delta_accuracy": +1
            }
        },
        {
            "trace_snippet": {
                "initial_correct": True,
                "previous_answer": "Paris", 
                "current_answer": "Lyon",
                "reprompt_id": "adversarial_challenge"
            },
            "expected_annotation": {
                "failure_mode": "overcorrected",
                "delta_accuracy": -1
            }
        },
        {
            "trace_snippet": {
                "initial_correct": False,
                "previous_answer": "Madrid",
                "current_answer": "Madrid",
                "reprompt_id": "critique_revise_specific"
            },
            "expected_annotation": {
                "failure_mode": "anchored",
                "delta_accuracy": 0
            }
        },
        {
            "trace_snippet": {
                "initial_correct": True,
                "previous_answer": "Paris",
                "current_answer": "Paris", 
                "reprompt_id": "concise_confirmation"
            },
            "expected_annotation": {
                "failure_mode": "unchanged_correct",
                "delta_accuracy": 0
            }
        }
    ]
    
    return samples

if __name__ == "__main__":
    # Test annotation system
    annotator = TraceAnnotatorEngine()
    samples = create_sample_annotations()
    
    print("Sample Trace Annotations:")
    for i, sample in enumerate(samples):
        print(f"\nExample {i+1}:")
        print(json.dumps(sample, indent=2))
