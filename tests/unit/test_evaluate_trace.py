"""
Unit tests for trace evaluation functionality.

Tests the core evaluation metrics including exact match, F1 score, 
token counting, and cost efficiency calculations.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from evaluate_trace import (
    evaluate_trace, 
    normalize_answer, 
    compute_f1_score,
    extract_final_answer,
    count_total_tokens,
    count_reprompts,
    evaluate_batch,
    create_sample_trace,
    TraceMetrics
)

class TestNormalizeAnswer:
    """Test answer normalization for comparison"""
    
    def test_basic_normalization(self):
        """Test basic string normalization"""
        assert normalize_answer("  Hello World  ") == "hello world"
        assert normalize_answer("Yes!") == "yes"
        assert normalize_answer("NO?") == "no"
    
    def test_numerical_extraction(self):
        """Test numerical answer extraction"""
        assert normalize_answer("The answer is 42.") == "42"
        assert normalize_answer("42") == "42"
        assert normalize_answer("The result is 3.14") == "3.14"
        assert normalize_answer("-5") == "-5"
        # Non-extractable cases should return full normalized text
        assert normalize_answer("I think it's complex") == "i think it's complex"
    
    def test_boolean_normalization(self):
        """Test yes/no answer normalization"""
        assert normalize_answer("Yes") == "yes"
        assert normalize_answer("y") == "yes"
        assert normalize_answer("true") == "yes"
        assert normalize_answer("1") == "yes"
        assert normalize_answer("No") == "no"
        assert normalize_answer("n") == "no"
        assert normalize_answer("false") == "no"
        assert normalize_answer("0") == "no"
    
    def test_empty_or_invalid_input(self):
        """Test handling of empty or invalid inputs"""
        assert normalize_answer("") == ""
        assert normalize_answer(None) == ""
        assert normalize_answer(123) == ""

class TestComputeF1Score:
    """Test F1 score computation between prediction and reference"""
    
    def test_exact_match(self):
        """Test F1 score for exact matches"""
        assert compute_f1_score("hello world", "hello world") == 1.0
        assert compute_f1_score("42", "42") == 1.0
    
    def test_partial_match(self):
        """Test F1 score for partial matches"""
        # 50% overlap: "hello" is common, "world" vs "there" differ
        # Precision = 1/2 = 0.5, Recall = 1/2 = 0.5, F1 = 0.5
        f1 = compute_f1_score("hello world", "hello there")
        assert f1 == 0.5
    
    def test_no_match(self):
        """Test F1 score for no matches"""
        assert compute_f1_score("hello", "goodbye") == 0.0
        assert compute_f1_score("42", "84") == 0.0
    
    def test_empty_strings(self):
        """Test F1 score for empty strings"""
        assert compute_f1_score("", "") == 1.0
        assert compute_f1_score("hello", "") == 0.0
        assert compute_f1_score("", "world") == 0.0

class TestExtractFinalAnswer:
    """Test extraction of final answer from trace"""
    
    def test_single_turn_trace(self):
        """Test extraction from single turn"""
        trace = {
            'turns': [
                {'answer': 'final answer'}
            ]
        }
        assert extract_final_answer(trace) == 'final answer'
    
    def test_multi_turn_trace(self):
        """Test extraction from multi-turn trace"""
        trace = {
            'turns': [
                {'answer': 'first answer'},
                {'revised_answer': 'final answer'}
            ]
        }
        assert extract_final_answer(trace) == 'final answer'
    
    def test_different_answer_keys(self):
        """Test handling of different answer key names"""
        trace = {
            'turns': [
                {'response': 'final response'}
            ]
        }
        assert extract_final_answer(trace) == 'final response'
    
    def test_invalid_trace(self):
        """Test handling of invalid traces"""
        assert extract_final_answer({}) == ""
        assert extract_final_answer({'turns': []}) == ""
        assert extract_final_answer({'turns': [{}]}) == ""

class TestCountTokens:
    """Test token counting functionality"""
    
    def test_explicit_token_counts(self):
        """Test counting explicit token fields"""
        trace = {
            'turns': [
                {'input_tokens': 10, 'output_tokens': 5},
                {'input_tokens': 15, 'output_tokens': 8}
            ]
        }
        assert count_total_tokens(trace) == 38  # 10+5+15+8
    
    def test_text_estimation(self):
        """Test token estimation from text length"""
        trace = {
            'turns': [
                {'question': 'What is 2+2?', 'answer': 'Four'}  # ~16 chars / 4 = 4 tokens
            ]
        }
        estimated = count_total_tokens(trace)
        assert estimated >= 1  # Should be at least 1
        assert estimated <= 10  # Should be reasonable estimate

class TestCountReprompts:
    """Test reprompt counting"""
    
    def test_explicit_reprompt_markers(self):
        """Test counting with explicit reprompt markers"""
        trace = {
            'turns': [
                {'answer': 'first'},
                {'reprompt': 'try again', 'answer': 'second'},
                {'is_reprompt': True, 'answer': 'third'}
            ]
        }
        assert count_reprompts(trace) == 2
    
    def test_implicit_reprompt_counting(self):
        """Test counting reprompts by turn count"""
        trace = {
            'turns': [
                {'answer': 'first'},
                {'answer': 'second'},
                {'answer': 'third'}
            ]
        }
        assert count_reprompts(trace) == 2  # All turns after first

class TestEvaluateTrace:
    """Test main trace evaluation function"""
    
    def test_correct_answer_no_reprompts(self):
        """Test evaluation of correct answer without reprompts"""
        trace = create_sample_trace(
            "What is 2+2?",
            [{'answer': '4', 'input_tokens': 10, 'output_tokens': 5}]
        )
        result = evaluate_trace(trace, '4')
        
        assert result['exact_match'] == 1.0
        assert result['f1'] == 1.0
        assert result['num_prompts'] == 0
        assert result['total_tokens'] == 15
        assert result['accuracy_per_cost'] == 1.0 / 15
    
    def test_incorrect_answer(self):
        """Test evaluation of incorrect answer"""
        trace = create_sample_trace(
            "What is 2+2?",
            [{'answer': '5', 'input_tokens': 10, 'output_tokens': 5}]
        )
        result = evaluate_trace(trace, '4')
        
        assert result['exact_match'] == 0.0
        assert result['f1'] == 0.0  # No token overlap between '5' and '4'
        assert result['num_prompts'] == 0
        assert result['accuracy_per_cost'] == 0.0
    
    def test_correction_after_reprompt(self):
        """Test evaluation of correction after reprompt"""
        trace = create_sample_trace(
            "What is 2+2?",
            [
                {'answer': '5', 'input_tokens': 10, 'output_tokens': 5},
                {'reprompt': 'Are you sure?', 'answer': '4', 'input_tokens': 15, 'output_tokens': 5}
            ]
        )
        result = evaluate_trace(trace, '4')
        
        assert result['exact_match'] == 1.0
        assert result['f1'] == 1.0
        assert result['num_prompts'] == 1
        assert result['total_tokens'] == 35  # 10+5+15+5
        assert result['accuracy_per_cost'] == 1.0 / 35
    
    def test_partial_match_f1(self):
        """Test F1 score for partial matches"""
        trace = create_sample_trace(
            "What color is the sky?",
            [{'answer': 'blue sky', 'input_tokens': 10, 'output_tokens': 5}]
        )
        result = evaluate_trace(trace, 'blue')
        
        assert result['exact_match'] == 0.0  # Not exact match
        assert result['f1'] > 0.0  # But has partial overlap
        assert result['f1'] < 1.0
    
    def test_boolean_qa_evaluation(self):
        """Test evaluation for boolean questions"""
        trace = create_sample_trace(
            "Is the sky blue?",
            [{'answer': 'Yes', 'input_tokens': 8, 'output_tokens': 3}]
        )
        result = evaluate_trace(trace, 'yes')
        
        assert result['exact_match'] == 1.0
        assert result['f1'] == 1.0
    
    def test_reference_as_dict(self):
        """Test handling reference as dictionary"""
        trace = create_sample_trace(
            "What is 2+2?",
            [{'answer': '4', 'input_tokens': 10, 'output_tokens': 5}]
        )
        reference = {'answer': '4', 'explanation': 'Simple addition'}
        result = evaluate_trace(trace, reference)
        
        assert result['exact_match'] == 1.0
        assert result['reference_answer'] == '4'

class TestEvaluateBatch:
    """Test batch evaluation functionality"""
    
    def test_batch_evaluation(self):
        """Test evaluation of multiple traces"""
        traces = [
            create_sample_trace("2+2?", [{'answer': '4', 'input_tokens': 10, 'output_tokens': 5}]),
            create_sample_trace("3+3?", [{'answer': '7', 'input_tokens': 10, 'output_tokens': 5}]),
            create_sample_trace("4+4?", [{'answer': '8', 'input_tokens': 10, 'output_tokens': 5}])
        ]
        references = ['4', '6', '8']
        
        result = evaluate_batch(traces, references)
        
        # Should have 2/3 correct (first and third)
        assert result['aggregate_metrics']['avg_exact_match'] == 2.0 / 3.0
        assert result['aggregate_metrics']['total_traces'] == 3
        assert len(result['per_trace_results']) == 3
    
    def test_batch_size_mismatch(self):
        """Test error handling for mismatched batch sizes"""
        traces = [create_sample_trace("test", [{'answer': 'test'}])]
        references = ['ref1', 'ref2']  # Different size
        
        with pytest.raises(ValueError):
            evaluate_batch(traces, references)

class TestTraceMetrics:
    """Test TraceMetrics dataclass"""
    
    def test_metrics_creation(self):
        """Test creation of TraceMetrics"""
        metrics = TraceMetrics(
            exact_match=1.0,
            f1=0.85,
            num_prompts=2,
            accuracy_per_cost=0.025,
            total_tokens=40,
            final_answer="42",
            reference_answer="42"
        )
        
        assert metrics.exact_match == 1.0
        assert metrics.f1 == 0.85
        assert metrics.num_prompts == 2
    
    def test_metrics_to_dict(self):
        """Test conversion of metrics to dictionary"""
        metrics = TraceMetrics(
            exact_match=1.0,
            f1=0.85,
            num_prompts=2,
            accuracy_per_cost=0.025,
            total_tokens=40,
            final_answer="42",
            reference_answer="42"
        )
        
        result_dict = metrics.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['exact_match'] == 1.0
        assert result_dict['f1'] == 0.85
        assert 'final_answer' in result_dict
        assert 'reference_answer' in result_dict

@pytest.fixture
def sample_traces():
    """Fixture providing sample traces for testing"""
    return [
        create_sample_trace(
            "Math problem with correction",
            [
                {'answer': '6', 'input_tokens': 10, 'output_tokens': 5},
                {'reprompt': 'Check again', 'answer': '5', 'input_tokens': 15, 'output_tokens': 5}
            ]
        ),
        create_sample_trace(
            "Boolean question",
            [{'answer': 'yes', 'input_tokens': 8, 'output_tokens': 3}]
        ),
        create_sample_trace(
            "Wrong answer maintained",
            [
                {'answer': '10', 'input_tokens': 10, 'output_tokens': 5},
                {'reprompt': 'Are you sure?', 'answer': '10', 'input_tokens': 15, 'output_tokens': 5}
            ]
        )
    ]

class TestIntegrationScenarios:
    """Integration tests for realistic scenarios"""
    
    def test_math_correction_scenario(self, sample_traces):
        """Test realistic math correction scenario"""
        trace = sample_traces[0]
        result = evaluate_trace(trace, '5')  # 2+3=5, not 6
        
        assert result['exact_match'] == 1.0  # Finally got it right
        assert result['num_prompts'] == 1    # One reprompt was used
        assert result['total_tokens'] == 35  # Total token cost
        assert result['accuracy_per_cost'] < 1.0 / 15  # Less efficient than single-turn
    
    def test_boolean_qa_scenario(self, sample_traces):
        """Test boolean QA scenario"""
        trace = sample_traces[1]
        result = evaluate_trace(trace, True)  # Pass boolean reference
        
        assert result['exact_match'] == 1.0
        assert result['num_prompts'] == 0
        assert result['total_tokens'] == 11
    
    def test_persistent_error_scenario(self, sample_traces):
        """Test scenario where error persists despite reprompting"""
        trace = sample_traces[2]
        result = evaluate_trace(trace, '8')  # Correct answer is 8, not 10
        
        assert result['exact_match'] == 0.0  # Still wrong after reprompt
        assert result['num_prompts'] == 1    # One reprompt was attempted
        assert result['accuracy_per_cost'] == 0.0  # No accuracy gain despite cost
    
    def test_cost_efficiency_comparison(self, sample_traces):
        """Test cost efficiency comparison between scenarios"""
        results = []
        references = ['5', 'yes', '8']
        
        for trace, ref in zip(sample_traces, references):
            result = evaluate_trace(trace, ref)
            results.append(result['accuracy_per_cost'])
        
        # Boolean QA (single turn, correct) should be most efficient
        # Math correction (reprompt, correct) should be less efficient
        # Persistent error (reprompt, wrong) should be least efficient
        assert results[1] > results[0] > results[2]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
