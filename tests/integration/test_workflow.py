import pytest
import sys
import pandas as pd
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.workflow import ClassificationWorkflow
except ImportError:
    pytest.skip("Workflow dependencies not available", allow_module_level=True)


class TestClassificationWorkflow:

    # Setup a temporary dataset for tests
    def setup_method(self):
        self.dataset_path = "/tmp/test_dataset.csv"
        data = {
            "prompt": [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "How do you make a basic pasta sauce?",
                "What are the main causes of climate change?",
                "Describe the process of photosynthesis.",
                "What is machine learning?",
                "How does the internet work?",
                "What causes rain?",
                "Explain gravity.",
                "What is photosynthesis?"
            ],
            "response": [
                "This is the first response.",
                "This might be problematic.",
                "Unclear and ambiguous response.",
                "Perfect response without errors.",
                "Answer is wavering and inconsistent.",
                "Another clear response.",
                "Well, it might work this way, but then again...",
                "Based on your question, the answer aligns with expectations.",
                "Simple, direct answer.",
                "This requires extensive analysis of multiple factors..."
            ],
            "error_type": [
                "no_error",
                "prompt_bias",
                "overthinking",
                "no_error",
                "answer_wavering",
                "no_error",
                "answer_wavering",
                "prompt_bias",
                "no_error",
                "overthinking"
            ]
        }
        pd.DataFrame(data).to_csv(self.dataset_path, index=False)

    def test_workflow_initialization(self):
        workflow = ClassificationWorkflow()
        info = workflow.get_workflow_info()
        assert "error_types" in info
        assert len(info["error_types"]) == 6

    def test_process_data(self):
        workflow = ClassificationWorkflow()
        dataset, stats = workflow.prepare_data(self.dataset_path)
        assert "total_samples" in stats
        assert stats["total_samples"] == 10

    # Test running a complete experiment
    def test_run_experiment(self):
        workflow = ClassificationWorkflow()
        result = workflow.run_classification_experiment(
            dataset_path=self.dataset_path,
            model_type="logistic_regression"
        )
        assert "accuracy" in result
        assert result["accuracy"] >= 0

    # Cleanup temporary files after tests
    def teardown_method(self):
        import os
        if os.path.exists(self.dataset_path):
            os.remove(self.dataset_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
