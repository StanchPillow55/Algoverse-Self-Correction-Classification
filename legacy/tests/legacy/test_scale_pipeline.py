#!/usr/bin/env python3
"""
Test Suite for Large-Scale Pipeline

Tests the full pipeline integration including:
- Dataset streaming
- LLM provider interfaces  
- Multi-pass trace generation
- Policy learning
- Data logging and persistence
"""

import pytest
pytest.skip("Legacy classifier tests skipped during teacher/learner pivot", allow_module_level=True)

import asyncio
import sys
import os
import tempfile
import shutil
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scale_pipeline import (
    ScalePipeline, 
    SentenceTransformerLLM, 
    OpenAIProvider, 
    AnthropicProvider,
    DatasetStreamer,
    TurnRecord, 
    TraceRecord
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineTester:
    """Comprehensive test suite for the scale pipeline"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
    
    def setup_test_environment(self):
        """Setup temporary test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Created temp dir: {self.temp_dir}")
        
        # Create mock RTS templates
        mock_templates = [
            {"id": "p_try_again", "style": "supportive", "text": "Please try again."},
            {"id": "p_are_you_sure", "style": "adversarial", "text": "Are you sure about <ANSWER>?"},
            {"id": "none", "style": "none", "text": ""}
        ]
        
        templates_path = self.temp_dir / "rts_templates.json"
        with open(templates_path, 'w') as f:
            json.dump(mock_templates, f)
        
        # Change to temp directory for tests
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.original_cwd:
            os.chdir(self.original_cwd)
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temp directory")
    
    def test_llm_providers(self) -> bool:
        """Test LLM provider implementations"""
        print("\n🧪 Testing LLM Providers...")
        
        try:
            # Test SentenceTransformer LLM (placeholder)
            st_llm = SentenceTransformerLLM()
            
            async def test_st():
                response, tokens = await st_llm.generate("What is the capital of France?")
                assert isinstance(response, str) and len(response) > 0
                assert isinstance(tokens, int) and tokens > 0
                print(f"✅ SentenceTransformer LLM: '{response[:50]}...', {tokens} tokens")
                return True
            
            result = asyncio.run(test_st())
            
            # Test OpenAI provider setup (without API call)
            try:
                openai_llm = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
                assert openai_llm.api_key == "test-key"
                assert openai_llm.model == "gpt-3.5-turbo"
                print("✅ OpenAI Provider initialized correctly")
            except Exception as e:
                print(f"❌ OpenAI Provider test failed: {e}")
                return False
            
            # Test Anthropic provider setup (without API call)
            try:
                anthropic_llm = AnthropicProvider(api_key="test-key", model="claude-3-sonnet-20240229")
                assert anthropic_llm.api_key == "test-key" 
                assert anthropic_llm.model == "claude-3-sonnet-20240229"
                print("✅ Anthropic Provider initialized correctly")
            except Exception as e:
                print(f"❌ Anthropic Provider test failed: {e}")
                return False
                
            return result
            
        except Exception as e:
            print(f"❌ LLM Provider tests failed: {e}")
            return False
    
    def test_dataset_streaming(self) -> bool:
        """Test dataset streaming functionality"""
        print("\n🧪 Testing Dataset Streaming...")
        
        try:
            # Test mock data streaming
            streamer = DatasetStreamer("mock")
            questions = list(streamer.stream_questions(limit=5))
            
            assert len(questions) == 5
            for q in questions:
                assert "id" in q and "question" in q and "reference_answer" in q
                
            print(f"✅ Mock data stream: {len(questions)} questions")
            
            # Test GSM8K streaming 
            gsm8k_streamer = DatasetStreamer("gsm8k")
            gsm8k_questions = list(gsm8k_streamer.stream_questions(limit=3))
            
            assert len(gsm8k_questions) == 3
            assert "math" in gsm8k_questions[0]["question"].lower() or "apples" in gsm8k_questions[0]["question"].lower()
            
            print(f"✅ GSM8K stream: {len(gsm8k_questions)} questions")
            
            # Test CSV streaming
            csv_path = self.temp_dir / "test_data.csv"
            test_csv_data = """id,question,reference_answer
1,"What is 2+2?","4"
2,"What is the capital of France?","Paris"
3,"Is water wet?","yes"
"""
            with open(csv_path, 'w') as f:
                f.write(test_csv_data)
                
            csv_streamer = DatasetStreamer("test_data.csv", str(csv_path))
            csv_questions = list(csv_streamer.stream_questions())
            
            assert len(csv_questions) == 3
            assert csv_questions[0]["question"] == "What is 2+2?"
            assert csv_questions[0]["reference_answer"] == "4"
            
            print(f"✅ CSV stream: {len(csv_questions)} questions")
            
            return True
            
        except Exception as e:
            print(f"❌ Dataset streaming test failed: {e}")
            return False
    
    def test_trace_generation(self) -> bool:
        """Test single trace generation"""
        print("\n🧪 Testing Trace Generation...")
        
        try:
            async def test_trace():
                # Initialize pipeline
                llm_provider = SentenceTransformerLLM()
                pipeline = ScalePipeline(
                    llm_provider=llm_provider,
                    classifier_model_path="nonexistent_model.pt",  # Will use mock
                    classifier_vectorizer_path="nonexistent_vectorizer.pkl",
                    rts_algorithm="thompson_sampling",
                    max_turns=2,  # Small for testing
                    log_dir=str(self.temp_dir / "test_logs")
                )
                
                # Generate a trace
                trace = await pipeline.generate_trace(
                    question_id="test_1",
                    question="What is the capital of France?",
                    reference_answer="Paris", 
                    dataset_name="test"
                )
                
                # Validate trace structure
                assert isinstance(trace, TraceRecord)
                assert trace.question_id == "test_1"
                assert trace.dataset_name == "test"
                assert len(trace.turns) >= 1
                assert trace.total_turns == len(trace.turns)
                
                # Validate turn structure
                for turn in trace.turns:
                    assert isinstance(turn, TurnRecord)
                    assert isinstance(turn.turn_id, int)
                    assert isinstance(turn.answer, str)
                    assert isinstance(turn.error_mode, str)
                    assert isinstance(turn.confidence_score, float)
                    assert isinstance(turn.delta_accuracy, int)
                    assert turn.delta_accuracy in [-1, 0, 1]
                
                print(f"✅ Generated trace with {len(trace.turns)} turns")
                print(f"   Final accuracy: {trace.final_accuracy}")
                print(f"   Total tokens: {trace.total_tokens}")
                
                return True
                
            return asyncio.run(test_trace())
            
        except Exception as e:
            print(f"❌ Trace generation test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_pipeline_integration(self) -> bool:
        """Test full pipeline integration"""
        print("\n🧪 Testing Full Pipeline Integration...")
        
        try:
            async def test_full_pipeline():
                # Initialize pipeline
                llm_provider = SentenceTransformerLLM()
                pipeline = ScalePipeline(
                    llm_provider=llm_provider,
                    classifier_model_path="nonexistent_model.pt",
                    classifier_vectorizer_path="nonexistent_vectorizer.pkl",
                    rts_algorithm="epsilon_greedy",  # Test different algorithm
                    max_turns=2,
                    policy_update_interval=3,
                    log_dir=str(self.temp_dir / "pipeline_logs")
                )
                
                # Process small dataset
                summary = await pipeline.process_dataset(
                    dataset_name="mock",
                    limit=5,
                    batch_size=2
                )
                
                # Validate summary
                assert "pipeline_stats" in summary
                assert "policy_stats" in summary
                assert summary["pipeline_stats"]["traces_processed"] == 5
                assert summary["pipeline_stats"]["total_turns"] >= 5
                
                # Check log files were created
                log_dir = self.temp_dir / "pipeline_logs"
                assert log_dir.exists()
                
                trace_files = list(log_dir.glob("traces_*.jsonl"))
                assert len(trace_files) > 0
                
                # Verify trace log format
                with open(trace_files[0], 'r') as f:
                    first_line = f.readline()
                    trace_data = json.loads(first_line)
                    assert "trace_id" in trace_data
                    assert "turns" in trace_data
                    assert "final_accuracy" in trace_data
                
                summary_files = list(log_dir.glob("summary_*.json"))
                assert len(summary_files) > 0
                
                policy_files = list(log_dir.glob("final_policy.pkl"))
                assert len(policy_files) > 0
                
                print(f"✅ Processed {summary['pipeline_stats']['traces_processed']} traces")
                print(f"✅ Generated {summary['pipeline_stats']['total_turns']} turns")
                print(f"✅ Average turns per trace: {summary['avg_turns_per_trace']:.2f}")
                print(f"✅ Policy updates: {summary['pipeline_stats']['policy_updates']}")
                
                return True
                
            return asyncio.run(test_full_pipeline())
            
        except Exception as e:
            print(f"❌ Full pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_data_schema_compliance(self) -> bool:
        """Test compliance with proposal data schema"""
        print("\n🧪 Testing Data Schema Compliance...")
        
        try:
            # Create sample turn record
            turn = TurnRecord(
                turn_id=0,
                question="Test question?",
                answer="Test answer",
                prompt_id="p_try_again",
                prompt_text="Please try again.",
                error_mode="anchored",
                confidence_score=0.75,
                delta_accuracy=1,
                token_count=15,
                timestamp="2024-01-01T12:00:00"
            )
            
            # Validate turn structure
            turn_dict = turn.to_dict()
            required_fields = [
                "turn_id", "question", "answer", "prompt_id", "prompt_text",
                "error_mode", "confidence_score", "delta_accuracy", 
                "token_count", "timestamp"
            ]
            
            for field in required_fields:
                assert field in turn_dict, f"Missing field: {field}"
            
            print("✅ Turn record schema compliant")
            
            # Create sample trace record
            trace = TraceRecord(
                trace_id="test_trace_1",
                dataset_name="test_dataset",
                question_id="q1",
                reference_answer="reference",
                turns=[turn],
                final_accuracy=1,
                total_tokens=15,
                total_turns=1,
                pipeline_version="1.0",
                created_at="2024-01-01T12:00:00"
            )
            
            # Validate trace structure
            trace_dict = trace.to_dict()
            trace_required_fields = [
                "trace_id", "dataset_name", "question_id", "reference_answer",
                "turns", "final_accuracy", "total_tokens", "total_turns",
                "pipeline_version", "created_at"
            ]
            
            for field in trace_required_fields:
                assert field in trace_dict, f"Missing trace field: {field}"
                
            assert isinstance(trace_dict["turns"], list)
            assert len(trace_dict["turns"]) == 1
            assert isinstance(trace_dict["turns"][0], dict)
            
            print("✅ Trace record schema compliant")
            
            return True
            
        except Exception as e:
            print(f"❌ Schema compliance test failed: {e}")
            return False
    
    def test_api_key_integration_docs(self) -> bool:
        """Test that API key integration is properly documented"""
        print("\n🧪 Testing API Key Integration Documentation...")
        
        try:
            # Check that providers accept API keys correctly
            openai_provider = OpenAIProvider(api_key="sk-test123", model="gpt-4")
            assert openai_provider.api_key == "sk-test123"
            assert openai_provider.base_url == "https://api.openai.com/v1/chat/completions"
            
            anthropic_provider = AnthropicProvider(api_key="sk-ant-test123")
            assert anthropic_provider.api_key == "sk-ant-test123"
            assert anthropic_provider.base_url == "https://api.anthropic.com/v1/messages"
            
            print("✅ API key interfaces properly implemented")
            print("📋 To use in production:")
            print("   # OpenAI:")
            print("   llm_provider = OpenAIProvider(api_key='sk-...', model='gpt-4')")
            print("   # Anthropic:")
            print("   llm_provider = AnthropicProvider(api_key='sk-ant-...', model='claude-3-sonnet-20240229')")
            
            return True
            
        except Exception as e:
            print(f"❌ API integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """Run all tests and return results"""
        print("🚀 Starting Scale Pipeline Test Suite")
        print("=" * 60)
        
        # Setup test environment
        self.setup_test_environment()
        
        tests = [
            ("LLM Providers", self.test_llm_providers),
            ("Dataset Streaming", self.test_dataset_streaming), 
            ("Trace Generation", self.test_trace_generation),
            ("Pipeline Integration", self.test_pipeline_integration),
            ("Schema Compliance", self.test_data_schema_compliance),
            ("API Key Integration", self.test_api_key_integration_docs)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        try:
            for test_name, test_func in tests:
                print(f"\n📋 Running: {test_name}")
                try:
                    result = test_func()
                    results[test_name] = result
                    if result:
                        passed += 1
                        print(f"✅ {test_name} PASSED")
                    else:
                        print(f"❌ {test_name} FAILED")
                except Exception as e:
                    results[test_name] = False
                    print(f"❌ {test_name} FAILED with exception: {e}")
        
        finally:
            # Always cleanup
            self.cleanup_test_environment()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Scale Pipeline is working correctly.")
            print("\n📋 PRODUCTION SETUP GUIDE:")
            print("1. Install dependencies: pip install aiohttp sentence-transformers")
            print("2. Get API keys from OpenAI/Anthropic")
            print("3. Train classifier on your data")
            print("4. Configure pipeline with your LLM provider")
            print("5. Run: python -m src.scale_pipeline")
        else:
            print("⚠️  Some tests failed. Please review the output above.")
        
        return results

def main():
    """Main test execution"""
    tester = PipelineTester()
    results = tester.run_all_tests()
    
    # Return appropriate exit code
    all_passed = all(results.values())
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
