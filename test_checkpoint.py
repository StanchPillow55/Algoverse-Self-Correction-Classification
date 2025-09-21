#!/usr/bin/env python3
"""
Test script for the checkpoint system.
Simulates failures and resumption to verify the system works correctly.
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.checkpoint import CheckpointManager, create_stable_run_id


def simulate_sample_processing(qid: str, should_fail: bool = False) -> dict:
    """Simulate processing a sample with optional failure."""
    time.sleep(0.1)  # Simulate work
    
    if should_fail:
        raise Exception(f"Simulated failure for {qid}")
    
    # Simulate successful processing
    return {
        "qid": str(qid),
        "status": "success",
        "final_accuracy": random.choice([0, 1]),
        "turns": [
            {
                "answer": f"answer_{qid}",
                "accuracy": random.choice([0, 1])
            }
        ],
        "simulated": True
    }


def test_checkpoint_basic_flow():
    """Test basic checkpoint functionality."""
    print("🧪 Testing basic checkpoint flow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_checkpoint.jsonl"
        manager = CheckpointManager(str(checkpoint_path), checkpoint_every=3)
        
        # Test 1: Fresh start
        completed = manager.load_completed_ids()
        assert len(completed) == 0, "Fresh start should have no completed IDs"
        print("  ✅ Fresh start working")
        
        # Test 2: Process some samples
        test_samples = ["sample1", "sample2", "sample3", "sample4", "sample5"]
        
        for i, qid in enumerate(test_samples):
            if not manager.is_completed(qid):
                result = simulate_sample_processing(qid)
                manager.append_result_atomic(result)
                
                # Test periodic state saving
                if manager.should_checkpoint_state():
                    manager.save_resume_state({"test": "metadata"})
        
        print(f"  ✅ Processed {len(test_samples)} samples")
        
        # Test 3: Resume from checkpoint
        manager2 = CheckpointManager(str(checkpoint_path), checkpoint_every=3)
        completed2 = manager2.load_completed_ids()
        
        assert len(completed2) == len(test_samples), f"Resume should find {len(test_samples)} completed samples"
        
        for qid in test_samples:
            assert manager2.is_completed(qid), f"Sample {qid} should be marked as completed"
        
        print("  ✅ Resume functionality working")
        
        # Test 4: Skip completed samples
        skipped = 0
        for qid in test_samples:
            if manager2.is_completed(qid):
                skipped += 1
        
        assert skipped == len(test_samples), "All samples should be skipped on resume"
        print("  ✅ Skip completed samples working")
        
        print("✅ Basic checkpoint flow: PASSED")


def test_error_handling():
    """Test error handling and recovery."""
    print("\n🧪 Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_error_checkpoint.jsonl"
        manager = CheckpointManager(str(checkpoint_path), checkpoint_every=2)
        
        test_samples = ["good1", "fail1", "good2", "fail2", "good3"]
        
        for qid in test_samples:
            if not manager.is_completed(qid):
                try:
                    should_fail = qid.startswith("fail")
                    result = simulate_sample_processing(qid, should_fail=should_fail)
                    manager.append_result_atomic(result)
                    print(f"  ✅ Processed {qid}")
                    
                except Exception as e:
                    manager.append_error_record(qid, e, retryable=True)
                    print(f"  ❌ Failed {qid}: {type(e).__name__}")
        
        # Verify checkpoint file contains both successes and errors
        stats = manager.get_stats()
        print(f"  📊 Stats: {stats['completed_samples']} completed, {stats['error_count']} errors")
        
        assert stats["completed_samples"] == 3, "Should have 3 successful samples"
        assert stats["error_count"] == 2, "Should have 2 error samples"
        
        print("✅ Error handling: PASSED")


def test_concurrent_writes():
    """Test concurrent write safety (simplified)."""
    print("\n🧪 Testing concurrent write safety...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_concurrent.jsonl"
        
        # Simulate concurrent writes by multiple managers
        managers = [
            CheckpointManager(str(checkpoint_path), checkpoint_every=10, enable_locking=True)
            for _ in range(3)
        ]
        
        # Each manager writes different samples
        for i, manager in enumerate(managers):
            for j in range(3):
                qid = f"manager{i}_sample{j}"
                result = simulate_sample_processing(qid)
                manager.append_result_atomic(result)
        
        # Verify all writes succeeded
        final_manager = CheckpointManager(str(checkpoint_path))
        completed = final_manager.load_completed_ids()
        
        assert len(completed) == 9, f"Expected 9 samples, got {len(completed)}"
        print("✅ Concurrent writes: PASSED")


def test_resume_state_persistence():
    """Test resume state save/load."""
    print("\n🧪 Testing resume state persistence...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "test_resume.jsonl"
        manager = CheckpointManager(str(checkpoint_path), checkpoint_every=1)
        
        # Set up initial state
        random.seed(42)
        original_state = random.getstate()
        
        # Process a sample and save state
        result = simulate_sample_processing("test_resume")
        manager.append_result_atomic(result)
        
        metadata = {"test": "resume_metadata", "experiment_id": "test_123"}
        manager.save_resume_state(metadata)
        
        # Create new manager and restore state
        manager2 = CheckpointManager(str(checkpoint_path))
        completed = manager2.load_completed_ids()
        resume_state = manager2.load_resume_state()
        
        assert len(completed) == 1, "Should find 1 completed sample"
        assert resume_state is not None, "Resume state should exist"
        assert resume_state["experiment"]["test"] == "resume_metadata"
        
        # Test random state restoration
        restored = manager2.restore_random_state(resume_state)
        assert restored, "Random state restoration should succeed"
        
        print("✅ Resume state persistence: PASSED")


def test_stable_run_ids():
    """Test stable run ID generation."""
    print("\n🧪 Testing stable run ID generation...")
    
    # Same inputs should generate same ID
    id1 = create_stable_run_id("gsm8k", "gpt-4", "config123")
    id2 = create_stable_run_id("gsm8k", "gpt-4", "config123")
    assert id1 == id2, "Same inputs should generate same run ID"
    
    # Different inputs should generate different IDs
    id3 = create_stable_run_id("gsm8k", "gpt-3.5", "config123")
    assert id1 != id3, "Different models should generate different run IDs"
    
    id4 = create_stable_run_id("math20", "gpt-4", "config123")
    assert id1 != id4, "Different datasets should generate different run IDs"
    
    print("✅ Stable run IDs: PASSED")


def run_integration_test():
    """Run a more realistic integration test."""
    print("\n🔬 Running integration test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Simulate interrupted experiment
        checkpoint_path = Path(temp_dir) / "integration_test.jsonl"
        
        # Phase 1: Initial run (interrupted after 7 samples)
        print("  📤 Phase 1: Initial run...")
        manager1 = CheckpointManager(str(checkpoint_path), checkpoint_every=3)
        
        total_samples = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]
        
        # Process first 7 samples (simulating interruption)
        for i, qid in enumerate(total_samples[:7]):
            if not manager1.is_completed(qid):
                try:
                    # Simulate occasional failures
                    should_fail = (i == 2 or i == 5)  # q3 and q6 fail
                    result = simulate_sample_processing(qid, should_fail=should_fail)
                    manager1.append_result_atomic(result)
                    
                    if manager1.should_checkpoint_state():
                        manager1.save_resume_state({
                            "phase": 1,
                            "batch": i // 3
                        })
                        
                except Exception as e:
                    manager1.append_error_record(qid, e, retryable=True)
        
        stats1 = manager1.get_stats()
        print(f"    Phase 1 complete: {stats1['completed_samples']} completed, {stats1['error_count']} errors")
        
        # Phase 2: Resume and continue
        print("  🔄 Phase 2: Resume and continue...")
        manager2 = CheckpointManager(str(checkpoint_path), checkpoint_every=3)
        completed = manager2.load_completed_ids()
        resume_state = manager2.load_resume_state()
        
        if resume_state:
            manager2.restore_random_state(resume_state)
        
        print(f"    Resumed with {len(completed)} completed samples")
        
        # Continue with remaining samples
        for qid in total_samples:
            if not manager2.is_completed(qid):
                try:
                    result = simulate_sample_processing(qid)
                    manager2.append_result_atomic(result)
                    
                    if manager2.should_checkpoint_state():
                        manager2.save_resume_state({
                            "phase": 2,
                            "resumed": True
                        })
                        
                except Exception as e:
                    manager2.append_error_record(qid, e, retryable=True)
        
        stats2 = manager2.get_stats()
        print(f"    Phase 2 complete: {stats2['completed_samples']} completed, {stats2['error_count']} errors")
        
        # Verify final state
        assert stats2['completed_samples'] >= 8, "Should have at least 8 successful samples"
        print("✅ Integration test: PASSED")


def main():
    parser = argparse.ArgumentParser(description="Test checkpoint system functionality")
    parser.add_argument("--test", choices=[
        "basic", "error", "concurrent", "resume", "runid", "integration", "all"
    ], default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    print("🧪 CHECKPOINT SYSTEM TEST SUITE")
    print("=" * 50)
    
    if args.test in ("basic", "all"):
        test_checkpoint_basic_flow()
    
    if args.test in ("error", "all"):
        test_error_handling()
    
    if args.test in ("concurrent", "all"):
        test_concurrent_writes()
    
    if args.test in ("resume", "all"):
        test_resume_state_persistence()
    
    if args.test in ("runid", "all"):
        test_stable_run_ids()
    
    if args.test in ("integration", "all"):
        run_integration_test()
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("\nYour checkpoint system is working correctly.")
    print("You can now run large-scale experiments with confidence,")
    print("knowing that interruptions won't lose your progress.")


if __name__ == "__main__":
    main()