"""
Unified Multi-Turn Error Handling Interface

Provides a consistent interface for handling API errors across different multi-turn experiment runners.
This module acts as a bridge between the core error handling system and various runner implementations.
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path

from .api_error_handler import APIErrorHandler, ErrorHandlingConfig, create_error_handler_from_config
from .checkpoint import CheckpointManager
from .api_health_monitor import APIHealthMonitor


class MultiTurnErrorHandler:
    """
    Unified interface for multi-turn error handling across different runners.
    
    Provides consistent error handling, checkpointing, and health monitoring
    for any multi-turn experiment pipeline.
    """
    
    def __init__(self, 
                 output_dir: Path,
                 experiment_id: str,
                 config: Optional[Dict[str, Any]] = None,
                 learner_type: str = "standard"):
        """
        Initialize multi-turn error handler.
        
        Args:
            output_dir: Output directory for checkpoints and error reports
            experiment_id: Unique experiment identifier
            config: Error handling configuration
            learner_type: Type of learner ("ensemble" or "standard")
        """
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.learner_type = learner_type
        
        # Initialize checkpoint manager
        checkpoint_file = self.output_dir / f"{experiment_id}_checkpoint.jsonl"
        self.checkpoint_manager = CheckpointManager(str(checkpoint_file))
        
        # Initialize API error handler
        error_config = config.get('error_handling', {}) if config else {}
        self.api_error_handler = create_error_handler_from_config(
            self.checkpoint_manager, 
            error_config
        )
        
        # Initialize health monitor
        health_config = config.get('health_monitoring', {}) if config else {}
        self.health_monitor = APIHealthMonitor(
            monitoring_window_hours=health_config.get('monitoring_window_hours', 24)
        )
        
        # Multi-turn specific state
        self.current_sample_id = None
        self.current_turn = 0
        self.sample_start_time = None
        
        # Check for previous termination
        self.was_terminated, self.termination_reason = self.checkpoint_manager.check_if_terminated()
    
    def initialize_experiment(self, dataset_name: str, provider: str, model: str) -> bool:
        """
        Initialize experiment and check if it should proceed.
        
        Returns:
            True if experiment should proceed, False if it should be skipped
        """
        if self.was_terminated:
            print(f"🚨 Experiment was previously terminated: {self.termination_reason}")
            print("Please review error logs and resolve issues before resuming.")
            return False
        
        # Check API health before starting
        should_pause, pause_reason = self.health_monitor.should_pause_experiment()
        if should_pause:
            print(f"⚠️ Experiment paused due to API health: {pause_reason}")
            print("Please check API status and try again later.")
            return False
        
        # Load existing progress
        completed_qids = self.checkpoint_manager.load_completed_ids()
        print(f"📋 Resuming with {len(completed_qids)} completed samples")
        
        return True
    
    def start_sample_processing(self, sample_id: str, question: str) -> bool:
        """
        Start processing a new sample.
        
        Returns:
            True if sample should be processed, False if it should be skipped
        """
        self.current_sample_id = sample_id
        self.current_turn = 0
        self.sample_start_time = time.time()
        
        # Skip if already completed
        if self.checkpoint_manager.is_completed(sample_id):
            print(f"⏭️ Skipping completed sample: {sample_id}")
            return False
        
        # Skip if too many previous errors
        if self.api_error_handler.should_skip_sample(sample_id):
            print(f"⏭️ Skipping sample due to too many errors: {sample_id}")
            return False
        
        # Check if experiment should terminate due to API errors
        if self.api_error_handler.experiment_terminated:
            print(f"🚨 Terminating experiment due to API errors: {self.api_error_handler.termination_reason}")
            self.checkpoint_manager.save_experiment_termination(
                self.api_error_handler.termination_reason,
                error_context=self.api_error_handler.export_error_report()
            )
            return False
        
        return True
    
    def handle_turn_start(self, turn_number: int) -> bool:
        """
        Handle the start of a new turn.
        
        Returns:
            True if turn should proceed, False if it should be skipped
        """
        self.current_turn = turn_number
        
        # Check if we should terminate before starting this turn
        if self.api_error_handler.experiment_terminated:
            return False
        
        return True
    
    def handle_api_call_error(self, error: Exception, provider: str, model: str) -> Tuple[bool, Optional[float]]:
        """
        Handle an API call error during a turn.
        
        Args:
            error: The exception that occurred
            provider: API provider name
            model: Model name
            
        Returns:
            (should_continue, backoff_delay)
        """
        # Record the request as failed
        self.health_monitor.record_request(provider, model, success=False, error=error)
        
        # Use API error handler to process the error
        should_continue, backoff_delay = self.api_error_handler.handle_api_error(
            error, provider, model, self.experiment_id, 
            self.current_sample_id or "unknown", self.current_turn
        )
        
        return should_continue, backoff_delay
    
    def handle_api_call_success(self, provider: str, model: str, response_time: float = 0):
        """Handle a successful API call."""
        self.health_monitor.record_request(provider, model, success=True, response_time=response_time)
    
    def complete_sample_processing(self, sample_data: Dict[str, Any]) -> bool:
        """
        Complete processing of a sample and save to checkpoint.
        
        Args:
            sample_data: Complete sample data including turns
            
        Returns:
            True if successful, False if failed
        """
        try:
            # Prepare result record for checkpoint
            result_record = {
                "qid": self.current_sample_id,
                "status": "success",
                "question": sample_data.get("question", ""),
                "reference": sample_data.get("reference", ""),
                "final_accuracy": sample_data.get("final_accuracy", 0),
                "turns_count": len(sample_data.get("turns", [])),
                "processing_time": time.time() - self.sample_start_time if self.sample_start_time else None
            }
            
            # Save to checkpoint
            self.checkpoint_manager.append_result_atomic(result_record)
            
            # Periodic checkpoint state saving
            if self.checkpoint_manager.should_checkpoint_state():
                experiment_metadata = {
                    "experiment_id": self.experiment_id,
                    "learner_type": self.learner_type,
                    "current_sample": self.current_sample_id,
                    "error_stats": self.api_error_handler.export_error_report()
                }
                self.checkpoint_manager.save_resume_state(experiment_metadata)
                print(f"📋 Checkpoint saved after sample {self.current_sample_id}")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to checkpoint sample {self.current_sample_id}: {e}")
            return False
    
    def handle_sample_error(self, error: Exception, provider: str, model: str) -> bool:
        """
        Handle a sample-level error (non-API error).
        
        Returns:
            True if experiment should continue, False if it should terminate
        """
        try:
            # Use error handler to manage the error
            should_continue, backoff_delay = self.handle_api_call_error(error, provider, model)
            
            if not should_continue:
                return False
            
            if backoff_delay and backoff_delay > 0:
                print(f"Backing off for {backoff_delay:.1f}s before continuing")
                time.sleep(backoff_delay)
            
            # Log error to checkpoint
            error_context = {
                "provider": provider,
                "model": model,
                "experiment_id": self.experiment_id,
                "learner_type": self.learner_type
            }
            self.checkpoint_manager.append_error_record(
                self.current_sample_id or "unknown", 
                error, 
                retryable=True, 
                error_context=error_context
            )
            
            return True
            
        except Exception as handler_error:
            print(f"Error in error handler: {handler_error}")
            return True  # Continue despite handler error
    
    def finalize_experiment(self, traces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Finalize the experiment with comprehensive reporting.
        
        Args:
            traces: List of all sample traces
            
        Returns:
            Dictionary with experiment results and error reports
        """
        try:
            # Calculate final metrics
            total_samples = len(traces)
            final_accuracy = sum(t.get("final_accuracy", 0) for t in traces) / max(1, total_samples)
            
            # Save final checkpoint
            final_experiment_metadata = {
                "experiment_completed": True,
                "experiment_id": self.experiment_id,
                "learner_type": self.learner_type,
                "total_samples_processed": total_samples,
                "final_accuracy": final_accuracy,
                "error_summary": self.api_error_handler.export_error_report()
            }
            self.checkpoint_manager.save_resume_state(final_experiment_metadata)
            
            # Generate health and error reports
            results = {"traces": traces, "summary": {"final_accuracy_mean": final_accuracy}}
            
            # Export health report
            health_report_file = self.health_monitor.export_health_report(self.output_dir)
            print(f"📋 API health report saved: {health_report_file}")
            
            # Generate API error report if there were issues
            if self.api_error_handler.total_api_errors > 0:
                error_report_file = self.output_dir / "api_error_report.json"
                with open(error_report_file, 'w') as f:
                    import json
                    json.dump(self.api_error_handler.export_error_report(), f, indent=2)
                print(f"📄 API error report saved: {error_report_file}")
                
                # Print recovery recommendations
                recommendations = self.api_error_handler.get_recovery_recommendations()
                if recommendations:
                    print("\n🔧 Recovery Recommendations:")
                    for rec in recommendations:
                        print(f"  • {rec}")
            
            # Add error handling results to output
            results["error_handling"] = {
                "total_api_errors": self.api_error_handler.total_api_errors,
                "experiment_terminated": self.api_error_handler.experiment_terminated,
                "health_report_file": str(health_report_file),
                "checkpoint_stats": self.checkpoint_manager.get_stats()
            }
            
            print(f"✅ Final checkpoint saved: {self.checkpoint_manager.get_stats()}")
            
            return results
            
        except Exception as e:
            print(f"Warning: Failed to finalize experiment properly: {e}")
            return {"traces": traces, "error": str(e)}
    
    def get_completed_qids(self) -> set:
        """Get set of completed question IDs."""
        return self.checkpoint_manager.completed_qids
    
    def is_experiment_terminated(self) -> Tuple[bool, Optional[str]]:
        """Check if experiment was terminated due to errors."""
        return self.api_error_handler.experiment_terminated, self.api_error_handler.termination_reason


def create_multi_turn_error_handler(output_dir: Path, 
                                   experiment_id: str,
                                   config: Optional[Dict[str, Any]] = None,
                                   learner_type: str = "standard") -> MultiTurnErrorHandler:
    """
    Factory function to create a multi-turn error handler.
    
    Args:
        output_dir: Output directory for checkpoints and reports
        experiment_id: Unique experiment identifier
        config: Complete experiment configuration
        learner_type: Type of learner ("ensemble" or "standard")
        
    Returns:
        Configured MultiTurnErrorHandler instance
    """
    return MultiTurnErrorHandler(output_dir, experiment_id, config, learner_type)


class MultiTurnAPIWrapper:
    """
    Wrapper for API calls in multi-turn context with automatic error handling.
    
    This class provides a clean interface for making API calls with automatic
    error handling, health monitoring, and recovery.
    """
    
    def __init__(self, error_handler: MultiTurnErrorHandler):
        self.error_handler = error_handler
    
    def safe_api_call(self, 
                     api_function: Callable,
                     provider: str,
                     model: str,
                     *args,
                     max_retries: int = 3,
                     **kwargs) -> Tuple[Any, bool]:
        """
        Make an API call with automatic error handling and retries.
        
        Args:
            api_function: The API function to call
            provider: API provider name
            model: Model name
            max_retries: Maximum number of retries
            *args, **kwargs: Arguments to pass to the API function
            
        Returns:
            (result, success): API result and success flag
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                result = api_function(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Record successful call
                self.error_handler.handle_api_call_success(provider, model, response_time)
                
                return result, True
                
            except Exception as e:
                last_error = e
                
                # Handle the error
                should_continue, backoff_delay = self.error_handler.handle_api_call_error(
                    e, provider, model
                )
                
                if not should_continue:
                    print(f"🚨 API error handler decided to terminate: {e}")
                    return None, False
                
                if attempt < max_retries:
                    if backoff_delay and backoff_delay > 0:
                        print(f"⏱️ Backing off for {backoff_delay:.1f}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(backoff_delay)
                    else:
                        # Default exponential backoff
                        delay = 2 ** attempt
                        print(f"⏱️ Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                else:
                    print(f"❌ Max retries exceeded for {provider}:{model}")
        
        return None, False