"""
Checkpoint utilities for resumable, idempotent writes.
Provides atomic JSONL writing and completion tracking for long-running experiments.
"""
import json
import os
import random
import time
import fcntl
from typing import Dict, Any, Set, Optional, Tuple, List
from pathlib import Path
import numpy as np
from datetime import datetime
import hashlib

class CheckpointError(Exception):
    """Base exception for checkpoint-related errors"""
    pass

class CheckpointManager:
    """Manages resumable experiment checkpoints with atomic writes and completion tracking."""
    
    def __init__(self, output_path: str, checkpoint_every: int = 10, enable_locking: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            output_path: Path to JSONL checkpoint file
            checkpoint_every: Write resume state every N samples
            enable_locking: Enable file locking for concurrent access
        """
        self.output_path = Path(output_path)
        self.checkpoint_every = checkpoint_every
        self.enable_locking = enable_locking
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint paths
        self.resume_state_path = self.output_path.parent / f"{self.output_path.stem}_resume_state.json"
        self.error_log_path = self.output_path.parent / f"{self.output_path.stem}_errors.log"
        
        # Internal state
        self.completed_qids: Set[str] = set()
        self.sample_count = 0
        self.error_count = 0
        self.last_checkpoint_time = time.time()
        
    def load_completed_ids(self) -> Set[str]:
        """Load set of completed question IDs from existing checkpoint file."""
        completed_ids = set()
        
        if not self.output_path.exists():
            print(f"📋 Starting fresh run - no existing checkpoint at {self.output_path}")
            return completed_ids
        
        print(f"📋 Loading existing checkpoint from {self.output_path}")
        
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        record = json.loads(line)
                        qid = record.get("qid")
                        status = record.get("status")
                        # Only count successful completions, not errors
                        if qid is not None and status == "success":
                            completed_ids.add(str(qid))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Skipping malformed JSON on line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint file: {e}")
        
        print(f"✅ Loaded {len(completed_ids)} completed samples from checkpoint")
        self.completed_qids = completed_ids
        return completed_ids
    
    def append_result_atomic(self, result: Dict[str, Any]) -> None:
        """
        Atomically append a single result to the checkpoint file.
        
        Args:
            result: Dictionary containing sample results
        """
        if not result.get("qid"):
            raise CheckpointError("Result must contain 'qid' field")
        
        # Add metadata
        result.update({
            "checkpoint_time": time.time(),
            "checkpoint_datetime": datetime.now().isoformat()
        })
        
        # Write directly to main file with locking for concurrent safety
        try:
            with open(self.output_path, "a", encoding="utf-8") as f:
                if self.enable_locking:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    if self.enable_locking:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Update internal state
            self.completed_qids.add(str(result["qid"]))
            self.sample_count += 1
            
        except Exception as e:
            raise CheckpointError(f"Failed to write checkpoint: {e}")
    
    def append_error_record(self, qid: str, error: Exception, retryable: bool = True, 
                           error_context: Dict[str, Any] = None) -> None:
        """
        Record an error for a specific sample with enhanced context.
        
        Args:
            qid: Question ID that failed
            error: Exception that occurred
            retryable: Whether this error should be retried on resume
            error_context: Additional context about the error (API provider, model, etc.)
        """
        error_record = {
            "qid": str(qid),
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "retryable": retryable,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat()
        }
        
        # Add additional context if provided
        if error_context:
            error_record.update({"error_context": error_context})
        
        try:
            # Write error record to checkpoint file but don't mark as completed
            # Add metadata
            error_record.update({
                "checkpoint_time": time.time(),
                "checkpoint_datetime": datetime.now().isoformat()
            })
            
            # Write directly to checkpoint file without calling append_result_atomic
            # to avoid marking as completed
            with open(self.output_path, "a", encoding="utf-8") as f:
                if self.enable_locking:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    if self.enable_locking:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Also log to error file for easier debugging
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(f"{error_record['datetime']} | {qid} | {type(error).__name__}: {str(error)}\n")
            
            self.error_count += 1
            print(f"❌ Recorded error for {qid}: {type(error).__name__}")
            
        except Exception as log_error:
            print(f"⚠️ Failed to log error for {qid}: {log_error}")
    
    def should_checkpoint_state(self) -> bool:
        """Check if we should save resume state based on checkpoint frequency."""
        return (self.sample_count % self.checkpoint_every == 0 or
                time.time() - self.last_checkpoint_time > 300)  # Also checkpoint every 5 minutes
    
    def save_resume_state(self, experiment_metadata: Dict[str, Any]) -> None:
        """
        Save current experiment state for resumability.
        
        Args:
            experiment_metadata: Experiment configuration and state
        """
        resume_state = {
            "last_update": datetime.now().isoformat(),
            "completed_count": len(self.completed_qids),
            "error_count": self.error_count,
            "sample_count": self.sample_count,
            "checkpoint_path": str(self.output_path),
            
            # Random state for deterministic resume
            "random_state": {
                "python_random": random.getstate(),
                "numpy_random": self._serialize_numpy_state(np.random.get_state()) if np.__version__ else None,
            },
            
            # Experiment metadata
            "experiment": experiment_metadata,
            
            # System info
            "system": {
                "git_commit": os.getenv("GIT_COMMIT", ""),
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "checkpoint_version": "1.0.0"
            }
        }
        
        # Write atomically
        tmp_path = str(self.resume_state_path) + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(resume_state, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, self.resume_state_path)
            self.last_checkpoint_time = time.time()
            
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            raise CheckpointError(f"Failed to save resume state: {e}")
    
    def load_resume_state(self) -> Optional[Dict[str, Any]]:
        """Load resume state if it exists."""
        if not self.resume_state_path.exists():
            return None
        
        try:
            with open(self.resume_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            print(f"📋 Resume state loaded: {state['completed_count']} completed, {state['error_count']} errors")
            return state
            
        except Exception as e:
            print(f"⚠️ Failed to load resume state: {e}")
            return None
    
    def restore_random_state(self, resume_state: Dict[str, Any]) -> bool:
        """
        Restore random state for deterministic resume.
        
        Args:
            resume_state: State dictionary from load_resume_state()
        
        Returns:
            True if state was restored successfully
        """
        try:
            random_state = resume_state.get("random_state", {})
            
            # Restore Python random state
            if "python_random" in random_state:
                state_data = random_state["python_random"]
                if isinstance(state_data, list):
                    # Convert lists to tuples recursively for older JSON format
                    def list_to_tuple(obj):
                        if isinstance(obj, list):
                            return tuple(list_to_tuple(item) for item in obj)
                        return obj
                    state_tuple = list_to_tuple(state_data)
                else:
                    state_tuple = state_data
                random.setstate(state_tuple)
            
            # Restore NumPy random state  
            if "numpy_random" in random_state and random_state["numpy_random"] is not None:
                np_state = self._deserialize_numpy_state(random_state["numpy_random"])
                if np_state:
                    np.random.set_state(np_state)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to restore random state: {e}")
            return False
    
    def is_completed(self, qid: str) -> bool:
        """Check if a sample is already completed."""
        return str(qid) in self.completed_qids
    
    def _serialize_numpy_state(self, state):
        """Serialize NumPy random state to JSON-compatible format."""
        if state is None:
            return None
        return {
            'state_type': state[0],
            'state_array': state[1].tolist(),
            'pos': state[2],
            'has_gauss': state[3],
            'cached_gaussian': state[4] if len(state) > 4 else None
        }
    
    def _deserialize_numpy_state(self, serialized_state):
        """Deserialize NumPy random state from JSON format."""
        if serialized_state is None:
            return None
        return (
            serialized_state['state_type'],
            np.array(serialized_state['state_array'], dtype=np.uint32),
            serialized_state['pos'],
            serialized_state['has_gauss'],
            serialized_state.get('cached_gaussian')
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current checkpoint statistics with enhanced error analysis."""
        error_analysis = self._analyze_errors()
        
        return {
            "completed_samples": len(self.completed_qids),
            "error_count": self.error_count,
            "total_processed": self.sample_count,
            "checkpoint_file": str(self.output_path),
            "resume_state_file": str(self.resume_state_path),
            "last_checkpoint": datetime.fromtimestamp(self.last_checkpoint_time).isoformat(),
            "error_analysis": error_analysis
        }

    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns from checkpoint file."""
        if not self.output_path.exists():
            return {"total_errors": 0, "error_types": {}, "api_errors": 0}
        
        error_types = {}
        api_error_count = 0
        retryable_errors = 0
        
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        if record.get("status") == "error":
                            error_type = record.get("error_type", "unknown")
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                            
                            # Count API-related errors
                            error_msg = record.get("error_message", "").lower()
                            if any(term in error_msg for term in ['api', 'rate limit', 'timeout', '429', '500', '502', '503']):
                                api_error_count += 1
                            
                            if record.get("retryable", True):
                                retryable_errors += 1
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Warning: Error analyzing checkpoint errors: {e}")
            return {"analysis_error": str(e)}
        
        return {
            "total_errors": sum(error_types.values()),
            "error_types": error_types,
            "api_errors": api_error_count,
            "retryable_errors": retryable_errors,
            "non_retryable_errors": sum(error_types.values()) - retryable_errors
        }
    
    def save_experiment_termination(self, termination_reason: str, 
                                  error_context: Dict[str, Any] = None) -> None:
        """Save experiment termination state to checkpoint."""
        termination_record = {
            "qid": "EXPERIMENT_TERMINATED",
            "status": "terminated",
            "termination_reason": termination_reason,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "completed_samples": len(self.completed_qids),
            "total_errors": self.error_count
        }
        
        if error_context:
            termination_record["termination_context"] = error_context
        
        try:
            self.append_result_atomic(termination_record)
            print(f"🚨 Experiment termination recorded: {termination_reason}")
        except Exception as e:
            print(f"Failed to record experiment termination: {e}")
    
    def check_if_terminated(self) -> Tuple[bool, Optional[str]]:
        """Check if experiment was previously terminated."""
        if not self.output_path.exists():
            return False, None
        
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        if record.get("status") == "terminated":
                            return True, record.get("termination_reason", "Unknown")
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        
        return False, None


def create_stable_run_id(dataset: str, model: str, config_hash: str = None) -> str:
    """
    Create a stable, deterministic run ID for resumable experiments.
    
    Args:
        dataset: Dataset name
        model: Model name
        config_hash: Optional hash of configuration parameters
    
    Returns:
        Stable run ID string
    """
    # Create base from dataset and model
    base = f"{dataset}_{model}"
    
    # Add config hash if provided
    if config_hash:
        base += f"_{config_hash[:8]}"
    
    # Add date (but not time) for daily separation
    date_str = datetime.now().strftime("%Y%m%d")
    
    return f"{date_str}_{base}"

def hash_config(config: Dict[str, Any]) -> str:
    """Create a hash of configuration parameters for stable run IDs."""
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode()).hexdigest()

# Backward compatibility functions
def append_jsonl_atomic(path: str, record: Dict[str, Any]) -> None:
    """Legacy function for backward compatibility."""
    manager = CheckpointManager(path, checkpoint_every=1, enable_locking=False)
    manager.append_result_atomic(record)

def load_completed_ids(path: str) -> Set[str]:
    """Legacy function for backward compatibility."""
    manager = CheckpointManager(path, enable_locking=False)
    return manager.load_completed_ids()