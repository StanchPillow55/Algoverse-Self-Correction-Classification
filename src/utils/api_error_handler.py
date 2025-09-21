"""
Enhanced API Error Handler with Checkpointing Integration

Provides comprehensive error handling for API issues in ensemble and multi-turn experiments,
with automatic checkpointing and graceful experiment termination.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .checkpoint import CheckpointManager, CheckpointError


class APIErrorType(Enum):
    """Classification of API error types"""
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication" 
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    QUOTA_EXCEEDED = "quota_exceeded"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INSUFFICIENT_CREDITS = "insufficient_credits"


class ErrorSeverity(Enum):
    """Severity levels for error handling decisions"""
    LOW = "low"          # Continue with warnings
    MEDIUM = "medium"    # Checkpoint and retry with backoff
    HIGH = "high"        # Checkpoint and terminate gracefully
    CRITICAL = "critical" # Immediate termination


@dataclass
class APIError:
    """Structured representation of an API error"""
    error_type: APIErrorType
    severity: ErrorSeverity
    provider: str
    model: str
    message: str
    timestamp: float
    retry_after: Optional[float] = None
    recoverable: bool = True
    experiment_id: str = "unknown"
    sample_id: str = "unknown"
    turn_number: int = 0


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling behavior"""
    max_api_errors_per_sample: int = 3
    max_total_api_errors: int = 50
    max_consecutive_failures: int = 5
    checkpoint_on_error: bool = True
    terminate_on_quota_exceeded: bool = True
    terminate_on_auth_failure: bool = True
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 300.0
    min_backoff_seconds: float = 1.0
    ensemble_failure_threshold: float = 0.5  # Terminate if >50% of ensemble models fail


class APIErrorHandler:
    """Enhanced error handler with checkpointing integration"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, config: Optional[ErrorHandlingConfig] = None):
        self.checkpoint_manager = checkpoint_manager
        self.config = config or ErrorHandlingConfig()
        
        # Error tracking
        self.error_history: List[APIError] = []
        self.errors_by_sample: Dict[str, List[APIError]] = {}
        self.consecutive_failures = 0
        self.total_api_errors = 0
        
        # State tracking
        self.experiment_terminated = False
        self.termination_reason = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_error_logging()
    
    def _setup_error_logging(self):
        """Set up dedicated error logging"""
        log_dir = self.checkpoint_manager.output_path.parent / "error_logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create error-specific log file
        error_log_file = log_dir / "api_errors.log"
        handler = logging.FileHandler(error_log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler if not already present
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(error_log_file) 
                  for h in self.logger.handlers):
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def classify_error(self, exception: Exception, provider: str = "unknown") -> APIErrorType:
        """Classify an API error based on exception details"""
        error_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # Rate limiting errors
        if any(term in error_str for term in ['rate limit', '429', 'too many requests', 'rate_limit_exceeded']):
            return APIErrorType.RATE_LIMIT
        
        # Authentication errors
        if any(term in error_str for term in ['unauthorized', '401', 'authentication', 'invalid api key', 'api_key']):
            return APIErrorType.AUTHENTICATION
        
        # Timeout errors
        if any(term in error_str for term in ['timeout', 'timed out', 'read timeout', 'connection timeout']):
            return APIErrorType.TIMEOUT
        
        # Service unavailable
        if any(term in error_str for term in ['503', '502', '500', '504', 'service unavailable', 'server error']):
            return APIErrorType.SERVICE_UNAVAILABLE
        
        # Quota/billing issues
        if any(term in error_str for term in ['quota', 'billing', 'insufficient funds', 'credits', 'usage limit']):
            return APIErrorType.QUOTA_EXCEEDED if 'quota' in error_str else APIErrorType.INSUFFICIENT_CREDITS
        
        # Context length issues
        if any(term in error_str for term in ['context length', 'context_length', 'maximum context', 'token limit']):
            return APIErrorType.CONTEXT_LENGTH_EXCEEDED
        
        # Network errors
        if any(term in error_str for term in ['connection', 'network', 'dns', 'host', 'unreachable']):
            return APIErrorType.NETWORK_ERROR
        
        return APIErrorType.UNKNOWN_ERROR
    
    def determine_severity(self, error_type: APIErrorType, consecutive_count: int) -> ErrorSeverity:
        """Determine error severity based on type and context"""
        # Critical errors that should terminate immediately
        if error_type in [APIErrorType.AUTHENTICATION, APIErrorType.QUOTA_EXCEEDED]:
            return ErrorSeverity.CRITICAL
        
        # High severity for persistent issues
        if consecutive_count >= self.config.max_consecutive_failures:
            return ErrorSeverity.HIGH
        
        # Medium severity for recoverable but serious issues
        if error_type in [APIErrorType.SERVICE_UNAVAILABLE, APIErrorType.INSUFFICIENT_CREDITS]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for common, recoverable issues
        return ErrorSeverity.LOW
    
    def handle_api_error(self, exception: Exception, provider: str, model: str, 
                        experiment_id: str, sample_id: str, turn_number: int) -> Tuple[bool, Optional[float]]:
        """
        Handle an API error with comprehensive logging and decision making.
        
        Returns:
            (should_continue, backoff_delay)
        """
        error_type = self.classify_error(exception, provider)
        severity = self.determine_severity(error_type, self.consecutive_failures)
        
        # Create structured error record
        api_error = APIError(
            error_type=error_type,
            severity=severity,
            provider=provider,
            model=model,
            message=str(exception),
            timestamp=time.time(),
            retry_after=self._extract_retry_after(exception),
            recoverable=self._is_recoverable_error(error_type),
            experiment_id=experiment_id,
            sample_id=sample_id,
            turn_number=turn_number
        )
        
        # Track the error
        self._track_error(api_error)
        
        # Log the error
        self.logger.error(f"API Error [{error_type.value}]: {provider}:{model} - {exception}")
        
        # Checkpoint current state if configured
        if self.config.checkpoint_on_error:
            self._emergency_checkpoint(api_error)
        
        # Make termination decision
        should_terminate = self._should_terminate_experiment(api_error)
        
        if should_terminate:
            self._terminate_experiment(api_error)
            return False, None
        
        # Calculate backoff delay if continuing
        backoff_delay = self._calculate_backoff_delay(api_error)
        
        # Reset consecutive failures if this is a different error type
        if len(self.error_history) > 1 and self.error_history[-2].error_type != error_type:
            self.consecutive_failures = 1
        else:
            self.consecutive_failures += 1
        
        return True, backoff_delay
    
    def _track_error(self, error: APIError):
        """Track error in internal state"""
        self.error_history.append(error)
        self.total_api_errors += 1
        
        # Track errors per sample
        if error.sample_id not in self.errors_by_sample:
            self.errors_by_sample[error.sample_id] = []
        self.errors_by_sample[error.sample_id].append(error)
    
    def _emergency_checkpoint(self, error: APIError):
        """Save emergency checkpoint when API error occurs"""
        try:
            emergency_metadata = {
                "emergency_checkpoint": True,
                "trigger_error": asdict(error),
                "total_api_errors": self.total_api_errors,
                "consecutive_failures": self.consecutive_failures,
                "error_summary": self._get_error_summary(),
                "timestamp": datetime.now().isoformat()
            }
            
            self.checkpoint_manager.save_resume_state(emergency_metadata)
            
            # Also log the error to checkpoint file
            self.checkpoint_manager.append_error_record(
                error.sample_id, 
                Exception(f"{error.error_type.value}: {error.message}"),
                retryable=error.recoverable
            )
            
            self.logger.info(f"Emergency checkpoint saved for error: {error.error_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency checkpoint: {e}")
    
    def _should_terminate_experiment(self, error: APIError) -> bool:
        """Determine if experiment should be terminated based on error"""
        
        # Immediate termination conditions
        if error.severity == ErrorSeverity.CRITICAL:
            self.termination_reason = f"Critical error: {error.error_type.value}"
            return True
        
        if error.error_type == APIErrorType.AUTHENTICATION and self.config.terminate_on_auth_failure:
            self.termination_reason = "Authentication failure"
            return True
        
        if error.error_type in [APIErrorType.QUOTA_EXCEEDED, APIErrorType.INSUFFICIENT_CREDITS] and self.config.terminate_on_quota_exceeded:
            self.termination_reason = "Quota/billing limits exceeded"
            return True
        
        # Threshold-based termination
        if self.total_api_errors >= self.config.max_total_api_errors:
            self.termination_reason = f"Total API errors exceeded threshold ({self.config.max_total_api_errors})"
            return True
        
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.termination_reason = f"Consecutive failures exceeded threshold ({self.config.max_consecutive_failures})"
            return True
        
        # Sample-specific error limits
        sample_errors = len(self.errors_by_sample.get(error.sample_id, []))
        if sample_errors >= self.config.max_api_errors_per_sample:
            self.logger.warning(f"Sample {error.sample_id} exceeded error limit, will skip")
            # Don't terminate entire experiment for single sample
        
        return False
    
    def _terminate_experiment(self, error: APIError):
        """Terminate experiment gracefully"""
        self.experiment_terminated = True
        
        termination_record = {
            "experiment_terminated": True,
            "termination_reason": self.termination_reason,
            "terminating_error": asdict(error),
            "total_api_errors": self.total_api_errors,
            "consecutive_failures": self.consecutive_failures,
            "termination_timestamp": datetime.now().isoformat(),
            "error_summary": self._get_error_summary()
        }
        
        # Save termination state
        try:
            termination_file = self.checkpoint_manager.output_path.parent / "experiment_termination.json"
            with open(termination_file, 'w') as f:
                json.dump(termination_record, f, indent=2)
            
            self.logger.critical(f"Experiment terminated: {self.termination_reason}")
            print(f"🚨 Experiment terminated due to API issues: {self.termination_reason}")
            
        except Exception as e:
            self.logger.error(f"Failed to save termination record: {e}")
    
    def _calculate_backoff_delay(self, error: APIError) -> float:
        """Calculate appropriate backoff delay"""
        if error.retry_after:
            return min(error.retry_after, self.config.max_backoff_seconds)
        
        # Exponential backoff based on consecutive failures
        base_delay = self.config.min_backoff_seconds
        exponential_delay = base_delay * (self.config.backoff_multiplier ** (self.consecutive_failures - 1))
        
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0.8, 1.2)
        delay = exponential_delay * jitter
        
        return min(delay, self.config.max_backoff_seconds)
    
    def _extract_retry_after(self, exception: Exception) -> Optional[float]:
        """Extract Retry-After header value from exception"""
        # This is similar to rate_limit.py implementation
        if hasattr(exception, 'response') and hasattr(exception.response, 'headers'):
            retry_after = exception.response.headers.get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
        
        # Check exception message
        error_str = str(exception).lower()
        if 'retry after' in error_str:
            import re
            match = re.search(r'retry after (\\d+(?:\\.\\d+)?)', error_str)
            if match:
                return float(match.group(1))
        
        return None
    
    def _is_recoverable_error(self, error_type: APIErrorType) -> bool:
        """Determine if an error type is generally recoverable"""
        non_recoverable = {
            APIErrorType.AUTHENTICATION,
            APIErrorType.QUOTA_EXCEEDED,
            APIErrorType.CONTEXT_LENGTH_EXCEEDED
        }
        return error_type not in non_recoverable
    
    def _get_error_summary(self) -> Dict[str, Any]:
        """Generate summary of error patterns"""
        if not self.error_history:
            return {}
        
        error_counts = {}
        for error in self.error_history:
            key = f"{error.provider}:{error.error_type.value}"
            error_counts[key] = error_counts.get(key, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_breakdown": error_counts,
            "affected_samples": len(self.errors_by_sample),
            "time_span_hours": (self.error_history[-1].timestamp - self.error_history[0].timestamp) / 3600,
            "most_common_error": max(error_counts.items(), key=lambda x: x[1]) if error_counts else None
        }
    
    def should_skip_sample(self, sample_id: str) -> bool:
        """Check if a sample should be skipped due to too many errors"""
        sample_errors = len(self.errors_by_sample.get(sample_id, []))
        return sample_errors >= self.config.max_api_errors_per_sample
    
    def check_ensemble_health(self, provider_errors: Dict[str, int]) -> bool:
        """Check if ensemble has too many failed providers"""
        total_providers = len(provider_errors)
        failed_providers = sum(1 for errors in provider_errors.values() 
                             if errors > 0)
        
        failure_rate = failed_providers / max(total_providers, 1)
        return failure_rate <= self.config.ensemble_failure_threshold
    
    def get_recovery_recommendations(self) -> List[str]:
        """Generate recommendations for recovering from API errors"""
        recommendations = []
        
        if not self.error_history:
            return ["No API errors detected"]
        
        error_summary = self._get_error_summary()
        most_common = error_summary.get("most_common_error")
        
        if most_common:
            error_type = most_common[0].split(':')[1]
            count = most_common[1]
            
            if error_type == "rate_limit":
                recommendations.append(f"Reduce request rate - {count} rate limit errors detected")
                recommendations.append("Consider increasing delays between requests")
            
            elif error_type == "authentication":
                recommendations.append("Check API credentials and permissions")
                recommendations.append("Verify API keys are valid and not expired")
            
            elif error_type == "quota_exceeded":
                recommendations.append("Check billing and usage limits")
                recommendations.append("Consider upgrading API plan or waiting for quota reset")
            
            elif error_type == "timeout":
                recommendations.append("Check network connectivity")
                recommendations.append("Consider increasing timeout values")
            
            elif error_type == "service_unavailable":
                recommendations.append("Wait for service to recover")
                recommendations.append("Monitor provider status pages")
        
        if self.experiment_terminated:
            recommendations.append("Experiment was terminated - address errors before resuming")
            recommendations.append(f"Termination reason: {self.termination_reason}")
        
        return recommendations
    
    def export_error_report(self) -> Dict[str, Any]:
        """Export comprehensive error report for analysis"""
        return {
            "experiment_status": "terminated" if self.experiment_terminated else "active",
            "termination_reason": self.termination_reason,
            "error_statistics": {
                "total_errors": len(self.error_history),
                "consecutive_failures": self.consecutive_failures,
                "affected_samples": len(self.errors_by_sample),
                "error_rate": len(self.error_history) / max(len(self.checkpoint_manager.completed_qids), 1)
            },
            "error_breakdown": self._get_error_summary(),
            "recovery_recommendations": self.get_recovery_recommendations(),
            "config": asdict(self.config),
            "detailed_errors": [asdict(error) for error in self.error_history[-10:]]  # Last 10 errors
        }


def create_error_handler_from_config(checkpoint_manager: CheckpointManager, 
                                    config_dict: Optional[Dict[str, Any]] = None) -> APIErrorHandler:
    """Factory function to create error handler from configuration"""
    if config_dict:
        config = ErrorHandlingConfig(**config_dict)
    else:
        config = ErrorHandlingConfig()
    
    return APIErrorHandler(checkpoint_manager, config)