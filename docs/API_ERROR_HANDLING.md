# API Error Handling and Checkpointing System

This document provides a comprehensive guide to the enhanced API error handling and checkpointing functionality added to the majority-vote ensemble and multi-turn experiment system.

## Overview

The system provides robust handling of API issues with the following key features:

- **Comprehensive Error Classification**: Automatically classifies API errors by type and severity
- **Smart Checkpointing**: Saves experiment progress before termination due to API issues
- **Health Monitoring**: Tracks API health across providers and models
- **Recovery Recommendations**: Provides actionable guidance for resolving API issues
- **Graceful Degradation**: Continues experiments when possible, even with partial API failures

## Components

### 1. Unified Multi-Turn Error Handler (`src/utils/multi_turn_error_handler.py`)

**NEW**: A unified interface that provides consistent error handling across different multi-turn experiment runners:

- **Consistent Interface**: Same error handling behavior for ensemble and standard (loop) runners
- **Experiment Lifecycle Management**: Handles initialization, sample processing, and finalization
- **Automatic Integration**: Seamlessly integrates with existing checkpoint and health monitoring systems
- **Runner-Agnostic**: Works with both ensemble voting and single-model multi-turn experiments

#### Key Features:
- `MultiTurnErrorHandler`: Central coordinator for all error handling activities
- `MultiTurnAPIWrapper`: Safe API call wrapper with automatic retry logic
- Factory functions for easy integration: `create_multi_turn_error_handler()`
- Supports both "ensemble" and "standard" learner types

### 2. API Error Handler (`src/utils/api_error_handler.py`)

The core error handling system that provides:

- **Error Classification**: Automatically identifies error types (rate limits, authentication, timeouts, etc.)
- **Severity Assessment**: Determines whether to continue, retry with backoff, or terminate
- **Intelligent Termination**: Makes termination decisions based on error patterns and thresholds
- **Emergency Checkpointing**: Saves experiment state when critical errors occur

#### Error Types Handled:
- `RATE_LIMIT`: API rate limiting errors (429, "too many requests")
- `AUTHENTICATION`: Invalid API keys or permissions (401, "unauthorized")
- `TIMEOUT`: Network or request timeouts
- `SERVICE_UNAVAILABLE`: Server errors (500, 502, 503, 504)
- `QUOTA_EXCEEDED`: Usage limits or billing issues
- `NETWORK_ERROR`: Connection and DNS issues
- `CONTEXT_LENGTH_EXCEEDED`: Input too long for model
- `INSUFFICIENT_CREDITS`: Account balance or credit issues

#### Configuration Options:
```json
{
  "error_handling": {
    "max_api_errors_per_sample": 3,
    "max_total_api_errors": 50,
    "max_consecutive_failures": 5,
    "checkpoint_on_error": true,
    "terminate_on_quota_exceeded": true,
    "terminate_on_auth_failure": true,
    "backoff_multiplier": 2.0,
    "max_backoff_seconds": 300.0,
    "min_backoff_seconds": 1.0,
    "ensemble_failure_threshold": 0.5
  }
}
```

### 2. Enhanced Checkpoint Manager (`src/utils/checkpoint.py`)

Extended checkpointing with API error state management:

- **Error Context Tracking**: Records detailed error information with each checkpoint
- **Termination State Management**: Saves reason for experiment termination
- **Error Pattern Analysis**: Analyzes error trends from checkpoint history
- **Resumable Experiments**: Allows experiments to resume from last successful checkpoint

#### Key Methods:
- `append_error_record()`: Record API errors with context
- `save_experiment_termination()`: Save termination state
- `check_if_terminated()`: Check if experiment was previously terminated
- `_analyze_errors()`: Analyze error patterns from checkpoint file

### 3. API Health Monitor (`src/utils/api_health_monitor.py`)

Monitors API health across providers and provides recovery guidance:

- **Provider Status Tracking**: Monitors success/failure rates per provider
- **Health Classification**: Categorizes providers as healthy, degraded, or down
- **Pattern Analysis**: Identifies trends in API errors
- **Recovery Recommendations**: Provides specific guidance for different error types

#### Health Status Levels:
- **Healthy**: < 5% failure rate, no consecutive failures
- **Degraded**: 5-20% failure rate, some issues but operational
- **Down**: > 20% failure rate or 10+ consecutive failures
- **Critical**: Multiple providers down or severe issues

### 4. Enhanced Ensemble Learner (`src/ensemble/learner.py`)

Resilient ensemble processing with error handling:

- **Model-Level Error Handling**: Handles failures of individual ensemble models
- **Graceful Degradation**: Continues with remaining healthy models
- **Ensemble Health Monitoring**: Tracks error rates across ensemble members
- **Smart Retry Logic**: Implements exponential backoff with jitter

### 5. Enhanced Multi-Turn Runners

**Both runners now use the unified error handling interface:**

#### Ensemble Runner (`src/ensemble/runner.py`)
- **Turn-Level Checkpointing**: Saves progress after each turn
- **Ensemble-Aware Error Handling**: Manages errors across multiple models
- **Voting Resilience**: Continues voting even when some models fail

#### Loop Runner (`src/loop/runner.py`) 
- **Unified Integration**: Now uses the same error handling as ensemble runner
- **Multi-Turn Error Tracking**: Tracks errors across turns for single-model experiments
- **Legacy Compatibility**: Maintains backward compatibility with existing configurations

**Common Features Across Both Runners:**
- Sample-Level Error Tracking
- Experiment Termination Logic
- Recovery State Preservation
- Health Monitoring Integration
- Consistent Checkpointing Behavior

## Usage Guide

### Basic Usage

1. **Run with Default Error Handling**:
```bash
python run_ensemble_experiments.py --config configs/ensemble_experiments/openai_basic.json
```

2. **Use Pre-defined Error Policies**:
```bash
# Conservative (strict error handling)
python run_ensemble_experiments.py --config config.json --error-policy conservative

# Aggressive (tolerates more errors)
python run_ensemble_experiments.py --config config.json --error-policy aggressive
```

3. **Custom Error Configuration**:
```bash
python run_ensemble_experiments.py --config config.json --error-config my_error_config.json
```

### Configuration Policies

#### Conservative Policy (`configs/error_handling/conservative_error_config.json`)
- Strict error limits
- Quick termination on issues
- Detailed logging
- Best for sensitive/critical experiments

#### Default Policy (`configs/error_handling/default_error_config.json`)
- Balanced error tolerance
- Reasonable retry limits
- Standard logging
- Good for most experiments

#### Aggressive Policy (`configs/error_handling/aggressive_error_config.json`)
- High error tolerance
- Extended retry attempts
- Minimal logging
- Best for long-running, robust experiments

### Experiment Recovery

When an experiment is terminated due to API issues:

1. **Check Error Reports**: Review generated error reports in the output directory
2. **Follow Recovery Recommendations**: Implement suggested fixes from recovery plans
3. **Resume Experiment**: Simply re-run the same command - it will resume from last checkpoint

### Output Files

The system generates several output files for error tracking and recovery:

- **`*_checkpoint.jsonl`**: Main checkpoint file with sample results and errors
- **`*_resume_state.json`**: Experiment state for resumability
- **`api_error_report.json`**: Comprehensive API error analysis
- **`api_health_report_*.json`**: API health status reports
- **`recovery_plan.json`**: Specific recovery recommendations
- **`experiment_termination.json`**: Termination details if experiment stopped
- **`error_logs/api_errors.log`**: Detailed error logs

## Error Recovery Workflow

1. **Error Detection**: System automatically detects and classifies API errors
2. **Impact Assessment**: Determines severity and whether to continue or terminate
3. **Checkpointing**: Saves current progress and error context
4. **Termination Decision**: Either continues with backoff or terminates gracefully
5. **Report Generation**: Creates detailed error reports and recovery recommendations
6. **Recovery Guidance**: Provides specific steps to resolve issues
7. **Resumption**: Allows seamless resumption once issues are resolved

## Monitoring and Alerts

### Real-time Monitoring
- Console output shows error status and health warnings
- Progress indicators include error counts and recovery actions
- Immediate alerts for critical issues requiring manual intervention

### Health Recommendations
The system provides actionable recommendations such as:
- "Reduce request rate - 15 rate limit errors detected"
- "Check API credentials - authentication errors detected"
- "Wait for service recovery - provider outage detected"
- "Increase timeout values - 8 timeout errors detected"

## Best Practices

1. **Start with Conservative Policy**: Use conservative settings for new experiments
2. **Monitor Error Reports**: Regularly check generated error reports
3. **Address Root Causes**: Fix underlying issues (API keys, billing, etc.) before resuming
4. **Use Appropriate Policies**: Match error policy to experiment criticality
5. **Keep Backups**: Maintain backups of checkpoint files for important experiments
6. **Review Health Reports**: Check API health reports to identify trends

## Integration with Existing Code

The error handling system integrates seamlessly with existing experiments:

- **Ensemble Configurations**: Simply add error handling config to existing ensemble configs
- **Backwards Compatibility**: All existing experiment configs continue to work
- **Optional Features**: Error handling can be disabled if needed
- **Progressive Enhancement**: Add error handling features gradually as needed

## Troubleshooting

### Common Issues and Solutions

1. **"Experiment previously terminated"**
   - Review error reports to understand termination reason
   - Address underlying issues (API keys, billing, etc.)
   - Resume with `--error-policy conservative` if needed

2. **"Too many API errors"**
   - Check API credentials and billing status
   - Review rate limits and usage
   - Use more conservative error settings

3. **"Provider health critical"**
   - Check provider status pages
   - Consider using different providers temporarily
   - Wait for service recovery

4. **"Checkpoint corruption"**
   - Use backup checkpoint files if available
   - Restart experiment with conservative settings
   - Contact support if issues persist

## Advanced Configuration

### Custom Error Handling Logic

You can create custom error handling configurations by extending the base configuration:

```json
{
  "error_handling": {
    "max_api_errors_per_sample": 5,
    "custom_error_patterns": ["custom_error_pattern"],
    "provider_specific_limits": {
      "openai": {"max_errors": 10},
      "anthropic": {"max_errors": 20}
    }
  },
  "health_monitoring": {
    "custom_health_checks": true,
    "alert_thresholds": {
      "error_rate": 0.1,
      "response_time": 15.0
    }
  }
}
```

This comprehensive error handling system ensures that your ensemble experiments are robust, resumable, and provide clear guidance for resolving API issues when they occur.