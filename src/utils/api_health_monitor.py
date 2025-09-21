"""
API Health Monitor and Recovery Utilities

Provides tools to monitor API health, detect patterns in failures, and recommend recovery strategies.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque

from .api_error_handler import APIErrorType, APIError


@dataclass
class ProviderStatus:
    """Status information for an API provider"""
    provider: str
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0
    avg_response_time: float = 0.0
    status: str = "unknown"  # healthy, degraded, down, unknown


@dataclass
class APIHealthStatus:
    """Overall API health status"""
    timestamp: float
    overall_status: str  # healthy, degraded, critical
    provider_statuses: Dict[str, ProviderStatus]
    recommendations: List[str]
    estimated_recovery_time: Optional[str] = None


class APIHealthMonitor:
    """Monitor API health across providers and models"""
    
    def __init__(self, monitoring_window_hours: int = 24, health_check_interval: int = 300):
        self.monitoring_window_hours = monitoring_window_hours
        self.health_check_interval = health_check_interval
        
        # Track provider health
        self.provider_statuses: Dict[str, ProviderStatus] = {}
        self.request_history: deque = deque(maxlen=1000)  # Recent request history
        
        # Health thresholds
        self.healthy_failure_rate_threshold = 0.05  # 5%
        self.degraded_failure_rate_threshold = 0.20  # 20%
        self.down_consecutive_failures_threshold = 10
        
        self.last_health_check = 0
    
    def record_request(self, provider: str, model: str, success: bool, 
                      response_time: float = 0, error: Optional[Exception] = None):
        """Record an API request and its outcome"""
        provider_key = f"{provider}:{model}"
        
        # Initialize provider status if needed
        if provider_key not in self.provider_statuses:
            self.provider_statuses[provider_key] = ProviderStatus(provider=provider_key)
        
        status = self.provider_statuses[provider_key]
        status.total_requests += 1
        
        # Update response time (moving average)
        if response_time > 0:
            if status.avg_response_time == 0:
                status.avg_response_time = response_time
            else:
                # Exponential moving average
                status.avg_response_time = 0.9 * status.avg_response_time + 0.1 * response_time
        
        current_time = time.time()
        
        if success:
            status.last_success = current_time
            status.consecutive_failures = 0
        else:
            status.last_failure = current_time
            status.total_failures += 1
            status.consecutive_failures += 1
        
        # Calculate failure rate
        if status.total_requests > 0:
            status.failure_rate = status.total_failures / status.total_requests
        
        # Update status classification
        status.status = self._classify_provider_status(status)
        
        # Record in request history
        self.request_history.append({
            "timestamp": current_time,
            "provider": provider_key,
            "success": success,
            "response_time": response_time,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None
        })
    
    def _classify_provider_status(self, status: ProviderStatus) -> str:
        """Classify provider status based on metrics"""
        # Check if provider is down (too many consecutive failures)
        if status.consecutive_failures >= self.down_consecutive_failures_threshold:
            return "down"
        
        # Check failure rate
        if status.failure_rate <= self.healthy_failure_rate_threshold:
            return "healthy"
        elif status.failure_rate <= self.degraded_failure_rate_threshold:
            return "degraded"
        else:
            return "down"
    
    def get_current_health(self) -> APIHealthStatus:
        """Get current overall API health status"""
        current_time = time.time()
        
        # Update health check timestamp
        self.last_health_check = current_time
        
        # Determine overall status
        provider_statuses_dict = {}
        healthy_count = 0
        degraded_count = 0
        down_count = 0
        
        for provider_key, status in self.provider_statuses.items():
            provider_statuses_dict[provider_key] = status
            
            if status.status == "healthy":
                healthy_count += 1
            elif status.status == "degraded":
                degraded_count += 1
            elif status.status == "down":
                down_count += 1
        
        # Determine overall status
        total_providers = len(self.provider_statuses)
        if total_providers == 0:
            overall_status = "unknown"
        elif down_count > 0:
            overall_status = "critical"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Generate recommendations
        recommendations = self._generate_health_recommendations(provider_statuses_dict, overall_status)
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(provider_statuses_dict, overall_status)
        
        return APIHealthStatus(
            timestamp=current_time,
            overall_status=overall_status,
            provider_statuses=provider_statuses_dict,
            recommendations=recommendations,
            estimated_recovery_time=recovery_time
        )
    
    def _generate_health_recommendations(self, statuses: Dict[str, ProviderStatus], 
                                       overall_status: str) -> List[str]:
        """Generate recommendations based on current health status"""
        recommendations = []
        
        if overall_status == "critical":
            recommendations.append("🚨 Critical API issues detected - consider pausing experiments")
            recommendations.append("Check provider status pages for service outages")
            
        if overall_status == "degraded":
            recommendations.append("⚠️ API performance degraded - reduce request rate")
            recommendations.append("Consider using only healthy providers temporarily")
        
        # Provider-specific recommendations
        for provider_key, status in statuses.items():
            provider_name = provider_key.split(':')[0]
            
            if status.status == "down":
                if status.consecutive_failures >= self.down_consecutive_failures_threshold:
                    recommendations.append(f"❌ {provider_name} is down - disable this provider")
                
            elif status.status == "degraded":
                if status.failure_rate > 0.1:
                    recommendations.append(f"⚠️ {provider_name} has high failure rate ({status.failure_rate:.1%}) - reduce load")
                
                if status.avg_response_time > 10.0:
                    recommendations.append(f"🐌 {provider_name} is slow (avg {status.avg_response_time:.1f}s) - increase timeouts")
        
        # Rate limiting recommendations
        recent_failures = self._get_recent_errors(hours=1)
        rate_limit_errors = sum(1 for req in recent_failures if 'rate' in (req.get('error_type', '') or '').lower())
        
        if rate_limit_errors > 5:
            recommendations.append(f"🚦 {rate_limit_errors} rate limit errors in last hour - reduce request frequency")
        
        # Authentication recommendations
        auth_errors = sum(1 for req in recent_failures if 'auth' in (req.get('error_type', '') or '').lower())
        if auth_errors > 0:
            recommendations.append(f"🔑 {auth_errors} authentication errors detected - check API keys")
        
        if not recommendations:
            recommendations.append("✅ No immediate actions required")
        
        return recommendations
    
    def _estimate_recovery_time(self, statuses: Dict[str, ProviderStatus], 
                               overall_status: str) -> Optional[str]:
        """Estimate time until providers recover"""
        if overall_status == "healthy":
            return None
        
        # Simple heuristic based on typical outage patterns
        down_providers = [s for s in statuses.values() if s.status == "down"]
        degraded_providers = [s for s in statuses.values() if s.status == "degraded"]
        
        if down_providers:
            # Major outages typically last 30 minutes to 2 hours
            return "30 minutes to 2 hours"
        
        elif degraded_providers:
            # Performance issues typically resolve within 15-30 minutes
            return "15-30 minutes"
        
        return None
    
    def _get_recent_errors(self, hours: int = 1) -> List[Dict]:
        """Get recent errors from request history"""
        cutoff_time = time.time() - (hours * 3600)
        return [req for req in self.request_history 
                if req['timestamp'] > cutoff_time and not req['success']]
    
    def get_error_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze error patterns to identify trends"""
        recent_errors = self._get_recent_errors(hours=self.monitoring_window_hours)
        
        if not recent_errors:
            return {"status": "no_errors", "patterns": []}
        
        # Group errors by type and provider
        error_by_type = defaultdict(int)
        error_by_provider = defaultdict(int)
        error_by_hour = defaultdict(int)
        
        for error in recent_errors:
            error_type = error.get('error_type', 'unknown')
            provider = error.get('provider', 'unknown')
            hour = int(error['timestamp']) // 3600
            
            error_by_type[error_type] += 1
            error_by_provider[provider] += 1
            error_by_hour[hour] += 1
        
        # Identify patterns
        patterns = []
        
        # Most common error types
        if error_by_type:
            most_common_error = max(error_by_type.items(), key=lambda x: x[1])
            patterns.append(f"Most common error: {most_common_error[0]} ({most_common_error[1]} occurrences)")
        
        # Provider with most errors
        if error_by_provider:
            worst_provider = max(error_by_provider.items(), key=lambda x: x[1])
            patterns.append(f"Provider with most errors: {worst_provider[0]} ({worst_provider[1]} errors)")
        
        # Time-based patterns
        if len(error_by_hour) > 1:
            error_hours = list(error_by_hour.keys())
            if max(error_hours) - min(error_hours) <= 2:
                patterns.append("Errors clustered in time - likely service outage")
            else:
                patterns.append("Errors distributed over time - likely systematic issue")
        
        return {
            "total_errors": len(recent_errors),
            "error_by_type": dict(error_by_type),
            "error_by_provider": dict(error_by_provider),
            "patterns": patterns,
            "analysis_window_hours": self.monitoring_window_hours
        }
    
    def should_pause_experiment(self) -> Tuple[bool, str]:
        """Determine if experiment should be paused due to API health"""
        health = self.get_current_health()
        
        if health.overall_status == "critical":
            return True, "Critical API issues detected across multiple providers"
        
        # Count down providers
        down_providers = [p for p in health.provider_statuses.values() if p.status == "down"]
        total_providers = len(health.provider_statuses)
        
        if total_providers > 0 and len(down_providers) / total_providers > 0.5:
            return True, f"More than half of providers ({len(down_providers)}/{total_providers}) are down"
        
        # Check recent error rate
        recent_errors = self._get_recent_errors(hours=1)
        recent_total = len([req for req in self.request_history 
                           if req['timestamp'] > time.time() - 3600])
        
        if recent_total > 10 and len(recent_errors) / recent_total > 0.5:
            return True, f"High recent error rate: {len(recent_errors)}/{recent_total} requests failed"
        
        return False, ""
    
    def export_health_report(self, output_dir: Path) -> Path:
        """Export comprehensive health report"""
        health = self.get_current_health()
        error_analysis = self.get_error_pattern_analysis()
        should_pause, pause_reason = self.should_pause_experiment()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "monitoring_window_hours": self.monitoring_window_hours,
            "health_status": asdict(health),
            "error_analysis": error_analysis,
            "experiment_recommendations": {
                "should_pause": should_pause,
                "pause_reason": pause_reason,
                "recovery_actions": health.recommendations
            },
            "provider_details": {
                provider: asdict(status) 
                for provider, status in health.provider_statuses.items()
            }
        }
        
        report_file = output_dir / f"api_health_report_{int(time.time())}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    def reset_health_data(self):
        """Reset all health monitoring data"""
        self.provider_statuses.clear()
        self.request_history.clear()
        self.last_health_check = 0
        print("🔄 API health monitoring data reset")


class RecoveryRecommendationEngine:
    """Generate specific recovery recommendations based on error patterns"""
    
    def __init__(self):
        self.recovery_strategies = {
            APIErrorType.RATE_LIMIT: self._rate_limit_recovery,
            APIErrorType.AUTHENTICATION: self._auth_recovery,
            APIErrorType.TIMEOUT: self._timeout_recovery,
            APIErrorType.SERVICE_UNAVAILABLE: self._service_recovery,
            APIErrorType.QUOTA_EXCEEDED: self._quota_recovery,
            APIErrorType.NETWORK_ERROR: self._network_recovery,
            APIErrorType.CONTEXT_LENGTH_EXCEEDED: self._context_recovery,
            APIErrorType.INSUFFICIENT_CREDITS: self._credits_recovery
        }
    
    def get_recovery_plan(self, errors: List[APIError]) -> Dict[str, Any]:
        """Generate comprehensive recovery plan from error history"""
        if not errors:
            return {"status": "no_errors", "actions": []}
        
        # Group errors by type
        error_by_type = defaultdict(list)
        for error in errors:
            error_by_type[error.error_type].append(error)
        
        recovery_actions = []
        immediate_actions = []
        preventive_actions = []
        
        # Get specific recommendations for each error type
        for error_type, error_list in error_by_type.items():
            if error_type in self.recovery_strategies:
                recommendations = self.recovery_strategies[error_type](error_list)
                recovery_actions.extend(recommendations.get("actions", []))
                immediate_actions.extend(recommendations.get("immediate", []))
                preventive_actions.extend(recommendations.get("preventive", []))
        
        return {
            "total_errors": len(errors),
            "error_types": len(error_by_type),
            "immediate_actions": list(set(immediate_actions)),
            "recovery_actions": list(set(recovery_actions)),
            "preventive_measures": list(set(preventive_actions)),
            "estimated_fix_time": self._estimate_fix_time(error_by_type)
        }
    
    def _rate_limit_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        count = len(errors)
        return {
            "immediate": [
                f"Reduce request rate immediately - {count} rate limit errors detected",
                "Implement exponential backoff with jitter"
            ],
            "actions": [
                "Increase delays between requests by 2-3x",
                "Consider using multiple API keys if allowed",
                "Implement request queue with rate limiting"
            ],
            "preventive": [
                "Monitor API usage against rate limits",
                "Set up alerts for approaching rate limits"
            ]
        }
    
    def _auth_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        return {
            "immediate": [
                "Stop all API requests immediately",
                "Verify API key validity and permissions"
            ],
            "actions": [
                "Check API key expiration date",
                "Verify API key has correct permissions",
                "Generate new API key if needed",
                "Update environment variables with valid keys"
            ],
            "preventive": [
                "Set up API key expiration monitoring",
                "Use key rotation strategy"
            ]
        }
    
    def _timeout_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        count = len(errors)
        return {
            "immediate": [
                f"Increase timeout values - {count} timeout errors detected"
            ],
            "actions": [
                "Double current timeout settings",
                "Check network connectivity",
                "Consider using CDN or closer API endpoints"
            ],
            "preventive": [
                "Monitor network latency to API endpoints",
                "Implement circuit breaker pattern"
            ]
        }
    
    def _service_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        return {
            "immediate": [
                "Pause experiment temporarily",
                "Check provider status page"
            ],
            "actions": [
                "Wait for service recovery (typically 15-60 minutes)",
                "Switch to backup providers if available",
                "Monitor provider status updates"
            ],
            "preventive": [
                "Set up provider status monitoring",
                "Implement multi-provider failover"
            ]
        }
    
    def _quota_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        return {
            "immediate": [
                "Stop all API requests immediately",
                "Check billing and quota status"
            ],
            "actions": [
                "Upgrade API plan if needed",
                "Wait for quota reset (if monthly/daily)",
                "Add payment method if billing issue"
            ],
            "preventive": [
                "Set up quota monitoring and alerts",
                "Implement usage tracking and budgets"
            ]
        }
    
    def _network_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        return {
            "immediate": [
                "Check internet connectivity",
                "Verify DNS resolution"
            ],
            "actions": [
                "Restart network connection",
                "Try different network/VPN",
                "Check firewall settings",
                "Verify proxy configuration"
            ],
            "preventive": [
                "Monitor network stability",
                "Implement network redundancy"
            ]
        }
    
    def _context_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        return {
            "immediate": [
                "Reduce input context length",
                "Split long inputs into smaller chunks"
            ],
            "actions": [
                "Implement context length validation",
                "Use text summarization for long inputs",
                "Switch to models with larger context windows"
            ],
            "preventive": [
                "Add context length checks before API calls",
                "Implement automatic text truncation"
            ]
        }
    
    def _credits_recovery(self, errors: List[APIError]) -> Dict[str, List[str]]:
        return {
            "immediate": [
                "Check account balance and billing",
                "Add credits/payment if needed"
            ],
            "actions": [
                "Top up account balance",
                "Switch to provider with available credits",
                "Implement spend monitoring"
            ],
            "preventive": [
                "Set up low balance alerts",
                "Implement automatic top-up if available"
            ]
        }
    
    def _estimate_fix_time(self, error_by_type: Dict[APIErrorType, List[APIError]]) -> str:
        """Estimate time to fix based on error types"""
        max_time = 0
        
        time_estimates = {
            APIErrorType.RATE_LIMIT: 5,     # minutes
            APIErrorType.AUTHENTICATION: 10,  # minutes
            APIErrorType.TIMEOUT: 15,       # minutes
            APIErrorType.SERVICE_UNAVAILABLE: 60,  # minutes
            APIErrorType.QUOTA_EXCEEDED: 1440,     # minutes (24 hours)
            APIErrorType.NETWORK_ERROR: 30,        # minutes
            APIErrorType.CONTEXT_LENGTH_EXCEEDED: 30,  # minutes
            APIErrorType.INSUFFICIENT_CREDITS: 60      # minutes
        }
        
        for error_type in error_by_type.keys():
            estimated_minutes = time_estimates.get(error_type, 15)
            max_time = max(max_time, estimated_minutes)
        
        if max_time < 60:
            return f"{max_time} minutes"
        elif max_time < 1440:
            hours = max_time // 60
            return f"{hours} hour{'s' if hours > 1 else ''}"
        else:
            days = max_time // 1440
            return f"{days} day{'s' if days > 1 else ''}"