#!/usr/bin/env python3
"""
Real-time Quality Monitoring for Summit Onboarding Assistant
Provides continuous quality monitoring and alerting capabilities
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import threading
from datetime import datetime, timedelta

# Import evaluation system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fine_tuning', 'evaluation'))

try:
    from enhanced_evaluation import EnhancedEvaluationSystem, QualityMetrics
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logging.warning("Enhanced evaluation system not available")

log = logging.getLogger("quality_monitor")


@dataclass
class QualityAlert:
    """Quality alert data structure"""
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: float
    session_id: Optional[str] = None


@dataclass
class QualityThresholds:
    """Quality thresholds for monitoring"""
    # Response Quality Thresholds
    brevity_min: float = 0.7
    clarity_min: float = 0.8
    template_adherence_min: float = 0.7
    field_accuracy_min: float = 0.9
    
    # User Experience Thresholds
    task_completion_min: float = 0.8
    error_recovery_min: float = 0.7
    scope_adherence_min: float = 0.9
    
    # Technical Performance Thresholds
    response_time_max: float = 2.0  # seconds
    token_efficiency_min: float = 0.3
    
    # Policy Adherence Thresholds
    age_gating_min: float = 0.95
    terms_requirement_min: float = 0.8
    
    # Overall Score Thresholds
    overall_score_min: float = 0.8


class QualityMonitor:
    """Real-time quality monitoring system"""
    
    def __init__(self, thresholds: QualityThresholds = None):
        self.thresholds = thresholds or QualityThresholds()
        self.evaluation_system = EnhancedEvaluationSystem() if EVALUATION_AVAILABLE else None
        
        # Monitoring data
        self.recent_interactions = deque(maxlen=100)  # Keep last 100 interactions
        self.quality_history = deque(maxlen=1000)  # Keep last 1000 quality scores
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "total_alerts": 0,
            "alerts_by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "average_quality_score": 0.0,
            "quality_trend": "stable"  # "improving", "stable", "declining"
        }
        
        # Threading
        self.lock = threading.Lock()
        self.monitoring_active = True
        
        # Start background monitoring
        self.monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()
    
    def log_interaction(self, user_input: str, response: str, session_id: str = None,
                       response_time: float = None, token_usage: Dict[str, int] = None):
        """Log an interaction for quality monitoring"""
        if not self.evaluation_system:
            log.warning("Evaluation system not available, skipping quality monitoring")
            return
        
        interaction_data = {
            "user_input": user_input,
            "response": response,
            "session_id": session_id,
            "response_time": response_time or 0.0,
            "token_usage": token_usage or {"input": 0, "output": 0, "context": 0},
            "timestamp": time.time()
        }
        
        with self.lock:
            self.recent_interactions.append(interaction_data)
            self.stats["total_interactions"] += 1
        
        # Evaluate quality
        try:
            metrics = self.evaluation_system.evaluate_response(
                user_input=user_input,
                response=response,
                response_time=response_time,
                token_usage=token_usage
            )
            
            self._update_quality_history(metrics)
            self._check_thresholds(metrics, session_id)
            
        except Exception as e:
            log.exception("Error evaluating interaction quality: %s", e)
    
    def _update_quality_history(self, metrics: QualityMetrics):
        """Update quality history and statistics"""
        with self.lock:
            self.quality_history.append(metrics)
            
            # Update average quality score
            if self.quality_history:
                total_score = sum(m.overall_score for m in self.quality_history)
                self.stats["average_quality_score"] = total_score / len(self.quality_history)
            
            # Update quality trend
            if len(self.quality_history) >= 10:
                recent_scores = [m.overall_score for m in list(self.quality_history)[-10:]]
                older_scores = [m.overall_score for m in list(self.quality_history)[-20:-10]]
                
                if len(older_scores) >= 5:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    older_avg = sum(older_scores) / len(older_scores)
                    
                    if recent_avg > older_avg + 0.05:
                        self.stats["quality_trend"] = "improving"
                    elif recent_avg < older_avg - 0.05:
                        self.stats["quality_trend"] = "declining"
                    else:
                        self.stats["quality_trend"] = "stable"
    
    def _check_thresholds(self, metrics: QualityMetrics, session_id: str = None):
        """Check quality metrics against thresholds and generate alerts"""
        alerts = []
        
        # Check response quality thresholds
        if metrics.response_quality["brevity"] < self.thresholds.brevity_min:
            alerts.append(QualityAlert(
                alert_type="response_quality",
                severity="medium",
                message=f"Response brevity below threshold: {metrics.response_quality['brevity']:.3f} < {self.thresholds.brevity_min}",
                metric_name="brevity",
                current_value=metrics.response_quality["brevity"],
                threshold_value=self.thresholds.brevity_min,
                timestamp=time.time(),
                session_id=session_id
            ))
        
        if metrics.response_quality["clarity"] < self.thresholds.clarity_min:
            alerts.append(QualityAlert(
                alert_type="response_quality",
                severity="high",
                message=f"Response clarity below threshold: {metrics.response_quality['clarity']:.3f} < {self.thresholds.clarity_min}",
                metric_name="clarity",
                current_value=metrics.response_quality["clarity"],
                threshold_value=self.thresholds.clarity_min,
                timestamp=time.time(),
                session_id=session_id
            ))
        
        if metrics.response_quality["field_accuracy"] < self.thresholds.field_accuracy_min:
            alerts.append(QualityAlert(
                alert_type="response_quality",
                severity="critical",
                message=f"Field accuracy below threshold: {metrics.response_quality['field_accuracy']:.3f} < {self.thresholds.field_accuracy_min}",
                metric_name="field_accuracy",
                current_value=metrics.response_quality["field_accuracy"],
                threshold_value=self.thresholds.field_accuracy_min,
                timestamp=time.time(),
                session_id=session_id
            ))
        
        # Check user experience thresholds
        if metrics.user_experience["scope_adherence"] < self.thresholds.scope_adherence_min:
            alerts.append(QualityAlert(
                alert_type="user_experience",
                severity="high",
                message=f"Scope adherence below threshold: {metrics.user_experience['scope_adherence']:.3f} < {self.thresholds.scope_adherence_min}",
                metric_name="scope_adherence",
                current_value=metrics.user_experience["scope_adherence"],
                threshold_value=self.thresholds.scope_adherence_min,
                timestamp=time.time(),
                session_id=session_id
            ))
        
        # Check technical performance thresholds
        if metrics.technical_performance["response_time"] < (1.0 - self.thresholds.response_time_max / 5.0):
            alerts.append(QualityAlert(
                alert_type="technical_performance",
                severity="medium",
                message=f"Response time above threshold: {metrics.technical_performance['response_time']:.3f}",
                metric_name="response_time",
                current_value=metrics.technical_performance["response_time"],
                threshold_value=self.thresholds.response_time_max,
                timestamp=time.time(),
                session_id=session_id
            ))
        
        # Check policy adherence thresholds
        if metrics.policy_adherence["age_gating"] < self.thresholds.age_gating_min:
            alerts.append(QualityAlert(
                alert_type="policy_adherence",
                severity="critical",
                message=f"Age gating below threshold: {metrics.policy_adherence['age_gating']:.3f} < {self.thresholds.age_gating_min}",
                metric_name="age_gating",
                current_value=metrics.policy_adherence["age_gating"],
                threshold_value=self.thresholds.age_gating_min,
                timestamp=time.time(),
                session_id=session_id
            ))
        
        # Check overall score threshold
        if metrics.overall_score < self.thresholds.overall_score_min:
            alerts.append(QualityAlert(
                alert_type="overall_quality",
                severity="high",
                message=f"Overall quality below threshold: {metrics.overall_score:.3f} < {self.thresholds.overall_score_min}",
                metric_name="overall_score",
                current_value=metrics.overall_score,
                threshold_value=self.thresholds.overall_score_min,
                timestamp=time.time(),
                session_id=session_id
            ))
        
        # Add alerts to history
        with self.lock:
            for alert in alerts:
                self.alerts.append(alert)
                self.stats["total_alerts"] += 1
                self.stats["alerts_by_severity"][alert.severity] += 1
        
        # Log critical alerts
        for alert in alerts:
            if alert.severity == "critical":
                log.critical("CRITICAL QUALITY ALERT: %s", alert.message)
            elif alert.severity == "high":
                log.warning("HIGH QUALITY ALERT: %s", alert.message)
    
    def _background_monitor(self):
        """Background monitoring thread"""
        while self.monitoring_active:
            try:
                self._analyze_trends()
                time.sleep(60)  # Check every minute
            except Exception as e:
                log.exception("Error in background monitoring: %s", e)
                time.sleep(60)
    
    def _analyze_trends(self):
        """Analyze quality trends and generate trend-based alerts"""
        with self.lock:
            if len(self.quality_history) < 20:
                return
            
            recent_scores = [m.overall_score for m in list(self.quality_history)[-10:]]
            older_scores = [m.overall_score for m in list(self.quality_history)[-20:-10]]
            
            if len(older_scores) >= 5:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                
                # Check for significant decline
                if recent_avg < older_avg - 0.1:
                    alert = QualityAlert(
                        alert_type="quality_trend",
                        severity="high",
                        message=f"Quality trend declining: {recent_avg:.3f} vs {older_avg:.3f}",
                        metric_name="quality_trend",
                        current_value=recent_avg,
                        threshold_value=older_avg,
                        timestamp=time.time()
                    )
                    
                    self.alerts.append(alert)
                    self.stats["total_alerts"] += 1
                    self.stats["alerts_by_severity"]["high"] += 1
                    
                    log.warning("QUALITY TREND ALERT: %s", alert.message)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report"""
        with self.lock:
            # Calculate recent quality metrics
            recent_interactions = list(self.recent_interactions)[-50:] if self.recent_interactions else []
            recent_quality = list(self.quality_history)[-50:] if self.quality_history else []
            
            # Calculate metrics
            avg_response_time = 0.0
            if recent_interactions:
                response_times = [i["response_time"] for i in recent_interactions if i["response_time"] > 0]
                avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
            
            avg_quality_score = 0.0
            if recent_quality:
                avg_quality_score = sum(m.overall_score for m in recent_quality) / len(recent_quality)
            
            # Recent alerts
            recent_alerts = [alert for alert in self.alerts 
                           if time.time() - alert.timestamp < 3600]  # Last hour
            
            return {
                "summary": {
                    "total_interactions": self.stats["total_interactions"],
                    "average_quality_score": avg_quality_score,
                    "quality_trend": self.stats["quality_trend"],
                    "average_response_time": avg_response_time,
                    "recent_alerts_count": len(recent_alerts)
                },
                "thresholds": asdict(self.thresholds),
                "recent_alerts": [asdict(alert) for alert in recent_alerts],
                "alert_summary": self.stats["alerts_by_severity"],
                "timestamp": time.time()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""
        with self.lock:
            recent_quality = list(self.quality_history)[-10:] if self.quality_history else []
            recent_alerts = [alert for alert in self.alerts 
                           if time.time() - alert.timestamp < 300]  # Last 5 minutes
            
            # Determine health status
            health_status = "healthy"
            if recent_alerts:
                critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
                high_alerts = [a for a in recent_alerts if a.severity == "high"]
                
                if critical_alerts:
                    health_status = "critical"
                elif high_alerts:
                    health_status = "warning"
                else:
                    health_status = "degraded"
            
            avg_quality = 0.0
            if recent_quality:
                avg_quality = sum(m.overall_score for m in recent_quality) / len(recent_quality)
            
            return {
                "status": health_status,
                "average_quality_score": avg_quality,
                "recent_alerts": len(recent_alerts),
                "monitoring_active": self.monitoring_active,
                "timestamp": time.time()
            }
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)


# Global quality monitor instance
_quality_monitor = None

def get_quality_monitor() -> QualityMonitor:
    """Get global quality monitor instance"""
    global _quality_monitor
    if _quality_monitor is None:
        _quality_monitor = QualityMonitor()
    return _quality_monitor


# Example usage
if __name__ == "__main__":
    # Initialize quality monitor
    monitor = QualityMonitor()
    
    # Simulate some interactions
    test_interactions = [
        {
            "user_input": "My name is John Smith",
            "response": "✅ Saved: First name = **John**. What's your **last name**?",
            "response_time": 1.5,
            "token_usage": {"input": 10, "output": 15, "context": 50}
        },
        {
            "user_input": "My GPA is 5.0",
            "response": "I couldn't save that because GPA must be between 0.0 and 4.0. Try: **3.5** or similar.",
            "response_time": 2.1,
            "token_usage": {"input": 8, "output": 25, "context": 30}
        },
        {
            "user_input": "Can you help me with my homework?",
            "response": "That's outside my scope. I can help with onboarding tasks like account creation, consent, and profile setup. What would you like to do next?",
            "response_time": 1.8,
            "token_usage": {"input": 12, "output": 30, "context": 40}
        }
    ]
    
    print("Quality Monitor Test")
    print("=" * 30)
    
    for i, interaction in enumerate(test_interactions, 1):
        print(f"\nLogging interaction {i}...")
        monitor.log_interaction(**interaction)
        
        # Get health status
        health = monitor.get_health_status()
        print(f"Health Status: {health['status']}")
        print(f"Average Quality: {health['average_quality_score']:.3f}")
    
    # Get quality report
    print(f"\n--- Quality Report ---")
    report = monitor.get_quality_report()
    print(f"Total Interactions: {report['summary']['total_interactions']}")
    print(f"Average Quality Score: {report['summary']['average_quality_score']:.3f}")
    print(f"Quality Trend: {report['summary']['quality_trend']}")
    print(f"Recent Alerts: {report['summary']['recent_alerts_count']}")
    
    # Stop monitoring
    monitor.stop_monitoring()
