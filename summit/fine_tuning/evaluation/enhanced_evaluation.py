#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics for Summit Onboarding Assistant
Provides comprehensive quality monitoring and evaluation capabilities
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

log = logging.getLogger("enhanced_evaluation")


class MetricType(Enum):
    """Types of evaluation metrics"""
    RESPONSE_QUALITY = "response_quality"
    USER_EXPERIENCE = "user_experience"
    TECHNICAL_PERFORMANCE = "technical_performance"
    POLICY_ADHERENCE = "policy_adherence"
    MULTILINGUAL_QUALITY = "multilingual_quality"


@dataclass
class EvaluationResult:
    """Result of an evaluation metric"""
    metric_type: MetricType
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any]
    timestamp: float


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for Summit responses"""
    response_quality: Dict[str, float]
    user_experience: Dict[str, float]
    technical_performance: Dict[str, float]
    policy_adherence: Dict[str, float]
    multilingual_quality: Dict[str, float]
    overall_score: float
    timestamp: float


class ResponseQualityEvaluator:
    """Evaluates response quality metrics"""
    
    def __init__(self):
        self.template_patterns = self._initialize_template_patterns()
        self.validation_patterns = self._initialize_validation_patterns()
    
    def _initialize_template_patterns(self) -> Dict[str, str]:
        """Initialize patterns for template adherence"""
        return {
            "field_confirmation": r"✅\s+Saved:\s+([^=]+)=\s*\*\*([^*]+)\*\*\.\s+Next:\s+(.+)",
            "validation_error": r"I couldn't save that because\s+(.+?)\.\s+Try:\s+(.+)",
            "parental_consent": r"Because you're under 18, I'll need a parent/guardian email",
            "out_of_scope": r"That's outside my scope\.\s+I can help with onboarding tasks",
            "email_autocorrect": r"That email looks mistyped\.\s+Did you mean\s+\*\*([^*]+)\*\*",
            "completion": r"All set!\s+Your profile is complete\.\s+Press\s+\*\*Continue\*\*"
        }
    
    def _initialize_validation_patterns(self) -> Dict[str, str]:
        """Initialize patterns for validation checks"""
        return {
            "email_format": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "gpa_range": r"[0-4](?:\.[0-9])?",
            "year_range": r"20[0-3][0-9]",
            "date_format": r"(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/(?:19|20)[0-9]{2}"
        }
    
    def evaluate_brevity(self, response: str) -> EvaluationResult:
        """Evaluate response brevity (target: <50 words)"""
        word_count = len(response.split())
        target_words = 50
        score = max(0, 1 - (word_count - target_words) / target_words) if word_count > target_words else 1.0
        
        return EvaluationResult(
            metric_type=MetricType.RESPONSE_QUALITY,
            metric_name="brevity",
            score=score,
            max_score=1.0,
            details={
                "word_count": word_count,
                "target_words": target_words,
                "score_explanation": f"{word_count} words (target: {target_words})"
            },
            timestamp=time.time()
        )
    
    def evaluate_clarity(self, response: str) -> EvaluationResult:
        """Evaluate response clarity patterns"""
        has_confirmation = "saved" in response.lower() or "✅" in response
        has_next_step = "next" in response.lower() or "continue" in response.lower()
        has_bold_formatting = "**" in response
        
        clarity_score = 0.0
        if has_confirmation:
            clarity_score += 0.4
        if has_next_step:
            clarity_score += 0.4
        if has_bold_formatting:
            clarity_score += 0.2
        
        return EvaluationResult(
            metric_type=MetricType.RESPONSE_QUALITY,
            metric_name="clarity",
            score=clarity_score,
            max_score=1.0,
            details={
                "has_confirmation": has_confirmation,
                "has_next_step": has_next_step,
                "has_bold_formatting": has_bold_formatting,
                "score_explanation": f"Confirmation: {has_confirmation}, Next step: {has_next_step}, Bold: {has_bold_formatting}"
            },
            timestamp=time.time()
        )
    
    def evaluate_template_adherence(self, response: str) -> EvaluationResult:
        """Evaluate adherence to response templates"""
        template_matches = 0
        total_templates = len(self.template_patterns)
        
        for template_name, pattern in self.template_patterns.items():
            if re.search(pattern, response, re.IGNORECASE):
                template_matches += 1
        
        score = template_matches / total_templates
        
        return EvaluationResult(
            metric_type=MetricType.RESPONSE_QUALITY,
            metric_name="template_adherence",
            score=score,
            max_score=1.0,
            details={
                "template_matches": template_matches,
                "total_templates": total_templates,
                "matched_templates": [name for name, pattern in self.template_patterns.items() 
                                    if re.search(pattern, response, re.IGNORECASE)]
            },
            timestamp=time.time()
        )
    
    def evaluate_field_accuracy(self, user_input: str, response: str) -> EvaluationResult:
        """Evaluate accuracy of field extraction and confirmation"""
        # Extract values from user input
        user_values = self._extract_user_values(user_input)
        response_values = self._extract_response_values(response)
        
        # Check for contradictions
        contradictions = 0
        for field, user_value in user_values.items():
            if field in response_values:
                response_value = response_values[field]
                if user_value.lower() != response_value.lower():
                    contradictions += 1
        
        accuracy_score = 1.0 - (contradictions / max(len(user_values), 1))
        
        return EvaluationResult(
            metric_type=MetricType.RESPONSE_QUALITY,
            metric_name="field_accuracy",
            score=accuracy_score,
            max_score=1.0,
            details={
                "user_values": user_values,
                "response_values": response_values,
                "contradictions": contradictions,
                "score_explanation": f"{contradictions} contradictions found"
            },
            timestamp=time.time()
        )
    
    def _extract_user_values(self, user_input: str) -> Dict[str, str]:
        """Extract field values from user input"""
        values = {}
        
        # Extract names
        name_match = re.search(r"(?:my name is|i am|name is)\s+([A-Za-z'-]+)", user_input, re.IGNORECASE)
        if name_match:
            values["first_name"] = name_match.group(1)
        
        # Extract email
        email_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", user_input)
        if email_match:
            values["email"] = email_match.group(1)
        
        # Extract DOB
        dob_match = re.search(r"(?:dob|date of birth|born on)\s+([0-9/]+)", user_input, re.IGNORECASE)
        if dob_match:
            values["dob"] = dob_match.group(1)
        
        return values
    
    def _extract_response_values(self, response: str) -> Dict[str, str]:
        """Extract field values from response"""
        values = {}
        
        # Extract from confirmation pattern
        confirmation_match = re.search(r"Saved:\s+([^=]+)=\s*\*\*([^*]+)\*\*", response)
        if confirmation_match:
            field = confirmation_match.group(1).strip()
            value = confirmation_match.group(2).strip()
            values[field.lower().replace(" ", "_")] = value
        
        return values


class UserExperienceEvaluator:
    """Evaluates user experience metrics"""
    
    def evaluate_task_completion(self, session_history: List[Dict]) -> EvaluationResult:
        """Evaluate task completion rate"""
        total_steps = len(session_history)
        completed_steps = sum(1 for step in session_history if step.get("completed", False))
        
        completion_rate = completed_steps / max(total_steps, 1)
        
        return EvaluationResult(
            metric_type=MetricType.USER_EXPERIENCE,
            metric_name="task_completion",
            score=completion_rate,
            max_score=1.0,
            details={
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "completion_rate": completion_rate
            },
            timestamp=time.time()
        )
    
    def evaluate_error_recovery(self, session_history: List[Dict]) -> EvaluationResult:
        """Evaluate error recovery success rate"""
        error_steps = [step for step in session_history if step.get("had_error", False)]
        recovered_steps = [step for step in error_steps if step.get("recovered", False)]
        
        recovery_rate = len(recovered_steps) / max(len(error_steps), 1)
        
        return EvaluationResult(
            metric_type=MetricType.USER_EXPERIENCE,
            metric_name="error_recovery",
            score=recovery_rate,
            max_score=1.0,
            details={
                "error_steps": len(error_steps),
                "recovered_steps": len(recovered_steps),
                "recovery_rate": recovery_rate
            },
            timestamp=time.time()
        )
    
    def evaluate_scope_adherence(self, user_inputs: List[str], responses: List[str]) -> EvaluationResult:
        """Evaluate correct handling of out-of-scope requests"""
        out_of_scope_count = 0
        correctly_handled = 0
        
        for user_input, response in zip(user_inputs, responses):
            if self._is_out_of_scope(user_input):
                out_of_scope_count += 1
                if self._is_correct_out_of_scope_response(response):
                    correctly_handled += 1
        
        adherence_rate = correctly_handled / max(out_of_scope_count, 1)
        
        return EvaluationResult(
            metric_type=MetricType.USER_EXPERIENCE,
            metric_name="scope_adherence",
            score=adherence_rate,
            max_score=1.0,
            details={
                "out_of_scope_requests": out_of_scope_count,
                "correctly_handled": correctly_handled,
                "adherence_rate": adherence_rate
            },
            timestamp=time.time()
        )
    
    def _is_out_of_scope(self, user_input: str) -> bool:
        """Check if user input is out of scope"""
        out_of_scope_keywords = [
            "homework", "scholarship", "game", "tech support", "coding",
            "weather", "joke", "cook", "capital", "translate"
        ]
        return any(keyword in user_input.lower() for keyword in out_of_scope_keywords)
    
    def _is_correct_out_of_scope_response(self, response: str) -> bool:
        """Check if response correctly handles out-of-scope request"""
        return "outside my scope" in response.lower() and "onboarding" in response.lower()


class TechnicalPerformanceEvaluator:
    """Evaluates technical performance metrics"""
    
    def evaluate_response_time(self, response_times: List[float]) -> EvaluationResult:
        """Evaluate response time performance"""
        if not response_times:
            return EvaluationResult(
                metric_type=MetricType.TECHNICAL_PERFORMANCE,
                metric_name="response_time",
                score=0.0,
                max_score=1.0,
                details={"error": "No response times provided"},
                timestamp=time.time()
            )
        
        avg_time = sum(response_times) / len(response_times)
        target_time = 2.0  # seconds
        score = max(0, 1 - (avg_time - target_time) / target_time) if avg_time > target_time else 1.0
        
        return EvaluationResult(
            metric_type=MetricType.TECHNICAL_PERFORMANCE,
            metric_name="response_time",
            score=score,
            max_score=1.0,
            details={
                "average_time": avg_time,
                "target_time": target_time,
                "response_times": response_times
            },
            timestamp=time.time()
        )
    
    def evaluate_token_efficiency(self, input_tokens: int, output_tokens: int, context_tokens: int) -> EvaluationResult:
        """Evaluate token usage efficiency"""
        total_tokens = input_tokens + output_tokens + context_tokens
        efficiency_score = output_tokens / max(total_tokens, 1)
        
        return EvaluationResult(
            metric_type=MetricType.TECHNICAL_PERFORMANCE,
            metric_name="token_efficiency",
            score=efficiency_score,
            max_score=1.0,
            details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context_tokens": context_tokens,
                "total_tokens": total_tokens,
                "efficiency_ratio": efficiency_score
            },
            timestamp=time.time()
        )


class PolicyAdherenceEvaluator:
    """Evaluates policy adherence metrics"""
    
    def evaluate_age_gating(self, dob_inputs: List[str], responses: List[str]) -> EvaluationResult:
        """Evaluate correct age gating implementation"""
        underage_count = 0
        correctly_gated = 0
        
        for dob_input, response in zip(dob_inputs, responses):
            if self._is_underage(dob_input):
                underage_count += 1
                if "parent" in response.lower() or "guardian" in response.lower():
                    correctly_gated += 1
        
        gating_accuracy = correctly_gated / max(underage_count, 1)
        
        return EvaluationResult(
            metric_type=MetricType.POLICY_ADHERENCE,
            metric_name="age_gating",
            score=gating_accuracy,
            max_score=1.0,
            details={
                "underage_count": underage_count,
                "correctly_gated": correctly_gated,
                "gating_accuracy": gating_accuracy
            },
            timestamp=time.time()
        )
    
    def evaluate_terms_requirement(self, responses: List[str]) -> EvaluationResult:
        """Evaluate terms acceptance requirement"""
        terms_mentioned = sum(1 for response in responses if "terms" in response.lower())
        total_responses = len(responses)
        
        terms_coverage = terms_mentioned / max(total_responses, 1)
        
        return EvaluationResult(
            metric_type=MetricType.POLICY_ADHERENCE,
            metric_name="terms_requirement",
            score=terms_coverage,
            max_score=1.0,
            details={
                "terms_mentioned": terms_mentioned,
                "total_responses": total_responses,
                "terms_coverage": terms_coverage
            },
            timestamp=time.time()
        )
    
    def _is_underage(self, dob_input: str) -> bool:
        """Check if DOB indicates underage user"""
        # Simple check for years indicating under 18
        year_match = re.search(r"(?:19|20)([0-9]{2})", dob_input)
        if year_match:
            year = int("20" + year_match.group(1)) if year_match.group(1).startswith(("0", "1")) else int("19" + year_match.group(1))
            current_year = 2024
            age = current_year - year
            return age < 18
        return False


class MultilingualQualityEvaluator:
    """Evaluates multilingual quality metrics"""
    
    def evaluate_language_consistency(self, responses: List[str], detected_languages: List[str]) -> EvaluationResult:
        """Evaluate consistency of language usage"""
        if not responses or not detected_languages:
            return EvaluationResult(
                metric_type=MetricType.MULTILINGUAL_QUALITY,
                metric_name="language_consistency",
                score=0.0,
                max_score=1.0,
                details={"error": "No responses or languages provided"},
                timestamp=time.time()
            )
        
        consistent_count = 0
        for response, language in zip(responses, detected_languages):
            if self._is_language_consistent(response, language):
                consistent_count += 1
        
        consistency_score = consistent_count / len(responses)
        
        return EvaluationResult(
            metric_type=MetricType.MULTILINGUAL_QUALITY,
            metric_name="language_consistency",
            score=consistency_score,
            max_score=1.0,
            details={
                "consistent_count": consistent_count,
                "total_responses": len(responses),
                "consistency_score": consistency_score
            },
            timestamp=time.time()
        )
    
    def _is_language_consistent(self, response: str, language: str) -> bool:
        """Check if response is consistent with detected language"""
        if language == "en":
            return not any(char in response for char in "ñáéíóúü¿¡")
        elif language == "es":
            return any(char in response for char in "ñáéíóúü¿¡")
        elif language == "fr":
            return any(char in response for char in "àâäéèêëïîôöùûüÿç")
        elif language == "pt":
            return any(char in response for char in "ãõáéíóúâêôç")
        return True


class EnhancedEvaluationSystem:
    """Main evaluation system that coordinates all evaluators"""
    
    def __init__(self):
        self.response_quality_evaluator = ResponseQualityEvaluator()
        self.user_experience_evaluator = UserExperienceEvaluator()
        self.technical_performance_evaluator = TechnicalPerformanceEvaluator()
        self.policy_adherence_evaluator = PolicyAdherenceEvaluator()
        self.multilingual_quality_evaluator = MultilingualQualityEvaluator()
    
    def evaluate_response(self, user_input: str, response: str, 
                        session_history: List[Dict] = None,
                        response_time: float = None,
                        token_usage: Dict[str, int] = None) -> QualityMetrics:
        """Evaluate a single response comprehensively"""
        
        session_history = session_history or []
        token_usage = token_usage or {"input": 0, "output": 0, "context": 0}
        
        # Response Quality Metrics
        brevity_result = self.response_quality_evaluator.evaluate_brevity(response)
        clarity_result = self.response_quality_evaluator.evaluate_clarity(response)
        template_result = self.response_quality_evaluator.evaluate_template_adherence(response)
        accuracy_result = self.response_quality_evaluator.evaluate_field_accuracy(user_input, response)
        
        response_quality = {
            "brevity": brevity_result.score,
            "clarity": clarity_result.score,
            "template_adherence": template_result.score,
            "field_accuracy": accuracy_result.score
        }
        
        # User Experience Metrics
        task_completion_result = self.user_experience_evaluator.evaluate_task_completion(session_history)
        error_recovery_result = self.user_experience_evaluator.evaluate_error_recovery(session_history)
        scope_adherence_result = self.user_experience_evaluator.evaluate_scope_adherence([user_input], [response])
        
        user_experience = {
            "task_completion": task_completion_result.score,
            "error_recovery": error_recovery_result.score,
            "scope_adherence": scope_adherence_result.score
        }
        
        # Technical Performance Metrics
        response_time_result = self.technical_performance_evaluator.evaluate_response_time([response_time] if response_time else [])
        token_efficiency_result = self.technical_performance_evaluator.evaluate_token_efficiency(
            token_usage["input"], token_usage["output"], token_usage["context"]
        )
        
        technical_performance = {
            "response_time": response_time_result.score,
            "token_efficiency": token_efficiency_result.score
        }
        
        # Policy Adherence Metrics
        age_gating_result = self.policy_adherence_evaluator.evaluate_age_gating([user_input], [response])
        terms_requirement_result = self.policy_adherence_evaluator.evaluate_terms_requirement([response])
        
        policy_adherence = {
            "age_gating": age_gating_result.score,
            "terms_requirement": terms_requirement_result.score
        }
        
        # Multilingual Quality Metrics
        language_consistency_result = self.multilingual_quality_evaluator.evaluate_language_consistency([response], ["en"])
        
        multilingual_quality = {
            "language_consistency": language_consistency_result.score
        }
        
        # Calculate overall score
        all_scores = list(response_quality.values()) + list(user_experience.values()) + \
                    list(technical_performance.values()) + list(policy_adherence.values()) + \
                    list(multilingual_quality.values())
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return QualityMetrics(
            response_quality=response_quality,
            user_experience=user_experience,
            technical_performance=technical_performance,
            policy_adherence=policy_adherence,
            multilingual_quality=multilingual_quality,
            overall_score=overall_score,
            timestamp=time.time()
        )
    
    def evaluate_session(self, session_data: Dict[str, Any]) -> Dict[str, QualityMetrics]:
        """Evaluate an entire session"""
        results = {}
        
        for interaction in session_data.get("interactions", []):
            user_input = interaction.get("user_input", "")
            response = interaction.get("response", "")
            response_time = interaction.get("response_time", 0)
            token_usage = interaction.get("token_usage", {})
            
            metrics = self.evaluate_response(
                user_input=user_input,
                response=response,
                session_history=session_data.get("history", []),
                response_time=response_time,
                token_usage=token_usage
            )
            
            results[f"interaction_{len(results)}"] = metrics
        
        return results
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, QualityMetrics]) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report"""
        if not evaluation_results:
            return {"error": "No evaluation results provided"}
        
        # Aggregate metrics across all interactions
        aggregated_metrics = {
            "response_quality": {},
            "user_experience": {},
            "technical_performance": {},
            "policy_adherence": {},
            "multilingual_quality": {}
        }
        
        overall_scores = []
        
        for interaction_id, metrics in evaluation_results.items():
            overall_scores.append(metrics.overall_score)
            
            for category in aggregated_metrics:
                category_metrics = getattr(metrics, category)
                for metric_name, score in category_metrics.items():
                    if metric_name not in aggregated_metrics[category]:
                        aggregated_metrics[category][metric_name] = []
                    aggregated_metrics[category][metric_name].append(score)
        
        # Calculate averages
        for category in aggregated_metrics:
            for metric_name in aggregated_metrics[category]:
                scores = aggregated_metrics[category][metric_name]
                aggregated_metrics[category][metric_name] = {
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        report = {
            "summary": {
                "total_interactions": len(evaluation_results),
                "overall_average_score": sum(overall_scores) / len(overall_scores),
                "overall_min_score": min(overall_scores),
                "overall_max_score": max(overall_scores)
            },
            "detailed_metrics": aggregated_metrics,
            "recommendations": self._generate_recommendations(aggregated_metrics),
            "timestamp": time.time()
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Check response quality
        if metrics["response_quality"].get("brevity", {}).get("average", 1.0) < 0.8:
            recommendations.append("Improve response brevity - responses are too long")
        
        if metrics["response_quality"].get("clarity", {}).get("average", 1.0) < 0.8:
            recommendations.append("Improve response clarity - add more confirmations and next steps")
        
        if metrics["response_quality"].get("template_adherence", {}).get("average", 1.0) < 0.7:
            recommendations.append("Improve template adherence - responses don't follow established patterns")
        
        # Check user experience
        if metrics["user_experience"].get("scope_adherence", {}).get("average", 1.0) < 0.9:
            recommendations.append("Improve out-of-scope handling - better detection and redirection needed")
        
        # Check technical performance
        if metrics["technical_performance"].get("response_time", {}).get("average", 1.0) < 0.8:
            recommendations.append("Optimize response time - responses are taking too long")
        
        # Check policy adherence
        if metrics["policy_adherence"].get("age_gating", {}).get("average", 1.0) < 0.9:
            recommendations.append("Improve age gating - better detection of underage users needed")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    # Initialize evaluation system
    evaluator = EnhancedEvaluationSystem()
    
    # Test cases
    test_cases = [
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
    
    print("Enhanced Evaluation System Test Results")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        metrics = evaluator.evaluate_response(**test_case)
        
        print(f"Overall Score: {metrics.overall_score:.3f}")
        print(f"Response Quality: {metrics.response_quality}")
        print(f"User Experience: {metrics.user_experience}")
        print(f"Technical Performance: {metrics.technical_performance}")
        print(f"Policy Adherence: {metrics.policy_adherence}")
    
    # Generate report
    session_data = {
        "interactions": test_cases,
        "history": []
    }
    
    session_results = evaluator.evaluate_session(session_data)
    report = evaluator.generate_evaluation_report(session_results)
    
    print(f"\n--- Evaluation Report ---")
    print(f"Total Interactions: {report['summary']['total_interactions']}")
    print(f"Overall Average Score: {report['summary']['overall_average_score']:.3f}")
    print(f"Recommendations: {report['recommendations']}")
