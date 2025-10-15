#!/usr/bin/env python3
"""
Standardized Response Templates for Summit Onboarding Assistant
Provides consistent response patterns for improved user experience
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ResponseType(Enum):
    """Types of responses Summit can generate"""
    FIELD_CONFIRMATION = "field_confirmation"
    VALIDATION_ERROR = "validation_error"
    PARENTAL_CONSENT = "parental_consent"
    OUT_OF_SCOPE = "out_of_scope"
    EMAIL_AUTOCORRECT = "email_autocorrect"
    COMPLETION = "completion"
    CLARIFICATION = "clarification"
    UPDATE_CONFIRMATION = "update_confirmation"
    HELP_GUIDANCE = "help_guidance"


@dataclass
class ResponseTemplate:
    """Template for generating consistent responses"""
    template: str
    required_fields: List[str]
    optional_fields: List[str] = None
    max_length: int = 100


class ResponseTemplateManager:
    """Manages standardized response templates"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[ResponseType, ResponseTemplate]:
        """Initialize all response templates"""
        return {
            ResponseType.FIELD_CONFIRMATION: ResponseTemplate(
                template="✅ Saved: {field} = **{value}**. Next: {next_step}.",
                required_fields=["field", "value", "next_step"],
                max_length=80
            ),
            
            ResponseType.VALIDATION_ERROR: ResponseTemplate(
                template="I couldn't save that because {reason}. Try: {correct_format}.",
                required_fields=["reason", "correct_format"],
                max_length=90
            ),
            
            ResponseType.PARENTAL_CONSENT: ResponseTemplate(
                template="Parental consent required. Add a parent/guardian email to send an approval link.",
                required_fields=[],
                max_length=85
            ),
            
            ResponseType.OUT_OF_SCOPE: ResponseTemplate(
                template="That's outside my scope. I can help with onboarding tasks like {examples}. What would you like to do next?",
                required_fields=["examples"],
                max_length=95
            ),
            
            ResponseType.EMAIL_AUTOCORRECT: ResponseTemplate(
                template="That email looks mistyped. Did you mean **{candidate}**? If yes, say 'Use that' to continue.",
                required_fields=["candidate"],
                max_length=85
            ),
            
            ResponseType.COMPLETION: ResponseTemplate(
                template="All set! Your profile is complete. Press **Continue** to proceed.",
                required_fields=[],
                max_length=70
            ),
            
            ResponseType.CLARIFICATION: ResponseTemplate(
                template="Got it. What's the {field_format}? ({example_format})",
                required_fields=["field_format", "example_format"],
                max_length=75
            ),
            
            ResponseType.UPDATE_CONFIRMATION: ResponseTemplate(
                template="Updated **{field}** to **{value}**. Anything else to change?",
                required_fields=["field", "value"],
                max_length=70
            ),
            
            ResponseType.HELP_GUIDANCE: ResponseTemplate(
                template="Here's how to {action}: {steps}",
                required_fields=["action", "steps"],
                max_length=90
            )
        }
    
    def generate_response(self, 
                         response_type: ResponseType, 
                         **kwargs) -> str:
        """Generate a response using the specified template"""
        template = self.templates.get(response_type)
        if not template:
            raise ValueError(f"Unknown response type: {response_type}")
        
        # Validate required fields
        missing_fields = [field for field in template.required_fields 
                         if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Generate response
        response = template.template.format(**kwargs)
        
        # Validate length
        if len(response) > template.max_length:
            response = response[:template.max_length-3] + "..."
        
        return response
    
    def get_field_confirmation(self, field: str, value: str, next_step: str) -> str:
        """Generate field confirmation response"""
        return self.generate_response(
            ResponseType.FIELD_CONFIRMATION,
            field=field,
            value=value,
            next_step=next_step
        )
    
    def get_validation_error(self, reason: str, correct_format: str) -> str:
        """Generate validation error response"""
        return self.generate_response(
            ResponseType.VALIDATION_ERROR,
            reason=reason,
            correct_format=correct_format
        )
    
    def get_parental_consent(self) -> str:
        """Generate parental consent response"""
        return self.generate_response(ResponseType.PARENTAL_CONSENT)
    
    def get_out_of_scope(self, examples: str = "account creation, consent, and profile setup") -> str:
        """Generate out-of-scope response"""
        return self.generate_response(
            ResponseType.OUT_OF_SCOPE,
            examples=examples
        )
    
    def get_email_autocorrect(self, candidate: str) -> str:
        """Generate email autocorrect response"""
        return self.generate_response(
            ResponseType.EMAIL_AUTOCORRECT,
            candidate=candidate
        )
    
    def get_completion(self) -> str:
        """Generate completion response"""
        return self.generate_response(ResponseType.COMPLETION)
    
    def get_clarification(self, field_format: str, example_format: str) -> str:
        """Generate clarification response"""
        return self.generate_response(
            ResponseType.CLARIFICATION,
            field_format=field_format,
            example_format=example_format
        )
    
    def get_update_confirmation(self, field: str, value: str) -> str:
        """Generate update confirmation response"""
        return self.generate_response(
            ResponseType.UPDATE_CONFIRMATION,
            field=field,
            value=value
        )
    
    def get_help_guidance(self, action: str, steps: str) -> str:
        """Generate help guidance response"""
        return self.generate_response(
            ResponseType.HELP_GUIDANCE,
            action=action,
            steps=steps
        )


# Field-specific templates
FIELD_TEMPLATES = {
    "first_name": {
        "confirmation": "✅ Saved: First name = **{value}**. What's your **last name**?",
        "validation": "I need a valid first name. Try: **John** or **Sarah**."
    },
    "last_name": {
        "confirmation": "✅ Saved: Last name = **{value}**. What's your **date of birth**? (MM/DD/YYYY)",
        "validation": "I need a valid last name. Try: **Smith** or **Johnson**."
    },
    "dob": {
        "confirmation": "✅ Saved: DOB = **{value}**. What's your **email address**?",
        "validation": "I need a valid date. Try: **05/12/2006** or **May 12, 2006**.",
        "parental_consent": "✅ Saved: DOB = **{value}**. Because you're under 18, I'll need a parent/guardian email for approval. What's their email?"
    },
    "email": {
        "confirmation": "✅ Saved: Email = **{value}**. Do you accept the **Terms of Service**? (yes/no)",
        "validation": "I need a valid email. Try: **user@example.com**.",
        "autocorrect": "That email looks mistyped. Did you mean **{candidate}**? If yes, say 'Use that' to continue."
    },
    "tos_accepted": {
        "confirmation": "✅ Terms accepted. What's your **education level**? (Middle School, High School, College, Coach, Pro)",
        "validation": "Please accept the terms to continue. Type **yes** to proceed."
    },
    "guardian_email": {
        "confirmation": "✅ Guardian email saved. Approval request sent. You can continue building your profile.",
        "validation": "I need a valid guardian email. Try: **parent@example.com**."
    },
    "education_level": {
        "confirmation": "✅ Education level set to **{value}**. Are you joining as a **Student** or **Student-Athlete**?",
        "validation": "Please select: Middle School, High School, College, Coach, or Pro."
    },
    "role": {
        "confirmation": "✅ Role set to **{value}**. What's your **gender**? (Male, Female, Non-binary, Prefer not to say)",
        "validation": "Please select: Student or Student-Athlete."
    },
    "gender": {
        "confirmation": "✅ Gender set to **{value}**. Tell me your **story** in 2-3 sentences.",
        "validation": "Please select: Male, Female, Non-binary, or Prefer not to say."
    },
    "bio": {
        "confirmation": "✅ Story saved. What are your **strengths**? (Pick up to 5)",
        "validation": "Please share your story in 2-3 sentences."
    },
    "strengths": {
        "confirmation": "✅ Strengths saved. What's your **school name**?",
        "validation": "Please select up to 5 strengths from the list."
    },
    "school": {
        "confirmation": "✅ School saved: **{value}**. What's your **graduation year**?",
        "validation": "Please enter your school name."
    },
    "graduation_year": {
        "confirmation": "✅ Graduation year set to **{value}**. Are you **studying abroad**? (yes/no)",
        "validation": "Please enter a valid graduation year (e.g., 2027)."
    },
    "abroad": {
        "confirmation": "✅ Studying abroad: **{value}**. What's your **GPA**? (optional)",
        "validation": "Please answer: yes or no."
    },
    "gpa": {
        "confirmation": "✅ GPA saved: **{value}**. What's your **primary sport**?",
        "validation": "Please enter a GPA between 0.0 and 4.0."
    },
    "primary_sport": {
        "confirmation": "✅ Sport selected: **{value}**. What are your **positions**?",
        "validation": "Please select a sport from the list."
    },
    "positions": {
        "confirmation": "✅ Positions saved. Tell me about your **teams** (team name + seasons).",
        "validation": "Please select positions for your sport."
    },
    "teams": {
        "confirmation": "✅ Teams saved. Add your **stats** (season, stat, value).",
        "validation": "Please enter team name and seasons."
    },
    "stats": {
        "confirmation": "✅ Stats saved. Any **awards**? (name + year)",
        "validation": "Please enter stats in format: season, stat name, value."
    },
    "awards": {
        "confirmation": "✅ Awards saved. Any **career experiences**? (job/intern/volunteer)",
        "validation": "Please enter award name and year."
    },
    "experiences": {
        "confirmation": "✅ Experiences saved. Any **career achievements**? (certifications, etc.)",
        "validation": "Please enter experience details."
    },
    "achievements": {
        "confirmation": "✅ Achievements saved. **All set!** Your profile is complete. Press **Continue**.",
        "validation": "Please enter achievement details."
    }
}


def get_field_template(field: str, action: str, **kwargs) -> str:
    """Get field-specific template"""
    if field not in FIELD_TEMPLATES:
        return f"✅ {field} saved. Continue to next step."
    
    if action not in FIELD_TEMPLATES[field]:
        return f"✅ {field} saved. Continue to next step."
    
    template = FIELD_TEMPLATES[field][action]
    return template.format(**kwargs)


# Validation rules for common fields
VALIDATION_RULES = {
    "email": {
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "error_message": "I need a valid email format. Try: **user@example.com**"
    },
    "gpa": {
        "min": 0.0,
        "max": 4.0,
        "error_message": "GPA must be between 0.0 and 4.0. Try: **3.5**"
    },
    "graduation_year": {
        "min": 2000,
        "max": 2040,
        "error_message": "Please enter a valid graduation year (2000-2040). Try: **2027**"
    },
    "dob": {
        "min_age": 13,
        "max_age": 100,
        "error_message": "Please enter a valid date of birth."
    }
}


def validate_field(field: str, value: Any) -> tuple[bool, Optional[str]]:
    """Validate field value and return (is_valid, error_message)"""
    if field not in VALIDATION_RULES:
        return True, None
    
    rules = VALIDATION_RULES[field]
    
    if field == "email":
        import re
        if not re.match(rules["pattern"], str(value)):
            return False, rules["error_message"]
    
    elif field == "gpa":
        try:
            gpa = float(value)
            if not (rules["min"] <= gpa <= rules["max"]):
                return False, rules["error_message"]
        except (ValueError, TypeError):
            return False, rules["error_message"]
    
    elif field == "graduation_year":
        try:
            year = int(value)
            if not (rules["min"] <= year <= rules["max"]):
                return False, rules["error_message"]
        except (ValueError, TypeError):
            return False, rules["error_message"]
    
    return True, None


if __name__ == "__main__":
    # Example usage
    manager = ResponseTemplateManager()
    
    # Test field confirmation
    response = manager.get_field_confirmation("first_name", "John", "What's your last name?")
    print(f"Field confirmation: {response}")
    
    # Test validation error
    response = manager.get_validation_error("invalid email format", "user@example.com")
    print(f"Validation error: {response}")
    
    # Test field-specific template
    response = get_field_template("email", "confirmation", value="john@example.com")
    print(f"Field template: {response}")
