#!/usr/bin/env python3
"""
Enhanced RAG Service for Summit Onboarding Assistant
Provides improved query rewriting, context utilization, and response generation
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("enhanced_rag_service")


class QueryIntent(Enum):
    """Types of user intents for better query rewriting"""
    FIELD_INPUT = "field_input"
    VALIDATION_ERROR = "validation_error"
    HELP_REQUEST = "help_request"
    PROFILE_CREATION = "profile_creation"
    PROFILE_UPDATE = "profile_update"
    OUT_OF_SCOPE = "out_of_scope"
    CLARIFICATION = "clarification"


@dataclass
class QueryContext:
    """Context for query rewriting"""
    user_message: str
    session_state: Dict[str, Any]
    current_field: Optional[str] = None
    extracted_fields: Dict[str, Any] = None
    validation_errors: List[str] = None


class EnhancedQueryRewriter:
    """Enhanced query rewriting for better retrieval"""
    
    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.field_mappings = self._initialize_field_mappings()
        self.context_keywords = self._initialize_context_keywords()
    
    def _initialize_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Initialize patterns for intent detection"""
        return {
            QueryIntent.FIELD_INPUT: [
                r"my name is",
                r"i am",
                r"email is",
                r"dob is",
                r"school is",
                r"gpa is",
                r"sport is",
                r"position is"
            ],
            QueryIntent.VALIDATION_ERROR: [
                r"invalid",
                r"error",
                r"wrong",
                r"incorrect",
                r"can't save",
                r"doesn't work"
            ],
            QueryIntent.HELP_REQUEST: [
                r"how do i",
                r"what should i",
                r"help me",
                r"can you help",
                r"what is",
                r"explain"
            ],
            QueryIntent.PROFILE_CREATION: [
                r"create profile",
                r"sign up",
                r"register",
                r"new account",
                r"start profile"
            ],
            QueryIntent.PROFILE_UPDATE: [
                r"change",
                r"update",
                r"edit",
                r"modify",
                r"fix"
            ],
            QueryIntent.OUT_OF_SCOPE: [
                r"homework",
                r"scholarship",
                r"game",
                r"tech support",
                r"coding",
                r"unrelated"
            ],
            QueryIntent.CLARIFICATION: [
                r"what do you mean",
                r"clarify",
                r"explain more",
                r"not sure"
            ]
        }
    
    def _initialize_field_mappings(self) -> Dict[str, List[str]]:
        """Initialize field-specific keywords for better retrieval"""
        return {
            "account": ["name", "email", "dob", "date of birth", "terms", "tos", "guardian", "parent"],
            "identity": ["education", "role", "gender", "student", "athlete"],
            "story": ["bio", "story", "strengths", "about me"],
            "academics": ["school", "graduation", "gpa", "abroad", "studying"],
            "athletics": ["sport", "position", "team", "stats", "awards", "performance"],
            "career": ["job", "internship", "volunteer", "experience", "achievement", "certification"]
        }
    
    def _initialize_context_keywords(self) -> Dict[str, List[str]]:
        """Initialize context-specific keywords"""
        return {
            "policies": ["terms", "privacy", "consent", "age", "guardian", "parental"],
            "validation": ["format", "valid", "required", "error", "invalid"],
            "procedures": ["how to", "steps", "process", "guide", "instructions"],
            "fields": ["field", "information", "data", "details", "profile"]
        }
    
    def detect_intent(self, query: str) -> QueryIntent:
        """Detect user intent from query"""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Default to help request for unclear intents
        return QueryIntent.HELP_REQUEST
    
    def rewrite_query_for_retrieval(self, context: QueryContext) -> str:
        """Rewrite user query for better retrieval"""
        intent = self.detect_intent(context.user_message)
        
        # Extract field information
        current_field = self._extract_current_field(context.user_message)
        field_category = self._get_field_category(current_field)
        
        # Build retrieval query based on intent
        if intent == QueryIntent.FIELD_INPUT:
            return self._rewrite_field_input_query(context, current_field, field_category)
        elif intent == QueryIntent.VALIDATION_ERROR:
            return self._rewrite_validation_query(context, current_field)
        elif intent == QueryIntent.HELP_REQUEST:
            return self._rewrite_help_query(context, current_field, field_category)
        elif intent == QueryIntent.PROFILE_CREATION:
            return "onboarding profile creation account setup requirements policies"
        elif intent == QueryIntent.PROFILE_UPDATE:
            return f"onboarding profile update {current_field} field modification"
        elif intent == QueryIntent.OUT_OF_SCOPE:
            return "onboarding scope limitations out of scope topics"
        else:
            return f"onboarding {current_field} {field_category} help guidance"
    
    def _extract_current_field(self, message: str) -> Optional[str]:
        """Extract the current field being discussed"""
        message_lower = message.lower()
        
        field_keywords = {
            "first_name": ["first name", "name", "given name"],
            "last_name": ["last name", "surname", "family name"],
            "email": ["email", "e-mail", "mail"],
            "dob": ["dob", "date of birth", "birthday", "birth date"],
            "school": ["school", "college", "university", "institution"],
            "gpa": ["gpa", "grade point average", "grades"],
            "sport": ["sport", "athletic", "sports"],
            "position": ["position", "role", "playing position"],
            "team": ["team", "club", "organization"],
            "stats": ["stats", "statistics", "performance", "numbers"],
            "awards": ["awards", "achievements", "recognition"],
            "bio": ["bio", "story", "about me", "personal story"],
            "strengths": ["strengths", "skills", "talents", "abilities"]
        }
        
        for field, keywords in field_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return field
        
        return None
    
    def _get_field_category(self, field: Optional[str]) -> str:
        """Get the category for a field"""
        if not field:
            return "general"
        
        for category, fields in self.field_mappings.items():
            if field in fields:
                return category
        
        return "general"
    
    def _rewrite_field_input_query(self, context: QueryContext, field: Optional[str], category: str) -> str:
        """Rewrite field input queries for better retrieval"""
        if not field:
            return f"onboarding {category} field requirements validation"
        
        return f"onboarding {field} field requirements validation format examples {category}"
    
    def _rewrite_validation_query(self, context: QueryContext, field: Optional[str]) -> str:
        """Rewrite validation error queries"""
        if not field:
            return "onboarding validation errors field requirements format"
        
        return f"onboarding {field} validation errors requirements format examples"
    
    def _rewrite_help_query(self, context: QueryContext, field: Optional[str], category: str) -> str:
        """Rewrite help requests for better retrieval"""
        if not field:
            return f"onboarding {category} help guidance instructions"
        
        return f"onboarding {field} help guidance instructions examples {category}"


class ContextUtilizer:
    """Enhanced context utilization for better responses"""
    
    def __init__(self):
        self.relevance_threshold = 0.7
        self.max_context_length = 800
    
    def filter_relevant_context(self, contexts: List[Dict], query: str, intent: QueryIntent) -> List[Dict]:
        """Filter contexts based on relevance and intent"""
        if not contexts:
            return []
        
        # Score contexts based on relevance
        scored_contexts = []
        for ctx in contexts:
            score = self._calculate_relevance_score(ctx, query, intent)
            if score >= self.relevance_threshold:
                scored_contexts.append((score, ctx))
        
        # Sort by relevance score
        scored_contexts.sort(key=lambda x: x[0], reverse=True)
        
        # Return top contexts
        return [ctx for _, ctx in scored_contexts[:3]]
    
    def _calculate_relevance_score(self, context: Dict, query: str, intent: QueryIntent) -> float:
        """Calculate relevance score for a context"""
        text = context.get("text", "").lower()
        query_lower = query.lower()
        
        # Base score from text similarity
        score = 0.0
        
        # Check for exact keyword matches
        query_words = set(query_lower.split())
        text_words = set(text.split())
        word_overlap = len(query_words.intersection(text_words))
        score += word_overlap / len(query_words) * 0.4
        
        # Check for field-specific relevance
        field_relevance = self._check_field_relevance(context, intent)
        score += field_relevance * 0.3
        
        # Check for policy/procedure relevance
        policy_relevance = self._check_policy_relevance(context, intent)
        score += policy_relevance * 0.3
        
        return min(score, 1.0)
    
    def _check_field_relevance(self, context: Dict, intent: QueryIntent) -> float:
        """Check field-specific relevance"""
        text = context.get("text", "").lower()
        
        field_keywords = {
            QueryIntent.FIELD_INPUT: ["field", "input", "data", "information"],
            QueryIntent.VALIDATION_ERROR: ["validation", "error", "format", "required"],
            QueryIntent.HELP_REQUEST: ["help", "guide", "instructions", "how to"],
            QueryIntent.PROFILE_CREATION: ["profile", "creation", "setup", "account"],
            QueryIntent.PROFILE_UPDATE: ["update", "modify", "change", "edit"]
        }
        
        keywords = field_keywords.get(intent, [])
        if not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return matches / len(keywords)
    
    def _check_policy_relevance(self, context: Dict, intent: QueryIntent) -> float:
        """Check policy/procedure relevance"""
        text = context.get("text", "").lower()
        
        policy_keywords = ["policy", "procedure", "rule", "requirement", "guideline"]
        matches = sum(1 for keyword in policy_keywords if keyword in text)
        
        return matches / len(policy_keywords)
    
    def format_context_for_prompt(self, contexts: List[Dict], query: str) -> str:
        """Format contexts for inclusion in prompts"""
        if not contexts:
            return "No relevant context found."
        
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            text = ctx.get("text", "")
            source = ctx.get("source", f"chunk_{i}")
            
            # Truncate if too long
            if len(text) > 200:
                text = text[:197] + "..."
            
            formatted_contexts.append(f"[{source}] {text}")
        
        return "\n".join(formatted_contexts)


class EnhancedRAGService:
    """Main enhanced RAG service"""
    
    def __init__(self):
        self.query_rewriter = EnhancedQueryRewriter()
        self.context_utilizer = ContextUtilizer()
    
    def process_query(self, user_message: str, session_state: Dict[str, Any], 
                     extracted_fields: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user query with enhanced RAG"""
        
        # Create query context
        context = QueryContext(
            user_message=user_message,
            session_state=session_state,
            extracted_fields=extracted_fields or {}
        )
        
        # Rewrite query for better retrieval
        rewritten_query = self.query_rewriter.rewrite_query_for_retrieval(context)
        
        # Detect intent
        intent = self.query_rewriter.detect_intent(user_message)
        
        # Generate enhanced prompt
        enhanced_prompt = self._generate_enhanced_prompt(
            user_message, rewritten_query, intent, context
        )
        
        return {
            "rewritten_query": rewritten_query,
            "intent": intent.value,
            "enhanced_prompt": enhanced_prompt,
            "context": context
        }
    
    def _generate_enhanced_prompt(self, user_message: str, rewritten_query: str, 
                                 intent: QueryIntent, context: QueryContext) -> str:
        """Generate enhanced prompt with better instructions"""
        
        # Base system prompt
        system_prompt = (
            "You are Summit, a specialized onboarding assistant. "
            "Your role is to guide users through profile completion with precision and efficiency.\n\n"
        )
        
        # Add intent-specific instructions
        intent_instructions = self._get_intent_instructions(intent)
        
        # Add context-specific instructions
        context_instructions = self._get_context_instructions(context)
        
        # Build the complete prompt
        prompt_parts = [
            system_prompt,
            intent_instructions,
            context_instructions,
            f"User query: {user_message}",
            f"Retrieval query: {rewritten_query}",
            "Provide a concise, actionable response following the established patterns."
        ]
        
        return "\n".join(filter(None, prompt_parts))
    
    def _get_intent_instructions(self, intent: QueryIntent) -> str:
        """Get intent-specific instructions"""
        instructions = {
            QueryIntent.FIELD_INPUT: (
                "The user is providing field information. Confirm what you've saved "
                "and ask for the next missing field. Use exact user-provided values."
            ),
            QueryIntent.VALIDATION_ERROR: (
                "The user encountered a validation error. Explain the issue clearly "
                "and provide the correct format with examples."
            ),
            QueryIntent.HELP_REQUEST: (
                "The user needs help or guidance. Provide clear, step-by-step "
                "instructions and ask one clarifying question if needed."
            ),
            QueryIntent.PROFILE_CREATION: (
                "The user wants to create a profile. Guide them through the "
                "onboarding process step by step."
            ),
            QueryIntent.PROFILE_UPDATE: (
                "The user wants to update their profile. Help them modify "
                "specific fields while maintaining data integrity."
            ),
            QueryIntent.OUT_OF_SCOPE: (
                "The user's request is out of scope. Politely redirect them "
                "to onboarding-related topics."
            ),
            QueryIntent.CLARIFICATION: (
                "The user needs clarification. Provide clear explanations "
                "and examples to help them understand."
            )
        }
        
        return instructions.get(intent, "Provide helpful guidance for onboarding.")
    
    def _get_context_instructions(self, context: QueryContext) -> str:
        """Get context-specific instructions"""
        instructions = []
        
        if context.extracted_fields:
            instructions.append(
                f"Extracted fields: {', '.join(context.extracted_fields.keys())}"
            )
        
        if context.session_state:
            completed_fields = [
                field for field, value in context.session_state.items() 
                if value is not None and value != ""
            ]
            if completed_fields:
                instructions.append(
                    f"Completed fields: {', '.join(completed_fields)}"
                )
        
        return "\n".join(instructions) if instructions else ""


# Example usage and testing
if __name__ == "__main__":
    # Initialize the service
    rag_service = EnhancedRAGService()
    
    # Test cases
    test_cases = [
        {
            "message": "My name is John Smith",
            "session_state": {},
            "extracted_fields": {"first_name": "John", "last_name": "Smith"}
        },
        {
            "message": "I can't save my email, it says invalid format",
            "session_state": {"first_name": "John"},
            "extracted_fields": {}
        },
        {
            "message": "How do I create my profile?",
            "session_state": {},
            "extracted_fields": {}
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        result = rag_service.process_query(**test_case)
        print(f"Intent: {result['intent']}")
        print(f"Rewritten Query: {result['rewritten_query']}")
        print(f"Enhanced Prompt:\n{result['enhanced_prompt']}")
        print("-" * 50)
