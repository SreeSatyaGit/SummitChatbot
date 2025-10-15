#!/usr/bin/env python3
"""
Enhanced Multilingual Support for Summit Onboarding Assistant
Provides cultural adaptations and improved language handling
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("multilingual_support")


class SupportedLanguage(Enum):
    """Supported languages for Summit"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    PORTUGUESE = "pt"
    GERMAN = "de"
    ITALIAN = "it"
    DUTCH = "nl"


@dataclass
class LanguageTemplate:
    """Language-specific response template"""
    language: SupportedLanguage
    confirmation_pattern: str
    validation_error_pattern: str
    out_of_scope_pattern: str
    parental_consent_pattern: str
    completion_pattern: str
    field_prompts: Dict[str, str]
    cultural_adaptations: Dict[str, str]


class MultilingualTemplateManager:
    """Manages multilingual response templates with cultural adaptations"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.language_detection_patterns = self._initialize_detection_patterns()
    
    def _initialize_templates(self) -> Dict[SupportedLanguage, LanguageTemplate]:
        """Initialize language-specific templates"""
        return {
            SupportedLanguage.ENGLISH: LanguageTemplate(
                language=SupportedLanguage.ENGLISH,
                confirmation_pattern="✅ Saved: {field} = **{value}**. Next: {next_step}.",
                validation_error_pattern="I couldn't save that because {reason}. Try: {correct_format}.",
                out_of_scope_pattern="That's outside my scope. I can help with onboarding tasks like {examples}. What would you like to do next?",
                parental_consent_pattern="Because you're under 18, I'll need a parent/guardian email for approval. What's their email?",
                completion_pattern="All set! Your profile is complete. Press **Continue** to proceed.",
                field_prompts={
                    "first_name": "What's your first name?",
                    "last_name": "What's your last name?",
                    "email": "What's your email address?",
                    "dob": "What's your date of birth? (MM/DD/YYYY)",
                    "education_level": "What's your education level? (Middle School, High School, College, Coach, Pro)",
                    "role": "Are you joining as a Student or Student-Athlete?",
                    "gender": "What's your gender? (Male, Female, Non-binary, Prefer not to say)",
                    "bio": "Tell me your story in 2-3 sentences",
                    "strengths": "What are your strengths? (Pick up to 5)",
                    "school": "What's your school name?",
                    "graduation_year": "What's your graduation year?",
                    "gpa": "What's your GPA? (optional)",
                    "primary_sport": "What's your primary sport?",
                    "positions": "What are your positions?",
                    "teams": "Tell me about your teams (team name + seasons)",
                    "stats": "Add your stats (season, stat, value)",
                    "awards": "Any awards? (name + year)",
                    "experiences": "Any career experiences? (job/intern/volunteer)",
                    "achievements": "Any career achievements? (certifications, etc.)"
                },
                cultural_adaptations={
                    "formality": "professional",
                    "directness": "direct",
                    "greeting": "Hello!",
                    "closing": "Thank you!"
                }
            ),
            
            SupportedLanguage.SPANISH: LanguageTemplate(
                language=SupportedLanguage.SPANISH,
                confirmation_pattern="✅ Guardado: {field} = **{value}**. Siguiente: {next_step}.",
                validation_error_pattern="No pude guardar eso porque {reason}. Intenta: {correct_format}.",
                out_of_scope_pattern="Eso está fuera de mi alcance. Puedo ayudar con tareas de incorporación como {examples}. ¿Qué te gustaría hacer a continuación?",
                parental_consent_pattern="Como eres menor de 18 años, necesitaré el email de un padre/tutor para aprobación. ¿Cuál es su email?",
                completion_pattern="¡Todo listo! Tu perfil está completo. Presiona **Continuar** para proceder.",
                field_prompts={
                    "first_name": "¿Cuál es tu nombre?",
                    "last_name": "¿Cuál es tu apellido?",
                    "email": "¿Cuál es tu correo electrónico?",
                    "dob": "¿Cuál es tu fecha de nacimiento? (DD/MM/YYYY)",
                    "education_level": "¿Cuál es tu nivel educativo? (Secundaria, Preparatoria, Universidad, Entrenador, Profesional)",
                    "role": "¿Te unes como Estudiante o Estudiante-Atleta?",
                    "gender": "¿Cuál es tu género? (Masculino, Femenino, No binario, Prefiero no decir)",
                    "bio": "Cuéntame tu historia en 2-3 oraciones",
                    "strengths": "¿Cuáles son tus fortalezas? (Elige hasta 5)",
                    "school": "¿Cuál es el nombre de tu escuela?",
                    "graduation_year": "¿En qué año te gradúas?",
                    "gpa": "¿Cuál es tu promedio? (opcional)",
                    "primary_sport": "¿Cuál es tu deporte principal?",
                    "positions": "¿Cuáles son tus posiciones?",
                    "teams": "Cuéntame sobre tus equipos (nombre del equipo + temporadas)",
                    "stats": "Agrega tus estadísticas (temporada, estadística, valor)",
                    "awards": "¿Algún premio? (nombre + año)",
                    "experiences": "¿Alguna experiencia laboral? (trabajo/pasantía/voluntariado)",
                    "achievements": "¿Algún logro profesional? (certificaciones, etc.)"
                },
                cultural_adaptations={
                    "formality": "formal",
                    "directness": "polite",
                    "greeting": "¡Hola!",
                    "closing": "¡Gracias!"
                }
            ),
            
            SupportedLanguage.FRENCH: LanguageTemplate(
                language=SupportedLanguage.FRENCH,
                confirmation_pattern="✅ Enregistré: {field} = **{value}**. Suivant: {next_step}.",
                validation_error_pattern="Je n'ai pas pu sauvegarder cela parce que {reason}. Essayez: {correct_format}.",
                out_of_scope_pattern="Cela dépasse mon champ d'action. Je peux aider avec les tâches d'intégration comme {examples}. Que souhaitez-vous faire ensuite?",
                parental_consent_pattern="Comme vous avez moins de 18 ans, j'aurai besoin de l'email d'un parent/tuteur pour approbation. Quel est leur email?",
                completion_pattern="Tout est prêt! Votre profil est complet. Appuyez sur **Continuer** pour procéder.",
                field_prompts={
                    "first_name": "Quel est votre prénom?",
                    "last_name": "Quel est votre nom de famille?",
                    "email": "Quelle est votre adresse e-mail?",
                    "dob": "Quelle est votre date de naissance? (JJ/MM/AAAA)",
                    "education_level": "Quel est votre niveau d'éducation? (Collège, Lycée, Université, Entraîneur, Professionnel)",
                    "role": "Rejoignez-vous en tant qu'Étudiant ou Étudiant-Athlète?",
                    "gender": "Quel est votre genre? (Masculin, Féminin, Non-binaire, Préfère ne pas dire)",
                    "bio": "Racontez-moi votre histoire en 2-3 phrases",
                    "strengths": "Quelles sont vos forces? (Choisissez jusqu'à 5)",
                    "school": "Quel est le nom de votre école?",
                    "graduation_year": "En quelle année obtenez-vous votre diplôme?",
                    "gpa": "Quelle est votre moyenne? (optionnel)",
                    "primary_sport": "Quel est votre sport principal?",
                    "positions": "Quelles sont vos positions?",
                    "teams": "Parlez-moi de vos équipes (nom de l'équipe + saisons)",
                    "stats": "Ajoutez vos statistiques (saison, statistique, valeur)",
                    "awards": "Des récompenses? (nom + année)",
                    "experiences": "Des expériences professionnelles? (emploi/stage/bénévolat)",
                    "achievements": "Des réalisations professionnelles? (certifications, etc.)"
                },
                cultural_adaptations={
                    "formality": "formal",
                    "directness": "polite",
                    "greeting": "Bonjour!",
                    "closing": "Merci!"
                }
            ),
            
            SupportedLanguage.PORTUGUESE: LanguageTemplate(
                language=SupportedLanguage.PORTUGUESE,
                confirmation_pattern="✅ Salvo: {field} = **{value}**. Próximo: {next_step}.",
                validation_error_pattern="Não consegui salvar isso porque {reason}. Tente: {correct_format}.",
                out_of_scope_pattern="Isso está fora do meu escopo. Posso ajudar com tarefas de integração como {examples}. O que você gostaria de fazer a seguir?",
                parental_consent_pattern="Como você tem menos de 18 anos, precisarei do email de um pai/responsável para aprovação. Qual é o email deles?",
                completion_pattern="Tudo pronto! Seu perfil está completo. Pressione **Continuar** para prosseguir.",
                field_prompts={
                    "first_name": "Qual é o seu nome?",
                    "last_name": "Qual é o seu sobrenome?",
                    "email": "Qual é o seu e-mail?",
                    "dob": "Qual é a sua data de nascimento? (DD/MM/AAAA)",
                    "education_level": "Qual é o seu nível de educação? (Ensino Médio, Universidade, Treinador, Profissional)",
                    "role": "Você está se juntando como Estudante ou Estudante-Atleta?",
                    "gender": "Qual é o seu gênero? (Masculino, Feminino, Não-binário, Prefiro não dizer)",
                    "bio": "Me conte sua história em 2-3 frases",
                    "strengths": "Quais são suas forças? (Escolha até 5)",
                    "school": "Qual é o nome da sua escola?",
                    "graduation_year": "Em que ano você se forma?",
                    "gpa": "Qual é a sua média? (opcional)",
                    "primary_sport": "Qual é o seu esporte principal?",
                    "positions": "Quais são suas posições?",
                    "teams": "Me conte sobre seus times (nome do time + temporadas)",
                    "stats": "Adicione suas estatísticas (temporada, estatística, valor)",
                    "awards": "Algum prêmio? (nome + ano)",
                    "experiences": "Alguma experiência profissional? (trabalho/estágio/voluntariado)",
                    "achievements": "Alguma conquista profissional? (certificações, etc.)"
                },
                cultural_adaptations={
                    "formality": "friendly",
                    "directness": "polite",
                    "greeting": "Olá!",
                    "closing": "Obrigado!"
                }
            )
        }
    
    def _initialize_detection_patterns(self) -> Dict[str, List[str]]:
        """Initialize language detection patterns"""
        return {
            "spanish": [
                r"\b(español|española|hola|gracias|por favor|sí|no|nombre|apellido|correo|fecha|nacimiento)\b",
                r"[ñáéíóúü¿¡]",
                r"\b(cuál|qué|dónde|cuándo|cómo|por qué)\b"
            ],
            "french": [
                r"\b(français|française|bonjour|merci|s'il vous plaît|oui|non|prénom|nom|email|date|naissance)\b",
                r"[àâäéèêëïîôöùûüÿç]",
                r"\b(quel|quelle|où|quand|comment|pourquoi)\b"
            ],
            "portuguese": [
                r"\b(português|portuguesa|olá|obrigado|por favor|sim|não|nome|sobrenome|email|data|nascimento)\b",
                r"[ãõáéíóúâêôç]",
                r"\b(qual|onde|quando|como|por que)\b"
            ],
            "german": [
                r"\b(deutsch|hallo|danke|bitte|ja|nein|name|nachname|email|datum|geburt)\b",
                r"[äöüß]",
                r"\b(was|wo|wann|wie|warum)\b"
            ],
            "italian": [
                r"\b(italiano|italiana|ciao|grazie|per favore|sì|no|nome|cognome|email|data|nascita)\b",
                r"[àèéìíîòóù]",
                r"\b(quale|dove|quando|come|perché)\b"
            ],
            "dutch": [
                r"\b(nederlands|hallo|dank je|alsjeblieft|ja|nee|naam|achternaam|email|datum|geboorte)\b",
                r"[ëïöü]",
                r"\b(wat|waar|wanneer|hoe|waarom)\b"
            ]
        }
    
    def detect_language(self, text: str) -> SupportedLanguage:
        """Detect language from text"""
        text_lower = text.lower()
        
        # Check for explicit language indicators
        for lang_code, patterns in self.language_detection_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return SupportedLanguage(lang_code)
        
        # Default to English
        return SupportedLanguage.ENGLISH
    
    def get_template(self, language: SupportedLanguage) -> LanguageTemplate:
        """Get template for specific language"""
        return self.templates.get(language, self.templates[SupportedLanguage.ENGLISH])
    
    def generate_confirmation(self, language: SupportedLanguage, field: str, value: str, next_step: str) -> str:
        """Generate field confirmation in specific language"""
        template = self.get_template(language)
        return template.confirmation_pattern.format(
            field=field,
            value=value,
            next_step=next_step
        )
    
    def generate_validation_error(self, language: SupportedLanguage, reason: str, correct_format: str) -> str:
        """Generate validation error in specific language"""
        template = self.get_template(language)
        return template.validation_error_pattern.format(
            reason=reason,
            correct_format=correct_format
        )
    
    def generate_out_of_scope(self, language: SupportedLanguage, examples: str = None) -> str:
        """Generate out-of-scope response in specific language"""
        template = self.get_template(language)
        if examples is None:
            examples = "account creation, consent, and profile setup"
        return template.out_of_scope_pattern.format(examples=examples)
    
    def generate_parental_consent(self, language: SupportedLanguage) -> str:
        """Generate parental consent request in specific language"""
        template = self.get_template(language)
        return template.parental_consent_pattern
    
    def generate_completion(self, language: SupportedLanguage) -> str:
        """Generate completion message in specific language"""
        template = self.get_template(language)
        return template.completion_pattern
    
    def get_field_prompt(self, language: SupportedLanguage, field: str) -> str:
        """Get field prompt in specific language"""
        template = self.get_template(language)
        return template.field_prompts.get(field, f"What's your {field}?")
    
    def adapt_cultural_context(self, language: SupportedLanguage, response: str) -> str:
        """Apply cultural adaptations to response"""
        template = self.get_template(language)
        adaptations = template.cultural_adaptations
        
        # Apply formality adaptations
        if adaptations["formality"] == "formal":
            response = self._make_formal(response, language)
        elif adaptations["formality"] == "friendly":
            response = self._make_friendly(response, language)
        
        # Apply directness adaptations
        if adaptations["directness"] == "polite":
            response = self._make_polite(response, language)
        
        return response
    
    def _make_formal(self, response: str, language: SupportedLanguage) -> str:
        """Make response more formal"""
        if language == SupportedLanguage.SPANISH:
            response = response.replace("tú", "usted")
            response = response.replace("tu", "su")
        elif language == SupportedLanguage.FRENCH:
            response = response.replace("tu", "vous")
            response = response.replace("ton", "votre")
        elif language == SupportedLanguage.PORTUGUESE:
            response = response.replace("você", "o senhor/a senhora")
        
        return response
    
    def _make_friendly(self, response: str, language: SupportedLanguage) -> str:
        """Make response more friendly"""
        if language == SupportedLanguage.PORTUGUESE:
            response = response.replace("o senhor/a senhora", "você")
        
        return response
    
    def _make_polite(self, response: str, language: SupportedLanguage) -> str:
        """Make response more polite"""
        if language == SupportedLanguage.SPANISH:
            response = response.replace("No puedo", "No puedo ayudarte con eso")
        elif language == SupportedLanguage.FRENCH:
            response = response.replace("Je ne peux pas", "Je ne peux malheureusement pas")
        elif language == SupportedLanguage.PORTUGUESE:
            response = response.replace("Não posso", "Infelizmente não posso")
        
        return response


class MultilingualFieldValidator:
    """Validates fields with language-specific rules"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize language-specific validation rules"""
        return {
            "email": {
                "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                "error_messages": {
                    "en": "I need a valid email format. Try: **user@example.com**.",
                    "es": "Necesito un formato de email válido. Intenta: **usuario@ejemplo.com**.",
                    "fr": "J'ai besoin d'un format d'email valide. Essayez: **utilisateur@exemple.com**.",
                    "pt": "Preciso de um formato de email válido. Tente: **usuario@exemplo.com**."
                }
            },
            "gpa": {
                "min": 0.0,
                "max": 4.0,
                "error_messages": {
                    "en": "GPA must be between 0.0 and 4.0. Try: **3.5** or similar.",
                    "es": "El promedio debe estar entre 0.0 y 4.0. Intenta: **3.5** o similar.",
                    "fr": "La moyenne doit être entre 0.0 et 4.0. Essayez: **3.5** ou similaire.",
                    "pt": "A média deve estar entre 0.0 e 4.0. Tente: **3.5** ou similar."
                }
            },
            "graduation_year": {
                "min": 2000,
                "max": 2040,
                "error_messages": {
                    "en": "Please enter a valid graduation year (2000-2040). Try: **2027**.",
                    "es": "Por favor ingresa un año de graduación válido (2000-2040). Intenta: **2027**.",
                    "fr": "Veuillez entrer une année de diplôme valide (2000-2040). Essayez: **2027**.",
                    "pt": "Por favor insira um ano de formatura válido (2000-2040). Tente: **2027**."
                }
            },
            "dob": {
                "pattern": r"(?:0[1-9]|1[0-2])/(?:0[1-9]|[12][0-9]|3[01])/(?:19|20)[0-9]{2}",
                "error_messages": {
                    "en": "I need a valid date format. Try: **05/12/2006**.",
                    "es": "Necesito un formato de fecha válido. Intenta: **12/05/2006**.",
                    "fr": "J'ai besoin d'un format de date valide. Essayez: **12/05/2006**.",
                    "pt": "Preciso de um formato de data válido. Tente: **12/05/2006**."
                }
            }
        }
    
    def validate_field(self, field: str, value: str, language: SupportedLanguage) -> Tuple[bool, Optional[str]]:
        """Validate field value with language-specific error messages"""
        if field not in self.validation_rules:
            return True, None
        
        rules = self.validation_rules[field]
        lang_code = language.value
        
        if field == "email":
            import re
            if not re.match(rules["pattern"], value):
                error_msg = rules["error_messages"].get(lang_code, rules["error_messages"]["en"])
                return False, error_msg
        
        elif field == "gpa":
            try:
                gpa = float(value)
                if not (rules["min"] <= gpa <= rules["max"]):
                    error_msg = rules["error_messages"].get(lang_code, rules["error_messages"]["en"])
                    return False, error_msg
            except (ValueError, TypeError):
                error_msg = rules["error_messages"].get(lang_code, rules["error_messages"]["en"])
                return False, error_msg
        
        elif field == "graduation_year":
            try:
                year = int(value)
                if not (rules["min"] <= year <= rules["max"]):
                    error_msg = rules["error_messages"].get(lang_code, rules["error_messages"]["en"])
                    return False, error_msg
            except (ValueError, TypeError):
                error_msg = rules["error_messages"].get(lang_code, rules["error_messages"]["en"])
                return False, error_msg
        
        elif field == "dob":
            import re
            if not re.match(rules["pattern"], value):
                error_msg = rules["error_messages"].get(lang_code, rules["error_messages"]["en"])
                return False, error_msg
        
        return True, None


class EnhancedMultilingualService:
    """Main multilingual service that coordinates all components"""
    
    def __init__(self):
        self.template_manager = MultilingualTemplateManager()
        self.field_validator = MultilingualFieldValidator()
    
    def process_multilingual_response(self, user_input: str, field: str, value: str, 
                                    next_step: str = None) -> str:
        """Process response with multilingual support"""
        # Detect language
        language = self.template_manager.detect_language(user_input)
        
        # Generate response
        if next_step:
            response = self.template_manager.generate_confirmation(language, field, value, next_step)
        else:
            response = self.template_manager.generate_completion(language)
        
        # Apply cultural adaptations
        response = self.template_manager.adapt_cultural_context(language, response)
        
        return response
    
    def validate_multilingual_field(self, field: str, value: str, user_input: str) -> Tuple[bool, Optional[str]]:
        """Validate field with multilingual error messages"""
        language = self.template_manager.detect_language(user_input)
        return self.field_validator.validate_field(field, value, language)
    
    def get_multilingual_prompt(self, field: str, user_input: str) -> str:
        """Get field prompt in user's language"""
        language = self.template_manager.detect_language(user_input)
        return self.template_manager.get_field_prompt(language, field)


# Example usage and testing
if __name__ == "__main__":
    # Initialize multilingual service
    multilingual_service = EnhancedMultilingualService()
    
    # Test cases
    test_cases = [
        {
            "user_input": "Mi nombre es María González",
            "field": "first_name",
            "value": "María",
            "next_step": "¿Cuál es tu apellido?"
        },
        {
            "user_input": "Mon nom est Pierre Dubois",
            "field": "first_name", 
            "value": "Pierre",
            "next_step": "Quel est votre nom de famille?"
        },
        {
            "user_input": "Meu nome é Ana Silva",
            "field": "first_name",
            "value": "Ana", 
            "next_step": "Qual é o seu sobrenome?"
        },
        {
            "user_input": "My name is John Smith",
            "field": "first_name",
            "value": "John",
            "next_step": "What's your last name?"
        }
    ]
    
    print("Enhanced Multilingual Service Test")
    print("=" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"User Input: {test_case['user_input']}")
        
        response = multilingual_service.process_multilingual_response(**test_case)
        print(f"Response: {response}")
        
        # Test validation
        is_valid, error_msg = multilingual_service.validate_multilingual_field(
            "email", "invalid-email", test_case['user_input']
        )
        print(f"Email Validation: {'Valid' if is_valid else f'Invalid - {error_msg}'}")
    
    # Test field prompts
    print(f"\n--- Field Prompts ---")
    languages = ["My name is John", "Mi nombre es Juan", "Mon nom est Jean", "Meu nome é João"]
    
    for lang_input in languages:
        language = multilingual_service.template_manager.detect_language(lang_input)
        prompt = multilingual_service.get_multilingual_prompt("email", lang_input)
        print(f"{language.value}: {prompt}")
