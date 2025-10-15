#!/usr/bin/env python3
"""
Generate Enhanced Training Data for Summit Onboarding Assistant
Creates comprehensive training examples with edge cases, validation errors, and multilingual support
"""

import json
import random
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingExample:
    tag: str
    prompt: str
    completion: str


class TrainingDataGenerator:
    """Generates enhanced training data for Summit"""
    
    def __init__(self):
        self.names = {
            "first": ["Sarah", "John", "Maria", "Alex", "Emma", "Michael", "Sofia", "David", "Isabella", "James"],
            "last": ["Johnson", "Smith", "Garcia", "Chen", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore"]
        }
        
        self.emails = {
            "domains": ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "school.edu"],
            "typos": {
                "gmail.com": ["gmial.com", "gmaill.com", "gmail.co"],
                "yahoo.com": ["yaho.com", "yahooo.com", "yahoo.co"],
                "hotmail.com": ["hotmial.com", "hotmaill.com", "hotmail.co"]
            }
        }
        
        self.sports = ["basketball", "football", "soccer", "baseball", "tennis", "swimming", "track", "volleyball"]
        
        self.positions = {
            "basketball": ["point guard", "shooting guard", "small forward", "power forward", "center"],
            "football": ["quarterback", "running back", "wide receiver", "tight end", "linebacker"],
            "soccer": ["goalkeeper", "defender", "midfielder", "forward", "striker"]
        }
        
        self.strengths = ["leadership", "teamwork", "communication", "creativity", "problem-solving", 
                         "time management", "adaptability", "critical thinking", "organization", "initiative"]
        
        self.schools = ["Lincoln High", "Washington High", "Jefferson High", "Roosevelt High", "Kennedy High"]
        
        self.languages = {
            "spanish": {
                "first_name": "¿Cuál es tu nombre?",
                "last_name": "¿Cuál es tu apellido?",
                "email": "¿Cuál es tu correo electrónico?",
                "dob": "¿Cuál es tu fecha de nacimiento?",
                "confirmation": "✅ Guardado: {field} = **{value}**. Siguiente: {next_step}."
            },
            "french": {
                "first_name": "Quel est votre prénom?",
                "last_name": "Quel est votre nom de famille?",
                "email": "Quel est votre adresse e-mail?",
                "dob": "Quelle est votre date de naissance?",
                "confirmation": "✅ Enregistré: {field} = **{value}**. Suivant: {next_step}."
            },
            "portuguese": {
                "first_name": "Qual é o seu nome?",
                "last_name": "Qual é o seu sobrenome?",
                "email": "Qual é o seu e-mail?",
                "dob": "Qual é a sua data de nascimento?",
                "confirmation": "✅ Salvo: {field} = **{value}**. Próximo: {next_step}."
            }
        }
    
    def generate_field_confirmation_examples(self) -> List[TrainingExample]:
        """Generate field confirmation examples"""
        examples = []
        
        # Basic field confirmations
        field_flow = [
            ("first_name", "What's your last name?"),
            ("last_name", "What's your date of birth? (MM/DD/YYYY)"),
            ("email", "Do you accept the Terms of Service? (yes/no)"),
            ("education_level", "Are you joining as a Student or Student-Athlete?"),
            ("role", "What's your gender? (Male, Female, Non-binary, Prefer not to say)"),
            ("gender", "Tell me your story in 2-3 sentences"),
            ("bio", "What are your strengths? (Pick up to 5)"),
            ("school", "What's your graduation year?"),
            ("graduation_year", "Are you studying abroad? (yes/no)"),
            ("gpa", "What's your primary sport?"),
            ("primary_sport", "What are your positions?"),
            ("positions", "Tell me about your teams (team name + seasons)"),
            ("teams", "Add your stats (season, stat, value)"),
            ("stats", "Any awards? (name + year)"),
            ("awards", "Any career experiences? (job/intern/volunteer)"),
            ("experiences", "Any career achievements? (certifications, etc.)"),
            ("achievements", "All set! Your profile is complete. Press **Continue**.")
        ]
        
        for field, next_step in field_flow:
            if field == "first_name":
                value = random.choice(self.names["first"])
                prompt = f"My name is {value}"
            elif field == "last_name":
                value = random.choice(self.names["last"])
                prompt = f"Last name is {value}"
            elif field == "email":
                domain = random.choice(self.emails["domains"])
                value = f"{random.choice(self.names['first']).lower()}.{random.choice(self.names['last']).lower()}@{domain}"
                prompt = f"My email is {value}"
            elif field == "education_level":
                value = random.choice(["High School", "College", "Middle School"])
                prompt = value
            elif field == "role":
                value = random.choice(["Student", "Student-Athlete"])
                prompt = value
            elif field == "gender":
                value = random.choice(["Male", "Female", "Non-binary", "Prefer not to say"])
                prompt = value
            elif field == "bio":
                value = "saved"
                prompt = "I'm passionate about sports and helping my community."
            elif field == "school":
                value = random.choice(self.schools)
                prompt = value
            elif field == "graduation_year":
                value = str(random.randint(2025, 2030))
                prompt = value
            elif field == "gpa":
                value = f"{random.uniform(2.0, 4.0):.1f}"
                prompt = value
            elif field == "primary_sport":
                value = random.choice(self.sports)
                prompt = value.title()
            elif field == "positions":
                value = "saved"
                sport = random.choice(self.sports)
                positions = self.positions.get(sport, ["player"])
                prompt = f"My positions: {', '.join(random.sample(positions, min(2, len(positions))))}"
            else:
                value = "saved"
                prompt = f"Added {field.replace('_', ' ')}"
            
            completion = f"✅ Saved: {field.replace('_', ' ').title()} = **{value}**. Next: {next_step}"
            
            examples.append(TrainingExample(
                tag=f"field_confirmation_{field}",
                prompt=prompt,
                completion=completion
            ))
        
        return examples
    
    def generate_validation_error_examples(self) -> List[TrainingExample]:
        """Generate validation error examples"""
        examples = []
        
        # Email validation errors
        invalid_emails = [
            "invalid-email",
            "user@",
            "@domain.com",
            "user@domain",
            "user space@domain.com"
        ]
        
        for email in invalid_emails:
            examples.append(TrainingExample(
                tag="validation_error_email",
                prompt=f"My email is {email}",
                completion="I couldn't save that because the email format is invalid. Try: **user@example.com**."
            ))
        
        # GPA validation errors
        invalid_gpas = ["5.0", "-1.0", "4.5", "abc", "10.0"]
        for gpa in invalid_gpas:
            examples.append(TrainingExample(
                tag="validation_error_gpa",
                prompt=f"My GPA is {gpa}",
                completion="I couldn't save that because GPA must be between 0.0 and 4.0. Try: **3.5** or similar."
            ))
        
        # Graduation year validation errors
        invalid_years = ["1990", "2050", "abc", "2020", "2035"]
        for year in invalid_years:
            examples.append(TrainingExample(
                tag="validation_error_year",
                prompt=f"My graduation year is {year}",
                completion="I couldn't save that because graduation year must be between 2000 and 2040. Try: **2027**."
            ))
        
        # DOB validation errors
        invalid_dobs = ["13/45/2006", "32/01/2006", "abc", "2006/05/12"]
        for dob in invalid_dobs:
            examples.append(TrainingExample(
                tag="validation_error_dob",
                prompt=f"My DOB is {dob}",
                completion="I couldn't save that because the date format is invalid. Try: **05/12/2006**."
            ))
        
        return examples
    
    def generate_email_autocorrect_examples(self) -> List[TrainingExample]:
        """Generate email autocorrect examples"""
        examples = []
        
        for domain, typos in self.emails["typos"].items():
            for typo in typos:
                name = f"{random.choice(self.names['first']).lower()}.{random.choice(self.names['last']).lower()}"
                incorrect_email = f"{name}@{typo}"
                correct_email = f"{name}@{domain}"
                
                examples.append(TrainingExample(
                    tag="email_autocorrect",
                    prompt=f"My email is {incorrect_email}",
                    completion=f"That email looks mistyped. Did you mean **{correct_email}**? If yes, say 'Use that' to continue."
                ))
        
        return examples
    
    def generate_parental_consent_examples(self) -> List[TrainingExample]:
        """Generate parental consent examples"""
        examples = []
        
        # Underage DOB examples
        underage_dobs = ["03/15/2010", "12/01/2009", "07/20/2011"]
        for dob in underage_dobs:
            examples.append(TrainingExample(
                tag="parental_consent_dob",
                prompt=f"I was born on {dob}",
                completion=f"✅ Saved: DOB = **{dob}**. Because you're under 18, I'll need a parent/guardian email for approval. What's their email?"
            ))
        
        # Guardian email examples
        guardian_emails = ["parent@example.com", "guardian@family.org", "mom@home.net"]
        for email in guardian_emails:
            examples.append(TrainingExample(
                tag="parental_consent_email",
                prompt=f"My guardian email is {email}",
                completion="✅ Guardian email saved. Approval request sent. You can continue building your profile."
            ))
        
        return examples
    
    def generate_out_of_scope_examples(self) -> List[TrainingExample]:
        """Generate out-of-scope examples"""
        examples = []
        
        out_of_scope_topics = [
            "Can you help me with my math homework?",
            "What's the weather like today?",
            "Tell me a joke",
            "How do I cook pasta?",
            "What's the capital of France?",
            "Can you write code for me?",
            "Help me with my essay",
            "What's the best movie to watch?",
            "How do I fix my computer?",
            "Can you translate this to Spanish?"
        ]
        
        for topic in out_of_scope_topics:
            examples.append(TrainingExample(
                tag="out_of_scope",
                prompt=topic,
                completion="That's outside my scope. I can help with onboarding tasks like account creation, consent, and profile setup. What would you like to do next?"
            ))
        
        return examples
    
    def generate_multilingual_examples(self) -> List[TrainingExample]:
        """Generate multilingual examples"""
        examples = []
        
        for lang, phrases in self.languages.items():
            # Spanish examples
            if lang == "spanish":
                examples.extend([
                    TrainingExample(
                        tag=f"multilingual_{lang}_name",
                        prompt="Mi nombre es María González",
                        completion="✅ Guardado: Nombre = **María**. ¿Cuál es tu **apellido**?"
                    ),
                    TrainingExample(
                        tag=f"multilingual_{lang}_lastname",
                        prompt="Mi apellido es Rodríguez",
                        completion="✅ Guardado: Apellido = **Rodríguez**. ¿Cuál es tu **fecha de nacimiento**? (MM/DD/YYYY)"
                    )
                ])
            
            # French examples
            elif lang == "french":
                examples.extend([
                    TrainingExample(
                        tag=f"multilingual_{lang}_name",
                        prompt="Mon nom est Pierre Dubois",
                        completion="✅ Enregistré: Nom = **Pierre**. Quel est votre **nom de famille**?"
                    ),
                    TrainingExample(
                        tag=f"multilingual_{lang}_lastname",
                        prompt="Mon nom de famille est Martin",
                        completion="✅ Enregistré: Nom de famille = **Martin**. Quelle est votre **date de naissance**? (MM/DD/YYYY)"
                    )
                ])
            
            # Portuguese examples
            elif lang == "portuguese":
                examples.extend([
                    TrainingExample(
                        tag=f"multilingual_{lang}_name",
                        prompt="Meu nome é Ana Silva",
                        completion="✅ Salvo: Nome = **Ana**. Qual é o seu **sobrenome**?"
                    ),
                    TrainingExample(
                        tag=f"multilingual_{lang}_lastname",
                        prompt="Meu sobrenome é Santos",
                        completion="✅ Salvo: Sobrenome = **Santos**. Qual é a sua **data de nascimento**? (MM/DD/YYYY)"
                    )
                ])
        
        return examples
    
    def generate_edge_case_examples(self) -> List[TrainingExample]:
        """Generate edge case examples"""
        examples = []
        
        # Ambiguous DOB
        examples.append(TrainingExample(
            tag="edge_case_ambiguous_dob",
            prompt="I was born on 12/05/2006",
            completion="Got it. Is that **December 5, 2006** or **May 12, 2006**? Please clarify the format."
        ))
        
        # Multiple fields at once
        examples.append(TrainingExample(
            tag="edge_case_multiple_fields",
            prompt="My name is Alex Chen, email alex.chen@school.edu, DOB 08/20/2005",
            completion="✅ Saved: First name = **Alex**, Last name = **Chen**, Email = **alex.chen@school.edu**, DOB = **08/20/2005**. Do you accept the **Terms of Service**? (yes/no)"
        ))
        
        # Update field
        examples.append(TrainingExample(
            tag="edge_case_update_field",
            prompt="Change my graduation year to 2028",
            completion="Updated **graduation year** to **2028**. Anything else to change?"
        ))
        
        # Completion check
        examples.append(TrainingExample(
            tag="edge_case_completion_check",
            prompt="Is my profile complete?",
            completion="Let me check... You still need to provide: [list missing fields]. Let's continue with [next field]."
        ))
        
        # Error recovery
        examples.append(TrainingExample(
            tag="error_recovery",
            prompt="I made a mistake with my email",
            completion="No problem! What's the correct email address?"
        ))
        
        return examples
    
    def generate_all_examples(self) -> List[TrainingExample]:
        """Generate all training examples"""
        all_examples = []
        
        all_examples.extend(self.generate_field_confirmation_examples())
        all_examples.extend(self.generate_validation_error_examples())
        all_examples.extend(self.generate_email_autocorrect_examples())
        all_examples.extend(self.generate_parental_consent_examples())
        all_examples.extend(self.generate_out_of_scope_examples())
        all_examples.extend(self.generate_multilingual_examples())
        all_examples.extend(self.generate_edge_case_examples())
        
        return all_examples
    
    def save_to_jsonl(self, examples: List[TrainingExample], filename: str):
        """Save examples to JSONL file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump({
                    "tag": example.tag,
                    "prompt": example.prompt,
                    "completion": example.completion
                }, f, ensure_ascii=False)
                f.write('\n')


def main():
    """Generate enhanced training data"""
    generator = TrainingDataGenerator()
    
    # Generate all examples
    examples = generator.generate_all_examples()
    
    # Save to file
    output_file = "enhanced_training_data_generated.jsonl"
    generator.save_to_jsonl(examples, output_file)
    
    print(f"Generated {len(examples)} training examples")
    print(f"Saved to {output_file}")
    
    # Print summary by tag
    tag_counts = {}
    for example in examples:
        tag_counts[example.tag] = tag_counts.get(example.tag, 0) + 1
    
    print("\nExamples by tag:")
    for tag, count in sorted(tag_counts.items()):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
