#!/usr/bin/env python
"""Summit AI conversational onboarding API.

Endpoints
---------
POST /sessions                      -> create a new onboarding chat session
GET  /sessions/{sid}                -> fetch current state
POST /sessions/{sid}/message        -> send a user message; returns assistant reply, extracted patch, merged state, nextPrompt
GET  /health

Notes
-----
- Uses your model worker (default http://127.0.0.1:8001/generate) for the *chat reply*.
- Runs a lightweight rule-based extractor in parallel to fill onboarding fields out-of-order.
- Maintains an in-memory session store (swap for Redis/DB in prod).
- Enhanced with standardized response templates for consistent user experience.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

try:
    import httpx
except Exception:
    httpx = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger("service.app")

# Add the fine_tuning directory to the path for importing response templates
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'fine_tuning', 'data', 'kb'))

try:
    from response_templates import ResponseTemplateManager, ResponseType, get_field_template, validate_field
    TEMPLATE_MANAGER = ResponseTemplateManager()
except ImportError:
    log.warning("Response templates not available, using fallback responses")
    TEMPLATE_MANAGER = None

# Import quality monitoring
try:
    from .quality_monitor import get_quality_monitor
    QUALITY_MONITOR = get_quality_monitor()
except ImportError:
    log.warning("Quality monitoring not available")
    QUALITY_MONITOR = None

# Import multilingual support
try:
    from .multilingual_support import EnhancedMultilingualService
    MULTILINGUAL_SERVICE = EnhancedMultilingualService()
except ImportError:
    log.warning("Multilingual support not available")
    MULTILINGUAL_SERVICE = None

app = FastAPI(title="Summit AI Conversational Onboarding")

# ---------------------------------------------------------------------
# Models: State & API shapes
# ---------------------------------------------------------------------

SPORTS = {
    "badminton","baseball","basketball","bmx","combat sports","crew","cricket","cross country",
    "esports","fishing","fencing","field hockey","flag football","football","golf",
}

BASKETBALL_POSITIONS = {
    "point guard","shooting guard","small forward","power forward","center",
}

STRENGTHS = {
    "content creation","choir","dancing","drawing","fashion","musical instruments",
    "activism","animal welfare","education access","literacy coaching",
}

EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
YEAR_RE = re.compile(r"\b(20[2-4][0-9])\b")  # 2020..2049
GPA_RE = re.compile(r"(?i)\bgpa\b[:\s]*([0-4](?:\.\d{1,2})?)")
DATE_ANY_RE = re.compile(r"(?i)\b(?:(\d{4})[-/](\d{1,2})[-/](\d{1,2})|([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})|(\d{1,2})[/-](\d{1,2})[/-](\d{2,4}))\b")

def today() -> date:
    return date.today()

def try_parse_date(s: str) -> Optional[date]:
    """Accepts YYYY-MM-DD, MM/DD/YYYY, 'May 12 2002' etc."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    # heuristic parse from regex groups
    m = DATE_ANY_RE.search(s)
    if not m:
        return None
    if m.group(1):  # yyyy-mm-dd style
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return date(y, mo, d)
    if m.group(4):  # Month name d, yyyy
        try:
            return datetime.strptime(f"{m.group(4)} {m.group(5)} {m.group(6)}", "%B %d %Y").date()
        except Exception:
            try:
                return datetime.strptime(f"{m.group(4)} {m.group(5)} {m.group(6)}", "%b %d %Y").date()
            except Exception:
                return None
    if m.group(7):  # mm/dd/yy or yyyy
        mo, d, y = int(m.group(7)), int(m.group(8)), int(m.group(9))
        if y < 100:
            y += 2000
        return date(y, mo, d)
    return None

def calc_age(dob: date) -> int:
    t = today()
    return t.year - dob.year - ((t.month, t.day) < (dob.month, dob.day))

def yn_to_bool(s: str) -> Optional[bool]:
    s = s.strip().lower()
    if s in {"y","yes","true"}: return True
    if s in {"n","no","false"}: return False
    return None

class Team(BaseModel):
    name: str
    seasons: List[str] = Field(default_factory=list)

class Stat(BaseModel):
    season: Optional[str] = None
    name: Optional[str] = None
    value: Optional[str] = None

class Award(BaseModel):
    name: str
    year: Optional[str] = None

class Experience(BaseModel):
    type: Optional[str] = None  # job|internship|volunteer
    org: Optional[str] = None
    title: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None

class Achievement(BaseModel):
    type: Optional[str] = None  # award|certification
    name: Optional[str] = None
    by: Optional[str] = None
    date: Optional[str] = None

class Account(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    dob: Optional[str] = None        # ISO
    email: Optional[str] = None
    tos_accepted: bool = False
    requires_parent: Optional[bool] = None
    guardian_email: Optional[str] = None

class Identity(BaseModel):
    education_level: Optional[str] = None  # middle_school|high_school|college|coach|pro
    role: Optional[str] = None             # student|student_athlete
    gender: Optional[str] = None           # male|female|non_binary|prefer_not_to_say

class Story(BaseModel):
    bio: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)

class Academics(BaseModel):
    school: Optional[str] = None
    graduation_year: Optional[int] = None
    abroad: Optional[bool] = None
    gpa: Optional[float] = None

class Athletics(BaseModel):
    primary_sport: Optional[str] = None
    positions: List[str] = Field(default_factory=list)
    teams: List[Team] = Field(default_factory=list)
    stats: List[Stat] = Field(default_factory=list)
    awards: List[Award] = Field(default_factory=list)

class Career(BaseModel):
    experiences: List[Experience] = Field(default_factory=list)
    achievements: List[Achievement] = Field(default_factory=list)

class OnboardingState(BaseModel):
    account: Account = Field(default_factory=Account)
    identity: Identity = Field(default_factory=Identity)
    story: Story = Field(default_factory=Story)
    academics: Academics = Field(default_factory=Academics)
    athletics: Athletics = Field(default_factory=Athletics)
    career: Career = Field(default_factory=Career)
    publish_blocked: bool = False

class CreateSessionResponse(BaseModel):
    session_id: str
    state: OnboardingState

class ChatRequest(BaseModel):
    # Primary shape uses 'message'. Accept legacy 'text' for backward compatibility.
    message: Optional[str] = None
    text: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    patch: Dict[str, Any]
    state: OnboardingState
    # nextPrompt removed from the public API payload

# ---------------------------------------------------------------------
# In-memory sessions (swap with DB/Redis in prod)
# ---------------------------------------------------------------------

from uuid import uuid4
SESSIONS: Dict[str, OnboardingState] = {}

# ---------------------------------------------------------------------
# State machine steps
# ---------------------------------------------------------------------

class Step(BaseModel):
    key: str
    prompt: str
    guard: Optional[str] = None  # python expression evaluated with 's' (state)

FLOW: List[Step] = [
    Step(key="account.first_name", prompt="What’s your first name?"),
    Step(key="account.last_name", prompt="And your last name?"),
    Step(key="account.dob", prompt="What’s your date of birth? (MM/DD/YYYY)"),
    Step(key="account.email", prompt="What email should we use for your account?"),
    Step(key="account.tos_accepted", prompt="Please confirm you agree to the Terms and Privacy (yes/no)."),
    Step(key="account.guardian_email",
         prompt="I’ll need a parent or guardian’s email to request approval.",
         guard="s.account.requires_parent is True and not s.account.guardian_email"),
    Step(key="identity.education_level", prompt="Which best describes you: Middle School, High School, College, or Coach?"),
    Step(key="identity.role", prompt="Are you joining as a Student or Student-Athlete?"),
    Step(key="identity.gender", prompt="How do you identify? (Male/Female/Non-Binary/Prefer not to say)"),
    Step(key="story.bio", prompt="Share your story in 2–3 sentences."),
    Step(key="story.strengths", prompt="Pick up to 5 strengths (Content Creation, Choir, Dancing, Drawing, Fashion, Musical Instruments, Activism, Animal Welfare, Education Access, Literacy Coaching)."),
    Step(key="academics.school", prompt="Where are you currently studying?"),
    Step(key="academics.graduation_year", prompt="What’s your graduation year?"),
    Step(key="academics.abroad", prompt="Are you studying outside your home country? (yes/no)"),
    Step(key="academics.gpa", prompt="What’s your GPA? (0.0–4.0)"),
    Step(key="athletics.primary_sport", prompt="What sport do you play?"),
    Step(key="athletics.positions", prompt="What position(s) do you play?"),
    Step(key="athletics.teams", prompt="What teams have you played for (team + seasons)?"),
    Step(key="athletics.stats", prompt="Any performance stats (season, stat, value)?"),
    Step(key="athletics.awards", prompt="Any athletic awards? (name + year)"),
    Step(key="career.experiences", prompt="Any jobs/internships/volunteer experience (org, title, dates)?"),
    Step(key="career.achievements", prompt="Any career awards or certifications (name, by, date)?"),
]

def get_value_by_keypath(state: OnboardingState, keypath: str) -> Any:
    obj = state
    for k in keypath.split("."):
        obj = getattr(obj, k)
    return obj

def compute_next_prompt(state: OnboardingState) -> Optional[str]:
    s = state
    for step in FLOW:
        if step.guard and not eval(step.guard):
            continue
        val = get_value_by_keypath(s, step.key)
        if val in (None, False, [], {}):
            return step.prompt
    return None

def generate_enhanced_response(user_msg: str, state: OnboardingState, patch: Dict[str, Any]) -> str:
    """Generate enhanced response using standardized templates and multilingual support"""
    if not TEMPLATE_MANAGER:
        return "Response templates not available"
    
    lower = user_msg.lower()
    
    # Check for out-of-scope messages
    onboarding_tokens = {
        "name", "first name", "last name", "dob", "date of birth", "email", "tos", "terms",
        "guardian", "parent", "school", "graduation", "gpa", "sport", "position", "positions",
        "team", "teams", "stats", "awards", "bio", "strength", "strengths", "education",
        "role", "gender", "internship", "job", "experience",
        # Multilingual tokens
        "nombre", "apellido", "correo", "fecha", "nacimiento", "escuela", "graduación",
        "nom", "prénom", "email", "date", "naissance", "école", "diplôme",
        "nome", "sobrenome", "data", "nascimento", "escola", "formatura"
    }
    
    is_onboarding_related = any(token in lower for token in onboarding_tokens)
    if not is_onboarding_related and len(lower.split()) > 4:
        if MULTILINGUAL_SERVICE:
            return MULTILINGUAL_SERVICE.template_manager.generate_out_of_scope(
                MULTILINGUAL_SERVICE.template_manager.detect_language(user_msg)
            )
        return TEMPLATE_MANAGER.get_out_of_scope()
    
    # Handle field confirmations with multilingual support
    if patch:
        for section, fields in patch.items():
            for field, value in fields.items():
                # Use multilingual service if available
                if MULTILINGUAL_SERVICE:
                    # Get next step based on field
                    next_steps = {
                        "first_name": "last_name",
                        "last_name": "dob", 
                        "dob": "email",
                        "email": "tos_accepted",
                        "tos_accepted": "education_level",
                        "guardian_email": "education_level",
                        "education_level": "role",
                        "role": "gender",
                        "gender": "bio",
                        "bio": "strengths",
                        "strengths": "school",
                        "school": "graduation_year",
                        "graduation_year": "abroad",
                        "abroad": "gpa",
                        "gpa": "primary_sport",
                        "primary_sport": "positions",
                        "positions": "teams",
                        "teams": "stats",
                        "stats": "awards",
                        "awards": "experiences",
                        "experiences": "achievements",
                        "achievements": None
                    }
                    
                    next_field = next_steps.get(field)
                    if next_field:
                        next_step = MULTILINGUAL_SERVICE.get_multilingual_prompt(next_field, user_msg)
                        return MULTILINGUAL_SERVICE.process_multilingual_response(user_msg, field, value, next_step)
                    else:
                        return MULTILINGUAL_SERVICE.process_multilingual_response(user_msg, field, value)
                
                # Fallback to standard templates
                if field == "first_name":
                    return TEMPLATE_MANAGER.get_field_confirmation("First name", value, "What's your last name?")
                elif field == "last_name":
                    return TEMPLATE_MANAGER.get_field_confirmation("Last name", value, "What's your date of birth? (MM/DD/YYYY)")
                elif field == "dob":
                    if fields.get("requires_parent"):
                        return TEMPLATE_MANAGER.get_parental_consent()
                    return TEMPLATE_MANAGER.get_field_confirmation("DOB", value, "What's your email address?")
                elif field == "email":
                    return TEMPLATE_MANAGER.get_field_confirmation("Email", value, "Do you accept the Terms of Service? (yes/no)")
                elif field == "tos_accepted":
                    return TEMPLATE_MANAGER.get_field_confirmation("Terms", "accepted", "What's your education level? (Middle School, High School, College, Coach, Pro)")
                elif field == "guardian_email":
                    return TEMPLATE_MANAGER.get_field_confirmation("Guardian email", value, "What's your education level?")
                elif field == "education_level":
                    return TEMPLATE_MANAGER.get_field_confirmation("Education level", value, "Are you joining as a Student or Student-Athlete?")
                elif field == "role":
                    return TEMPLATE_MANAGER.get_field_confirmation("Role", value, "What's your gender? (Male, Female, Non-binary, Prefer not to say)")
                elif field == "gender":
                    return TEMPLATE_MANAGER.get_field_confirmation("Gender", value, "Tell me your story in 2-3 sentences")
                elif field == "bio":
                    return TEMPLATE_MANAGER.get_field_confirmation("Story", "saved", "What are your strengths? (Pick up to 5)")
                elif field == "strengths":
                    return TEMPLATE_MANAGER.get_field_confirmation("Strengths", "saved", "What's your school name?")
                elif field == "school":
                    return TEMPLATE_MANAGER.get_field_confirmation("School", value, "What's your graduation year?")
                elif field == "graduation_year":
                    return TEMPLATE_MANAGER.get_field_confirmation("Graduation year", value, "Are you studying abroad? (yes/no)")
                elif field == "abroad":
                    return TEMPLATE_MANAGER.get_field_confirmation("Studying abroad", value, "What's your GPA? (optional)")
                elif field == "gpa":
                    return TEMPLATE_MANAGER.get_field_confirmation("GPA", value, "What's your primary sport?")
                elif field == "primary_sport":
                    return TEMPLATE_MANAGER.get_field_confirmation("Sport", value, "What are your positions?")
                elif field == "positions":
                    return TEMPLATE_MANAGER.get_field_confirmation("Positions", "saved", "Tell me about your teams (team name + seasons)")
                elif field == "teams":
                    return TEMPLATE_MANAGER.get_field_confirmation("Teams", "saved", "Add your stats (season, stat, value)")
                elif field == "stats":
                    return TEMPLATE_MANAGER.get_field_confirmation("Stats", "saved", "Any awards? (name + year)")
                elif field == "awards":
                    return TEMPLATE_MANAGER.get_field_confirmation("Awards", "saved", "Any career experiences? (job/intern/volunteer)")
                elif field == "experiences":
                    return TEMPLATE_MANAGER.get_field_confirmation("Experiences", "saved", "Any career achievements? (certifications, etc.)")
                elif field == "achievements":
                    return TEMPLATE_MANAGER.get_completion()
    
    # Check for validation errors
    if "email" in lower and "@" in user_msg:
        # Check for common email typos
        email_match = EMAIL_RE.search(user_msg)
        if email_match:
            email = email_match.group(0)
            if "gmail" in email.lower() and "gmial" in email.lower():
                corrected = email.replace("gmial", "gmail")
                return TEMPLATE_MANAGER.get_email_autocorrect(corrected)
            elif "yahoo" in email.lower() and "yaho" in email.lower():
                corrected = email.replace("yaho", "yahoo")
                return TEMPLATE_MANAGER.get_email_autocorrect(corrected)
    
    # Default response
    next_prompt = compute_next_prompt(state)
    if next_prompt:
        return f"I'm here to help with your onboarding. {next_prompt}"
    else:
        return TEMPLATE_MANAGER.get_completion()

# ---------------------------------------------------------------------
# Lightweight rule-based extractor
# ---------------------------------------------------------------------

def extract_patch(text: str) -> Dict[str, Any]:
    """Heuristic extraction for common fields. Non-destructive: returns only what it finds."""
    txt = text.strip()
    lower = txt.lower()
    patch: Dict[str, Any] = {}

    # Names: "I am Javier Lopez" / "I'm Javier" / "First name Javier, Last name Lopez"
    m = re.search(r"(?i)\b(first\s*name)\s*[:=]?\s*([A-Za-z'-]+)", txt)
    if m:
        patch.setdefault("account", {})["first_name"] = m.group(2).strip().title()
    m = re.search(r"(?i)\b(last\s*name)\s*[:=]?\s*([A-Za-z'-]+)", txt)
    if m:
        patch.setdefault("account", {})["last_name"] = m.group(2).strip().title()

    m = re.search(r"(?i)\bI(?:'m| am)\s+([A-Z][a-z'-]+)\s+([A-Z][a-z'-]+)", txt)
    if m:
        patch.setdefault("account", {})["first_name"] = patch.get("account", {}).get("first_name") or m.group(1)
        patch.setdefault("account", {})["last_name"] = patch.get("account", {}).get("last_name") or m.group(2)

    # Email
    m = EMAIL_RE.search(txt)
    if m:
        patch.setdefault("account", {})["email"] = m.group(0)

    # DOB
    dob = try_parse_date(txt)
    if dob:
        patch.setdefault("account", {})["dob"] = dob.isoformat()
        # age-gate calculation
        patch["account"]["requires_parent"] = calc_age(dob) < 18

    # TOS yes/no
    if "terms" in lower or "privacy" in lower or "agree" in lower:
        yn = yn_to_bool(lower.split()[-1])  # crude, also catches "yes" messages
        if yn is None and "agree" in lower:
            yn = True
        if yn is not None:
            patch.setdefault("account", {})["tos_accepted"] = bool(yn)

    # Guardian email
    if "parent" in lower or "guardian" in lower:
        m2 = EMAIL_RE.search(txt)
        if m2:
            patch.setdefault("account", {})["guardian_email"] = m2.group(0)

    # Education level
    for label, canon in [
        ("middle school","middle_school"),
        ("junior high","middle_school"),
        ("high school","high_school"),
        ("college","college"),
        ("coach","coach"),
        ("professional","pro"),
    ]:
        if label in lower:
            patch.setdefault("identity", {})["education_level"] = canon

    # Role
    if "student-athlete" in lower or "student athlete" in lower:
        patch.setdefault("identity", {})["role"] = "student_athlete"
    elif "student" in lower:
        patch.setdefault("identity", {})["role"] = "student"

    # Gender
    for g in ["male","female","non-binary","nonbinary","prefer not to say"]:
        if g in lower:
            val = "non_binary" if "non" in g else g.replace(" ", "_")
            if g == "prefer not to say": val = "prefer_not_to_say"
            patch.setdefault("identity", {})["gender"] = val

    # Story bio (catch "my story:" or "story:")
    m = re.search(r"(?i)\b(my\s+story|story)\s*[:\-]\s*(.+)$", txt)
    if m:
        patch.setdefault("story", {})["bio"] = m.group(2).strip()

    # Strengths
    chosen = [s for s in STRENGTHS if s in lower]
    if chosen:
        patch.setdefault("story", {})["strengths"] = chosen

    # School
    m = re.search(r"(?i)\bschool\s*[:\-]\s*([^.,\n]+)", txt)
    if m:
        patch.setdefault("academics", {})["school"] = m.group(1).strip()
    # Grad year
    m = YEAR_RE.search(txt)
    if m:
        patch.setdefault("academics", {})["graduation_year"] = int(m.group(1))
    # Abroad yes/no
    if "abroad" in lower or "outside your home country" in lower:
        yn = yn_to_bool(lower.split()[-1])
        if yn is not None:
            patch.setdefault("academics", {})["abroad"] = yn
    # GPA
    m = GPA_RE.search(txt)
    if m:
        try:
            gpa = float(m.group(1))
            if 0 <= gpa <= 4.0:
                patch.setdefault("academics", {})["gpa"] = gpa
        except Exception:
            pass

    # Sport
    for s in SPORTS:
        if s in lower:
            patch.setdefault("athletics", {})["primary_sport"] = s
            break
    # Positions
    pos_found = [p for p in BASKETBALL_POSITIONS if p in lower]
    if pos_found:
        patch.setdefault("athletics", {})["positions"] = pos_found

    # Teams (very light: "team: NAME")
    m = re.search(r"(?i)\bteam\s*[:\-]\s*([^.,\n]+)", txt)
    if m:
        patch.setdefault("athletics", {}).setdefault("teams", []).append({"name": m.group(1).strip(), "seasons": []})
    # Stats (e.g., Points Per Game 25)
    m = re.search(r"(?i)\b(points per game|assists per game|blocks per game)\s*[:=]?\s*(\d+(?:\.\d+)?)", txt)
    if m:
        patch.setdefault("athletics", {}).setdefault("stats", []).append(
            {"season": None, "name": m.group(1).title(), "value": m.group(2)}
        )
    # Awards
    if "award" in lower or "mvp" in lower or "captain" in lower:
        year = None
        ym = YEAR_RE.search(txt)
        if ym:
            year = ym.group(1)
        # pick first phrase up to comma
        name = re.split(r"[.,;]\s*", txt, 1)[0]
        patch.setdefault("athletics", {}).setdefault("awards", []).append({"name": name.strip(), "year": year})

    # Career experience
    if any(k in lower for k in ["job","internship","volunteer"]):
        kind = "job" if "job" in lower else "internship" if "internship" in lower else "volunteer"
        org = None
        m = re.search(r"(?i)\b(company|organization|org)\s*[:\-]\s*([^.,\n]+)", txt)
        if m: org = m.group(2).strip()
        title = None
        m = re.search(r"(?i)\b(position|title|role)\s*[:\-]\s*([^.,\n]+)", txt)
        if m: title = m.group(2).strip()
        start = None; end = None
        if "start" in lower: 
            m = re.search(r"(?i)\bstart\s*[:\-]\s*([^.,\n]+)", txt); start = m.group(1).strip() if m else None
        if "end" in lower:
            m = re.search(r"(?i)\bend\s*[:\-]\s*([^.,\n]+)", txt); end = m.group(1).strip() if m else "Present"
        if org or title or start or end:
            patch.setdefault("career", {}).setdefault("experiences", []).append(
                {"type": kind, "org": org, "title": title, "start": start, "end": end or "Present"}
            )

    # Career achievements
    if "certification" in lower or "employee of the month" in lower or "award" in lower:
        ach_type = "certification" if "certification" in lower else "award"
        by = None
        m = re.search(r"(?i)\bby\s+([A-Za-z0-9'&.\-\s]+)", txt)
        if m: by = m.group(1).strip()
        name = None
        m = re.search(r"(?i)\b(certification|award|employee of the month|cpr certified|best intern)\b", lower)
        if m: name = m.group(0).title()
        date_txt = None
        dm = DATE_ANY_RE.search(txt)
        if dm: date_txt = dm.group(0)
        if name or by or date_txt:
            patch.setdefault("career", {}).setdefault("achievements", []).append(
                {"type": ach_type, "name": name, "by": by, "date": date_txt}
            )

    return patch

def validate_and_merge(state: OnboardingState, patch: Dict[str, Any]) -> OnboardingState:
    """Apply a shallow merge with basic validation/normalization."""
    s = state.dict()
    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict):
                dst[k] = merge(dst.get(k, {}) if isinstance(dst.get(k), dict) else {}, v)
            elif isinstance(v, list):
                # merge lists by value (avoid duplicates)
                cur = dst.get(k, [])
                for item in v:
                    if item not in cur:
                        cur.append(item)
                dst[k] = cur
            else:
                dst[k] = v
        return dst

    merged = merge(s, patch)

    # Email basic check
    email = merged["account"].get("email")
    if email and not EMAIL_RE.match(email):
        merged["account"]["email"] = None

    # DOB -> ISO + requires_parent flag
    dob_s = merged["account"].get("dob")
    if dob_s:
        try:
            dob = try_parse_date(dob_s) or datetime.fromisoformat(dob_s).date()
            merged["account"]["dob"] = dob.isoformat()
            merged["account"]["requires_parent"] = calc_age(dob) < 18
        except Exception:
            merged["account"]["dob"] = None

    # GPA clamp
    gpa = merged["academics"].get("gpa")
    if gpa is not None:
        try:
            gpa = float(gpa)
            merged["academics"]["gpa"] = max(0.0, min(4.0, gpa))
        except Exception:
            merged["academics"]["gpa"] = None

    # Normalize sport/positions to canon labels
    sport = merged["athletics"].get("primary_sport")
    if sport:
        sport_l = sport.lower()
        merged["athletics"]["primary_sport"] = next((s for s in SPORTS if s == sport_l), sport_l)

    merged["athletics"]["positions"] = sorted({p for p in merged["athletics"].get("positions", [])})

    # Publish block based on parental approval
    merged["publish_blocked"] = bool(merged["account"].get("requires_parent") and not merged["account"].get("guardian_email"))

    return OnboardingState.parse_obj(merged)

# ---------------------------------------------------------------------
# Model worker call (chat reply)
# ---------------------------------------------------------------------

def worker_url() -> str:
    # Use compose-provided env var if set; falls back to local default
    return os.environ.get("SUMMIT_MODEL_SERVICE_URL", "http://127.0.0.1:8001/generate")


async def call_worker(user_text: str) -> str:
    """
    Send the user's message to the RAG model server.
    RAG server expects a JSON body with 'query' (required). We also send 'prompt' for compatibility.
    """
    if httpx is None:
        raise HTTPException(status_code=500, detail="httpx not available in this environment")

    # Optional decoding knobs via env (safe defaults)
    body = {
        "query": user_text,
        "prompt": user_text,  # tolerated by our RAG server, ignored elsewhere
        "task": "onboarding_guide",
        "top_k": int(os.getenv("RAG_TOP_K", "3")),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "128")),
        "temperature": float(os.getenv("TEMPERATURE", "0.0")),
        "top_p": float(os.getenv("TOP_P", "0.9")),
        "repetition_penalty": float(os.getenv("REPETITION_PENALTY", "1.1")),
        "no_repeat_ngram_size": int(os.getenv("NO_REPEAT_NGRAM_SIZE", "3")),
    }

    r = None
    try:
        headers = {}
        # Forward RAG API key if present in environment
        # Read RAG API key from environment or Docker secret file
        rag_key = os.environ.get("RAG_API_KEY") or os.environ.get("SUMMIT_RAG_API_KEY")
        try:
            if os.path.exists("/run/secrets/rag_api_key"):
                with open("/run/secrets/rag_api_key", "r") as fh:
                    sk = fh.read().strip()
                    if sk:
                        rag_key = sk
        except Exception:
            pass
        if rag_key:
            headers["Authorization"] = f"Bearer {rag_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(worker_url(), json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
            log.info("worker response json: %s", data)

            # Accept multiple possible response shapes
            if isinstance(data, dict):
                # our RAG server usually returns "answer"
                if isinstance(data.get("answer"), str) and data["answer"].strip():
                    return data["answer"]
                # fallbacks some servers use
                if isinstance(data.get("text"), str) and data["text"].strip():
                    return data["text"]
                if isinstance(data.get("output"), str) and data["output"].strip():
                    return data["output"]
                if isinstance(data.get("response"), str) and data["response"].strip():
                    return data["response"]
                if isinstance(data.get("generated_texts"), list) and data["generated_texts"]:
                    return str(data["generated_texts"][0])

    except Exception as e:
        # Log raw response if available
        try:
            raw = r.text if r is not None else None
            if raw:
                log.error("Model worker raw response before failure: %s", raw)
        except Exception:
            pass
        log.exception("Model worker call failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to call model worker")

    raise HTTPException(status_code=502, detail="worker did not return generated text")

# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------

@app.get("/health")
def health():
    health_status = {"status": "ok"}
    
    # Add quality monitoring health if available
    if QUALITY_MONITOR:
        try:
            quality_health = QUALITY_MONITOR.get_health_status()
            health_status.update({
                "quality_status": quality_health["status"],
                "average_quality_score": quality_health["average_quality_score"],
                "recent_alerts": quality_health["recent_alerts"]
            })
        except Exception as e:
            log.exception("Error getting quality health status: %s", e)
            health_status["quality_status"] = "error"
    
    return health_status

@app.get("/quality/report")
def get_quality_report():
    """Get comprehensive quality report"""
    if not QUALITY_MONITOR:
        raise HTTPException(status_code=503, detail="Quality monitoring not available")
    
    try:
        report = QUALITY_MONITOR.get_quality_report()
        return report
    except Exception as e:
        log.exception("Error generating quality report: %s", e)
        raise HTTPException(status_code=500, detail="Error generating quality report")

@app.get("/quality/health")
def get_quality_health():
    """Get quality monitoring health status"""
    if not QUALITY_MONITOR:
        raise HTTPException(status_code=503, detail="Quality monitoring not available")
    
    try:
        health = QUALITY_MONITOR.get_health_status()
        return health
    except Exception as e:
        log.exception("Error getting quality health: %s", e)
        raise HTTPException(status_code=500, detail="Error getting quality health")

@app.post("/sessions", response_model=CreateSessionResponse)
def create_session():
    sid = uuid4().hex
    SESSIONS[sid] = OnboardingState()
    return {"session_id": sid, "state": SESSIONS[sid]}

@app.get("/sessions/{sid}", response_model=OnboardingState)
def get_session(sid: str):
    state = SESSIONS.get(sid)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")
    return state

@app.post("/sessions/{sid}/message", response_model=ChatResponse)
async def post_message(sid: str, body: Dict[str, Any]):
    state = SESSIONS.get(sid)
    if not state:
        raise HTTPException(status_code=404, detail="unknown session")

    # Accept new shape (message) or legacy shape (text)
    if isinstance(body, dict):
        user_msg = str(body.get("message") or body.get("text") or "").strip()
    else:
        # fallback for pydantic model compatibility
        user_msg = (getattr(body, "message", None) or getattr(body, "text", None) or "").strip()
    if not user_msg:
        raise HTTPException(status_code=422, detail="Empty message")

    lower = user_msg.lower()

    # Heuristic: tokens/topics that indicate onboarding relevance
    onboarding_tokens = {
        "name", "first name", "last name", "dob", "date of birth", "email", "tos", "terms",
        "guardian", "parent", "school", "graduation", "gpa", "sport", "position", "positions",
        "team", "teams", "stats", "awards", "bio", "strength", "strengths", "education",
        "role", "gender", "internship", "job", "experience"
    }
    # brief check for sports-specific tokens
    onboarding_tokens.update({s for s in SPORTS})
    onboarding_tokens.update({p for p in BASKETBALL_POSITIONS})

    def is_onboarding_related(text_lower: str) -> bool:
        # any token present => considered relevant
        for t in onboarding_tokens:
            if t in text_lower:
                return True
        # also treat short direct answers (single word / short phrase) as relevant
        if len(text_lower.split()) <= 4 and not text_lower.endswith("?"):
            return True
        return False

    # Run lightweight extraction first (deterministic)
    patch = extract_patch(user_msg)
    # Merge + validate immediately so state is updated deterministically
    new_state = validate_and_merge(state, patch)

    # Compute next prompt (based on new_state)
    next_prompt = compute_next_prompt(new_state)

    # Generate enhanced response using templates
    enhanced_reply = generate_enhanced_response(user_msg, new_state, patch)
    
    # If message appears out-of-scope (no extracted fields and not onboarding-related),
    # do NOT call the model worker — return a brief onboarding-only disclaimer.
    if not patch and not is_onboarding_related(lower):
        reply = enhanced_reply if TEMPLATE_MANAGER else "Sorry, I can only help with onboarding. I can't assist with that request."
        # persist state (unchanged) and return
        SESSIONS[sid] = new_state
        return ChatResponse(reply=reply, patch=patch, state=new_state)

    # For relevant messages, call the model worker in parallel
    worker_task = asyncio.create_task(call_worker(user_msg))

    # Await assistant reply
    try:
        model_reply = await worker_task
        # Use enhanced template response if available, otherwise fall back to model response
        reply = enhanced_reply if TEMPLATE_MANAGER else model_reply
    except HTTPException:
        # fall back to deterministic UX when worker is down
        reply = enhanced_reply if TEMPLATE_MANAGER else (next_prompt or "Thanks! Your info has been saved.")

    # Log interaction for quality monitoring
    if QUALITY_MONITOR:
        try:
            QUALITY_MONITOR.log_interaction(
                user_input=user_msg,
                response=reply,
                session_id=sid,
                response_time=0.0,  # Could be measured if needed
                token_usage={"input": len(user_msg.split()), "output": len(reply.split()), "context": 0}
            )
        except Exception as e:
            log.exception("Error logging interaction for quality monitoring: %s", e)

    # Persist
    SESSIONS[sid] = new_state

    return ChatResponse(
        reply=reply,
        patch=patch,
        state=new_state,
    )


# ---------------------------------------------------------------------
# Local runner
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log.info("Starting Summit AI Onboarding API on %s:%s (worker %s)", host, port, worker_url())
    uvicorn.run("service.app:app", host=host, port=port, log_level="info")
