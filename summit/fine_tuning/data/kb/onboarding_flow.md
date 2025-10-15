# Onboarding Flow (Step Order & Logic)

**Goal:** collect required fields with minimal friction. Confirm each field, then prompt for the next missing one.

## Order (with gate conditions)
1. **First name** → “What’s your first name?”
2. **Last name** → “And your last name?”
3. **Date of birth** (MM/DD/YYYY) → infer `requires_parent` if age < 18
4. **Email** (format check)
5. **Terms consent** (yes/no)
6. **Guardian email** → *only if* `requires_parent == true`
7. **Education level** (Middle School, High School, College, Coach, Pro)
8. **Role** (Student, Student-Athlete)
9. **Gender** (Male, Female, Non‑Binary, Prefer not to say)
10. **Story (bio)** (2–3 sentences)
11. **Strengths** (pick up to 5 from the allowed list)
12. **School**
13. **Graduation year** (YYYY; reasonable range)
14. **Studying abroad** (yes/no)
15. **GPA** (0.0–4.0; optional)
16. **Primary sport** (canonical list)
17. **Positions** (e.g., point guard, center)
18. **Teams** (team name + seasons)
19. **Stats** (season, stat, value)
20. **Athletic awards** (name + year)
21. **Career experiences** (job/intern/volunteer; org, title, dates)
22. **Career achievements** (award/certification; name, by, date)

## Next-step logic (pseudo)
```
for step in ORDER:
    if step.guard and not guard_ok(state): continue
    if state[step.key] is missing/empty:
        ask(step.prompt); break
else:
    say("All set! Your profile is complete.")
```

## Examples
- After saving DOB for a 16-year-old:  
  “Saved: DOB **05/12/2009**. Because you’re under 18, I’ll need a parent or guardian’s email for approval. What’s their email?”

- After saving Education level:  
  “Education level set to **High School**. Are you joining as a **Student** or **Student‑Athlete**?”