# Field Reference (Types, Validation, Examples)

| Section | Field | Type | Validation / Canonicalization | Examples |
|---|---|---|---|---|
| Account | first_name | string | Trimmed; Title Case | Aisha, Javier |
| Account | last_name | string | Trimmed; Title Case | Khan, Lopez |
| Account | dob | date | Accept YYYY-MM-DD, MM/DD/YYYY, “May 12 2006”; store ISO | 2006-05-12 |
| Account | email | string | RFC-like regex | user@example.com |
| Account | tos_accepted | bool | yes/no | yes |
| Account | requires_parent | bool | computed from dob (<18) | true/false |
| Account | guardian_email | string | required if requires_parent | parent@domain.com |
| Identity | education_level | enum | middle_school, high_school, college, coach, pro | high_school |
| Identity | role | enum | student, student_athlete | student_athlete |
| Identity | gender | enum | male, female, non_binary, prefer_not_to_say | prefer_not_to_say |
| Story | bio | string | 1–3 short sentences | “I love helping others…” |
| Story | strengths | list<enum> | pick ≤5 from allowed list | drawing, activism |
| Academics | school | string | free text | Central High |
| Academics | graduation_year | int | 2000–2040 (configurable) | 2027 |
| Academics | abroad | bool | yes/no | no |
| Academics | gpa | float | 0.0–4.0 | 3.6 |
| Athletics | primary_sport | enum | see list below | basketball |
| Athletics | positions | list<enum> | sport‑specific | point guard |
| Athletics | teams | list<object> | {name, seasons[]} | Tigers; 2023 |
| Athletics | stats | list<object> | {season?, name, value} | 2024, PPG, 18.4 |
| Athletics | awards | list<object> | {name, year?} | MVP, 2023 |
| Career | experiences | list<object> | {type, org, title, start, end} | job, Zaxby’s, Crew, 2024–Present |
| Career | achievements | list<object> | {type, name, by?, date?} | cert, CPR, Red Cross, 2023 |

## Allowed values

**education_level:** `middle_school | high_school | college | coach | pro`  
**role:** `student | student_athlete`  
**gender:** `male | female | non_binary | prefer_not_to_say`  
**strengths:** `content creation, choir, dancing, drawing, fashion, musical instruments, activism, animal welfare, education access, literacy coaching`  
**primary_sport (sample):** `badminton, baseball, basketball, bmx, combat sports, crew, cricket, cross country, esports, fishing, fencing, field hockey, flag football, football, golf`  
**basketball positions:** `point guard, shooting guard, small forward, power forward, center`

## Normalization rules
- **Names:** Title Case.  
- **DOB:** store as ISO `YYYY-MM-DD`.  
- **Email:** reject if regex fails; ask to re-enter.  
- **GPA:** clamp to `[0.0, 4.0]`.  
- **Guardian flow:** if `requires_parent == true` and `guardian_email` missing → request it before publishing.