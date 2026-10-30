"""Conservative language cues for uncertainty and past symptoms, before name matching.

These are linguistic rules, not clinical rules. Past mentions are retained for clarification;
they never assert a current symptom or a current denial. Only explicit cues are handled.
"""
import re

# Do not split ตั้งแต่ (since). A contrast starts a new assertion scope.
CONTRAST = re.compile(r"(?<!ตั้ง)แต่|\bbut\b|\bhowever\b|;", re.IGNORECASE)
UNCERTAIN = re.compile(
    r"^(?:ไม่แน่ใจ(?:ว่า)?|ไม่ทราบว่า|ไม่รู้ว่า|สงสัยว่า|อาจจะ|"
    r"(?:not\s+sure|unsure)(?:\s+(?:if|whether))?\b|maybe\b|possibly\b)\s*", re.IGNORECASE)
PAST = re.compile(r"^(?:เมื่อก่อน(?:เคย)?|แต่ก่อน|เคย|used\s+to\b|previously\b)\s*", re.IGNORECASE)
CURRENT = re.compile(r"^(?:ตอนนี้|ปัจจุบัน|ขณะนี้|currently\b|now\b)\s*", re.IGNORECASE)
RESOLVED = re.compile(r"(?:หายไปแล้ว|หายสนิทแล้ว|หายแล้ว|has\s+resolved|is\s+gone)$", re.IGNORECASE)
ONGOING = re.compile(r"(?:ยังไม่หาย(?:เลย)?|ไม่หายสักที|ไม่หาย|has\s+not\s+resolved)$", re.IGNORECASE)
RESTART = re.compile(
    r"(?:และ|\band\s+)(?=(?:ไม่แน่ใจ|ไม่ทราบว่า|ไม่รู้ว่า|สงสัยว่า|อาจจะ|เคย|เมื่อก่อน|ตอนนี้|"
    r"not\s+sure|unsure|maybe|previously|currently))", re.IGNORECASE)


def assertion_clauses(text):
    """Return (text to match, state override) pairs. None keeps ordinary negation parsing."""
    clauses = []
    # Protect the historical prefix แต่ก่อน from the contrast splitter.
    chunks = [text] if text.strip().startswith("แต่ก่อน") else CONTRAST.split(text)
    for chunk in chunks:
        for part in RESTART.split(chunk):
            part = part.strip()
            if not part:
                continue
            state = None
            uncertain = UNCERTAIN.match(part)
            if uncertain:
                state = "unsure"
                part = part[uncertain.end():].strip()
            past = PAST.match(part)
            if past:
                state = state or "historical"
                part = part[past.end():].strip()
            part = CURRENT.sub("", part).strip()
            ongoing = ONGOING.search(part)
            resolved = None if ongoing else RESOLVED.search(part)
            if ongoing:
                part = part[:ongoing.start()].strip()
                # An explicit continuing symptom is current even after "used to".
                state = "unsure" if state == "unsure" else None
            elif resolved:
                part = part[:resolved.start()].strip()
                state = "unsure" if state == "unsure" else "historical"
                if not part and clauses:
                    # "ไอ แต่ตอนนี้หายแล้ว": the immediately preceding clause is resolved.
                    previous, _ = clauses[-1]
                    clauses[-1] = (previous, state)
            if part:
                clauses.append((part, state))
    return clauses
