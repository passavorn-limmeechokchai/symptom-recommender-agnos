"""Small text helpers for the normalizer: the comparison form of a text, and reading numbers."""
import re

INFINITY = float("inf")

# How many days one unit of an HPI question is (the `unit` of a question in a chapter file).
QUESTION_UNIT_DAYS = {"hours": 1 / 24, "days": 1, "weeks": 7, "months": 30, "years": 365}


def key(text):
    """The form every name and every input is compared in.

    Lower case, no spaces or punctuation, no ๆ, and no letter repeated 3 or more times:
    "ปวดดดดท้อง" -> "ปวดท้อง", "ไอแห้งๆ" -> "ไอแห้ง", "Low-grade fever" -> "lowgradefever".
    """
    text = text.lower().replace("ๆ", "")
    text = re.sub(r"[^\w฀-๿]", "", text)   # keep letters, digits and Thai
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return text


def has_thai(text):
    for char in text:
        if "฀" <= char <= "๿":
            return True
    return False


def read_duration(text, lexicon):
    """How long, as (shortest, longest) in days, or None if the text gives no duration.

        "2 สัปดาห์"               -> (14, 14)
        "2-4 สัปดาห์"             -> (14, 28)     a number without a unit takes the next unit
        "ตั้งแต่ 3 วัน ถึง 2 เดือน"   -> (3, 60)
        "ไม่เกิน 5 วัน"            -> (0, 5)
        "มากกว่า 2 เดือน"          -> (60, infinity)
    """
    lowered = text.lower()
    units = sorted(lexicon.time_units, key=len, reverse=True)   # "days" before "day"
    unit_pattern = "|".join(re.escape(unit) for unit in units)
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(" + unit_pattern + r")?")

    days = []
    waiting = []   # numbers seen before their unit
    for found in pattern.finditer(lowered):
        number = float(found.group(1))
        unit = found.group(2)
        if unit is None:
            waiting.append(number)
            continue
        for earlier in waiting:
            days.append(earlier * lexicon.time_units[unit])
        waiting = []
        days.append(number * lexicon.time_units[unit])
    if not days:
        return None

    if contains_any(lowered, lexicon.less_than):
        return 0, max(days)
    if contains_any(lowered, lexicon.more_than):
        return min(days), INFINITY
    return min(days), max(days)


def read_temperature(text):
    """Body temperature in °C as (lowest, highest), or None.

    "38.5" -> (38.5, 38.5), "37.5-38.2" -> (37.5, 38.2); numbers outside 34-43 are ignored.
    """
    values = []
    for number in re.findall(r"\d+(?:\.\d+)?", text):
        if 34 <= float(number) <= 43:
            values.append(float(number))
    if not values:
        return None
    return min(values), max(values)


def categories_for(question, low, high):
    """The question's categories that a value, or a range of values, falls in.

    A single value (low == high) on a boundary, or inside a grey zone, gets both neighbours.
    A range only counts for a category it overlaps by more than a boundary point.
    """
    found = []
    for name, rule in question.categories.items():
        c_low, c_high = category_range(rule)
        if low == high:
            inside = c_low <= low <= c_high
        else:
            inside = min(high, c_high) - max(low, c_low) > 0
        if inside:
            found.append(name)
    if low == high:
        for zone in question.grey_zones:
            if zone["from"] <= low <= zone["to"]:
                for name in zone["both"]:
                    if name not in found:
                        found.append(name)
    return found


def category_range(rule):
    """{"below": 3} -> (-inf, 3), {"from": 3, "to": 8} -> (3, 8), {"above": 8} -> (8, inf)."""
    if "below" in rule:
        return -INFINITY, rule["below"]
    if "above" in rule:
        return rule["above"], INFINITY
    return rule.get("from", -INFINITY), rule.get("to", INFINITY)


def contains_any(text, words):
    for word in words:
        if word in text:
            return True
    return False
