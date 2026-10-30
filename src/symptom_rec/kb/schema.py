"""The shape of each block in the knowledge base files.

Each class lists the keys one YAML block may have and the type of each value.
When the files are loaded, pydantic checks every block against its class: a missing key,
a value of the wrong type, or a key that is not listed is reported as an error.
Checks that need more than one file (for example "this symptom id exists") are in validate.py.
"""
from pydantic import BaseModel, ConfigDict

# The sections of a chapter file, in the order they must appear.
CHAPTER_SECTIONS = [
    "chapter", "title", "chief_complaint", "also_entry", "status",
    "definitions", "hpi", "steps", "gates", "must_ask", "tree_selection", "branch_selection",
    "branch_hints", "trees", "scores", "confirm", "findings", "branch_findings", "reviewer_findings",
    "history_clues", "exam_clues",
]


class NoExtraKeys(BaseModel):
    """Base for every class below: a key that is not listed in the class is an error."""

    model_config = ConfigDict(extra="forbid")


# ── kb/symptoms.yaml ───────────────────────────────────────────────────────
class Symptom(NoExtraKeys):
    id: str
    th: str
    en: str
    kind: str                       # symptom | history | measurement
    parent: str | None = None       # the more general symptom, if any
    refines: dict | None = None     # {question: answer} that makes it more specific than the parent
    aliases: list[str] = []
    sex: str | None = None          # female | male
    red_flag: bool = False
    urgent: bool = False            # present -> see a doctor now, whatever the chief complaint


# ── kb/diseases.yaml ───────────────────────────────────────────────────────
class Disease(NoExtraKeys):
    id: str
    th: str
    en: str
    kind: str                       # disease | group
    examples: list[str] = []
    sex: str | None = None          # female | male
    must_not_miss: bool = False


# ── kb/chapters/*.yaml ─────────────────────────────────────────────────────
# A condition (`when`) is either the word "always" or a mapping such as
# {present: [fever], cough.duration: acute}. validate.py checks what is inside it.

class Title(NoExtraKeys):
    th: str
    en: str


class HpiQuestion(NoExtraKeys):
    ask: str
    options: list[str] | None = None    # answers to pick from, or
    unit: str | None = None             # a number in this unit,
    categories: dict | None = None      # sorted into these named ranges
    grey_zones: list[dict] = []         # ranges where two neighbouring categories both apply
    multiple: bool = False              # more than one answer allowed
    allow_unknown: bool = False         # "not sure" is a valid answer
    sex: str | None = None              # asked only of this sex


class Step(NoExtraKeys):
    id: str


class Gate(NoExtraKeys):
    when: str | dict
    ask: list[str]
    if_any_yes: str | dict              # "urgent", or {prioritize: <branch>}


class MustAsk(NoExtraKeys):
    when: str | dict
    ask: list[str]
    rules_out: list[str] = []
    red_flag_when: dict | None = None


class TreeRule(NoExtraKeys):
    when: str | dict
    use: list[str]


class BranchHint(NoExtraKeys):
    when: str | dict
    boost: str


class Score(NoExtraKeys):
    applies_to: str
    when: str | dict = "always"
    items: list[dict]
    threshold: dict


class Confirm(NoExtraKeys):
    supports: list[str] = []
    against: list[str] = []
    hpi: dict = {}
    if_against: dict = {}


class Finding(NoExtraKeys):
    symptoms: list[str] = []
    signs: list[str] = []


class HistoryClue(NoExtraKeys):
    findings: list[str]
    suggests: list[str]


class ExamClue(NoExtraKeys):
    sign: str
    suggests: list[str]
    askable_as: str | None = None


class Chapter(NoExtraKeys):
    file: str                                   # file name without .yaml, e.g. "03_fever"
    chapter: int | str
    title: Title
    chief_complaint: str
    also_entry: list[str]
    status: str                                 # draft | reviewed
    definitions: dict
    hpi: dict[str, dict[str, HpiQuestion]]      # symptom -> question name -> question
    steps: list[Step]
    gates: dict[str, Gate]
    must_ask: list[MustAsk]
    tree_selection: list[TreeRule]
    branch_selection: dict                      # {} or {attribute: ..., map: ...}
    branch_hints: list[BranchHint]
    trees: dict
    scores: dict[str, Score]
    confirm: dict[str, Confirm]
    findings: dict[str, Finding]
    branch_findings: dict[str, Finding]
    reviewer_findings: dict[str, Finding]
    history_clues: list[HistoryClue]
    exam_clues: list[ExamClue]


# ── kb/lexicon.yaml ────────────────────────────────────────────────────────
class BodyPart(NoExtraKeys):
    words: list[str]
    # The symptom id each sensation at this body part means. A list means the first symptom,
    # with the others offered as alternatives worth asking about.
    pain: str | list[str] | None = None
    sore: str | list[str] | None = None        # if not given, read like pain
    burning: str | list[str] | None = None
    itch: str | list[str] | None = None
    swelling: str | list[str] | None = None
    numbness: str | list[str] | None = None


SENSATIONS = ["pain", "sore", "burning", "itch", "swelling", "numbness"]


class Lexicon(NoExtraKeys):
    fillers_start: list[str]
    fillers_end: list[str]
    negations: list[str]
    sensations: dict[str, list[str]]    # sensation (one of SENSATIONS) -> its words
    body_parts: dict[str, BodyPart]
    time_units: dict[str, float]        # word -> days
    less_than: list[str]
    more_than: list[str]
    ambiguous: dict[str, list[str]] = {}  # phrase -> the symptoms it can mean; the patient chooses


# ── tests/cases/vignettes.yaml ─────────────────────────────────────────────
class History(NoExtraKeys):
    hpi: dict = {}
    present: list[str] = []
    absent: list[str] = []
    unsure: list[str] = []
    historical: list[str] = []
    unmapped: list[str] = []


class Expect(NoExtraKeys):
    top3_includes: str | None = None
    top3_includes_any: list[str] = []
    ask_first: list[str] = []
    not_ask_first: list[str] = []
    urgent: bool | None = None
    never_recommend: list[str] = []
    never_in_ddx: list[str] = []
    duration_categories: list[str] = []
    switch_chief_complaint: str | None = None


class Vignette(NoExtraKeys):
    id: str
    level: str                      # simple | book
    sex: str
    age: float
    chief_complaint: str
    history: History
    expect: Expect | None = None
    exam: list[str] = []
    book_ddx: list[str] = []
    final_diagnosis: str | None = None
