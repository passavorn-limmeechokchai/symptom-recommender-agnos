"""Read the knowledge base YAML files into Python objects."""
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..paths import KB_DIR, VIGNETTES_FILE
from .schema import CHAPTER_SECTIONS, Chapter, Disease, Lexicon, Symptom, Vignette


class KBError(Exception):
    """The files could not be loaded. `problems` lists every reason."""

    def __init__(self, problems):
        super().__init__("\n".join(problems))
        self.problems = problems


class KnowledgeBase:
    """Everything in kb/, plus the lookups other modules need."""

    def __init__(self, symptoms, diseases, chapters, alias_sources):
        self.symptoms = symptoms            # symptom id -> Symptom
        self.diseases = diseases            # disease id -> Disease
        self.chapters = chapters            # file name, e.g. "03_fever" -> Chapter
        self.alias_sources = alias_sources  # (symptom id, alias) -> "p.248", "reviewer" or ""

        # "<symptom>.<question>" -> HpiQuestion, e.g. "cough.duration"
        self.questions = {}
        for chapter in chapters.values():
            for symptom_id, questions in chapter.hpi.items():
                for name, question in questions.items():
                    self.questions[symptom_id + "." + name] = question

        # parent symptom -> its more specific children
        self.children = {}
        for symptom in symptoms.values():
            if symptom.parent:
                self.children.setdefault(symptom.parent, []).append(symptom.id)

        # symptom -> the chapters that open when a patient starts with it
        self.entry_chapters = {}
        for name, chapter in chapters.items():
            for symptom_id in [chapter.chief_complaint] + chapter.also_entry:
                self.entry_chapters.setdefault(symptom_id, []).append(name)

        # Thai name or alias -> symptom ids that use it
        self.terms = {}
        for symptom in symptoms.values():
            for term in [symptom.th] + symptom.aliases:
                ids = self.terms.setdefault(term, [])
                if symptom.id not in ids:
                    ids.append(symptom.id)


def load_kb(kb_dir=KB_DIR):
    kb_dir = Path(kb_dir)
    problems = []

    symptoms = {}
    for symptom_id, body in read_yaml(kb_dir / "symptoms.yaml", problems).items():
        try:
            symptoms[symptom_id] = Symptom(id=symptom_id, **body)
        except ValidationError as e:
            problems += describe(f"symptoms.{symptom_id}", e)

    diseases = {}
    for disease_id, body in read_yaml(kb_dir / "diseases.yaml", problems).items():
        try:
            diseases[disease_id] = Disease(id=disease_id, **body)
        except ValidationError as e:
            problems += describe(f"diseases.{disease_id}", e)

    chapters = {}
    for path in sorted((kb_dir / "chapters").glob("*.yaml")):
        name = path.stem
        body = read_yaml(path, problems)
        if list(body) != CHAPTER_SECTIONS:
            missing = [s for s in CHAPTER_SECTIONS if s not in body]
            unknown = [s for s in body if s not in CHAPTER_SECTIONS]
            problems.append(f"{name}: sections must be {CHAPTER_SECTIONS} in that order; "
                            f"missing {missing}, unknown {unknown}")
            continue
        try:
            chapters[name] = Chapter(file=name, **body)
        except ValidationError as e:
            problems += describe(name, e)

    if problems:
        raise KBError(problems)
    alias_sources = read_alias_sources(kb_dir / "symptoms.yaml")
    return KnowledgeBase(symptoms, diseases, chapters, alias_sources)


def load_lexicon(kb_dir=KB_DIR):
    problems = []
    body = read_yaml(Path(kb_dir) / "lexicon.yaml", problems)
    if not problems:
        try:
            return Lexicon(**body)
        except ValidationError as e:
            problems += describe("lexicon", e)
    raise KBError(problems)


def load_vignettes(path=VIGNETTES_FILE):
    problems = []
    cases = []
    for n, body in enumerate(read_yaml(Path(path), problems) or []):
        try:
            cases.append(Vignette(**body))
        except ValidationError as e:
            problems += describe(f"vignette {body.get('id', n)}", e)
    if problems:
        raise KBError(problems)
    return cases


def read_yaml(path, problems):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        problems.append(f"{path.name}: cannot read: {e}")
        return {}


def describe(where, error):
    """Turn a pydantic error into lines like 'symptoms.fever: kind: Field required'."""
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"{where}: {location}: {item['msg']}")
    return lines


# An alias line in symptoms.yaml, with an optional source comment:
#     - หน้ามืด               # p.248
ALIAS_LINE = re.compile(r"^\s+- (.+?)\s*(?:#\s*(.+))?$")
ENTRY_LINE = re.compile(r"^([a-z_0-9]+):\s*$")


def read_alias_sources(symptoms_file):
    """(symptom id, alias) -> the source written in the alias's comment, or "" if none.

    YAML loaders drop comments, so this reads the file as text.
    """
    sources = {}
    current_symptom = None
    for line in symptoms_file.read_text(encoding="utf-8").split("\n"):
        entry = ENTRY_LINE.match(line)
        if entry:
            current_symptom = entry.group(1)
            continue
        alias = ALIAS_LINE.match(line)
        if alias and current_symptom:
            sources[(current_symptom, alias.group(1))] = (alias.group(2) or "").strip()
    return sources
