"""Checks that look across files, which a single YAML block cannot show on its own.

Run from the repo root:
    python -m symptom_rec.kb.validate                          # knowledge base + test cases
    python -m symptom_rec.kb.validate --data data/raw/x.csv    # also check aliases against the data
"""
import argparse
import sys
from pathlib import Path

from ..data.sessions import load_sessions
from ..paths import KB_DIR, VIGNETTES_FILE
from .loader import KBError, load_kb, load_lexicon, load_vignettes
from .schema import SENSATIONS
from .trees import all_branches, leaf_lists, shape_problems

SYMPTOM_KINDS = ["symptom", "history", "measurement"]
DISEASE_KINDS = ["disease", "group"]
SEXES = ["female", "male"]
STATUSES = ["draft", "reviewed"]
CONDITION_KEYS = ["present", "absent", "sex", "age_above", "age_below", "age_between", "no_symptoms_of"]
SCORE_ITEM_KEYS = ["symptom", "absent", "age_between", "hpi"]
SOURCE_PREFIXES = ["p", "g", "eye", "ent", "uro"]   # book, guideline, eye, ENT and urology notes


class Checker:
    """Collects errors and warnings while the checks run."""

    def __init__(self, kb):
        self.kb = kb
        self.errors = []
        self.warnings = []
        self.used_symptoms = set()

    def symptom(self, symptom_id, where):
        self.used_symptoms.add(symptom_id)
        if symptom_id not in self.kb.symptoms:
            self.errors.append(f"{where}: unknown symptom '{symptom_id}'")

    def disease(self, disease_id, where):
        if disease_id not in self.kb.diseases:
            self.errors.append(f"{where}: unknown disease '{disease_id}'")

    def one_of(self, value, allowed, where):
        if value is not None and value not in allowed:
            self.errors.append(f"{where}: '{value}' must be one of {allowed}")

    def answer(self, key, values, where):
        """`key` is "<symptom>.<question>"; every text value must be one of its answers."""
        question = self.kb.questions.get(key)
        if question is None:
            self.errors.append(f"{where}: no HPI question '{key}'")
            return
        allowed = answers_of(question)
        for value in values:
            if isinstance(value, str) and value not in allowed:
                self.errors.append(f"{where}: '{value}' is not an answer of {key} {sorted(allowed)}")

    def condition(self, when, branches, where):
        if when == "always":
            return
        if not isinstance(when, dict):
            self.errors.append(f"{where}: a condition is 'always' or a mapping")
            return
        for key, value in when.items():
            if key in ("present", "absent"):
                for symptom_id in value:
                    self.symptom(symptom_id, where)
            elif key == "sex":
                self.one_of(value, SEXES, where)
            elif key == "no_symptoms_of":
                if value not in branches:
                    self.errors.append(f"{where}: unknown branch '{value}'")
            elif key in CONDITION_KEYS:
                pass   # ages are plain numbers
            elif "." in key:
                self.answer(key, value if isinstance(value, list) else [value], where)
            else:
                self.errors.append(f"{where}: condition '{key}' must be written as <symptom>.<question>")


def answers_of(question):
    allowed = set(question.options or []) | set(question.categories or {})
    if question.allow_unknown:
        allowed.add("unknown")
    return allowed


# ── symptoms.yaml and diseases.yaml ────────────────────────────────────────
def check_registries(c):
    for s in c.kb.symptoms.values():
        where = f"symptoms.{s.id}"
        c.one_of(s.kind, SYMPTOM_KINDS, where)
        c.one_of(s.sex, SEXES, where)
        if bool(s.parent) != bool(s.refines):
            c.errors.append(f"{where}: parent and refines go together")
        if s.parent:
            c.symptom(s.parent, where + ".parent")
        for name in [s.th, s.en] + s.aliases:
            if "#" in name:   # "- word# p.1": YAML needs a space before a comment
                c.errors.append(f"{where}: '{name}' contains '#'; put a space before the comment")
    for d in c.kb.diseases.values():
        c.one_of(d.kind, DISEASE_KINDS, f"diseases.{d.id}")
        c.one_of(d.sex, SEXES, f"diseases.{d.id}")
    for term, ids in c.kb.terms.items():
        if len(ids) > 1:
            c.warnings.append(f"'{term}' names more than one symptom: {ids}")
    for (symptom_id, alias), source in c.kb.alias_sources.items():
        if source and source != "reviewer" and source.split(".")[0] not in SOURCE_PREFIXES:
            c.errors.append(f"symptoms.{symptom_id}: alias '{alias}' has an unknown source '{source}'")


# ── one chapter file ───────────────────────────────────────────────────────
def check_chapter(c, name, ch):
    """Check one chapter. Returns the disease ids its trees use."""
    c.one_of(ch.status, STATUSES, f"{name}.status")

    problems = shape_problems(ch.trees)
    for p in problems:
        c.errors.append(f"{name}.trees: {p}")
    if problems:
        return set()
    branches = all_branches(ch.trees)
    placed = set()
    for path, diseases in leaf_lists(ch.trees).items():
        if len(diseases) != len(set(diseases)):
            c.errors.append(f"{name}: {path} lists a disease twice")
        for disease_id in diseases:
            c.disease(disease_id, f"{name}: {path}")
            placed.add(disease_id)

    c.symptom(ch.chief_complaint, f"{name}.chief_complaint")
    for symptom_id in ch.also_entry:
        c.symptom(symptom_id, f"{name}.also_entry")

    for symptom_id, questions in ch.hpi.items():
        c.symptom(symptom_id, f"{name}.hpi")
        for q_name, q in questions.items():
            where = f"{name}.hpi.{symptom_id}.{q_name}"
            c.one_of(q.sex, SEXES, where)
            if not q.options and not q.categories:
                c.errors.append(f"{where}: needs options or categories")
            for zone in q.grey_zones:
                if sorted(zone) != ["both", "from", "to"]:
                    c.errors.append(f"{where}: a grey zone has keys from, to, both")
                elif not set(zone["both"]) <= set(q.categories or {}):
                    c.errors.append(f"{where}: grey zone uses unknown category {zone['both']}")

    for g_name, gate in ch.gates.items():
        where = f"{name}.gates.{g_name}"
        c.condition(gate.when, branches, where)
        for symptom_id in gate.ask:
            c.symptom(symptom_id, where)
        action = gate.if_any_yes
        if action != "urgent" and not (isinstance(action, dict) and action.get("prioritize") in branches):
            c.errors.append(f"{where}: if_any_yes must be 'urgent' or {{prioritize: <branch>}}")

    for n, rule in enumerate(ch.must_ask):
        where = f"{name}.must_ask[{n}]"
        c.condition(rule.when, branches, where)
        if rule.red_flag_when is not None:
            c.condition(rule.red_flag_when, branches, where + ".red_flag_when")
        for symptom_id in rule.ask:
            c.symptom(symptom_id, where)
        for disease_id in rule.rules_out:
            c.disease(disease_id, where)

    for n, rule in enumerate(ch.tree_selection):
        where = f"{name}.tree_selection[{n}]"
        c.condition(rule.when, branches, where)
        for tree in rule.use:
            if tree not in ch.trees:
                c.errors.append(f"{where}: unknown tree '{tree}'")

    if ch.branch_selection:
        check_branch_selection(c, name, ch.branch_selection, branches)

    for n, hint in enumerate(ch.branch_hints):
        where = f"{name}.branch_hints[{n}]"
        c.condition(hint.when, branches, where)
        if hint.boost not in branches:
            c.errors.append(f"{where}: unknown branch '{hint.boost}'")

    for s_name, score in ch.scores.items():
        where = f"{name}.scores.{s_name}"
        c.disease(score.applies_to, where)
        c.condition(score.when, branches, where)
        for key in score.threshold:
            c.one_of(key, ["above", "at_least"], where + ".threshold")
        for item in score.items:
            for key in item:
                c.one_of(key, SCORE_ITEM_KEYS, where + ".items")
            for key in ("symptom", "absent"):
                if key in item:
                    c.symptom(item[key], where)
            for hpi_key in item.get("hpi", {}):
                c.answer(hpi_key, [], where)

    for symptom_id, confirm in ch.confirm.items():
        where = f"{name}.confirm.{symptom_id}"
        for s in [symptom_id] + confirm.supports + confirm.against + list(confirm.if_against.values()):
            c.symptom(s, where)
        for hpi_key, sides in confirm.hpi.items():
            for side, values in sides.items():
                c.one_of(side, ["supports", "against"], where)
                c.answer(hpi_key, values, where)

    for section in ("findings", "reviewer_findings"):
        for disease_id, finding in getattr(ch, section).items():
            c.disease(disease_id, f"{name}.{section}")
            for symptom_id in finding.symptoms:
                c.symptom(symptom_id, f"{name}.{section}.{disease_id}")
    for path, finding in ch.branch_findings.items():
        if path not in branches:
            c.errors.append(f"{name}.branch_findings: '{path}' is not a tree branch")
        for symptom_id in finding.symptoms:
            c.symptom(symptom_id, f"{name}.branch_findings.{path}")

    for n, clue in enumerate(ch.history_clues):
        for symptom_id in clue.findings:
            c.symptom(symptom_id, f"{name}.history_clues[{n}]")
        for disease_id in clue.suggests:
            c.disease(disease_id, f"{name}.history_clues[{n}]")
    for clue in ch.exam_clues:
        if clue.askable_as:
            c.symptom(clue.askable_as, f"{name}.exam_clues.{clue.sign}")

    return placed


def check_branch_selection(c, name, selection, branches):
    where = f"{name}.branch_selection"
    if sorted(selection) != ["attribute", "map"]:
        c.errors.append(f"{where}: needs exactly attribute and map")
        return
    c.answer(selection["attribute"], list(selection["map"]), where)
    branch_names = {path.split(".")[-1] for path in branches}
    for picked in selection["map"].values():
        for branch in picked:
            if branch not in branch_names:
                c.errors.append(f"{where}: unknown branch '{branch}'")


# ── tests/cases/vignettes.yaml ─────────────────────────────────────────────
def check_vignettes(c, cases):
    for v in cases:
        where = f"vignette {v.id}"
        c.one_of(v.level, ["simple", "book"], where)
        c.one_of(v.sex, SEXES, where)
        if v.level == "simple" and v.expect is None:
            c.errors.append(f"{where}: a simple case needs expect")
        if v.level == "book" and not v.final_diagnosis:
            c.errors.append(f"{where}: a book case needs final_diagnosis")
        c.symptom(v.chief_complaint, where)
        if v.chief_complaint not in c.kb.entry_chapters:
            c.errors.append(f"{where}: no chapter opens on '{v.chief_complaint}'")
        for symptom_id in v.history.present + v.history.absent:
            c.symptom(symptom_id, where)
        for key, value in v.history.hpi.items():
            c.answer(key, value if isinstance(value, list) else [value], where)
        diseases = v.book_ddx + [v.final_diagnosis]
        e = v.expect
        if e is not None:
            for symptom_id in e.ask_first + e.not_ask_first + e.never_recommend:
                c.symptom(symptom_id, where)
            if e.switch_chief_complaint:
                c.symptom(e.switch_chief_complaint, where)
            diseases += [e.top3_includes] + e.top3_includes_any + e.never_in_ddx
        for disease_id in diseases:
            if disease_id:
                c.disease(disease_id, where)


# ── everything together ────────────────────────────────────────────────────
def validate(kb, vignettes=None, lexicon=None):
    """Run every check. Returns a Checker; look at .errors and .warnings."""
    c = Checker(kb)
    check_registries(c)

    placed = set()
    for name, chapter in kb.chapters.items():
        placed |= check_chapter(c, name, chapter)
    unplaced = sorted(set(kb.diseases) - placed)
    if unplaced:
        c.errors.append(f"diseases in no DDx tree: {unplaced}")

    # a child symptom's refines must be an answer of its parent's HPI question
    for s in kb.symptoms.values():
        for q_name, answer in (s.refines or {}).items():
            key = f"{s.parent}.{q_name}"
            if key not in kb.questions:
                c.warnings.append(f"symptoms.{s.id}: no HPI question {key} to refine")
            elif answer not in answers_of(kb.questions[key]):
                c.errors.append(f"symptoms.{s.id}: {key} has no answer '{answer}'")

    # a question asked in two chapters must be the same question in both
    first_seen = {}
    for name, chapter in kb.chapters.items():
        for symptom_id, questions in chapter.hpi.items():
            for q_name, question in questions.items():
                key = symptom_id + "." + q_name
                shape = question.model_dump(exclude={"ask"})
                if key not in first_seen:
                    first_seen[key] = (name, shape)
                elif first_seen[key][1] != shape:
                    c.errors.append(
                        f"{name}.hpi.{key}: differs from the same question in {first_seen[key][0]}"
                    )

    unused = sorted(set(kb.symptoms) - c.used_symptoms - set(kb.entry_chapters))
    if unused:
        c.warnings.append(f"symptoms no chapter uses: {unused}")

    if vignettes is not None:
        check_vignettes(c, vignettes)
    if lexicon is not None:
        check_lexicon(c, lexicon)
    return c


def check_lexicon(c, lexicon):
    for sensation in lexicon.sensations:
        c.one_of(sensation, SENSATIONS, "lexicon.sensations")
    for part_name, part in lexicon.body_parts.items():
        for sensation in SENSATIONS:
            value = getattr(part, sensation)   # the field named by the sensation, e.g. part.pain
            if isinstance(value, str):
                value = [value]
            for symptom_id in value or []:
                if symptom_id not in c.kb.symptoms:
                    where = f"lexicon.body_parts.{part_name}.{sensation}"
                    c.errors.append(f"{where}: unknown symptom '{symptom_id}'")
    for phrase, symptom_ids in lexicon.ambiguous.items():
        for symptom_id in symptom_ids:
            if symptom_id not in c.kb.symptoms:
                c.errors.append(f"lexicon.ambiguous.{phrase}: unknown symptom '{symptom_id}'")


# ── leakage check against the Agnos data ───────────────────────────────────
def data_strings(csv_path):
    """Every symptom name and search term that appears in the Agnos data."""
    seen = set()
    for s in load_sessions(csv_path):
        seen.update(s.typed)
        seen.update(s.confirmed())
        seen.update(s.absent)
        seen.update(s.unsure)
    return seen


def check_alias_leakage(kb, seen):
    """An alias that also appears in the data must say where it came from. Returns the errors."""
    errors = []
    for (symptom_id, alias), source in kb.alias_sources.items():
        if alias in seen and not source:
            message = f"symptoms.{symptom_id}: alias '{alias}' appears in the data but has no source comment"
            errors.append(message)
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kb", type=Path, default=KB_DIR)
    parser.add_argument("--vignettes", type=Path, default=VIGNETTES_FILE)
    parser.add_argument("--data", type=Path, help="Agnos CSV; adds the alias leakage check")
    args = parser.parse_args(argv)

    try:
        kb = load_kb(args.kb)
        cases = load_vignettes(args.vignettes) if args.vignettes.exists() else None
        lexicon = load_lexicon(args.kb) if (args.kb / "lexicon.yaml").exists() else None
    except KBError as e:
        for problem in e.problems:
            print("ERROR", problem)
        return 1

    c = validate(kb, cases, lexicon)
    if args.data:
        c.errors += check_alias_leakage(kb, data_strings(args.data))

    summary = f"{len(kb.chapters)} chapters, {len(kb.symptoms)} symptoms, {len(kb.diseases)} diseases"
    if cases is not None:
        summary += f", {len(cases)} vignettes"
    print(summary)
    for w in c.warnings:
        print("WARN ", w)
    for e in c.errors:
        print("ERROR", e)
    if c.errors:
        print(f"{len(c.errors)} error(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
