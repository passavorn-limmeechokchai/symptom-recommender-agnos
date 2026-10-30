import csv
import json

import pytest
import yaml

from symptom_rec.kb.loader import KBError, load_kb, load_vignettes
from symptom_rec.kb.schema import CHAPTER_SECTIONS
from symptom_rec.kb.trees import all_branches
from symptom_rec.kb.validate import check_alias_leakage, data_strings, validate
from symptom_rec.paths import DATA_DIR

# A tiny made-up knowledge base: 4 symptoms, 2 diseases, 1 chapter.
SYMPTOMS_YAML = """\
fever:
  th: ไข้
  en: fever
  kind: symptom
  aliases:
    - ตัวร้อน               # p.1

cough:
  th: ไอ
  en: cough
  kind: symptom

dry_cough:
  th: ไอแห้งๆ
  en: dry cough
  kind: symptom
  parent: cough
  refines: {character: dry}

sputum:
  th: มีเสมหะ
  en: sputum
  kind: symptom
  aliases:
    - เสมหะ
"""

DISEASES = {
    "flu": {"th": "ไข้หวัดใหญ่", "en": "influenza", "kind": "disease"},
    "pneumonia": {"th": "ปอดอักเสบ", "en": "pneumonia", "kind": "disease"},
}


def make_chapter(**changes):
    """A valid chapter; pass section=value to change one section."""
    chapter = {
        "chapter": 1,
        "title": {"th": "ไอ", "en": "Cough"},
        "chief_complaint": "cough",
        "also_entry": [],
        "status": "draft",
        "definitions": {},
        "hpi": {"cough": {"character": {"ask": "ไอแบบไหน", "options": ["dry", "productive"]}}},
        "steps": [{"id": "start"}],
        "gates": {},
        "must_ask": [],
        "tree_selection": [{"when": "always", "use": ["main"]}],
        "branch_selection": {},
        "branch_hints": [],
        "trees": {"main": {"viral": ["flu"], "bacterial": ["pneumonia"]}},
        "scores": {},
        "confirm": {},
        "findings": {"flu": {"symptoms": ["fever"]}, "pneumonia": {"symptoms": ["sputum"]}},
        "branch_findings": {},
        "reviewer_findings": {},
        "history_clues": [],
        "exam_clues": [],
    }
    chapter.update(changes)
    return chapter


def write_kb(folder, chapter):
    (folder / "chapters").mkdir()
    (folder / "symptoms.yaml").write_text(SYMPTOMS_YAML, encoding="utf-8")
    (folder / "diseases.yaml").write_text(yaml.safe_dump(DISEASES, allow_unicode=True), encoding="utf-8")
    chapter_text = yaml.safe_dump(chapter, allow_unicode=True, sort_keys=False)
    (folder / "chapters" / "01_cough.yaml").write_text(chapter_text, encoding="utf-8")
    return folder


def errors_for(tmp_path, **changes):
    kb = load_kb(write_kb(tmp_path, make_chapter(**changes)))
    return validate(kb).errors


# ── the real knowledge base ────────────────────────────────────────────────
def test_real_kb_is_valid():
    result = validate(load_kb(), load_vignettes())
    assert result.errors == []


def test_real_kb_aliases_do_not_leak_from_the_data():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        pytest.skip("Agnos data not present (it is never committed)")
    assert check_alias_leakage(load_kb(), data_strings(csv_files[0])) == []


# ── loading ────────────────────────────────────────────────────────────────
def test_small_kb_loads_and_is_valid(tmp_path):
    kb = load_kb(write_kb(tmp_path, make_chapter()))
    assert validate(kb).errors == []
    assert kb.terms["ตัวร้อน"] == ["fever"]
    assert kb.children["cough"] == ["dry_cough"]
    assert kb.entry_chapters["cough"] == ["01_cough"]
    assert kb.alias_sources[("fever", "ตัวร้อน")] == "p.1"
    assert kb.alias_sources[("sputum", "เสมหะ")] == ""
    assert kb.questions["cough.character"].options == ["dry", "productive"]


def test_sections_must_be_in_order(tmp_path):
    chapter = make_chapter()
    reordered = {}
    for section in reversed(CHAPTER_SECTIONS):
        reordered[section] = chapter[section]
    with pytest.raises(KBError, match="sections must be"):
        load_kb(write_kb(tmp_path, reordered))


def test_missing_section_is_an_error(tmp_path):
    chapter = make_chapter()
    del chapter["scores"]
    with pytest.raises(KBError, match="missing \\['scores'\\]"):
        load_kb(write_kb(tmp_path, chapter))


def test_unknown_key_is_an_error(tmp_path):
    chapter = make_chapter(branch_hints=[{"when": "always", "favor": "main.viral"}])
    with pytest.raises(KBError, match="favor"):
        load_kb(write_kb(tmp_path, chapter))


# ── cross-file checks ──────────────────────────────────────────────────────
def test_short_condition_is_an_error(tmp_path):
    errors = errors_for(tmp_path, tree_selection=[{"when": {"character": "dry"}, "use": ["main"]}])
    assert any("must be written as <symptom>.<question>" in e for e in errors)


def test_unknown_symptom_is_an_error(tmp_path):
    errors = errors_for(tmp_path, findings={"flu": {"symptoms": ["fevr"]}, "pneumonia": {}})
    assert any("unknown symptom 'fevr'" in e for e in errors)


def test_answer_must_be_an_option(tmp_path):
    errors = errors_for(tmp_path, tree_selection=[{"when": {"cough.character": "wet"}, "use": ["main"]}])
    assert any("'wet' is not an answer of cough.character" in e for e in errors)


def test_findings_hold_diseases_and_branch_findings_hold_branches(tmp_path):
    errors = errors_for(
        tmp_path,
        findings={"main.viral": {"symptoms": ["fever"]}, "pneumonia": {}},
        branch_findings={"flu": {"symptoms": ["fever"]}},
    )
    assert any("unknown disease 'main.viral'" in e for e in errors)
    assert any("'flu' is not a tree branch" in e for e in errors)


def test_disease_outside_every_tree_is_an_error(tmp_path):
    errors = errors_for(tmp_path, trees={"main": {"viral": ["flu"]}})
    assert "diseases in no DDx tree: ['pneumonia']" in errors


def test_comment_without_a_space_is_an_error(tmp_path):
    folder = write_kb(tmp_path, make_chapter())
    text = SYMPTOMS_YAML.replace("    - เสมหะ\n", "    - เสมหะ# p.1\n")
    (folder / "symptoms.yaml").write_text(text, encoding="utf-8")
    errors = validate(load_kb(folder)).errors
    assert any("contains '#'" in e for e in errors)


def test_bad_status_is_an_error(tmp_path):
    errors = errors_for(tmp_path, status="done")
    assert any("'done' must be one of ['draft', 'reviewed']" in e for e in errors)


# ── trees ──────────────────────────────────────────────────────────────────
def test_all_branches():
    trees = {"main": {"viral": ["flu"], "bacterial": ["pneumonia"]}}
    assert all_branches(trees) == {
        "main": ["flu", "pneumonia"],
        "main.viral": ["flu"],
        "main.bacterial": ["pneumonia"],
    }


# ── leakage check ──────────────────────────────────────────────────────────
def test_alias_found_in_data_needs_a_source(tmp_path):
    kb = load_kb(write_kb(tmp_path, make_chapter()))
    errors = check_alias_leakage(kb, {"ตัวร้อน", "เสมหะ"})
    assert errors == ["symptoms.sputum: alias 'เสมหะ' appears in the data but has no source comment"]


def test_data_strings_reads_symptom_names_and_search_terms(tmp_path):
    path = tmp_path / "fake.csv"
    summary = {"yes_symptoms": [{"text": "ไอ"}], "no_symptoms": [{"text": "ไข้"}], "idk_symptoms": []}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gender", "age", "summary", "search_term"])
        writer.writerow(["male", 30, json.dumps(summary, ensure_ascii=False), "ไอแห้ง, เจ็บคอ"])
    assert data_strings(path) == {"ไอ", "ไข้", "ไอแห้ง", "เจ็บคอ"}
