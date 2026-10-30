"""Only invented review data; real phrases are written to ignored build/ by the CLI."""
import csv
import json

import pytest

from symptom_rec.data.sessions import Session
from symptom_rec.kb.loader import load_kb
from symptom_rec.normalize.normalizer import Match
from symptom_rec.normalize.review import (
    COLUMNS,
    load_review,
    make_manifest,
    prepare_review,
    score_review,
)
from symptom_rec.normalize.text import key


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def sessions():
    return [Session(i, "female", 40, texts, "ไอ", {}, [], [], [], None)
            for i, texts in enumerate([["ไอ", " ไอ "], ["ไข้", "ไอ"], ["เมื่อก่อนเคยไอ"],
                                       ["ไม่แน่ใจว่ามีไข้"], ["ผื่น"], ["ปวดหัว"]])]


def change_sheet(directory, subset, change):
    path = directory / f"{subset}.tsv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    change(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_phrase_grouping_and_alias_validation_isolation(kb):
    train = sessions()
    manifest = make_manifest(train, kb)
    assert manifest == make_manifest(train, kb)
    assert len(manifest["tasks"]) == 6
    cough = next(t for t in manifest["tasks"] if t["key"] == key("ไอ"))
    assert cough["occurrences"] == 3 and len(cough["texts"]) == 2
    validation = {t["key"] for t in manifest["tasks"] if t["subset"] == "validation"}
    for s in train:
        if s.row in manifest["alias_train_rows"]:
            assert not validation & {key(t) for t in s.typed}


def test_review_starts_without_gold_and_preserves_annotations(kb, tmp_path):
    manifest = make_manifest(sessions(), kb)
    prepare_review(tmp_path, manifest, kb)
    reviewed, pending = load_review(tmp_path, manifest, kb, "development")
    assert not reviewed and pending == 5
    result = score_review(None, reviewed, pending)
    assert result["status"] == "awaiting_review"
    assert all(value is None for value in result["metrics"].values())

    def annotate(rows):
        rows[0].update(decision="map", labels="cough:present", reviewer="fixture")
    change_sheet(tmp_path, "development", annotate)
    before = (tmp_path / "development.tsv").read_bytes()
    prepare_review(tmp_path, manifest, kb)
    assert (tmp_path / "development.tsv").read_bytes() == before
    reviewed, pending = load_review(tmp_path, manifest, kb, "development")
    assert len(reviewed) == 1 and pending == 4


def test_manifest_rejects_changed_training_identity(kb, tmp_path):
    manifest = make_manifest(sessions(), kb)
    prepare_review(tmp_path, manifest, kb)
    changed = dict(manifest, train_sha256="changed")
    with pytest.raises(ValueError, match="differs"):
        prepare_review(tmp_path, changed, kb)


@pytest.mark.parametrize("edit", [
    {"texts": '["edited input"]'}, {"occurrences": "999"},
    {"decision": "map", "labels": "made_up:present", "reviewer": "fixture"},
    {"decision": "map", "labels": "cough:maybe", "reviewer": "fixture"},
    {"decision": "map", "labels": "cough:present"},
    {"decision": "map", "reviewer": "fixture"},
    {"decision": "not_symptom", "labels": "cough:present", "reviewer": "fixture"},
])
def test_review_rejects_edited_inputs_or_invalid_gold(kb, tmp_path, edit):
    manifest = make_manifest(sessions(), kb)
    prepare_review(tmp_path, manifest, kb)
    change_sheet(tmp_path, "development", lambda rows: rows[0].update(edit))
    with pytest.raises(ValueError):
        load_review(tmp_path, manifest, kb, "development")


def test_review_rejects_missing_or_duplicate_rows(kb, tmp_path):
    manifest = make_manifest(sessions(), kb)
    prepare_review(tmp_path, manifest, kb)
    change_sheet(tmp_path, "development", lambda rows: rows.pop())
    with pytest.raises(ValueError, match="missing"):
        load_review(tmp_path, manifest, kb, "development")


class Predictions:
    def __init__(self, kb, outputs):
        self.kb = kb
        self.outputs = outputs

    def root(self, sid):
        from symptom_rec.engine.facts import root
        return root(self.kb, sid)

    def read_symptoms(self, text):
        return self.outputs[text]


def task(text, occurrences=1):
    return {"texts": [text], "counts": {text: occurrences}, "occurrences": occurrences}


def test_metrics_penalize_extras_misses_wrong_assertion_and_bad_autofill(kb):
    reviewed = [(task("invented-a", 10), "map", {("cough", "present")}),
                (task("invented-b"), "map", {("fever", "unsure")}),
                (task("invented-c"), "map", {("cough", "present")}),
                (task("invented-d"), "clarify", set())]
    model = Predictions(kb, {
        "invented-a": [Match("", "cough", "fixture", 1), Match("", "fever", "fixture", 1)],
        "invented-b": [Match("", "fever", "fixture", 1, absent=True)],
        "invented-c": [Match("", None, "fixture", 0, candidates=["cough"])],
        "invented-d": [Match("", "cough", "fixture", 1)],
    })
    result = score_review(model, reviewed, 0)
    assert result["counts"]["true_positive"] == 1
    assert result["counts"]["false_positive"] == 3
    assert result["counts"]["false_negative"] == 2
    metrics = result["metrics"]
    assert metrics["assignment_precision"] == 0.25
    assert metrics["assignment_recall"] == pytest.approx(1 / 3)
    assert metrics["exact_match_per_phrase"] == 0
    assert metrics["candidate_recall_at3_when_deferred"] == 1
    assert metrics["auto_assignment_coverage"] == 0.75
    assert metrics["appropriate_deferral"] == 0
    json.dumps(result, allow_nan=False)


def test_phrase_average_and_occurrence_average_are_distinct(kb):
    reviewed = [(task("invented-a", 9), "map", {("cough", "present")}),
                (task("invented-b", 1), "map", {("fever", "present")})]
    model = Predictions(kb, {"invented-a": [Match("", "cough", "fixture", 1)], "invented-b": []})
    metrics = score_review(model, reviewed, 0)["metrics"]
    assert metrics["exact_match_per_phrase"] == 0.5
    assert metrics["exact_match_per_occurrence"] == 0.9
