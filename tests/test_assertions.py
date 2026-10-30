"""Invented language examples; no clinical rules or confidential phrases are added."""
import pytest
from fastapi.testclient import TestClient

from symptom_rec.api.app import create_app
from symptom_rec.api.service import RecommenderService
from symptom_rec.data.sessions import Session
from symptom_rec.engine.adapter import EngineRecommender
from symptom_rec.engine.engine import Engine
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.learn import learn_aliases
from symptom_rec.normalize.normalizer import Normalizer


@pytest.fixture(scope="module")
def normalizer():
    return Normalizer(load_kb(), load_lexicon())


@pytest.mark.parametrize("text,expected", [
    ("ไม่แน่ใจว่ามีไข้", {("fever", "unsure")}),
    ("ไม่แน่ใจว่าไม่มีไข้", {("fever", "unsure")}),
    ("อาจจะมีไข้", {("fever", "unsure")}),
    ("ไม่ทราบว่ามีไข้", {("fever", "unsure")}),
    ("not sure if I have fever", {("fever", "unsure")}),
    ("maybe fever", {("fever", "unsure")}),
    ("เมื่อก่อนเคยไอ แต่ตอนนี้หายแล้ว", {("cough", "historical")}),
    ("ไอ แต่ตอนนี้หายแล้ว", {("cough", "historical")}),
    ("เคยไอ", {("cough", "historical")}),
    ("ไอหายแล้ว", {("cough", "historical")}),
    ("used to cough", {("cough", "historical")}),
    ("cough is gone", {("cough", "historical")}),
    ("ไอยังไม่หาย", {("cough", "present")}),
    ("cough has not resolved", {("cough", "present")}),
    ("ตอนนี้ไม่มีไข้", {("fever", "absent")}),
    ("ไม่มีไข้แต่ไอ", {("fever", "absent"), ("cough", "present")}),
    ("ไม่แน่ใจว่ามีไข้แต่ไอ", {("fever", "unsure"), ("cough", "present")}),
    ("ไอและไม่แน่ใจว่ามีไข้", {("fever", "unsure"), ("cough", "present")}),
    ("เคยไอ แต่ตอนนี้ยังไอ", {("cough", "historical"), ("cough", "present")}),
    ("มีประวัติเป็นเบาหวาน", {("diabetes", "present")}),
    ("ปัสสาวะไม่ออก", {("unable_to_urinate", "present")}),
    ("หายใจลำบาก", {("dyspnea", "present")}),
])
def test_assertion_scope(normalizer, text, expected):
    matches = normalizer.read_symptoms(text)
    assert {(m.symptom, m.state) for m in matches if m.symptom} == expected
    assert set(normalizer.symptom_ids(text)) == {sid for sid, state in expected if state == "present"}


def test_noncurrent_context_cannot_be_erased_by_a_learned_alias(normalizer):
    learned = Normalizer(normalizer.kb, normalizer.lexicon, {"ไม่แน่ใจว่ามีไข้": "cough"})
    assert learned.symptom_ids("ไม่แน่ใจว่ามีไข้") == []
    assert next(m for m in learned.read_symptoms("ไม่แน่ใจว่ามีไข้") if m.symptom).state == "unsure"


def session(row):
    return Session(row, "female", 40, ["ไม่แน่ใจว่ามีไข้", "เคยไอ", "ไม่มีไข้"],
                   "ไข้", {}, ["ไอ"], [], [], None)


def test_alias_learning_excludes_nonpositive_context(normalizer):
    assert learn_aliases(normalizer, [session(0), session(1)]) == {}


def test_proxy_reporting_does_not_count_uncertainty_or_past_as_positive(normalizer):
    from symptom_rec.evaluation import evaluate_normalizer
    s = session(0)
    result = evaluate_normalizer(normalizer, [s], set())
    assert result["signed"]["unsure_assignments"] == 1
    assert result["signed"]["historical_assignments"] == 1
    assert result["signed"].get("positive_assignments", 0) == 0
    assert result["signed"]["negative_assignments"] == 1


def test_history_and_memory_do_not_convert_noncurrent_mentions_into_labels(normalizer):
    chief, history = normalizer.read_history(
        ["เมื่อก่อนเคยไอ แต่ตอนนี้หายแล้ว", "ไม่แน่ใจว่ามีไข้", "ไม่มีเสมหะ"], {}, [])
    assert chief is None
    assert history.present == [] and history.unmapped == []
    assert history.historical == ["cough"]
    assert history.unsure == ["fever"] and history.absent == ["sputum"]
    s = Session(0, "female", 40, [], "เคยไอ", {}, ["ไม่แน่ใจว่ามีไข้"], [], [], None)
    adapter = EngineRecommender(Engine(normalizer.kb), normalizer)
    adapter.learn([s])
    assert adapter.engine.memory.counts.sessions_with == {}


def test_uncertain_or_past_hpi_is_not_a_current_answer(normalizer):
    assert normalizer.read_answer("cough", "ไม่แน่ใจว่าไอมา 2 สัปดาห์") == {}
    assert normalizer.read_answer("cough", "เคยไอแห้ง แต่ตอนนี้หายแล้ว") == {}
    assert normalizer.read_answer("cough", "ไอมาตั้งแต่ 3 วัน ถึง 2 สัปดาห์")


@pytest.fixture(scope="module")
def client(normalizer):
    service = RecommenderService(normalizer.kb, normalizer, Engine(normalizer.kb, alpha=1.0))
    with TestClient(create_app(service)) as client:
        yield client


def request(client, chief="cough", history=None):
    response = client.post("/v1/recommend", json={"sex": "female", "age": 40,
                                               "chief_complaint": chief, "history": history or {}})
    assert response.status_code == 200, response.text
    return response.json()


def test_api_preserves_states_and_history_roundtrip(client):
    result = request(client, history={"present": ["ไม่แน่ใจว่ามีไข้", "เคยปวดท้อง"]})
    history = result["normalized_history"]
    assert history["unsure"] == ["fever"]
    assert history["historical"] == ["abdominal_pain"]
    assert history["absent"] == []
    assert history["present"] == ["cough"]
    again = request(client, history=history)
    assert again["normalized_history"] == history
    assert again["recommendations"] == result["recommendations"]


@pytest.mark.parametrize("text", ["ไม่แน่ใจว่ามีไข้", "เมื่อก่อนเคยไอ แต่ตอนนี้หายแล้ว"])
def test_api_noncurrent_cc_does_not_select_a_chapter(client, text):
    result = request(client, text)
    assert result["chief_complaint"] is None and result["needs_clarification"]
    assert result["chapters"] == []


def test_uncertain_global_flag_does_not_become_an_emergency(client):
    result = request(client, history={"present": ["ไม่แน่ใจว่าอยากตาย"]})
    assert not result["urgent"]
    assert "suicidal_ideation" in result["normalized_history"]["unsure"]
    current = request(client, history={"present": ["อยากตาย"]})
    assert current["urgent"]
