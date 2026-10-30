"""HTTP contract tests use invented inputs and no private training artifacts."""
import json

import pytest
from fastapi.testclient import TestClient

from symptom_rec.api.app import create_app
from symptom_rec.api.service import RecommenderService, load_service
from symptom_rec.engine.engine import Engine
from symptom_rec.engine.facts import Facts
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.paths import KB_DIR


@pytest.fixture(scope="module")
def service():
    kb = load_kb()
    return RecommenderService(kb, Normalizer(kb, load_lexicon()), Engine(kb, alpha=1.0))


@pytest.fixture(scope="module")
def client(service):
    with TestClient(create_app(service)) as client:
        yield client


def payload(**changes):
    return {"sex": "female", "age": 40, "chief_complaint": "cough", **changes}


def post(client, **changes):
    response = client.post("/v1/recommend", json=payload(**changes))
    assert response.status_code == 200, response.text
    return response.json()


def test_health_and_openapi(client, service):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_mode"] == "book_only"
    assert health["alpha"] == 1.0
    assert health["symptoms"] == len(service.kb.symptoms)
    assert not health["semantic_enabled"]
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) == {"/health", "/v1/symptoms", "/v1/questions", "/v1/recommend"}
    assert client.get("/docs").status_code == 200


def test_full_ranking_matches_engine(client, service):
    result = post(client, chief_complaint="ไอ")
    expected = service.engine.run(Facts(service.kb, "female", 40, "cough", [], [], {}))
    assert result["chief_complaint"]["id"] == "cough"
    assert result["needs_clarification"] is False
    ranked = [(item["symptom"]["id"], item["score"]) for item in result["recommendations"]]
    assert ranked == expected.ranked(1.0)
    assert len(ranked) > 5
    assert all(item["score"] == item["book_score"] for item in result["recommendations"])
    assert any(item["book_reasons"] for item in result["recommendations"])


def test_questions_and_history_roundtrip(client):
    first = post(client)
    lookup = client.post("/v1/questions", json=payload()).json()
    assert lookup["hpi_questions"] == first["hpi_questions"]
    duration = next(q for q in lookup["hpi_questions"] if q["id"] == "cough.duration")
    assert duration["unit"] == "weeks"
    history = dict(first["normalized_history"])
    history["hpi"] = {"cough.duration": 1, "cough.character": "dry"}
    second = post(client, history=history)
    assert second["normalized_history"]["hpi"]["cough.character"] == ["dry"]
    assert len(second["ddx"]) < len(first["ddx"])
    assert "cough.duration" not in {q["id"] for q in second["hpi_questions"]}
    third = post(client, history=second["normalized_history"])
    assert third["recommendations"] == second["recommendations"]
    assert third["ddx"] == second["ddx"]
    # A different request must not change the next caller's result.
    assert post(client) == first


def test_correcting_hpi_does_not_retain_inferred_symptoms(client):
    first = post(client, history={"hpi": {"cough.character": "dry"}})
    history = first["normalized_history"]
    history["hpi"]["cough.character"] = "productive"
    corrected = post(client, history=history)
    fresh = post(client, history={"hpi": {"cough.character": "productive"}})
    assert corrected["ddx"] == fresh["ddx"]
    assert corrected["recommendations"] == fresh["recommendations"]


def test_ambiguous_cc_offers_choices_without_assigning(client):
    result = post(client, chief_complaint="หายใจไม่สะดวก")
    assert result["chief_complaint"] is None
    assert result["needs_clarification"]
    match = result["normalization"][0]
    assert match["layer"] == "ambiguous"
    assert {c["id"] for c in match["candidates"]} == {"nasal_congestion", "dyspnea"}


def test_negation_and_unknown_are_not_positive(client):
    result = post(client, history={"present": ["ไม่มีไข้"], "unsure": ["sputum", "dyspnea"]})
    assert "fever" in result["normalized_history"]["absent"]
    assert "fever" not in result["normalized_history"]["present"]
    assert not result["urgent"]
    assert not {"fever", "sputum", "dyspnea"} & {r["symptom"]["id"] for r in result["recommendations"]}
    flag = next(r for r in result["red_flags"] if r["symptom"]["id"] == "dyspnea")
    assert flag["answered"] and flag["answer"] == "unknown"


@pytest.mark.parametrize("chief", ["cough", "zzzzqqqq", "vaginal_bleeding"])
def test_global_urgency_survives_unknown_or_inapplicable_cc(client, chief):
    result = post(client, chief_complaint=chief, sex="male", history={"present": ["suicidal_ideation"]})
    assert result["urgent"]
    assert "suicidal_ideation" in result["urgent_reasons"]
    if chief != "cough":
        assert result["needs_clarification"]
        assert result["chapters"] == []


def test_chapter_gate_and_sex_filter(client):
    result = post(client, chief_complaint="abdominal_pain", sex="male",
                  history={"present": ["cold_extremities"]})
    assert result["urgent"]
    assert "unstable" in result["urgent_reasons"]
    assert all(r["symptom"]["sex"] != "female" for r in result["recommendations"] + result["red_flags"])


def test_unknown_cc_does_not_promote_history(client):
    result = post(client, chief_complaint="zzzzqqqq", history={"present": ["fever"]})
    assert result["chief_complaint"] is None
    assert result["needs_clarification"]
    assert result["chapters"] == []


def test_free_text_hpi_and_unread_answers(client):
    result = post(client, answers={"cough": ["ไอมา 1 สัปดาห์", "zzzzqqqq"]})
    assert result["normalized_history"]["hpi"]["cough.duration"] == ["acute"]
    assert result["unread_answers"] == {"cough": ["zzzzqqqq"]}


def test_conflicting_free_text_hpi_requires_correction(client):
    answers = {"sputum": ["เสมหะสีขาวใส", "เสมหะสีเขียวเหลือง"]}
    assert client.post("/v1/recommend", json=payload(answers=answers)).status_code == 422
    result = post(client, answers=answers, history={"hpi": {"sputum.color": "white_clear"}})
    assert result["normalized_history"]["hpi"]["sputum.color"] == ["white_clear"]


def test_large_history_remains_a_valid_response(client, service):
    symptoms = [sid for sid, symptom in service.kb.symptoms.items()
                if not symptom.parent and sid not in service.kb.children and symptom.sex is None][:200]
    assert len(symptoms) == 200
    result = post(client, history={"present": symptoms})
    assert set(symptoms) <= set(result["normalized_history"]["present"])


@pytest.mark.parametrize("changes", [
    {"age": -1}, {"age": True}, {"age": "40"}, {"sex": "other"}, {"chief_complaint": "  "},
    {"top_k": 5}, {"chief_complaint": "label:invented-missing"},
    {"history": {"present": [" "]}}, {"history": {"absent": ["cough"]}},
    {"history": {"present": ["dry_cough"], "absent": ["cough"]}},
    {"history": {"absent": ["fever"], "unsure": ["fever"]}},
    {"history": {"hpi": {"invented.question": "yes"}}},
    {"history": {"hpi": {"cough.character": "invented-option"}}},
    {"history": {"hpi": {"cough.character": 2}}},
    {"history": {"hpi": {"cough.duration": -1}}},
    {"history": {"hpi": {"cough.duration": True}}},
])
def test_invalid_inputs_are_422(client, changes):
    assert client.post("/v1/recommend", json=payload(**changes)).status_code == 422


def test_catalog_search_and_pagination(client):
    result = client.get("/v1/symptoms", params={"q": "cough", "limit": 1}).json()
    assert result["total"] > 1
    assert len(result["items"]) == 1
    second = client.get("/v1/symptoms", params={"q": "cough", "limit": 1, "offset": 1}).json()
    assert second["items"] != result["items"]
    assert second["total"] == result["total"]
    male = client.get("/v1/symptoms", params={"sex": "male"}).json()
    assert all(item["sex"] != "female" for item in male["items"])
    assert client.get("/v1/symptoms", params={"offset": -1}).status_code == 422


def write_artifacts(service, directory):
    directory.mkdir(parents=True, exist_ok=True)
    engine = Engine(service.kb, alpha=0.1)
    engine.memory.add(("cough", ["cough", "sputum", "label:invented-example-symptom"]))
    engine.memory.save(directory / "memory.json")
    (directory / "aliases.json").write_text("{}")
    (directory / "alpha.json").write_text(json.dumps({"alpha": 0.1}))


def test_load_hybrid_and_data_labels(service, tmp_path):
    write_artifacts(service, tmp_path)
    with TestClient(create_app(kb_dir=KB_DIR, build_dir=tmp_path)) as client:
        assert client.get("/health").json()["model_mode"] == "hybrid"
        result = post(client)
        assert result["alpha"] == 0.1
        label = "label:invented-example-symptom"
        assert label in {r["symptom"]["id"] for r in result["recommendations"]}
        result = post(client, chief_complaint=label, history={"absent": ["sputum"]})
        assert result["chief_complaint"]["id"] == label
        assert result["chapters"] == []
        assert "sputum" not in {r["symptom"]["id"] for r in result["recommendations"]}


def test_load_missing_and_partial_artifacts(tmp_path):
    assert load_service(KB_DIR, tmp_path).mode == "book_only"
    (tmp_path / "alpha.json").write_text('{"alpha": 0.1}')
    with pytest.raises(RuntimeError, match="Incomplete"):
        load_service(KB_DIR, tmp_path)


def test_invalid_artifact_alpha(service, tmp_path):
    write_artifacts(service, tmp_path)
    (tmp_path / "alpha.json").write_text('{"alpha": 2}')
    with pytest.raises(RuntimeError, match="alpha"):
        load_service(KB_DIR, tmp_path)


def test_semantic_loads_its_own_profile(service, tmp_path, monkeypatch):
    from symptom_rec.normalize import semantic
    write_artifacts(service, tmp_path / "semantic")
    (tmp_path / "semantic" / "config.json").write_text("{}")
    monkeypatch.setattr(semantic, "load_search", lambda kb, path: object())
    loaded = load_service(KB_DIR, tmp_path, semantic=True)
    assert loaded.mode == "hybrid"
    assert loaded.engine.alpha == 0.1
    assert loaded.normalizer.semantic is not None


def test_semantic_requires_built_profile(tmp_path):
    with pytest.raises(RuntimeError, match="Semantic mode requires"):
        load_service(KB_DIR, tmp_path, semantic=True)
