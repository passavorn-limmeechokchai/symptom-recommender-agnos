import pytest

from symptom_rec.baselines.counting import Cooccurrence, Popularity, SymptomCounts
from symptom_rec.data.sessions import Session
from symptom_rec.evaluation import (
    bootstrap,
    difference,
    evaluate,
    first_step_cases,
    hide_one_cases,
    hit_rate,
    middle_95,
    session_hit_rate,
)


def make_session(row, symptoms, copy_of=None):
    """A made-up session: the first symptom is the chief complaint."""
    return Session(
        row=row, sex="female", age=30, typed=symptoms, chief_complaint=symptoms[0],
        answers={}, associated=symptoms[1:], absent=[], unsure=[], previous_treatment=None,
        copy_of=copy_of,
    )


# 6 cough sessions with sputum, 2 with fever, 1 with blood in sputum (only in this one session)
SESSIONS = (
    [make_session(row, ["cough", "sputum"]) for row in range(6)]
    + [make_session(6, ["cough", "fever"]), make_session(7, ["cough", "fever"])]
    + [make_session(8, ["cough", "blood in sputum"])]
    + [make_session(9, ["rash"])]
)


# ── counting ───────────────────────────────────────────────────────────────
def test_counts():
    counts = SymptomCounts(SESSIONS)
    assert counts.sessions_with["cough"] == 9
    assert counts.together["cough"]["sputum"] == 6
    assert counts.together["sputum"]["cough"] == 6


def test_leave_out_and_put_back():
    model = Cooccurrence(SESSIONS)
    model.leave_out(SESSIONS[8])
    assert model.counts.together["cough"]["blood in sputum"] == 0
    model.put_back(SESSIONS[8])
    assert model.counts.together["cough"]["blood in sputum"] == 1


# ── baselines ──────────────────────────────────────────────────────────────
def test_popularity_skips_given_symptoms():
    model = Popularity(SESSIONS)
    assert model.recommend({"given": ["cough"]}, 2) == ["sputum", "fever"]


def test_cooccurrence_ranks_by_what_comes_with_the_given_symptoms():
    model = Cooccurrence(SESSIONS)
    assert model.recommend({"given": ["cough"]}, 3) == ["sputum", "fever", "blood in sputum"]


def test_cooccurrence_fills_up_with_popular_symptoms():
    model = Cooccurrence(SESSIONS)
    # rash never comes with anything, so all places come from popularity
    assert model.recommend({"given": ["rash"]}, 2) == ["cough", "sputum"]


# ── hide-one test ──────────────────────────────────────────────────────────
def test_chief_complaint_is_never_hidden():
    session = make_session(0, ["cough", "sputum", "fever"])
    shown_and_hidden = []
    for shown, hidden in hide_one_cases(session):
        shown_and_hidden.append((shown["given"], hidden))
    assert shown_and_hidden == [(["cough", "fever"], "sputum"), (["cough", "sputum"], "fever")]
    assert hide_one_cases(make_session(1, ["rash"])) == []


def test_first_step_shows_only_the_chief_complaint():
    session = make_session(0, ["cough", "sputum", "fever"])
    session.answers = {"cough": ["2 days"], "sputum": ["white"]}
    cases = first_step_cases(session)
    assert [hidden for _, hidden in cases] == ["sputum", "fever"]
    for shown, _ in cases:
        assert shown["given"] == ["cough"]
        assert shown["answers"] == {"cough": ["2 days"]}   # never the hidden symptom's answers


def test_leave_one_out_stops_a_session_finding_itself():
    model = Cooccurrence(SESSIONS)
    blood = [SESSIONS[8]]
    # counted with itself, the only session with blood in sputum finds it in the top 3
    assert hit_rate(evaluate(model, blood), 3) == 1
    # taken out first, nothing links cough to blood in sputum any more
    assert hit_rate(evaluate(model, blood, leave_one_out=True), 3) == 0
    # and the counts are back afterwards
    assert model.counts.together["cough"]["blood in sputum"] == 1


def test_leave_one_out_also_takes_out_exact_copies():
    sessions = SESSIONS + [make_session(10, ["cough", "blood in sputum"], copy_of=8)]
    model = Cooccurrence(sessions)
    results = evaluate(model, sessions, leave_one_out=True)
    for r in results:
        if r["row"] in (8, 10):
            assert r["hits"][3] == 0


def test_sessions_without_associated_symptoms_are_not_tested():
    results = evaluate(Popularity(SESSIONS), SESSIONS)
    assert len(results) == 9  # the rash session has nothing to hide


# ── bootstrap ──────────────────────────────────────────────────────────────
def test_bootstrap_uses_the_same_draws_for_every_model():
    results = evaluate(Cooccurrence(SESSIONS), SESSIONS)
    rates = bootstrap({"a": results, "b": results}, k=3, rounds=50)
    assert rates["a"] == rates["b"]
    assert set(difference(rates["a"], rates["b"])) == {0}


def test_middle_95():
    values = list(range(1000))
    assert middle_95(values) == (25, 974)


def test_session_mean_gives_each_session_equal_weight():
    results = [{"cases": 1, "hits": {5: 0}}, {"cases": 3, "hits": {5: 3}}]
    assert hit_rate(results, 5) == 0.75
    assert session_hit_rate(results, 5) == 0.5


def test_frozen_cases_override_new_case_generation():
    session = make_session(0, ["cough", "sputum"])
    fixed = {0: hide_one_cases(session)}
    session.associated = []
    model = Popularity(SESSIONS)
    results = evaluate(model, [session], cases_by_row=fixed)
    assert results[0]["cases"] == 1
    assert results[0]["hits"][3] == 1


def test_reference_targets_do_not_change_with_a_new_normalizer():
    from symptom_rec.engine.adapter import EngineRecommender
    from symptom_rec.engine.engine import Engine
    from symptom_rec.kb.loader import load_kb, load_lexicon
    from symptom_rec.normalize.normalizer import Normalizer

    kb = load_kb()
    model = EngineRecommender(Engine(kb), Normalizer(kb, load_lexicon()), reference_ids={"ไข้": ["cough"]})
    assert model.found("ไข้", ["cough"])
    assert not model.found("ไข้", ["fever"])


def test_bootstrap_rejects_unpaired_sessions():
    a = [{"row": 0, "cases": 1, "hits": {5: 1}}]
    b = [{"row": 1, "cases": 1, "hits": {5: 1}}]
    with pytest.raises(ValueError, match="same sessions"):
        bootstrap({"a": a, "b": b}, 5)


def test_normalizer_proxy_records_negation_without_calling_it_unrecognized():
    from symptom_rec.evaluation import evaluate_normalizer
    from symptom_rec.kb.loader import load_kb, load_lexicon
    from symptom_rec.normalize.normalizer import Normalizer

    normalizer = Normalizer(load_kb(), load_lexicon())
    session = make_session(0, ["ไอ"])
    session.typed = ["ไม่มีไข้"]
    session.absent = ["ไข้"]
    result = evaluate_normalizer(normalizer, [session], set())
    assert result["typed"]["unseen"]["not placed"] == 1  # legacy positive-id metric
    assert result["signed"]["negative_assignments"] == 1
    assert result["signed"]["agrees_with_record"] == 1
    assert result["signed"].get("deferred", 0) == 0
