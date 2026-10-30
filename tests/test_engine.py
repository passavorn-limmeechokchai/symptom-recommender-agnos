import pytest

from symptom_rec.engine.engine import Engine
from symptom_rec.engine.facts import Facts, holds
from symptom_rec.engine.memory import Memory
from symptom_rec.evaluation import check_vignette
from symptom_rec.kb.loader import load_kb, load_vignettes

# Expectations of the vignettes that are safety rules: they must hold with or without data.
SAFETY = ["ask_first", "not_ask_first", "urgent", "never_recommend", "never_in_ddx", "switch_chief_complaint",
          "duration_categories"]


@pytest.fixture(scope="module")
def kb():
    return load_kb()


@pytest.fixture(scope="module")
def engine(kb):
    return Engine(kb)   # no data learned: every disease starts equal


def make_facts(kb, chief_complaint, present=(), absent=(), hpi=None, sex="female", age=40):
    return Facts(kb, sex, age, chief_complaint, list(present), list(absent), hpi or {})


def ddx_ids(result):
    ids = []
    for disease, _ in result.ddx:
        ids.append(disease)
    return ids


def probability(result, disease):
    for d, p in result.ddx:
        if d == disease:
            return p
    return 0.0


# ── facts and conditions ───────────────────────────────────────────────────
def test_an_answer_names_a_specific_form(kb):
    facts = make_facts(kb, "cough", hpi={"cough.character": "dry"})
    assert "dry_cough" in facts.present


def test_a_specific_form_answers_its_parents_question(kb):
    facts = make_facts(kb, "cough", present=["dry_cough"])
    assert facts.hpi["cough.character"] == ["dry"]


def test_numbers_become_categories(kb):
    facts = make_facts(kb, "cough", hpi={"cough.duration": 1})   # cough.duration is in weeks
    assert facts.hpi["cough.duration"] == ["acute"]


def test_conditions(kb):
    assert holds({"cough.duration": "acute"}, make_facts(kb, "cough")) is None     # not answered yet
    assert holds({"abdominal_pain.location": "unknown"}, make_facts(kb, "abdominal_pain")) is True
    assert holds({"present": ["cough"]}, make_facts(kb, "cough", present=["dry_cough"])) is True
    assert holds({"sex": "male"}, make_facts(kb, "cough")) is False


# ── the DDx ────────────────────────────────────────────────────────────────
def test_hpi_answer_narrows_the_ddx(engine, kb):
    unknown = engine.run(make_facts(kb, "cough"))
    acute = engine.run(make_facts(kb, "cough", hpi={"cough.duration": 1}))
    assert len(acute.ddx) < len(unknown.ddx)
    assert "common_cold" in ddx_ids(acute)
    assert "tuberculosis" not in ddx_ids(acute)


def test_evidence_moves_the_ddx(engine, kb):
    without = engine.run(make_facts(kb, "cough", hpi={"cough.duration": 5}))
    with_drug = engine.run(make_facts(kb, "cough", present=["ace_inhibitor"], hpi={"cough.duration": 5}))
    assert probability(with_drug, "drug_induced_cough") > probability(without, "drug_induced_cough")


# ── what to ask ────────────────────────────────────────────────────────────
def test_red_flags_follow_the_answers(engine, kb):
    acute = engine.run(make_facts(kb, "cough", hpi={"cough.duration": 1}))
    assert "hemoptysis" in acute.red_flag_ids()
    assert "heavy_smoking" not in acute.red_flag_ids()    # asked for chronic cough only


def test_red_flags_are_asked_while_the_answer_is_not_known(engine, kb):
    result = engine.run(make_facts(kb, "cough"))
    assert "dyspnea" in result.red_flag_ids()            # acute cough gate
    assert "heavy_smoking" in result.red_flag_ids()      # chronic cough must_ask


def test_gate_answered_yes_is_urgent(engine, kb):
    result = engine.run(make_facts(kb, "abdominal_pain", present=["cold_extremities"]))
    assert result.urgent == ["unstable"]


def test_some_symptoms_are_urgent_whatever_the_complaint(engine, kb):
    result = engine.run(make_facts(kb, "cough", present=["suicidal_ideation"]))
    assert "suicidal_ideation" in result.urgent


def test_nothing_for_the_other_sex(engine, kb):
    result = engine.run(make_facts(kb, "abdominal_pain", sex="male", age=30))
    ranked = []
    for symptom_id, _ in result.ranked(0.5):
        ranked.append(symptom_id)
    for symptom in kb.symptoms.values():
        if symptom.sex == "female":
            assert symptom.id not in result.red_flag_ids()
            assert symptom.id not in ranked


def test_ranking_leaves_out_what_is_known_or_asked(engine, kb):
    result = engine.run(make_facts(kb, "cough", present=["sputum"], hpi={"cough.duration": 1}))
    ranked = []
    for symptom_id, _ in result.ranked(1.0):
        ranked.append(symptom_id)
    assert "cough" not in ranked
    assert "sputum" not in ranked
    assert "dyspnea" not in ranked    # asked as a red flag instead
    assert result.hpi_questions == ["cough.character", "cough.timing"]


@pytest.mark.parametrize("intent", ["voluntary", "involuntary"])
def test_weight_gain_keeps_its_ddx_after_a_weight_loss_answer(engine, kb, intent):
    facts = make_facts(kb, "weight_gain", hpi={"weight_loss.intent": intent})
    result = engine.run(facts)
    assert {"heart_failure", "hypothyroidism", "primary_obesity"} <= set(ddx_ids(result))
    assert "eating_disorder" not in ddx_ids(result)


@pytest.mark.parametrize(
    "complaint, questions",
    [("weight_loss", ["weight_loss.intent", "weight_loss.appetite"]),
     ("weight_gain", ["weight_gain.speed"])],
)
def test_weight_change_hpi_follows_the_current_complaint(engine, kb, complaint, questions):
    result = engine.run(make_facts(kb, complaint))
    assert result.hpi_questions == questions


@pytest.mark.parametrize("complaint", ["vaginal_bleeding", "heavy_menstrual_bleeding"])
def test_a_complaint_for_the_other_sex_does_not_open_a_chapter(engine, kb, complaint):
    result = engine.run(make_facts(kb, complaint, sex="male"))
    assert result.chapters == []
    assert result.ddx == []
    assert result.red_flags == []
    assert result.hpi_questions == []


def test_global_urgency_still_runs_when_the_complaint_is_for_the_other_sex(engine, kb):
    facts = make_facts(kb, "vaginal_bleeding", sex="male", present=["suicidal_ideation"])
    assert "suicidal_ideation" in engine.run(facts).urgent


def test_pregnancy_is_not_excluded_by_denial_or_sterilization(engine, kb):
    facts = make_facts(
        kb, "vaginal_bleeding", absent=["pregnancy", "missed_period"],
        hpi={"vaginal_bleeding.source": "vaginal", "vaginal_bleeding.contraception": "tubal_ligation"},
    )
    result = engine.run(facts)
    assert {"miscarriage", "molar_pregnancy", "ectopic_pregnancy"} <= set(ddx_ids(result))
    assert "vaginal_bleeding.pregnancy_test" in result.hpi_questions


@pytest.mark.parametrize("current_status", ["none", "light_or_usual", "acute_heavy"])
def test_vaginal_bleeding_urgency_depends_on_current_bleeding(engine, kb, current_status):
    facts = make_facts(
        kb, "heavy_menstrual_bleeding", hpi={
            "vaginal_bleeding.current_status": current_status,
            "vaginal_bleeding.course": "6_months_or_more",
        },
    )
    assert bool(engine.run(facts).urgent) == (current_status == "acute_heavy")


@pytest.mark.parametrize("complaint", ["dark_urine", "cola_colored_urine"])
def test_urine_color_does_not_assert_blood_or_open_jaundice(engine, kb, complaint):
    facts = make_facts(kb, complaint)
    result = engine.run(facts)
    assert "hematuria" not in facts.known
    assert result.chapters == ["g6_hematuria"]
    assert complaint + ".confirmation" in result.hpi_questions


@pytest.mark.parametrize("complaint", ["hematuria", "dark_urine", "cola_colored_urine"])
@pytest.mark.parametrize("confirmation", ["rbc_found", "no_rbc_heme_positive", "no_rbc_heme_negative"])
def test_urine_microscopy_and_heme_select_different_causes(engine, kb, complaint, confirmation):
    facts = make_facts(kb, complaint, hpi={complaint + ".confirmation": confirmation})
    ddx = set(ddx_ids(engine.run(facts)))
    if confirmation == "rbc_found":
        assert {"renal_stone", "bladder_cancer"} <= ddx
        assert not {"hemoglobinuria", "myoglobinuria", "non_heme_pigmenturia"} & ddx
    elif confirmation == "no_rbc_heme_positive":
        assert ddx == {"hemoglobinuria", "myoglobinuria"}
    else:
        assert ddx == {"non_heme_pigmenturia"}


@pytest.mark.parametrize("complaint", ["difficulty_urinating", "unable_to_urinate", "urinary_retention"])
@pytest.mark.parametrize("course", ["acute_painful", "chronic_painless", "acute_on_chronic"])
def test_acute_and_chronic_retention_stay_distinct_across_entries(engine, kb, complaint, course):
    facts = make_facts(kb, complaint, hpi={
        "difficulty_urinating.bladder": "full", "difficulty_urinating.course": course,
    })
    assert bool(engine.run(facts).urgent) == (course != "chronic_painless")


@pytest.mark.parametrize("complaint", ["dysuria", "difficulty_urinating", "hematuria", "flank_pain"])
def test_colic_with_fever_is_urgent_across_urinary_entries(engine, kb, complaint):
    facts = make_facts(kb, complaint, present=["colicky_flank_pain", "fever"])
    assert engine.run(facts).urgent


# ── what is learned from data ──────────────────────────────────────────────
def test_votes_raise_the_diseases_that_explain_the_symptoms(engine, kb):
    memory = Memory(kb, engine.books)
    before = memory.weights("19_cough")
    reading = ("cough", ["cough", "sputum"])
    memory.add(reading)
    memory.add(reading)
    after = memory.weights("19_cough")
    book = engine.books["19_cough"]
    for disease in before:
        if book.explains(disease, "sputum"):
            assert after[disease] > before[disease]
        else:
            assert after[disease] < before[disease]
    memory.add(reading, -1)
    memory.add(reading, -1)
    assert memory.weights("19_cough") == pytest.approx(before)
    assert memory.counts.together["cough"]["sputum"] == 0


def test_data_score_comes_from_the_counts(kb):
    engine = Engine(kb)
    engine.memory.add(("cough", ["cough", "sputum"]))
    engine.memory.add(("cough", ["cough", "label:อาการที่ KB ไม่มี"]))
    result = engine.run(make_facts(kb, "cough"))
    assert result.data == {"sputum": 0.5, "label:อาการที่ KB ไม่มี": 0.5}


# ── the vignettes ──────────────────────────────────────────────────────────
def test_safety_expectations_of_every_vignette(engine):
    for case in load_vignettes():
        checks = check_vignette(engine, case, 0.5)
        for name in SAFETY:
            if name in checks:
                assert checks[name], f"{case.id}: {name}"
