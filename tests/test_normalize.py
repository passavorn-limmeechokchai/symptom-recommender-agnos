import pytest

from symptom_rec.data.sessions import Session
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.kb.schema import HpiQuestion
from symptom_rec.normalize.learn import learn_aliases
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.normalize.text import categories_for, key, read_duration

# Every input below is made up; none is copied from the Agnos data.


@pytest.fixture(scope="module")
def normalizer():
    return Normalizer(load_kb(), load_lexicon())


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon()


# ── symptoms ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text, expected, layer",
    [
        ("ตัวร้อน", ["fever"], "exact"),                                  # alias
        ("Sore Throat", ["sore_throat"], "exact"),                         # English name
        ("ปวดดดดท้อง", ["abdominal_pain"], "exact"),                       # repeated letters
        ("มีเสมหะมาก", ["copious_sputum"], "exact"),                       # a name that ends like a filler
        ("รู้สึกคลื่นไส้", ["nausea"], "fillers"),
        ("ท้องแสบ", ["burning_abdominal_pain"], "body_part"),
        ("ปวดหัวเข่า", ["knee_pain"], "body_part"),
        ("ปวดคอ", ["neck_pain"], "body_part"),                            # ปวด at the neck
        ("เจ็บในลำคอ", ["sore_throat"], "body_part"),                     # เจ็บ inside the throat
        ("อาเจียนแสบท้อง", ["vomiting", "burning_abdominal_pain"], "split"),
        ("ปวดหัว ตัวร้อน", ["headache", "fever"], "split"),
        ("ผื่นคัน", ["rash", "itch"], "split"),
        ("fever and cough", ["fever", "cough"], "split"),
        ("ปวดหอง", ["abdominal_pain"], "fuzzy"),                           # typo, same Thai sound
        ("ประวัติเป็นเบาหวาน", ["diabetes"], "fillers"),                    # history item
        ("diabetic (type 2)", ["diabetes"], "exact"),                      # brackets separate
        ("อยากตายมาก", ["suicidal_ideation"], "fillers"),
        ("ฉี่ไม่ออก", ["unable_to_urinate"], "exact"),
        ("ปัสสาวะเล็ด", ["urinary_incontinence"], "exact"),
        ("ไอแล้วปัสสาวะเล็ด", ["stress_urinary_leak"], "exact"),
        ("ปัสสาวะบ่อย", ["urinary_frequency"], "exact"),
        ("ปัสสาวะปริมาณมากกว่าปกติ", ["polyuria"], "exact"),
    ],
)
def test_symptoms(normalizer, text, expected, layer):
    matches = normalizer.read_symptoms(text)
    assert normalizer.symptom_ids(text) == expected
    assert matches[0].layer == layer


def test_negation_marks_the_symptom_absent(normalizer):
    [match] = normalizer.read_symptoms("ไม่มีไข้")
    assert match.symptom == "fever"
    assert match.absent
    assert normalizer.symptom_ids("ไม่มีไข้") == []


def test_burning_throat_leans_to_reflux_but_keeps_sore_throat(normalizer):
    [match] = normalizer.read_symptoms("แสบคอ")
    assert match.symptom == "heartburn"
    assert match.candidates == ["sore_throat"]


def test_ambiguous_phrase_is_asked_back(normalizer):
    [match] = normalizer.read_symptoms("หายใจไม่สะดวก")
    assert match.symptom is None
    assert match.layer == "ambiguous"
    assert match.candidates == ["nasal_congestion", "dyspnea"]


def test_commas_separate_symptoms(normalizer):
    assert normalizer.symptom_ids("ไอ, น้ำมูกไหล") == ["cough", "rhinorrhea"]


def test_more_specific_symptom_replaces_its_parent(normalizer):
    # เวียนศีรษะ (dizziness) + บ้านหมุน (vertigo, a kind of dizziness) -> vertigo only
    assert normalizer.symptom_ids("เวียนศีรษะบ้านหมุน") == ["vertigo"]


def test_split_keeps_symptom_with_extra_words(normalizer):
    matches = normalizer.read_symptoms("ไอตอนตีสอง")
    assert matches[0].symptom == "cough"
    assert matches[0].score < 0.9


def test_word_is_not_cut_into_a_symptom_and_a_non_symptom(normalizer):
    # "ไอศกรีม" (ice cream) is not ไอ (cough) + ศกรีม
    assert normalizer.symptom_ids("กินไอศกรีม") == []


def test_body_part_left_over_only_offers_candidates(normalizer):
    # the lexicon does not say what itching of the back is, so it does not guess plain itch
    [match] = normalizer.read_symptoms("คันที่หลัง")
    assert match.symptom is None
    assert match.candidates == ["itch"]


def test_unknown_text_gives_candidates_not_a_guess(normalizer):
    [match] = normalizer.read_symptoms("ก้อนนูนแข็งที่หลังหู")
    assert match.symptom is None
    assert len(match.candidates) == 3


# ── HPI answers ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "symptom, text, expected",
    [
        ("cough", "ไอมาได้ 5 วัน", {"cough.duration": ["acute"]}),
        ("cough", "ประมาณ 9 สัปดาห์", {"cough.duration": ["chronic", "subacute"]}),   # grey zone 6-10 weeks
        ("cough", "ไอแห้งๆ ตอนกลางคืน", {"cough.character": ["dry"]}),
        ("abdominal_pain", "เป็นมาไม่ถึง 1 วัน", {"abdominal_pain.duration": ["acute"]}),
        ("abdominal_pain", "แถวๆรอบสะดือ", {"abdominal_pain.location": ["periumbilical"]}),
        ("fever", "วัดได้ 38.9 องศา", {"fever.temperature": ["moderate"]}),
        ("dizziness", "รู้สึกบ้านหมุน", {"dizziness.character": ["rotational"]}),
        ("dizziness", "บ้านไม่หมุน", {}),                                                # negated
        ("knee_pain", "สัก 3 วันแล้ว", {"joint_pain.duration": ["acute"]}),            # asked of the parent
    ],
)
def test_answers(normalizer, symptom, text, expected):
    result = normalizer.read_answer(symptom, text)
    for question in result:
        result[question] = sorted(result[question])
    for question in expected:
        expected[question] = sorted(expected[question])
    assert result == expected


# ── whole sessions ─────────────────────────────────────────────────────────
def test_read_session(normalizer):
    session = Session(
        row=0, sex="male", age=40, typed=["ไอ"], chief_complaint="ไอ",
        answers={"ไอ": ["ระยะเวลา 4 วัน"], "มีเสมหะ": [], "ก้อนนูนแข็งที่หลังหู": []},
        associated=["มีเสมหะ", "ก้อนนูนแข็งที่หลังหู"], absent=["ไข้"], unsure=[], previous_treatment=None,
    )
    chief_complaint, history = normalizer.read_session(session)
    assert chief_complaint == "cough"
    assert history.present == ["cough", "sputum"]
    assert history.absent == ["fever"]
    assert history.hpi == {"cough.duration": ["acute"]}
    assert history.unmapped == ["ก้อนนูนแข็งที่หลังหู"]


# ── text helpers ───────────────────────────────────────────────────────────
def test_key():
    assert key("ปวดดดดท้อง") == "ปวดท้อง"
    assert key("ไอแห้งๆ") == "ไอแห้ง"
    assert key("Low-grade fever") == "lowgradefever"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2 สัปดาห์", (14, 14)),
        ("2-4 สัปดาห์", (14, 28)),
        ("ตั้งแต่ 3 วัน ถึง 2 เดือน", (3, 60)),
        ("ไม่เกิน 5 วัน", (0, 5)),
        ("มากกว่า 2 เดือน", (60, float("inf"))),
        ("about 3 days", (3, 3)),
        ("อายุ 40", None),                      # a number without a time unit
    ],
)
def test_read_duration(lexicon, text, expected):
    assert read_duration(text, lexicon) == expected


def test_categories_for():
    question = HpiQuestion(
        ask="how long", unit="weeks",
        categories={"acute": {"below": 3}, "subacute": {"from": 3, "to": 8}, "chronic": {"above": 8}},
        grey_zones=[{"from": 2, "to": 4, "both": ["acute", "subacute"]}],
    )
    assert categories_for(question, 1, 1) == ["acute"]
    assert categories_for(question, 2.5, 2.5) == ["acute", "subacute"]   # grey zone
    assert categories_for(question, 1, 3) == ["acute"]                   # touches 3 but does not cross it
    assert categories_for(question, 1, 5) == ["acute", "subacute"]
    assert categories_for(question, 10, float("inf")) == ["chronic"]


# ── learning from what patients typed ──────────────────────────────────────
def make_session(row, typed, recorded):
    return Session(
        row=row, sex="female", age=30, typed=typed, chief_complaint=recorded[0],
        answers={}, associated=recorded[1:], absent=[], unsure=[], previous_treatment=None,
    )


def test_learn_a_term_the_rules_cannot_place(normalizer):
    sessions = [
        make_session(0, ["ไอ", "หายใจทางจมูกไม่ได้"], ["ไอ", "คัดจมูก"]),
        make_session(1, ["หายใจทางจมูกไม่ได้"], ["คัดจมูก"]),
        make_session(2, ["ปวดแปลบๆที่ตาตุ่ม"], ["ปวดข้อเท้า"]),          # typed in one session only
    ]
    # cough is explained by the typed "ไอ", so only nasal congestion is left for the new term
    assert learn_aliases(normalizer, sessions) == {"หายใจทางจมูกไม่ได้": "nasal_congestion"}


def test_never_learn_into_the_family_the_rules_already_read(normalizer):
    # the rules read "ไอแห้งๆ" as dry cough; the record's plain cough must not replace it
    sessions = [make_session(0, ["ไอแห้งๆ"], ["ไอ"]), make_session(1, ["ไอแห้งๆ"], ["ไอ"])]
    assert learn_aliases(normalizer, sessions) == {}


def test_ambiguous_phrase_is_never_learned(normalizer):
    sessions = [make_session(0, ["หายใจไม่สะดวก"], ["คัดจมูก"]), make_session(1, ["หายใจไม่สะดวก"], ["คัดจมูก"])]
    assert learn_aliases(normalizer, sessions) == {}


def test_learned_terms_are_used(normalizer):
    learned = Normalizer(normalizer.kb, normalizer.lexicon, {key("หายใจทางจมูกไม่ได้"): "nasal_congestion"})
    [match] = learned.read_symptoms("หายใจทางจมูกไม่ได้")
    assert (match.symptom, match.layer) == ("nasal_congestion", "learned")
