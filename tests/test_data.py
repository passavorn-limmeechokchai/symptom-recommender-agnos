import json
from pathlib import Path

import pytest

from symptom_rec.data.sessions import Session, load_sessions, parse_row
from symptom_rec.data.split import load_split, save_split, split_sessions
from symptom_rec.paths import DATA_DIR

# Made-up sessions in the same format as the Agnos CSV. Row 4 is an exact copy of row 1.
FIXTURE = Path(__file__).parent / "fixtures" / "fake_sessions.csv"


def make_session(row, chief_complaint):
    return Session(
        row=row, sex="female", age=30, typed=[chief_complaint], chief_complaint=chief_complaint,
        answers={chief_complaint: []}, associated=[], absent=[], unsure=[], previous_treatment=None,
    )


def make_sessions(counts):
    """{"cough": 50, "rash": 3} -> 53 sessions with those chief complaints."""
    sessions = []
    for chief_complaint, n in counts.items():
        for _ in range(n):
            sessions.append(make_session(len(sessions), chief_complaint))
    return sessions


# ── reading sessions ───────────────────────────────────────────────────────
def test_row_is_read_as_a_history():
    s = load_sessions(FIXTURE)[0]
    assert s.sex == "male"
    assert s.age == 30
    assert s.typed == ["ไอ", "มีเสมหะ"]
    assert s.chief_complaint == "ไอ"
    assert s.associated == ["เสมหะ"]
    assert s.confirmed() == ["ไอ", "เสมหะ"]
    assert s.answers == {"ไอ": ["ระยะเวลา 3 วัน", "ลักษณะ มีเสมหะ"], "เสมหะ": []}
    assert s.absent == ["ไข้"]
    assert s.previous_treatment == "การรักษาก่อนหน้า ยังไม่ได้รักษา"


def test_exact_copy_is_kept_and_marked():
    sessions = load_sessions(FIXTURE)
    assert len(sessions) == 6
    assert sessions[4].copy_of == 1
    assert sessions[4].group() == sessions[1].group() == 1
    assert sessions[1].copy_of is None


def test_blank_search_terms_are_dropped():
    s = load_sessions(FIXTURE)[2]
    assert s.typed == ["ไข้", "เจ็บคอ"]
    assert s.unsure == ["ไอ"]


def test_previous_treatment_in_english_is_kept_apart():
    s = load_sessions(FIXTURE)[3]
    assert s.confirmed() == ["Headache"]
    assert s.previous_treatment == "Previous treatment none"


def test_symptom_listed_twice_is_kept_once_with_all_answers():
    summary = {"yes_symptoms": [{"text": "ปวดขา", "answers": ["ข้าง ซ้าย"]}, {"text": "ปวดขา", "answers": []}]}
    row = {"gender": "male", "age": "40", "summary": json.dumps(summary, ensure_ascii=False),
           "search_term": "ปวดขา"}
    s = parse_row(0, row)
    assert s.confirmed() == ["ปวดขา"]
    assert s.answers == {"ปวดขา": ["ข้าง ซ้าย"]}


def test_missing_answers_and_missing_treatment_are_allowed():
    s = load_sessions(FIXTURE)[5]
    assert s.answers == {"เวียนศีรษะ": []}
    assert s.previous_treatment is None


# ── train/test split ───────────────────────────────────────────────────────
def test_split_is_stratified_by_chief_complaint():
    sessions = make_sessions({"cough": 50, "fever": 20, "rash": 3, "itch": 2})
    train, test = split_sessions(sessions)
    test_ccs = [sessions[row].chief_complaint for row in test]
    assert test_ccs.count("cough") == 10
    assert test_ccs.count("fever") == 4
    # rash and itch are pooled into one group of 5, so one of them goes to test
    assert test_ccs.count("rash") + test_ccs.count("itch") == 1
    assert len(train) + len(test) == 75


def test_split_is_repeatable_and_does_not_overlap():
    sessions = make_sessions({"cough": 50, "fever": 20})
    train, test = split_sessions(sessions)
    assert split_sessions(sessions) == (train, test)
    assert set(train) & set(test) == set()
    assert sorted(train + test) == list(range(70))


def test_copies_go_where_their_original_goes():
    sessions = make_sessions({"cough": 50})
    for copy_row in range(50, 60):
        copy = make_session(copy_row, "cough")
        copy.copy_of = copy_row - 50
        sessions.append(copy)
    train, test = split_sessions(sessions)
    for copy in sessions[50:]:
        assert (copy.row in test) == (copy.copy_of in test)
    # the split itself is made on the 50 originals
    assert len([row for row in test if row < 50]) == 10


def test_split_file_round_trip(tmp_path):
    path = tmp_path / "build" / "split.json"
    save_split(path, [0, 2, 3], [1])
    assert load_split(path) == ([0, 2, 3], [1])


# ── the real data ──────────────────────────────────────────────────────────
def test_real_data_reads_and_splits():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        pytest.skip("Agnos data not present (it is never committed)")
    sessions = load_sessions(csv_files[0])
    for s in sessions:
        assert s.chief_complaint is not None
        assert s.previous_treatment is not None
    train, test = split_sessions(sessions)
    assert 0.18 < len(test) / len(sessions) < 0.22
