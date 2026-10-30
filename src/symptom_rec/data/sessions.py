"""Read the Agnos CSV into sessions shaped like a clinical history.

One CSV row is one patient session:
    gender, age    -> sex, age
    search_term    -> what the patient typed, comma separated
    summary        -> JSON with the symptoms the patient confirmed (yes_symptoms),
                      denied (no_symptoms) or was unsure of (idk_symptoms)

Read as a history:
    first confirmed symptom      -> chief complaint
    answers of each symptom      -> HPI (duration, location, character, ...)
    other confirmed symptoms     -> associated symptoms
    denied symptoms              -> pertinent negatives
    the "previous treatment" item at the end -> kept apart; it is asked of everyone

Names and answers are kept exactly as they are in the data. Mapping them to knowledge-base
ids is the normalizer's job, so how well that works can be measured.
"""
import csv
import json

from ..paths import DATA_DIR

# The fixed last item of every session, in Thai or in English.
PREVIOUS_TREATMENT = ["การรักษาก่อนหน้า", "Previous treatment"]


class Session:
    def __init__(self, row, sex, age, typed, chief_complaint, answers, associated, absent, unsure,
                 previous_treatment, copy_of=None):
        self.row = row                                # row number in the CSV, counting from 0
        self.sex = sex                                # "male" or "female"
        self.age = age
        self.typed = typed                            # search terms the patient typed, in order
        self.chief_complaint = chief_complaint        # first confirmed symptom, or None
        self.answers = answers                        # symptom -> its answers, e.g. {"ไอ": ["ระยะเวลา 2 วัน"]}
        self.associated = associated                  # the other confirmed symptoms, in order
        self.absent = absent                          # symptoms the patient said they do not have
        self.unsure = unsure                          # symptoms the patient was not sure about
        self.previous_treatment = previous_treatment  # the answer to that item, or None
        self.copy_of = copy_of                        # row number of an earlier identical row, or None

    def confirmed(self):
        """Chief complaint followed by the associated symptoms."""
        if self.chief_complaint is None:
            return []
        return [self.chief_complaint] + self.associated

    def group(self):
        """Sessions that are exact copies of each other share the same group number."""
        if self.copy_of is None:
            return self.row
        return self.copy_of


def parse_row(row_number, row):
    """One CSV row (a dict of column -> text) -> Session."""
    summary = json.loads(row["summary"])

    typed = []
    for term in row["search_term"].split(","):
        if term.strip():
            typed.append(term.strip())

    confirmed = []
    answers = {}
    previous_treatment = None
    for item in summary.get("yes_symptoms", []):
        name = item["text"]
        item_answers = item.get("answers", [])
        if name in PREVIOUS_TREATMENT:
            if item_answers:
                previous_treatment = item_answers[0]
            continue
        # A few sessions list the same symptom twice; keep it once, with the answers of both.
        if name not in confirmed:
            confirmed.append(name)
            answers[name] = []
        answers[name] += item_answers

    absent = []
    for item in summary.get("no_symptoms", []):
        absent.append(item["text"])

    unsure = []
    for item in summary.get("idk_symptoms", []):
        unsure.append(item["text"])

    if confirmed:
        chief_complaint = confirmed[0]
        associated = confirmed[1:]
    else:
        chief_complaint = None
        associated = []

    return Session(
        row=row_number,
        sex=row["gender"],
        age=int(row["age"]),
        typed=typed,
        chief_complaint=chief_complaint,
        answers=answers,
        associated=associated,
        absent=absent,
        unsure=unsure,
        previous_treatment=previous_treatment,
    )


def load_sessions(csv_path):
    """Every row of the CSV as a Session.

    The data has no patient id, so a row that is an exact copy of an earlier row may be a
    double submission or another patient who gave the same answers. It is kept, so no real
    patient is lost from the counts, and marked with copy_of, so the train/test split and
    leave-one-out evaluation can keep copies together.
    """
    sessions = []
    first_row = {}  # the whole row's text -> the row number where it first appeared
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f)):
            session = parse_row(row_number, row)
            whole_row = (row["gender"], row["age"], row["summary"], row["search_term"])
            if whole_row in first_row:
                session.copy_of = first_row[whole_row]
            else:
                first_row[whole_row] = row_number
            sessions.append(session)
    return sessions


def find_csv(folder=DATA_DIR):
    """The one CSV file in data/raw/. Stops with a clear message if there is none or several."""
    files = sorted(folder.glob("*.csv"))
    if len(files) != 1:
        raise SystemExit(f"expected exactly one CSV in {folder}, found {len(files)}")
    return files[0]
