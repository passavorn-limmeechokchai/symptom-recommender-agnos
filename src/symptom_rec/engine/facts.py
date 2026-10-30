"""What is known about one patient, in KB ids, and how a chapter's `when` conditions are read."""
from ..normalize.text import categories_for


class Facts:
    def __init__(self, kb, sex, age, chief_complaint, present, absent, hpi, labels=None):
        self.kb = kb
        self.sex = sex
        self.age = age
        self.chief_complaint = chief_complaint    # symptom id, or None
        self.absent = set(absent)
        # symptoms the knowledge base does not have, by their name in the data ("label:<name>");
        # only the data part of the engine can use them
        self.labels = labels or []

        # "<symptom>.<question>" -> answers (categories or options); numbers are kept too
        self.hpi = {}
        self.numbers = {}
        for question, value in hpi.items():
            self.add_answer(question, value)

        # Symptoms the patient has: as told, plus specific forms their answers name.
        self.present = []
        for symptom_id in [chief_complaint] + list(present):
            if symptom_id and symptom_id not in self.present:
                self.present.append(symptom_id)
        for symptom_id in list(self.present):
            symptom = kb.symptoms[symptom_id]
            # a specific form answers its parent's question: dry_cough -> cough.character = dry
            if symptom.parent and symptom.refines:
                for question, answer in symptom.refines.items():
                    self.add_answer(symptom.parent + "." + question, answer)
        targets = list(self.present)
        for symptom_id in self.present:
            for ancestor in ancestors(kb, symptom_id):
                if ancestor not in targets:
                    targets.append(ancestor)
        for symptom_id in targets:
            # an answer names a specific form: cough.character = dry -> dry_cough
            # include the parent's other forms when the patient started with a specific one
            for child_id in kb.children.get(symptom_id, []):
                if self.answers_name(kb.symptoms[child_id]) and child_id not in self.present:
                    self.present.append(child_id)

        # present symptoms and every more general form of them, for conditions such as
        # {present: [cough]} when the patient said "dry cough"
        self.known = set()
        for symptom_id in self.present:
            self.known.add(symptom_id)
            for ancestor in ancestors(kb, symptom_id):
                self.known.add(ancestor)

    def add_answer(self, question, value):
        if value is None:
            value = "unknown"   # a vignette's null: the patient does not know
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self.numbers[question] = value
            spec = self.kb.questions.get(question)
            if spec is not None and spec.categories:
                answers = categories_for(spec, value, value)
            else:
                answers = [str(value)]
        elif isinstance(value, list):
            answers = value
        else:
            answers = [value]
        known = self.hpi.setdefault(question, [])
        for answer in answers:
            if answer not in known:
                known.append(answer)

    def answers_name(self, child):
        """Do the HPI answers name this specific form? (all of its `refines` answered that way)"""
        if not child.refines:
            return False
        for question, answer in child.refines.items():
            if answer not in self.hpi.get(child.parent + "." + question, []):
                return False
        return True


def holds(when, facts, branch_symptoms=None):
    """Is a `when` condition true for this patient?

    True or False, or None when it depends on an HPI question that has no answer yet. A question
    with no answer counts as "unknown", so a rule written for "unknown" is true.
    `branch_symptoms` (branch path -> symptom ids its diseases list) is needed for no_symptoms_of.
    """
    if when == "always":
        return True
    result = True
    for key, value in when.items():
        answer = check(key, value, facts, branch_symptoms)
        if answer is False:
            return False
        if answer is None:
            result = None
    return result


def check(key, value, facts, branch_symptoms):
    if key == "present":
        for symptom_id in value:
            if symptom_id not in facts.known:
                return False
        return True
    if key == "absent":
        for symptom_id in value:
            if symptom_id not in facts.absent:
                return False
        return True
    if key == "sex":
        return facts.sex == value
    if key == "age_above":
        return facts.age > value
    if key == "age_below":
        return facts.age < value
    if key == "age_between":
        return value[0] <= facts.age <= value[1]
    if key == "no_symptoms_of":
        for symptom_id in branch_symptoms.get(value, set()):
            if symptom_id in facts.known:
                return False
        return True

    # "<symptom>.<question>": answer, or a list of answers any of which will do
    wanted = value if isinstance(value, list) else [value]
    answers = facts.hpi.get(key)
    if answers is None or answers == ["unknown"]:
        # not answered, or "do not know": a rule written for "unknown" is true, others undecided
        if "unknown" in wanted:
            return True
        return None
    for answer in wanted:
        if answer in answers:
            return True
    return False


def ancestors(kb, symptom_id):
    """The more general forms of a symptom, nearest first: knee_pain -> [joint_pain]."""
    found = []
    parent = kb.symptoms[symptom_id].parent
    while parent:
        found.append(parent)
        parent = kb.symptoms[parent].parent
    return found


def root(kb, symptom_id):
    """The most general form of a symptom; a "label:<name>" (not in the KB) is its own root."""
    if symptom_id not in kb.symptoms:
        return symptom_id
    found = ancestors(kb, symptom_id)
    if found:
        return found[-1]
    return symptom_id
