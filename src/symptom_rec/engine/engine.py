"""The engine: from what the patient has told so far to what to ask next.

run(facts) gives a Result with:
    urgent            gates answered yes: the patient should be seen now
    red_flags         questions the book says to ask first (gates and must_ask), yes/no
    hpi_questions     HPI questions of the chief complaint not answered yet
    ddx               diseases with probabilities, from the chief complaint's chapter(s) (ddx.py)
    ranked(α)         every other symptom, most likely first:

                          score = α x book_score + (1 - α) x data_score

                      book_score   how likely the DDx makes the symptom: the sum of the
                                   probabilities of the diseases that list it
                      data_score   how often it came together with the patient's symptoms in
                                   the training data (memory.py)
                      Each is divided by its largest value, so both run from 0 to 1. Symptoms
                      neither part scores follow, most common first, so the list is complete.

Symptoms are recommended at their most general form ("sputum", not "white sputum"); the specific
forms are HPI answers, asked once the patient says yes.
"""
from .ddx import ChapterBook, any_present, chapters_for
from .facts import ancestors, holds, root
from .memory import Memory


class Result:
    def __init__(self):
        self.chief_complaint = None
        self.chapters = []
        self.urgent = []            # names of gates answered yes
        self.switch_to = None       # chief complaint to switch to (confirm.if_against), or None
        self.red_flags = []         # [{"symptom", "source", "answered"}]
        self.hpi_questions = []     # "<symptom>.<question>"
        self.ddx = []               # [(disease, probability)], most likely first
        self.book = {}              # symptom -> book_score
        self.book_reasons = {}      # symptom -> diseases that list it, most likely first
        self.data = {}              # symptom -> data_score
        self.popular = []           # every symptom in the data, most common first
        self.skip = set()           # symptoms not to recommend (answered, red flags, other sex)

    def red_flag_ids(self):
        ids = []
        for flag in self.red_flags:
            ids.append(flag["symptom"])
        return ids

    def ranked(self, alpha):
        """[(symptom, score)], every candidate, best first."""
        book_max = max(self.book.values(), default=0) or 1
        data_max = max(self.data.values(), default=0) or 1
        scored = []
        for symptom_id in set(self.book) | set(self.data):
            if symptom_id in self.skip:
                continue
            score = alpha * self.book.get(symptom_id, 0) / book_max
            score += (1 - alpha) * self.data.get(symptom_id, 0) / data_max
            scored.append((-score, symptom_id))
        scored.sort()   # highest score first; ties by name, so results never change
        ranked = []
        listed = set()
        for negative_score, symptom_id in scored:
            ranked.append((symptom_id, -negative_score))
            listed.add(symptom_id)
        for symptom_id in self.popular:
            if symptom_id not in listed and symptom_id not in self.skip:
                ranked.append((symptom_id, 0.0))
                listed.add(symptom_id)
        return ranked


class Engine:
    def __init__(self, kb, memory=None, alpha=0.5):
        self.kb = kb
        self.books = {}
        for name in kb.chapters:
            self.books[name] = ChapterBook(kb, name)
        self.memory = memory or Memory(kb, self.books)
        self.alpha = alpha

    def run(self, facts):
        kb = self.kb
        result = Result()
        result.chief_complaint = facts.chief_complaint
        complaint = kb.symptoms.get(facts.chief_complaint)
        applicable = complaint is None or complaint.sex is None or complaint.sex == facts.sex
        if applicable:
            result.chapters = chapters_for(kb, facts.chief_complaint)

        for name in result.chapters:
            self.ask_first(result, self.books[name], facts)
        # urgent whatever the complaint (symptoms.yaml `urgent`), e.g. thoughts of self-harm
        for symptom_id in sorted(facts.known):
            if kb.symptoms[symptom_id].urgent:
                result.urgent.append(symptom_id)
        if applicable:
            result.hpi_questions = self.unanswered_hpi(facts)

        # DDx: every chapter the chief complaint opens; a disease in two chapters keeps its higher score
        scores = {}
        for name in result.chapters:
            for disease, score in self.books[name].scores(facts, self.memory.weights(name)).items():
                scores[disease] = max(scores.get(disease, 0), score)
        total = sum(scores.values()) or 1
        ddx = []
        for disease, score in scores.items():
            ddx.append((-score / total, disease))
        ddx.sort()
        for negative, disease in ddx:
            result.ddx.append((disease, -negative))

        # book_score: the DDx probability of each symptom, at its most general form
        for disease, probability in result.ddx:
            disease_roots = set()
            for name in result.chapters:
                for symptom_id in self.books[name].listed.get(disease, set()):
                    disease_roots.add(root(kb, symptom_id))
            for symptom_id in disease_roots:
                result.book[symptom_id] = result.book.get(symptom_id, 0) + probability
                result.book_reasons.setdefault(symptom_id, []).append(disease)

        # data_score: how often each symptom came with the patient's symptoms
        counts = self.memory.counts
        given = set(facts.labels)
        for symptom_id in facts.present:
            given.add(root(kb, symptom_id))
        for symptom_id in given:
            with_it = counts.sessions_with.get(symptom_id, 0)
            if with_it <= 0:
                continue
            for other, together in counts.together.get(symptom_id, {}).items():
                if together > 0:
                    result.data[other] = result.data.get(other, 0) + together / with_it
        popular = []
        for symptom_id, n in counts.sessions_with.items():
            if n > 0:
                popular.append((-n, symptom_id))
        popular.sort()
        for _, symptom_id in popular:
            result.popular.append(symptom_id)

        # never recommend what is answered, asked as a red flag, or for the other sex
        result.skip = set(given) | set(result.red_flag_ids())
        for symptom_id in facts.absent:
            result.skip.add(root(kb, symptom_id))
        for symptom in kb.symptoms.values():
            if symptom.sex and symptom.sex != facts.sex:
                result.skip.add(symptom.id)
        return result

    def ask_first(self, result, book, facts):
        """Gates and must_ask: the questions to ask before anything else, and urgent gates."""
        chapter = book.chapter
        for gate_name, gate in chapter.gates.items():
            if holds(gate.when, facts, book.branch_symptoms) is False:
                continue   # asked when the condition holds or is not known yet
            self.add_red_flags(result, gate.ask, gate_name, facts)
            if gate.if_any_yes == "urgent" and any_present(gate.ask, facts):
                result.urgent.append(gate_name)
        for n, rule in enumerate(chapter.must_ask):
            if holds(rule.when, facts, book.branch_symptoms) is not False:
                self.add_red_flags(result, rule.ask, f"{book.name}.must_ask[{n}]", facts)

        confirm = chapter.confirm.get(facts.chief_complaint)
        if confirm and confirm.if_against.get("switch_chief_complaint"):
            if count_evidence(confirm, facts, "against") > count_evidence(confirm, facts, "supports"):
                result.switch_to = confirm.if_against["switch_chief_complaint"]

    def add_red_flags(self, result, symptom_ids, source, facts):
        for symptom_id in symptom_ids:
            sex = self.kb.symptoms[symptom_id].sex
            if sex and sex != facts.sex:
                continue
            if symptom_id in result.red_flag_ids():
                continue
            answered = symptom_id in facts.known or symptom_id in facts.absent
            result.red_flags.append({"symptom": symptom_id, "source": source, "answered": answered})

    def unanswered_hpi(self, facts):
        if facts.chief_complaint is None:
            return []
        targets = [facts.chief_complaint] + ancestors(self.kb, facts.chief_complaint)
        questions = []
        for question in self.kb.questions:
            symptom_id = question.split(".")[0]
            if symptom_id in targets and question not in facts.hpi:
                spec = self.kb.questions[question]
                if spec.sex is None or spec.sex == facts.sex:
                    questions.append(question)
        return questions


def count_evidence(confirm, facts, side):
    """How many `supports` (or `against`) findings of a confirm block the patient has."""
    n = 0
    for symptom_id in getattr(confirm, side):
        if symptom_id in facts.known:
            n += 1
    for question, sides in confirm.hpi.items():
        for answer in sides.get(side, []):
            if answer in facts.hpi.get(question, []):
                n += 1
    return n
