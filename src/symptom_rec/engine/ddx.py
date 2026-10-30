"""The differential diagnosis (DDx) of one chapter for one patient.

    1. Trees: every tree_selection rule that is true adds its trees. If none is true (the HPI is
       not answered yet), every tree is kept.
    2. branch_selection keeps the branches the answer names (dizziness time course, joint count).
    3. Diseases for the other sex are dropped.
    4. Each disease starts at its weight learned from the data: how common it is among users
       (memory.py). Without data every disease starts equal.
    5. Each piece of evidence multiplies the weight by EVIDENCE for the disease, or divides by it
       against:
         a symptom the patient has that the chapter lists for the disease          x 2
         a symptom the patient does not have that the chapter lists for it         / 2
         a branch_hints rule that is true                                          x 2 for the branch
         a gate with if_any_yes: prioritize, answered yes                          x 2 for the branch
         each finding of a history clue the patient has                            x 2 for its diseases
         a clinical score (for example Centor) at its threshold                    x 2 for its disease
"""
from ..kb.trees import all_branches, leaf_lists
from .facts import ancestors, holds

EVIDENCE = 2


class ChapterBook:
    """One chapter, prepared for quick look-ups."""

    def __init__(self, kb, name):
        self.kb = kb
        self.name = name
        self.chapter = kb.chapters[name]

        # disease -> its branch paths, e.g. "acute.non_life_threatening.infection.upper_respiratory"
        self.paths = {}
        for path, diseases in leaf_lists(self.chapter.trees).items():
            for disease in diseases:
                self.paths.setdefault(disease, []).append(path)

        # disease -> the symptom ids the chapter lists for it
        self.listed = {}
        for disease, disease_paths in self.paths.items():
            listed = set()
            for section in (self.chapter.findings, self.chapter.reviewer_findings):
                if disease in section:
                    listed |= set(section[disease].symptoms)
            for branch, finding in self.chapter.branch_findings.items():
                for path in disease_paths:
                    if path == branch or path.startswith(branch + "."):
                        listed |= set(finding.symptoms)
            self.listed[disease] = listed

        # disease -> listed ids plus their more general forms (white_sputum -> sputum)
        self.covers = {}
        for disease, listed in self.listed.items():
            covers = set(listed)
            for symptom_id in listed:
                covers |= set(ancestors(kb, symptom_id))
            self.covers[disease] = covers

        # branch path -> symptom ids its diseases list (for no_symptoms_of)
        self.branch_symptoms = {}
        for path, diseases in all_branches(self.chapter.trees).items():
            symptoms = set()
            for disease in diseases:
                symptoms |= self.listed.get(disease, set())
            self.branch_symptoms[path] = symptoms

    def own_complaint(self):
        """The chapter's own chief complaint and its more general forms. Every disease of the
        chapter has it, so it is not evidence; a second way in (eyelid lump in the eye chapter)
        is, since only some diseases explain it."""
        chief_complaint = self.chapter.chief_complaint
        return set([chief_complaint]) | set(ancestors(self.kb, chief_complaint))

    def explains(self, disease, symptom_id):
        """Does the chapter list this symptom for the disease, as written or more specifically
        (white_sputum explains "sputum") or more generally (joint_pain explains "knee_pain")?"""
        if symptom_id in self.covers[disease]:
            return True
        for ancestor in ancestors(self.kb, symptom_id):
            if ancestor in self.listed[disease]:
                return True
        return False

    def under(self, disease, branch):
        for path in self.paths[disease]:
            if path == branch or path.startswith(branch + "."):
                return True
        return False

    # ── the DDx ────────────────────────────────────────────────────────────
    def chosen_trees(self, facts):
        trees = []
        for rule in self.chapter.tree_selection:
            if holds(rule.when, facts, self.branch_symptoms) is True:
                for tree in rule.use:
                    if tree not in trees:
                        trees.append(tree)
        if not trees:
            trees = list(self.chapter.trees)
        return trees

    def candidates(self, facts):
        """Diseases in the chosen trees and branches, without those for the other sex."""
        trees = self.chosen_trees(facts)
        picked_branches = self.picked_branches(facts)
        diseases = []
        for disease, disease_paths in self.paths.items():
            in_tree = False
            for path in disease_paths:
                if path.split(".")[0] in trees and branch_allowed(path, picked_branches):
                    in_tree = True
            sex = self.kb.diseases[disease].sex
            if in_tree and (sex is None or sex == facts.sex):
                diseases.append(disease)
        return diseases

    def picked_branches(self, facts):
        """Branch names that branch_selection keeps, or None when it keeps everything."""
        selection = self.chapter.branch_selection
        if not selection:
            return None
        answers = facts.hpi.get(selection["attribute"], [])
        picked = []
        for answer in answers:
            picked += selection["map"].get(answer, [])
        return picked or None

    def scores(self, facts, weights):
        """Disease -> score (not normalized): its learned weight times the evidence."""
        scores = {}
        for disease in self.candidates(facts):
            scores[disease] = weights.get(disease, 1.0)

        evidence = []
        for symptom_id in facts.present:
            if symptom_id not in self.own_complaint():
                evidence.append(symptom_id)

        for disease in scores:
            for symptom_id in evidence:
                if self.explains(disease, symptom_id):
                    scores[disease] *= EVIDENCE
            for symptom_id in facts.absent:
                if symptom_id in self.covers[disease]:
                    scores[disease] /= EVIDENCE

        for hint in self.chapter.branch_hints:
            if holds(hint.when, facts, self.branch_symptoms) is True:
                self.boost_branch(scores, hint.boost)
        for gate in self.chapter.gates.values():
            if isinstance(gate.if_any_yes, dict) and any_present(gate.ask, facts):
                if holds(gate.when, facts, self.branch_symptoms) is not False:
                    self.boost_branch(scores, gate.if_any_yes["prioritize"])
        for clue in self.chapter.history_clues:
            for finding in clue.findings:   # each finding counts on its own
                if finding in facts.known:
                    for disease in clue.suggests:
                        if disease in scores:
                            scores[disease] *= EVIDENCE
        for score in self.chapter.scores.values():
            if score.applies_to in scores and score_reached(score, facts):
                scores[score.applies_to] *= EVIDENCE
        return scores

    def boost_branch(self, scores, branch):
        for disease in scores:
            if self.under(disease, branch):
                scores[disease] *= EVIDENCE


def chapters_for(kb, symptom_id):
    """Chapters a chief complaint opens; a specific form opens its parent's (vertigo -> dizziness)."""
    if symptom_id is None:
        return []
    for candidate in [symptom_id] + ancestors(kb, symptom_id):
        if candidate in kb.entry_chapters:
            return kb.entry_chapters[candidate]
    return []


def branch_allowed(path, picked_branches):
    if picked_branches is None:
        return True
    for part in path.split("."):
        if part in picked_branches:
            return True
    return False


def any_present(symptom_ids, facts):
    for symptom_id in symptom_ids:
        if symptom_id in facts.known:
            return True
    return False


def all_present(symptom_ids, facts):
    for symptom_id in symptom_ids:
        if symptom_id not in facts.known:
            return False
    return True


def score_reached(score, facts):
    """A clinical score such as Centor: count the items that hold and compare with the threshold."""
    if holds(score.when, facts) is not True:
        return False
    points = 0
    for item in score.items:
        if "symptom" in item and item["symptom"] in facts.known and number_ok(item.get("hpi"), facts):
            points += 1
        elif "absent" in item and item["absent"] in facts.absent:
            points += 1
        elif "age_between" in item and item["age_between"][0] <= facts.age <= item["age_between"][1]:
            points += 1
    threshold = score.threshold
    if "at_least" in threshold:
        return points >= threshold["at_least"]
    if "above" in threshold:
        return points > threshold["above"]
    return False


def number_ok(conditions, facts):
    """{fever.temperature: {above: 38}}: true if the number given is above 38, or if no number was given."""
    if not conditions:
        return True
    for question, rule in conditions.items():
        value = facts.numbers.get(question)
        if value is None:
            continue
        if "above" in rule and not value > rule["above"]:
            return False
    return True
