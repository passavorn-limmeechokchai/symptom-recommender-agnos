"""What the engine learns from the training sessions, in KB ids.

counts   how many sessions had each symptom, and each pair of symptoms. This is the data_score,
         like the co-occurrence baseline but on KB symptoms, so "Fever" and "เป็นไข้" count together.
         Symptoms are counted at their most general form (white_sputum counts as sputum). A symptom
         the KB does not have is counted by its name in the data, as "label:<name>", so the data
         part still works where the book has nothing.

votes    how common each disease of a chapter is among users. Each session gives one vote to the
         diseases of its chief complaint's chapter, shared by how many of its other symptoms the
         chapter lists for each disease:
             cough + sputum + runny nose; common cold lists both, bronchitis lists sputum
             -> common cold gets 2/3 of the vote, bronchitis 1/3
         weight = (votes + VOTE_PRIOR) / (all votes in the chapter + VOTE_PRIOR x number of diseases)
         VOTE_PRIOR is the number of votes every disease starts with. It keeps every disease
         possible, and decides how far the data's few votes per chapter may outweigh the book's
         own clues: with 1, they overruled textbook cases such as "ACE inhibitor -> drug-induced
         cough" (vignettes 61/68 in the top 3); with 3, the vignettes pass (66/68) for about one
         point of hit rate on the training set.

A session can be taken out and put back (leave-one-out), and everything can be saved to
build/memory.json so the API does not need the data.
"""
import json

from ..baselines.counting import SymptomCounts
from .ddx import chapters_for
from .facts import ancestors, root

VOTE_PRIOR = 3


class Memory:
    def __init__(self, kb, books):
        self.kb = kb
        self.books = books
        self.counts = SymptomCounts([])
        self.votes = {}     # chapter -> {disease: votes}

    def add(self, reading, amount=1):
        """Count one session, read as (chief complaint id, present symptom ids and "label:<name>"s);
        amount -1 takes it out."""
        chief_complaint, present = reading
        roots = []
        for symptom_id in present:
            if root(self.kb, symptom_id) not in roots:
                roots.append(root(self.kb, symptom_id))
        self.counts.add(roots, amount)

        if chief_complaint is None:
            return
        # the chief complaint chose the chapter, so only the other symptoms vote
        skip = set([chief_complaint]) | set(ancestors(self.kb, chief_complaint))
        others = []
        for symptom_id in present:
            if symptom_id in self.kb.symptoms and symptom_id not in skip:
                others.append(symptom_id)
        for name in chapters_for(self.kb, chief_complaint):
            book = self.books[name]
            matches = {}
            for disease in book.paths:
                matches[disease] = 0
                for symptom_id in others:
                    if book.explains(disease, symptom_id):
                        matches[disease] += 1
            total = sum(matches.values())
            if total == 0:
                continue
            votes = self.votes.setdefault(name, {})
            for disease, n in matches.items():
                votes[disease] = votes.get(disease, 0) + amount * n / total

    def weights(self, name):
        """Disease -> learned weight for one chapter (see the module docstring)."""
        diseases = self.books[name].paths
        votes = self.votes.get(name, {})
        total = sum(votes.values())
        weights = {}
        for disease in diseases:
            weights[disease] = (votes.get(disease, 0) + VOTE_PRIOR) / (total + VOTE_PRIOR * len(diseases))
        return weights

    def save(self, path):
        content = {
            "sessions_with": self.counts.sessions_with,
            "together": self.counts.together,
            "votes": self.votes,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")

    def load(self, path):
        content = json.loads(path.read_text(encoding="utf-8"))
        self.counts.sessions_with = content["sessions_with"]
        self.counts.together = content["together"]
        self.votes = content["votes"]
