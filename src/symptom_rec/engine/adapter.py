"""The engine as the evaluation harness sees it: Agnos names in, ranked KB symptoms out.

The shown symptoms and their answers go through the normalizer, and the engine ranks KB symptoms.
A name the normalizer cannot place is kept as "label:<name>", which only the data part can use.

A hidden symptom counts as found when
    - its KB symptom (at its most general form), or its label, is in the top K, or
    - it is a more general form of a shown symptom: the patient said knee pain, so the app knows
      about joint pain already.
It counts as asked (reported apart, not as found) when it is a red-flag question, or a more
specific form of a shown symptom, which the HPI questions ask about (joint pain -> which joint).
"""
from .facts import Facts, ancestors, root

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class EngineRecommender:
    def __init__(self, engine, normalizer, alpha=None, reference_ids=None):
        self.engine = engine
        self.normalizer = normalizer
        self.alpha = engine.alpha if alpha is None else alpha
        self.readings = {}      # session row -> (chief complaint id, present ids), read once
        self.ids = {}           # name -> KB ids, read once
        self.result = None      # the engine's last answer
        self.shown_ids = []     # KB ids of the last shown symptoms
        self.reference_ids = reference_ids  # frozen targets; independent of the tested normalizer

    # ── learning ───────────────────────────────────────────────────────────
    def reading(self, session):
        if session.row not in self.readings:
            chief_complaint, history = self.normalizer.read_session(session)
            present = list(history.present)
            for name in history.unmapped:
                present.append("label:" + name)
            self.readings[session.row] = (chief_complaint, present)
        return self.readings[session.row]

    def learn(self, sessions):
        for s in sessions:
            self.engine.memory.add(self.reading(s), 1)

    def leave_out(self, session):
        self.engine.memory.add(self.reading(session), -1)

    def put_back(self, session):
        self.engine.memory.add(self.reading(session), 1)

    # ── recommending ───────────────────────────────────────────────────────
    def run(self, shown):
        chief_complaint, history = self.normalizer.read_history(shown["given"], shown["answers"], [])
        labels = []
        for name in history.unmapped:
            labels.append("label:" + name)
        facts = Facts(self.engine.kb, shown["sex"], shown["age"], chief_complaint,
                      history.present, history.absent, history.hpi, labels)
        self.shown_ids = list(history.present)
        self.result = self.engine.run(facts)
        for sid in history.unsure:
            self.result.skip.add(root(self.engine.kb, sid))
        return self.result

    def recommend(self, shown, k):
        ranked = []
        for symptom_id, _ in self.run(shown).ranked(self.alpha)[:k]:
            ranked.append(symptom_id)
        return ranked

    def found(self, hidden, ranked):
        kb = self.engine.kb
        ids = self.target_ids(hidden)
        if not ids:
            return "label:" + hidden in ranked
        for symptom_id in ids:
            if root(kb, symptom_id) in ranked:
                return True
            for shown_id in self.shown_ids:
                if symptom_id == shown_id or symptom_id in ancestors(kb, shown_id):
                    return True   # already told: knee pain was shown, joint pain is hidden
        return False

    def also_asked(self, hidden):
        kb = self.engine.kb
        asked = self.result.red_flag_ids()
        for symptom_id in self.target_ids(hidden):
            if symptom_id in asked or root(kb, symptom_id) in asked:
                return True
            for shown_id in self.shown_ids:
                if shown_id in ancestors(kb, symptom_id):
                    return True   # an HPI question of the shown symptom asks about it
        return False

    def hidden_ids(self, name):
        if name not in self.ids:
            self.ids[name] = self.normalizer.symptom_ids(name)
        return self.ids[name]

    def target_ids(self, name):
        if self.reference_ids is not None:
            return self.reference_ids[name]
        return self.hidden_ids(name)


def tune_alpha(recommender, sessions, make_cases, k=5):
    """HitRate@k for every α in ALPHAS, leave-one-out on the given (training) sessions.

    Each case runs the engine once; only the final ranking depends on α.
    Returns {α: hit rate}.
    """
    copies = {}
    for s in sessions:
        copies.setdefault(s.group(), []).append(s)
    hits = {}
    for alpha in ALPHAS:
        hits[alpha] = 0
    cases_seen = 0
    for s in sessions:
        cases = make_cases(s)
        if not cases:
            continue
        for c in copies[s.group()]:
            recommender.leave_out(c)
        for shown, hidden in cases:
            result = recommender.run(shown)
            cases_seen += 1
            for alpha in ALPHAS:
                top = []
                for symptom_id, _ in result.ranked(alpha)[:k]:
                    top.append(symptom_id)
                if recommender.found(hidden, top):
                    hits[alpha] += 1
        for c in copies[s.group()]:
            recommender.put_back(c)
    rates = {}
    for alpha in ALPHAS:
        rates[alpha] = hits[alpha] / cases_seen
    return rates


def best_alpha(rates):
    """The α with the highest hit rate; on a tie, the one closest to 0.5 (an even mix)."""
    best = None
    for alpha, rate in rates.items():
        if best is None or rate > rates[best] or (rate == rates[best] and abs(alpha - 0.5) < abs(best - 0.5)):
            best = alpha
    return best
