"""Two recommenders that use only the Agnos data. They are the baselines the engine must beat.

Popularity:    the symptoms that most sessions confirmed, whatever the patient said.
Co-occurrence: the symptoms most often confirmed together with the ones the patient gave.

Both only count sessions. A session can therefore be taken out of the counts and put back,
which leave-one-out evaluation needs (see evaluation.py).

Every recommender in this project has the same methods (see evaluation.py):
    recommend(shown, k)    -> up to k symptom names, best first; shown["given"] are the symptoms
                              the patient gave
    found(hidden, ranked)  -> is the hidden symptom among the ranked ones
    also_asked(hidden)     -> was it asked as a red-flag question (never, for these two)
    leave_out(session)     -> forget this session's counts
    put_back(session)      -> count it again
"""


class SymptomCounts:
    """How many sessions confirmed each symptom, and each pair of symptoms."""

    def __init__(self, sessions):
        self.sessions_with = {}  # symptom -> number of sessions that confirmed it
        self.together = {}       # symptom a -> {symptom b: number of sessions that confirmed both}
        for s in sessions:
            self.add(s.confirmed(), 1)

    def add(self, symptoms, amount):
        """Count one session's symptoms (each listed once); amount -1 takes them out again."""
        for a in symptoms:
            self.sessions_with[a] = self.sessions_with.get(a, 0) + amount
            neighbours = self.together.setdefault(a, {})
            for b in symptoms:
                if b != a:
                    neighbours[b] = neighbours.get(b, 0) + amount


class CountingRecommender:
    """What both baselines share: the counts, and taking a session out and back in."""

    def __init__(self, sessions):
        self.counts = SymptomCounts(sessions)

    def leave_out(self, session):
        self.counts.add(session.confirmed(), -1)

    def put_back(self, session):
        self.counts.add(session.confirmed(), 1)

    def found(self, hidden, ranked):
        return hidden in ranked

    def also_asked(self, hidden):
        return False


class Popularity(CountingRecommender):
    def recommend(self, shown, k):
        return most_popular(self.counts, shown["given"], k)


class Cooccurrence(CountingRecommender):
    """score(symptom) = sum over the given symptoms g of
                        (sessions that confirmed g and symptom) / (sessions that confirmed g)

    If fewer than k symptoms score above 0 (a rare chief complaint, for example),
    the remaining places are filled with the most popular symptoms.
    """

    def recommend(self, shown, k):
        given = shown["given"]
        scores = {}
        for g in given:
            n_g = self.counts.sessions_with.get(g, 0)
            if n_g == 0:
                continue
            for name, n_both in self.counts.together.get(g, {}).items():
                if name not in given and n_both > 0:
                    scores[name] = scores.get(name, 0) + n_both / n_g
        ranked = top_k(scores, k)
        for name in most_popular(self.counts, given, k):
            if len(ranked) == k:
                break
            if name not in ranked:
                ranked.append(name)
        return ranked


def most_popular(counts, given, k):
    """The k symptoms confirmed by the most sessions, leaving out those already given."""
    scores = {}
    for name, n in counts.sessions_with.items():
        if name not in given and n > 0:
            scores[name] = n
    return top_k(scores, k)


def top_k(scores, k):
    """The k names with the highest score. Ties are broken by name, so results never change."""
    ordered = []
    for name, score in scores.items():
        ordered.append((-score, name))
    ordered.sort()
    best = []
    for _, name in ordered[:k]:
        best.append(name)
    return best
