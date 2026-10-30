"""Learn extra names for symptoms from what patients typed in the training set.

For each typed term the rules cannot place, or place in a different symptom family from the
record (for example "หายใจทางจมูกไม่ได้"), look at the training sessions where it was typed:

    n = sessions where the term was typed
    m = of those, sessions whose recorded symptoms include symptom s
        (not counting symptoms that another typed term of the same session already explains)

    learn  term -> s   when n >= MIN_SESSIONS and m / n >= MIN_SHARE, for the most common s

A phrase the lexicon marks as ambiguous is never learned. A term is never learned into the
family the rules already read it as: "ไอแห้งๆ" stays dry cough rather than the more general
cough, and a term the rules read as two symptoms keeps both.

Only the training set is used, and the result is written to build/, never into kb/: the
knowledge base stays free of the data's wording, and the test set measures how well the
learned terms carry over to new patients.
"""
import json

from .text import key

MIN_SESSIONS = 2
MIN_SHARE = 0.6


def learn_aliases(normalizer, sessions):
    """{typed term key: symptom id} learned from the sessions."""
    typed_in = {}     # term key -> number of sessions where it was typed
    recorded = {}     # term key -> {symptom id: number of those sessions that recorded it}
    for s in sessions:
        recorded_ids = []
        for name in s.confirmed():
            for symptom_id in normalizer.symptom_ids(name):
                if symptom_id not in recorded_ids:
                    recorded_ids.append(symptom_id)

        read_as = {}  # term key -> what the rules read it as
        for term in s.typed:
            if any(m.state != "present" for m in normalizer.read_symptoms(term)):
                continue  # a positive-only alias must never erase uncertainty, negation or past context
            read_as[key(term)] = normalizer.symptom_ids(term)

        for term, ids in read_as.items():
            if ids and all_in(ids, recorded_ids):
                continue   # the rules already agree with the record
            explained_by_others = []
            for other, other_ids in read_as.items():
                if other != term:
                    explained_by_others += other_ids
            typed_in[term] = typed_in.get(term, 0) + 1
            counts = recorded.setdefault(term, {})
            for symptom_id in recorded_ids:
                if symptom_id not in explained_by_others:
                    counts[symptom_id] = counts.get(symptom_id, 0) + 1

    aliases = {}
    for term, n in typed_in.items():
        if n < MIN_SESSIONS or not recorded[term] or term in normalizer.ambiguous:
            continue
        rule_roots = []
        for symptom_id in normalizer.symptom_ids(term):
            rule_roots.append(normalizer.root(symptom_id))
        best = None
        for symptom_id, m in recorded[term].items():
            if best is None or m > recorded[term][best] or (m == recorded[term][best] and symptom_id < best):
                best = symptom_id
        if recorded[term][best] / n >= MIN_SHARE and normalizer.root(best) not in rule_roots:
            aliases[term] = best
    return aliases


def all_in(ids, other_ids):
    for symptom_id in ids:
        if symptom_id not in other_ids:
            return False
    return True


def save_aliases(path, aliases):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aliases, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


def load_aliases(path):
    """The learned terms, or {} if build/ has none yet."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
