"""Measure how often a recommender finds a symptom the patient really had.

Two ways of hiding symptoms, both scored the same way:

    hide one       in a session with a chief complaint and associated symptoms, hide one associated
                   symptom and show the rest. A session with 3 associated symptoms gives 3 cases.
    first step     show only the chief complaint and its HPI answers, and hide every associated
                   symptom: what the app knows when it first recommends. Each hidden symptom is a case.

    HitRate@K = cases where the hidden symptom is in the top K / all cases

The chief complaint is never hidden, because the patient always gives it. A recommender sees the
Agnos names of the shown symptoms and their HPI answers, never the hidden symptom or its answers.
The normalizer is measured separately.

Leave-one-out (used on the training set, for example to choose α)
    Before a training session is tested, it is taken out of the model's counts, together with
    any exact copies of it, and put back afterwards. Otherwise the model would find the hidden
    symptom only because it has already counted this very session.

Bootstrap (used on the test set)
    Sessions are drawn with replacement 1,000 times and the hit rate is recomputed on each draw.
    The middle 95% of those values is the reported interval. Every model uses the same draws,
    so the difference between two models gets an interval as well.

A recommender has: recommend(shown, k) -> up to k symptoms, best first; found(hidden, ranked);
also_asked(hidden) -> was the hidden symptom asked as a red-flag question instead;
leave_out(session) and put_back(session).
"""
import random

from .normalize.text import key

K_VALUES = [3, 5, 8]


def hide_one_cases(session):
    """Every (shown, hidden) pair from one session: one associated symptom hidden, the rest shown."""
    confirmed = session.confirmed()
    cases = []
    for hidden in confirmed[1:]:
        given = []
        for name in confirmed:
            if name != hidden:
                given.append(name)
        cases.append((shown(session, given), hidden))
    return cases


def first_step_cases(session):
    """Only the chief complaint and its answers are shown; every associated symptom is hidden."""
    confirmed = session.confirmed()
    cases = []
    for hidden in confirmed[1:]:
        cases.append((shown(session, confirmed[:1]), hidden))
    return cases


def shown(session, given):
    """What a recommender may see: sex, age, the shown symptoms and their own answers."""
    answers = {}
    for name in given:
        answers[name] = session.answers.get(name, [])
    return {"sex": session.sex, "age": session.age, "given": given, "answers": answers}


def evaluate(model, sessions, leave_one_out=False, make_cases=hide_one_cases, cases_by_row=None):
    """Run the test on every session that has at least one associated symptom.

    cases_by_row may supply frozen raw cases, overriding generation from the current sessions.
    Returns one result per tested session, in the order given:
        {"row": 12, "cases": 2, "hits": {3: 1, 5: 2, 8: 2}, "asked": 0}
    "asked" counts hidden symptoms missed in the top 5 but asked as a red-flag question.
    """
    copies = {}  # group -> the sessions that are exact copies of each other
    for s in sessions:
        copies.setdefault(s.group(), []).append(s)

    results = []
    for s in sessions:
        cases = make_cases(s) if cases_by_row is None else cases_by_row[s.row]
        if not cases:
            continue
        if leave_one_out:
            for c in copies[s.group()]:
                model.leave_out(c)

        hits = {}
        for k in K_VALUES:
            hits[k] = 0
        asked = 0
        for case_shown, hidden in cases:
            ranked = model.recommend(case_shown, max(K_VALUES))
            for k in K_VALUES:
                if model.found(hidden, ranked[:k]):
                    hits[k] += 1
            if not model.found(hidden, ranked[:5]) and model.also_asked(hidden):
                asked += 1

        if leave_one_out:
            for c in copies[s.group()]:
                model.put_back(c)
        results.append({"row": s.row, "cases": len(cases), "hits": hits, "asked": asked})
    return results


def hit_rate(results, k):
    hits = 0
    cases = 0
    for r in results:
        hits += r["hits"][k]
        cases += r["cases"]
    return hits / cases


def session_hit_rate(results, k):
    """Mean within-session hit rate: every eligible session has equal weight."""
    return sum(r["hits"][k] / r["cases"] for r in results) / len(results)


def bootstrap(results_by_model, k, rounds=1000, seed=0, metric=hit_rate):
    """Metric on resampled sessions: model name -> a list with one value per round.

    results_by_model: model name -> evaluate() results, all on the same sessions.
    """
    names = list(results_by_model)
    n_sessions = len(results_by_model[names[0]])
    expected_rows = [(r.get("row"), r["cases"]) for r in results_by_model[names[0]]]
    for results in results_by_model.values():
        if [(r.get("row"), r["cases"]) for r in results] != expected_rows:
            raise ValueError("Paired bootstrap requires the same sessions and case counts for every model")
    rng = random.Random(seed)
    rates = {}
    for name in names:
        rates[name] = []
    for _ in range(rounds):
        draw = []
        for _ in range(n_sessions):
            draw.append(rng.randrange(n_sessions))
        for name in names:
            picked = []
            for i in draw:
                picked.append(results_by_model[name][i])
            rates[name].append(metric(picked, k))
    return rates


def middle_95(values):
    """(low, high): the range that holds the middle 95% of the values."""
    ordered = sorted(values)
    low = ordered[int(len(ordered) * 0.025)]
    high = ordered[int(len(ordered) * 0.975) - 1]
    return low, high


def difference(rates_a, rates_b):
    """Round-by-round difference a - b of two models' bootstrap hit rates."""
    result = []
    for i in range(len(rates_a)):
        result.append(rates_a[i] - rates_b[i])
    return result


# ── normalizer ─────────────────────────────────────────────────────────────
TYPED_RESULTS = ["same symptom", "related", "different", "not placed", "no recorded name placed"]


def evaluate_normalizer(normalizer, sessions, seen_terms, reference_ids=None):
    """Counts for the normalizer part of the report.

    names     every symptom name in the sessions, in the data's own wording -> the layer that
              placed it, or "not placed"
    typed     each term the patient typed, against all the names recorded in its session -> one of
              TYPED_RESULTS ("related": one is a more specific form of the other). Grouped as
              "same text" (typed exactly like one of the recorded names, which says little),
              "seen" (typed in some training session too) or "unseen".
    answers   HPI answers of placed symptoms: "read" if at least one KB answer was found in it
    signed    polarity-aware assignment, deferral and candidate consistency against session records

    reference_ids freezes the reference side so a tested normalizer cannot redefine its targets.
    """
    names = {}
    typed = {"same text": {}, "seen": {}, "unseen": {}}
    answers = {"read": 0, "not read": 0}
    signed = {}
    signed_by_group = {group: {} for group in typed}

    def reference(name):
        return normalizer.symptom_ids(name) if reference_ids is None else reference_ids[name]

    def signed_count(group, name):
        count(signed, name)
        count(signed_by_group[group], name)

    for s in sessions:
        for name in s.confirmed() + s.absent:
            count(names, placed_by(normalizer, name))

        recorded_ids = []
        recorded_keys = []
        for name in s.confirmed():
            recorded_keys.append(key(name))
            recorded_ids += reference(name)
        expected = {(normalizer.root(sid), state)
                    for state, labels in [("present", s.confirmed()), ("absent", s.absent)]
                    for name in labels for sid in reference(name)}
        for term in s.typed:
            if key(term) in recorded_keys:
                group = "same text"
            elif key(term) in seen_terms:
                group = "seen"
            else:
                group = "unseen"
            count(typed[group], compare_typed(normalizer, term, recorded_ids))
            matches = normalizer.read_symptoms(term)
            assigned = {(normalizer.root(m.symptom), m.state) for m in matches if m.symptom}
            if not assigned:
                signed_count(group, "deferred")
                candidates = {(normalizer.root(sid), m.state) for m in matches for sid in m.candidates}
                if candidates & expected:
                    signed_count(group, "deferred_candidate_agrees")
            for sid, state in assigned:
                name = {"present": "positive_assignments", "absent": "negative_assignments",
                        "unsure": "unsure_assignments", "historical": "historical_assignments"}[state]
                signed_count(group, name)
                if (sid, state) in expected:
                    signed_count(group, "agrees_with_record")
                elif (state in {"present", "absent"}
                      and (sid, "present" if state == "absent" else "absent") in expected):
                    signed_count(group, "opposes_record")
                else:
                    signed_count(group, "not_in_record")

        for name in s.confirmed():
            ids = normalizer.symptom_ids(name)
            if not ids:
                continue
            for answer in s.answers.get(name, []):
                if normalizer.read_answer(ids[0], answer):
                    answers["read"] += 1
                else:
                    answers["not read"] += 1
    return {"names": names, "typed": typed, "answers": answers,
            "signed": signed, "signed_by_group": signed_by_group}


def compare_typed(normalizer, term, recorded_ids):
    """One of TYPED_RESULTS."""
    if not recorded_ids:
        return "no recorded name placed"
    ids = normalizer.symptom_ids(term)
    if not ids:
        return "not placed"
    for symptom_id in ids:
        if symptom_id in recorded_ids:
            return "same symptom"
    recorded_roots = []
    for symptom_id in recorded_ids:
        recorded_roots.append(normalizer.root(symptom_id))
    for symptom_id in ids:
        if normalizer.root(symptom_id) in recorded_roots:
            return "related"
    return "different"


def typed_terms(sessions):
    """Keys of every term typed in the sessions."""
    terms = set()
    for s in sessions:
        for term in s.typed:
            terms.add(key(term))
    return terms


def placed_by(normalizer, name):
    for match in normalizer.read_symptoms(name):
        if match.symptom and match.state == "present":
            return match.layer
    return "not placed"


def count(counter, name):
    counter[name] = counter.get(name, 0) + 1


# ── clinical cases (vignettes) ─────────────────────────────────────────────
def check_vignette(engine, case, alpha):
    """Run the engine on one vignette. Returns {expectation: True/False} for its `expect` block,
    and for a book case {"final_diagnosis_rank": rank in the DDx, or None if not in it}."""
    from .engine.facts import Facts  # here, so the baselines do not need the engine

    history = case.history
    facts = Facts(engine.kb, case.sex, case.age, case.chief_complaint, history.present,
                  history.absent, history.hpi)
    result = engine.run(facts)
    ddx = []
    for disease, _ in result.ddx:
        ddx.append(disease)
    asked = result.red_flag_ids()
    recommended = []
    for symptom_id, _ in result.ranked(alpha):
        recommended.append(symptom_id)

    checks = {}
    expect = case.expect
    if expect is not None:
        if expect.top3_includes:
            checks["top3_includes"] = expect.top3_includes in ddx[:3]
        if expect.top3_includes_any:
            checks["top3_includes"] = any_in(expect.top3_includes_any, ddx[:3])
        if expect.ask_first:
            checks["ask_first"] = all_in_list(expect.ask_first, asked)
        if expect.not_ask_first:
            checks["not_ask_first"] = not any_in(expect.not_ask_first, asked)
        if expect.urgent is not None:
            checks["urgent"] = bool(result.urgent) == expect.urgent
        if expect.never_recommend:
            checks["never_recommend"] = not any_in(expect.never_recommend, recommended + asked)
        if expect.never_in_ddx:
            checks["never_in_ddx"] = not any_in(expect.never_in_ddx, ddx)
        if expect.duration_categories:
            answers = []
            for question, values in facts.hpi.items():
                if question.endswith(".duration"):
                    if values == ["unknown"]:   # not known: every category is still possible
                        values = list(engine.kb.questions[question].categories)
                    answers += values
            checks["duration_categories"] = sorted(answers) == sorted(expect.duration_categories)
        if expect.switch_chief_complaint:
            checks["switch_chief_complaint"] = result.switch_to == expect.switch_chief_complaint
    if case.final_diagnosis:
        checks["final_diagnosis_rank"] = None
        if case.final_diagnosis in ddx:
            checks["final_diagnosis_rank"] = ddx.index(case.final_diagnosis) + 1
    return checks


def any_in(items, collection):
    for item in items:
        if item in collection:
            return True
    return False


def all_in_list(items, collection):
    for item in items:
        if item not in collection:
            return False
    return True
