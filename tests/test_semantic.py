import numpy as np
import pytest

from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.normalize.semantic import SemanticSearch


class Encoder:
    """Controlled embeddings, no model downloads or patient data."""
    def encode(self, texts):
        return np.array([[1.0, 0.0] if text == "ไข้" else [0.0, 1.0] for text in texts])


class Search:
    def __init__(self, accepts=True):
        self.accept = accepts
        self.queries = []

    def search(self, text):
        self.queries.append(text)
        return [("dyspnea", 0.94), ("fatigue", 0.7), ("cough", 0.6)]

    def accepts(self, ranked):
        return self.accept


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def test_semantic_index_groups_aliases_and_specific_forms_by_parent(kb, tmp_path):
    search = SemanticSearch(kb, Encoder(), threshold=0.9, margin=0.1, cache_dir=tmp_path)
    ranked = search.search("ไข้")
    assert ranked[0] == ("fever", 1.0)
    assert len({sid for sid, _ in ranked}) == len(ranked)
    assert search.accepts(ranked)


def test_semantic_fallback_preserves_rules_and_explicit_ambiguity(kb):
    search = Search()
    normalizer = Normalizer(kb, load_lexicon(), semantic=search)
    assert normalizer.symptom_ids("ไอ") == ["cough"]
    assert normalizer.symptom_ids("ไม่มีไข้") == []
    assert normalizer.read_symptoms("หายใจไม่สะดวก")[0].layer == "ambiguous"
    assert search.queries == []


def test_semantic_acceptance_and_deferral_preserve_polarity(kb):
    search = Search()
    normalizer = Normalizer(kb, load_lexicon(), semantic=search)
    # Exercise the fallback with fabricated text rather than depend on fuzzy tokenization.
    from symptom_rec.normalize.normalizer import Match
    original = [Match("คำบรรยายใหม่", None, "none", 0)]
    [match] = normalizer.semantic_fallback("คำบรรยายใหม่", "คำบรรยายใหม่", True, original)
    assert match.symptom == "dyspnea"
    assert match.absent
    assert match.layer == "semantic"
    search.accept = False
    [match] = normalizer.semantic_fallback("คำบรรยายใหม่", "คำบรรยายใหม่", False, original)
    assert match.symptom is None
    assert match.layer == "semantic_candidates"
    assert match.candidates == ["dyspnea", "fatigue", "cough"]


def test_semantic_requires_both_score_and_margin(kb):
    search = SemanticSearch(kb, Encoder(), threshold=0.9, margin=0.1)
    assert not search.accepts([("fever", 0.95), ("cough", 0.94)])
    assert not search.accepts([("fever", 0.8), ("cough", 0.5)])
    assert not search.accepts([])


@pytest.mark.parametrize("prefix,state", [("ไม่แน่ใจว่า", "unsure"), ("เคย", "historical")])
def test_even_accepted_semantic_fallback_keeps_noncurrent_assertions(kb, prefix, state):
    search = Search()
    normalizer = Normalizer(kb, load_lexicon(), semantic=search)
    matches = normalizer.read_symptoms(prefix + "zzzzqqqq")
    assert any(m.symptom == "dyspnea" for m in matches)
    assert all(m.state == state for m in matches)
    assert normalizer.symptom_ids(prefix + "zzzzqqqq") == []
    assert search.queries and all(prefix not in q for q in search.queries)


def test_semantic_policy_stays_candidate_only_without_validation_support():
    from scripts.prepare_semantic import choose_policy

    observations = [{"score": 0.97, "gap": 0.1, "agrees": i % 3 != 0} for i in range(12)]
    policy, _ = choose_policy(observations)
    assert policy["threshold"] > 1
    observations = [{"score": 0.97, "gap": 0.1, "agrees": True} for _ in range(12)]
    policy, _ = choose_policy(observations)
    assert policy["accepted"] == policy["agrees"] == 12
    assert policy["threshold"] <= 1
