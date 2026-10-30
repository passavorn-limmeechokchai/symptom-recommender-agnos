"""Controlled vectors and invented symptoms; no model downloads or private data."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scripts.prepare_retrieval import choose_policy

from symptom_rec.api.app import create_app
from symptom_rec.api.service import load_service
from symptom_rec.kb.schema import Symptom
from symptom_rec.normalize import retrieval
from symptom_rec.normalize.retrieval import MODEL_ID, MODEL_REVISION, fit_search, kb_signature, load_retrieval
from symptom_rec.paths import KB_DIR


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.extend(texts)
        return np.array([[1., 0., 0.] if t in {"ไข้", "fever", "invented-hot"} else [0., 1., 0.]
                         for t in texts], dtype=np.float32)


@pytest.fixture
def kb():
    symptoms = [Symptom(id="fever", th="ไข้", en="fever", kind="symptom"),
                Symptom(id="cough", th="ไอ", en="cough", kind="symptom"),
                Symptom(id="dry_cough", th="ไอแห้ง", en="dry cough", kind="symptom", parent="cough")]
    return SimpleNamespace(symptoms={s.id: s for s in symptoms})


def test_classifier_groups_specific_forms_and_caches_queries(kb):
    encoder = Encoder()
    model = fit_search(kb, encoder)
    model.threshold, model.margin = .25, .2
    ranked = model.search("invented-hot")
    assert ranked[0][0] == "fever"
    assert model.accepts(ranked)
    assert {sid for sid, _ in ranked} == {"cough", "fever"}
    before = len(encoder.calls)
    assert model.search("invented-hot") == ranked
    assert len(encoder.calls) == before
    assert not model.accepts([("fever", .5), ("cough", .49)])
    assert not model.accepts([("fever", .1), ("cough", -.5)])
    assert not model.accepts([])


@pytest.mark.parametrize("regularization", [0, -1, float("nan")])
def test_regularization_validation(kb, regularization):
    with pytest.raises(ValueError, match="Regularization"):
        fit_search(kb, Encoder(), regularization)


def save_profile(kb, directory):
    model = fit_search(kb, Encoder())
    model.save(directory / "classifier.npz")
    config = {"version": 1, "model": MODEL_ID, "revision": MODEL_REVISION,
              "kb_sha256": kb_signature(kb),
              "weights_sha256": hashlib.sha256((directory / "classifier.npz").read_bytes()).hexdigest(),
              "policy": {"threshold": .25, "margin": .2}}
    (directory / "config.json").write_text(json.dumps(config))


def test_profile_roundtrip_and_checksum(kb, tmp_path, monkeypatch):
    save_profile(kb, tmp_path)
    monkeypatch.setattr(retrieval, "E5Encoder", lambda path: Encoder())
    model = load_retrieval(kb, tmp_path / "config.json")
    assert model.search("invented-hot")[0][0] == "fever"
    assert model.accepts(model.search("invented-hot"))
    with (tmp_path / "classifier.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_retrieval(kb, tmp_path / "config.json")


def test_profile_rejects_stale_kb(kb, tmp_path):
    save_profile(kb, tmp_path)
    kb.symptoms["fever"].aliases.append("invented-new-alias")
    with pytest.raises(ValueError, match="KB names"):
        load_retrieval(kb, tmp_path / "config.json")


def test_policy_requires_unique_phrases_and_macro_agreement():
    repeated = [{"key": "one", "score": .5, "gap": .3, "agrees": True} for _ in range(100)]
    assert choose_policy(repeated)["accepted"] == 0
    wrong = [{"key": str(i), "score": .5, "gap": .3, "agrees": False} for i in range(4)]
    assert choose_policy(repeated + wrong)["accepted"] == 0  # high micro score alone cannot qualify
    good = [{"key": str(i // 2), "score": .5, "gap": .3, "agrees": True} for i in range(12)]
    chosen = choose_policy(good)
    assert chosen["accepted"] == 12 and chosen["unique"] == 6
    assert chosen["macro_agreement"] == 1


def test_api_profile_is_explicit_and_visible(tmp_path, monkeypatch):
    class Search:
        def search(self, text):
            return [("fever", .9), ("cough", .1)]

        def accepts(self, ranked):
            return True

    directory = tmp_path / "retrieval"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "classifier.npz").write_bytes(b"fixture")
    monkeypatch.setattr(retrieval, "load_retrieval", lambda kb, path: Search())
    assert load_service(KB_DIR, tmp_path).profile == "rules"
    with TestClient(create_app(kb_dir=KB_DIR, build_dir=tmp_path, profile="retrieval")) as client:
        assert client.get("/health").json()["normalizer_profile"] == "retrieval"
        result = client.post("/v1/recommend", json={"sex": "female", "age": 40,
                                                  "chief_complaint": "zzzzqqqq"})
        assert result.status_code == 200
        assert result.json()["chief_complaint"]["id"] == "fever"
        assert result.json()["normalizer_profile"] == "retrieval"


def test_unknown_profile_fails_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Unknown normalizer"):
        load_service(KB_DIR, tmp_path, profile="invented")
