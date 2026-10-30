"""Local E5 embeddings plus a regularized KB-name classifier; no remote inference.

Every KB name/alias trains a root-symptom target. Weak training-session references select
regularization and the acceptance gate separately. Scores are decision scores, not probabilities.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..engine.facts import root
from .semantic import E5Encoder

MODEL_ID = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def documents(kb):
    texts, targets = [], []
    for symptom in kb.symptoms.values():
        for text in [symptom.th, symptom.en] + symptom.aliases:
            texts.append(text)
            targets.append(root(kb, symptom.id))
    return texts, targets


def kb_signature(kb):
    data = json.dumps(documents(kb), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


class CachedEncoder:
    """Training cache under private build/. Runtime uses E5Encoder directly."""
    def __init__(self, model_dir, directory):
        self.model_dir = model_dir
        self.directory = directory
        self.encoder = None
        self.cache = {}
        metadata = directory / "texts.json"
        if metadata.exists():
            content = json.loads(metadata.read_text(encoding="utf-8"))
            if content["model"] != MODEL_ID or content["revision"] != MODEL_REVISION:
                raise ValueError("Embedding cache belongs to a different model")
            vectors = np.load(directory / "vectors.npy", allow_pickle=False)
            self.cache.update(zip(content["texts"], vectors, strict=True))

    def encode(self, texts):
        missing = sorted(set(texts) - self.cache.keys())
        if missing:
            if self.encoder is None:
                self.encoder = E5Encoder(self.model_dir)
            self.cache.update(zip(missing, self.encoder.encode(missing), strict=True))
        return np.stack([self.cache[t] for t in texts])

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        texts = sorted(self.cache)
        np.save(self.directory / "vectors.npy", np.stack([self.cache[t] for t in texts]))
        (self.directory / "texts.json").write_text(json.dumps(
            {"model": MODEL_ID, "revision": MODEL_REVISION, "texts": texts},
            ensure_ascii=False), encoding="utf-8")


class RidgeSearch:
    def __init__(self, kb, encoder, mean, weights, roots, threshold=1e9, margin=0.0):
        self.kb = kb
        self.encoder = encoder
        self.mean = mean
        self.weights = weights
        self.roots = list(roots)
        self.threshold = threshold
        self.margin = margin
        self.cache = {}

    def prepare(self, texts):
        missing = sorted(set(texts) - self.cache.keys())
        if missing:
            self.cache.update(zip(missing, self.encoder.encode(missing), strict=True))

    def search(self, text, k=3):
        self.prepare([text])
        features = self.cache[text] - self.mean
        features = features / max(float(np.linalg.norm(features)), 1e-8)
        scores = features @ self.weights
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.roots[i], float(scores[i])) for i in order]

    def accepts(self, ranked):
        if not ranked:
            return False
        gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
        return ranked[0][1] >= self.threshold and gap >= self.margin

    def save(self, path):
        np.savez(path, mean=self.mean, weights=self.weights, roots=np.asarray(self.roots))


def fit_search(kb, encoder, regularization=1.0):
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be finite and positive")
    texts, targets = documents(kb)
    vectors = encoder.encode(texts)
    mean = vectors.mean(axis=0)
    features = vectors - mean
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    roots = sorted(set(targets))
    positions = {sid: i for i, sid in enumerate(roots)}
    labels = np.zeros((len(texts), len(roots)), dtype=np.float32)
    labels[np.arange(len(texts)), [positions[sid] for sid in targets]] = 1
    gram = features.T @ features + regularization * np.eye(features.shape[1])
    weights = np.linalg.solve(gram, features.T @ labels)
    return RidgeSearch(kb, encoder, mean, weights, roots)


def load_retrieval(kb, config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("version") != 1 or config.get("model") != MODEL_ID
            or config.get("revision") != MODEL_REVISION):
        raise ValueError("Retrieval model identity differs from the prepared profile")
    if config["kb_sha256"] != kb_signature(kb):
        raise ValueError("KB names/parents changed; rerun scripts/prepare_retrieval.py")
    weights_path = config_path.parent / "classifier.npz"
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != config["weights_sha256"]:
        raise ValueError("Retrieval classifier checksum differs from its profile")
    arrays = np.load(weights_path, allow_pickle=False)
    roots = arrays["roots"].tolist()
    if roots != sorted(set(documents(kb)[1])):
        raise ValueError("Retrieval targets differ from current KB roots")
    mean, weights = arrays["mean"], arrays["weights"]
    if (weights.shape != (len(mean), len(roots)) or not np.isfinite(weights).all()
            or not np.isfinite(mean).all()):
        raise ValueError("Invalid retrieval classifier arrays")
    policy = config["policy"]
    if not all(np.isfinite(policy[n]) for n in ["threshold", "margin"]) or policy["margin"] < 0:
        raise ValueError("Invalid retrieval acceptance policy")
    encoder = E5Encoder(config_path.parent / "model-base")
    return RidgeSearch(kb, encoder, mean, weights, roots, policy["threshold"], policy["margin"])
