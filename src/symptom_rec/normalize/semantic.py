"""Local E5 retrieval over source-written symptom names; never calls an inference API."""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..engine.facts import root

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class E5Encoder:
    def __init__(self, model_dir, batch_size=32):
        # Optional dependencies: rules-only operation and unit tests do not need them.
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True,
                                              use_safetensors=True).eval()

    def encode(self, texts):
        """Mean pooling and normalized vectors, following the E5 model card."""
        torch = self.torch
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = ["query: " + text for text in texts[start:start + self.batch_size]]
            tokens = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=128, return_tensors="pt")
            with torch.inference_mode():
                states = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).bool()
                pooled = states.masked_fill(~mask, 0).sum(1) / mask.sum(1)
                vectors.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        return np.concatenate(vectors)


class SemanticSearch:
    def __init__(self, kb, encoder, threshold=1.1, margin=0.0, cache_dir=None):
        self.kb = kb
        self.encoder = encoder
        self.threshold = threshold  # >1 = candidates only until validation approves auto-assignment
        self.margin = margin
        self.cache = {}
        self.texts = []
        self.ids = []
        for symptom in kb.symptoms.values():
            for text in [symptom.th, symptom.en] + symptom.aliases:
                self.texts.append(text)
                self.ids.append(root(kb, symptom.id))
        signature = hashlib.sha256(json.dumps(
            [MODEL_ID, MODEL_REVISION, self.texts, self.ids], ensure_ascii=False).encode()).hexdigest()
        cache_file = Path(cache_dir) / (signature + ".npy") if cache_dir else None
        if cache_file and cache_file.exists():
            self.vectors = np.load(cache_file, allow_pickle=False)
        else:
            self.vectors = encoder.encode(self.texts)
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, self.vectors, allow_pickle=False)

    def prepare(self, texts):
        """Batch query vectors locally; this does not fit a model or use any labels."""
        missing = sorted(set(texts) - self.cache.keys())
        if missing:
            self.cache.update(zip(missing, self.encoder.encode(missing), strict=True))

    def search(self, text, k=3):
        self.prepare([text])
        similarities = self.vectors @ self.cache[text]
        best = {}
        for symptom_id, score in zip(self.ids, similarities, strict=True):
            best[symptom_id] = max(best.get(symptom_id, -1.0), float(score))
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:k]

    def accepts(self, ranked):
        if not ranked:
            return False
        gap = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0
        return ranked[0][1] >= self.threshold and gap >= self.margin


def load_search(kb, config_path, train=None):
    """Load an explicitly prepared local model and validate the calibration's training identity."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["model"] != MODEL_ID or config["revision"] != MODEL_REVISION:
        raise ValueError("Semantic model identity differs from the calibrated model")
    if train is not None:
        from ..data.protocol import session_digest
        if config["train_sha256"] != session_digest(train):
            raise ValueError("Semantic calibration differs from this training set; recalibrate on train only")
    policy = config["policy"]
    return SemanticSearch(kb, E5Encoder(config_path.parent / "model"),
                          policy["threshold"], policy["margin"], config_path.parent / "index")
