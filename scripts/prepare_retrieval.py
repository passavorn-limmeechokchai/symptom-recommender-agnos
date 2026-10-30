"""Fit a KB-name classifier and select its gate within training; never inspect test results."""
import argparse
import hashlib
import json
from collections import defaultdict

from symptom_rec.data.protocol import fixed_split, session_digest
from symptom_rec.data.sessions import find_csv, load_sessions
from symptom_rec.data.validation import training_folds
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.learn import learn_aliases
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.normalize.retrieval import (
    MODEL_ID,
    MODEL_REVISION,
    CachedEncoder,
    fit_search,
    kb_signature,
)
from symptom_rec.normalize.text import key
from symptom_rec.paths import BUILD_DIR


class Recorder:
    def __init__(self):
        self.queries = []

    def search(self, text):
        self.queries.append(text)
        return []


def development_queries(sessions, kb, lexicon):
    """Fold-local aliases, grouped copies and fixed rule-derived session references."""
    reference = Normalizer(kb, lexicon)
    queries = []
    folds = []
    for fold, (train, validation) in enumerate(training_folds(sessions)):
        aliases = learn_aliases(reference, train)
        recorder = Recorder()
        normalizer = Normalizer(kb, lexicon, aliases, recorder)
        seen = {key(t) for s in train for t in s.typed}
        folds.append({"train": len(train), "validation": len(validation), "aliases": len(aliases)})
        for session in validation:
            expected = {(reference.root(sid), absent)
                        for absent, names in [(False, session.confirmed()), (True, session.absent)]
                        for name in names for sid in reference.symptom_ids(name)}
            if not expected:
                continue
            for term in session.typed:
                recorder.queries = []
                matches = normalizer.read_symptoms(term)
                if any(m.symptom or m.layer == "ambiguous" or m.state in {"unsure", "historical"}
                       for m in matches):
                    continue
                for text in recorder.queries:
                    queries.append({"text": text, "key": key(term), "fold": fold,
                                    "absent": bool(matches and matches[0].absent),
                                    "expected": sorted(expected), "seen": key(term) in seen})
    return queries, folds


def choose_policy(observations, minimum_agreement=0.9, minimum_unique=5):
    eligible = []
    for threshold in [0, .025, .05, .075, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9]:
        for margin in [0, .01, .02, .03, .04, .05, .075, .1, .15, .2, .3]:
            accepted = [o for o in observations if o["score"] >= threshold and o["gap"] >= margin]
            grouped = defaultdict(list)
            for obs in accepted:
                grouped[obs["key"]].append(obs["agrees"])
            if len(grouped) < minimum_unique:
                continue
            agrees = sum(o["agrees"] for o in accepted)
            agreement = agrees / len(accepted)
            macro = sum(sum(v) / len(v) for v in grouped.values()) / len(grouped)
            if min(agreement, macro) >= minimum_agreement:
                eligible.append({"threshold": threshold, "margin": margin, "accepted": len(accepted),
                                 "agrees": agrees, "unique": len(grouped),
                                 "agreement": agreement, "macro_agreement": macro})
    if not eligible:
        return {"threshold": 1e9, "margin": 0, "accepted": 0, "agrees": 0, "unique": 0,
                "agreement": 0, "macro_agreement": 0}
    return max(eligible, key=lambda p: (p["agrees"], p["macro_agreement"], p["agreement"],
                                       p["threshold"], p["margin"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download only pinned public model files")
    args = parser.parse_args()
    directory = BUILD_DIR / "retrieval"
    directory.mkdir(exist_ok=True)
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False,
                          local_dir=directory / "model-base",
                          allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                          "tokenizer_config.json", "special_tokens_map.json"])
        return
    if not (BUILD_DIR / "split.json").exists():
        raise SystemExit("Original saved split required")
    sessions = load_sessions(find_csv())
    train_rows, _ = fixed_split(sessions, BUILD_DIR / "split.json")
    by_row = {s.row: s for s in sessions}
    train = [by_row[row] for row in train_rows]
    manifest_path = BUILD_DIR / "normalization_review" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["train_sha256"] != session_digest(train):
        raise SystemExit("Review partition belongs to different training data")
    source_rows = set(manifest["alias_train_rows"])
    if not source_rows <= set(train_rows):
        raise SystemExit("Review source rows must be a subset of the original training split")
    source = [s for s in train if s.row in source_rows]
    kb, lexicon = load_kb(), load_lexicon()
    queries, folds = development_queries(source, kb, lexicon)
    encoder = CachedEncoder(directory / "model-base", directory / "embedding_cache")
    if queries:
        encoder.encode([q["text"] for q in queries])
    summaries = []
    details = []
    models = {}
    for regularization in [.01, .1, 1, 10, 100]:
        search = fit_search(kb, encoder, regularization)
        observations = []
        for q in queries:
            ranked = search.search(q["text"])
            expected = {(sid, bool(absent)) for sid, absent in q["expected"]}
            observations.append({"key": q["key"], "score": ranked[0][1],
                                 "gap": ranked[0][1] - ranked[1][1],
                                 "agrees": (ranked[0][0], q["absent"]) in expected,
                                 "candidate_hit": any((sid, q["absent"]) in expected for sid, _ in ranked)})
        policy = choose_policy(observations)
        summary = {"regularization": regularization, "policy": policy,
                   "queries": len(queries), "top1": sum(o["agrees"] for o in observations),
                   "top3": sum(o["candidate_hit"] for o in observations)}
        summaries.append(summary)
        details.append({"regularization": regularization, "observations": observations})
        models[regularization] = search
    chosen = max(summaries, key=lambda s: (s["policy"]["agrees"], s["policy"]["macro_agreement"],
                                          s["policy"]["agreement"], s["policy"]["threshold"]))
    search = models[chosen["regularization"]]
    search.save(directory / "classifier.npz")
    encoder.save()
    config = {"version": 1, "model": MODEL_ID, "revision": MODEL_REVISION,
              "kb_sha256": kb_signature(kb), "train_sha256": session_digest(train),
              "source_train_sha256": session_digest(source), "source_train_rows": sorted(source_rows),
              "review_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "regularization": chosen["regularization"], "policy": chosen["policy"],
              "fold_seed": 1729, "folds": folds, "development": summaries,
              "weights_sha256": hashlib.sha256((directory / "classifier.npz").read_bytes()).hexdigest(),
              "reference": "weak session consistency; not physician-labelled accuracy"}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "calibration.json").write_text(
        json.dumps({"queries": queries, "observations": details}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"source_sessions": len(source),
                      "heldout_phrase_groups": sum(t["subset"] == "validation" for t in manifest["tasks"]),
                      "chosen": chosen, "profile": "retrieval"}, indent=2))


if __name__ == "__main__":
    main()
