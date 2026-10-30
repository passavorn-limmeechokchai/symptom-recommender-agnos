"""Download public E5 weights, or calibrate local semantic fallback using training folds only.

Install .[semantic], then run --download once. Calibration never reads the held-out rows.
"""
import argparse
import json
from pathlib import Path

from symptom_rec.data.protocol import evaluation_lock, fixed_split, session_digest
from symptom_rec.data.sessions import find_csv, load_sessions
from symptom_rec.data.validation import training_folds
from symptom_rec.engine.facts import root
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.learn import learn_aliases, load_aliases
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.normalize.semantic import MODEL_ID, MODEL_REVISION, E5Encoder, SemanticSearch
from symptom_rec.paths import BUILD_DIR
from symptom_rec.reporting import write_report_section


def validation_examples(sessions, kb, lexicon, search):
    """Out-of-fold fallback queries with weak, polarity-aware session references."""
    examples = []
    fold_sizes = []
    reference = Normalizer(kb, lexicon)
    for train, validation in training_folds(sessions):
        aliases = learn_aliases(reference, train)
        normalizer = Normalizer(kb, lexicon, aliases)
        fold_sizes.append({"train": len(train), "validation": len(validation),
                           "validation_with_associated": sum(bool(s.associated) for s in validation),
                           "aliases_from_fold_train": len(aliases)})
        for session in validation:
            expected = set()
            for absent, names in [(False, session.confirmed()), (True, session.absent)]:
                for name in names:
                    for sid in reference.symptom_ids(name):
                        expected.add((root(kb, sid), absent))
            if not expected:
                continue
            for term in session.typed:
                matches = normalizer.read_symptoms(term)
                if any(m.state in {"unsure", "historical"} for m in matches):
                    continue  # weak binary session references cannot calibrate these assertions
                if any(m.symptom or m.layer == "ambiguous" for m in matches):
                    continue
                # Match the exact fallback query and negation handling used at runtime.
                text = term
                absent = bool(matches and matches[0].absent)
                if absent:
                    from symptom_rec.normalize.text import key
                    text = key(term)
                    for word in normalizer.negations:
                        if text.startswith(word) and len(text) > len(word):
                            text = text[len(word):]
                            break
                examples.append((text, absent, expected))
    search.prepare([text for text, _, _ in examples])
    observations = []
    for text, absent, expected in examples:
        ranked = search.search(text)
        observations.append({"score": ranked[0][1], "gap": ranked[0][1] - ranked[1][1],
                             "agrees": (ranked[0][0], absent) in expected,
                             "candidate_hit": any((sid, absent) in expected for sid, _ in ranked)})
    return observations, fold_sizes


def choose_policy(observations, minimum_agreement=0.9, minimum_accepted=5):
    """Select the most agreeing assignments subject to a conservative agreement floor.

    The labels are session-derived proxies, not physician adjudications.
    A candidate-only policy is retained if no setting qualifies.
    """
    grid = []
    for threshold in [0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95]:
        for margin in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
            accepted = [x for x in observations if x["score"] >= threshold and x["gap"] >= margin]
            agrees = sum(x["agrees"] for x in accepted)
            grid.append({"threshold": threshold, "margin": margin, "accepted": len(accepted),
                         "agrees": agrees, "agreement": agrees / len(accepted) if accepted else 0})
    eligible = [x for x in grid if x["accepted"] >= minimum_accepted
                and x["agreement"] >= minimum_agreement]
    if eligible:
        chosen = max(eligible, key=lambda x: (x["agrees"], x["agreement"], x["threshold"], x["margin"]))
    else:
        chosen = {"threshold": 1.1, "margin": 0.0, "accepted": 0, "agrees": 0, "agreement": 0}
    return chosen, grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download public model files only")
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    model_dir = BUILD_DIR / "semantic" / "model"
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, token=False, local_dir=model_dir,
                          allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                          "tokenizer_config.json", "special_tokens_map.json", "README.md"])
        print(f"Public model downloaded to {model_dir}")
        return
    import torch
    torch.set_num_threads(4)
    sessions = load_sessions(args.data or find_csv())
    train_rows, test_rows = fixed_split(sessions, BUILD_DIR / "split.json")
    train_set = set(train_rows)
    train = [s for s in sessions if s.row in train_set]
    kb = load_kb()
    lexicon = load_lexicon()
    reference = Normalizer(kb, lexicon, load_aliases(BUILD_DIR / "aliases.json"))
    evaluation_lock(sessions, train_rows, test_rows, reference, BUILD_DIR / "evaluation_lock.json")
    search = SemanticSearch(kb, E5Encoder(model_dir), cache_dir=BUILD_DIR / "semantic" / "index")
    observations, folds = validation_examples(train, kb, lexicon, search)
    chosen, grid = choose_policy(observations)
    content = {"model": MODEL_ID, "revision": MODEL_REVISION,
               "train_sha256": session_digest(train), "train_rows": train_rows,
               "fold_seed": 1729, "folds": folds, "policy": chosen,
               "fallback_validation_queries": len(observations),
               "candidate_hits": sum(x["candidate_hit"] for x in observations),
               "minimum_agreement": 0.9, "minimum_accepted": 5,
               "reference": "rules-only session labels; weak polarity-aware agreement, not gold accuracy",
               "grid": grid}
    (BUILD_DIR / "semantic" / "config.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in content.items() if k not in ("grid", "train_rows")}))
    # Full selection details are already saved in the private config.json.
    write_report_section("semantic-validation", concise_report(content))


def concise_report(content):
    chosen = content["policy"]
    queries = content["fallback_validation_queries"]
    report = ["# E5-small calibration", "",
              f"Use `{content['model']}` on CPU as a fallback after rules. Select thresholds using "
              "five grouped training folds, relearning aliases within each fold.", "",
              f"There were {queries} fallback queries; the reference appeared in the top three "
              f"candidates in {content['candidate_hits']}/{queries} queries. "
              "Selection requires at least five accepted queries and ≥90% proxy agreement.", "",
              f"The selected policy accepted {chosen['accepted']} queries "
              f"with {chosen['agrees']} agreements. "
              + ("It therefore supplies candidates only." if not chosen["accepted"]
                 else f"Threshold: {chosen['threshold']}; margin: {chosen['margin']}.") +
              " Calibration results select the policy; they are not an independent accuracy estimate."]
    return "\n".join(report)


if __name__ == "__main__":
    main()
