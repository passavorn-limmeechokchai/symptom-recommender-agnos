"""Prepare private physician review sheets, or score reviewed phrases without session proxies."""
import argparse
import json
from pathlib import Path

from symptom_rec.data.protocol import fixed_split
from symptom_rec.data.sessions import find_csv, load_sessions
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.learn import learn_aliases
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.normalize.review import load_review, make_manifest, prepare_review, score_review
from symptom_rec.paths import BUILD_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--directory", type=Path, default=BUILD_DIR / "normalization_review")
    parser.add_argument("--subset", choices=["development", "validation"], default="development")
    parser.add_argument("--allow-partial", action="store_true")
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument("--semantic", action="store_true", help="Candidate-only E5-small")
    comparison.add_argument("--retrieval", action="store_true", help="Frozen experimental E5-base classifier")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(BUILD_DIR.resolve()):
        raise SystemExit("Private review data must stay under ignored build/")
    if not (BUILD_DIR / "split.json").exists():
        raise SystemExit("Saved split required; build the original model first")
    sessions = load_sessions(args.data or find_csv())
    train_rows, _ = fixed_split(sessions, BUILD_DIR / "split.json")
    by_row = {s.row: s for s in sessions}
    train = [by_row[row] for row in train_rows]
    kb = load_kb()
    manifest = make_manifest(train, kb)
    if args.action == "prepare":
        print(json.dumps(prepare_review(directory, manifest, kb)))
        return
    saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if saved != manifest:
        raise SystemExit("Review manifest differs from the saved training data/split/KB")
    reviewed, pending = load_review(directory, saved, kb, args.subset)
    if reviewed and pending and not args.allow_partial:
        raise SystemExit("Review is incomplete; finish the sheet or explicitly use --allow-partial")
    lexicon = load_lexicon()
    baseline = Normalizer(kb, lexicon)
    alias_rows = set(manifest["alias_train_rows"])
    aliases = learn_aliases(baseline, [s for s in train if s.row in alias_rows])
    search = None
    if args.semantic:
        from symptom_rec.normalize.semantic import load_search
        search = load_search(kb, BUILD_DIR / "semantic" / "config.json", train)
        if search.threshold <= 1:
            raise SystemExit("Review comparison requires candidate-only semantic mode until gold calibration")
    if args.retrieval:
        from symptom_rec.normalize.retrieval import load_retrieval
        config_path = BUILD_DIR / "retrieval" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        import hashlib
        if config["review_manifest_sha256"] != hashlib.sha256(
                (directory / "manifest.json").read_bytes()).hexdigest():
            raise SystemExit("Retrieval profile was calibrated with a different review partition")
        search = load_retrieval(kb, config_path)
    normalizer = Normalizer(kb, lexicon, aliases, search)
    result = score_review(normalizer, reviewed, pending)
    result.update({"subset": args.subset, "aliases": len(aliases),
                   "alias_training_sessions": len(alias_rows), "semantic": args.semantic,
                   "retrieval": args.retrieval,
                   "label_source": "physician review; unreviewed rows are never treated as gold",
                   "scope": "training-derived development benchmark; fresh data needed for confirmation"})
    suffix = "_retrieval" if args.retrieval else "_semantic" if args.semantic else ""
    (directory / f"metrics_{args.subset}{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
