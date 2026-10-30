"""Build everything the recommender learns from the Agnos data, into build/.

    python scripts/build.py                     # uses the one CSV in data/raw/
    python scripts/build.py --data other.csv

Steps (learning uses training rows only):
    1. reuse/create build/split.json and freeze test cases/targets in build/evaluation_lock.json
    2. learn extra symptom names from what patients typed -> build/aliases.json
    3. learn symptom counts and disease weights for the engine -> build/memory.json
    4. choose α, the share of the book in the ranking, by leave-one-out hit rate -> build/alpha.json

With --semantic, use the separately calibrated local fallback and write its artifacts under
build/semantic/, retaining the same test split and reference mappings.
"""
import argparse
import json
from pathlib import Path

from symptom_rec.data.protocol import evaluation_lock, fixed_split
from symptom_rec.data.sessions import find_csv, load_sessions
from symptom_rec.data.split import MIN_GROUP_SIZE
from symptom_rec.engine.adapter import EngineRecommender, best_alpha, tune_alpha
from symptom_rec.engine.engine import Engine
from symptom_rec.evaluation import hide_one_cases
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.learn import learn_aliases, save_aliases
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.paths import BUILD_DIR


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, help="Agnos CSV (default: the one in data/raw/)")
    parser.add_argument("--semantic", action="store_true",
                        help="Use train-calibrated local semantic fallback")
    args = parser.parse_args()

    csv_path = args.data or find_csv()
    sessions = load_sessions(csv_path)
    copies = 0
    for s in sessions:
        if s.copy_of is not None:
            copies += 1
    print(f"{len(sessions)} sessions; {copies} are exact copies of an earlier row (kept, split together)")

    # 1. reuse the saved train/test assignment
    train_rows, test_rows = fixed_split(sessions, BUILD_DIR / "split.json")
    counts = {}  # counted like split_sessions does: originals only
    for s in sessions:
        if s.copy_of is None:
            counts[s.chief_complaint] = counts.get(s.chief_complaint, 0) + 1
    common = [cc for cc in counts if counts[cc] >= MIN_GROUP_SIZE]
    print(f"{len(counts)} chief complaints, {len(common)} with at least {MIN_GROUP_SIZE} sessions")
    print(f"split: {len(train_rows)} train, {len(test_rows)} test -> build/split.json")

    # 2. learned symptom names, from the training set only
    train_set = set(train_rows)
    train = []
    for s in sessions:
        if s.row in train_set:
            train.append(s)
    kb = load_kb()
    lexicon = load_lexicon()
    aliases = learn_aliases(Normalizer(kb, lexicon), train)
    reference = Normalizer(kb, lexicon, load_reference_aliases(aliases))
    lock = evaluation_lock(sessions, train_rows, test_rows, reference, BUILD_DIR / "evaluation_lock.json")
    print(f"Locked evaluation: {len(lock['cases']['hide_one'])} hidden-symptom cases")
    artifact_dir = BUILD_DIR / "semantic" if args.semantic else BUILD_DIR
    artifact_dir.mkdir(exist_ok=True)
    save_aliases(artifact_dir / "aliases.json", aliases)
    print(f"learned {len(aliases)} symptom names from typed terms -> {artifact_dir.name}/aliases.json")
    search = None
    if args.semantic:
        import torch

        from symptom_rec.normalize.semantic import load_search
        torch.set_num_threads(4)
        search = load_search(kb, artifact_dir / "config.json", train)
        search.prepare([name for s in train for name in s.confirmed()])

    # 3. what the engine learns from the data
    engine = Engine(kb)
    recommender = EngineRecommender(engine, Normalizer(kb, lexicon, aliases, search),
                                    reference_ids=lock["reference_ids"])
    recommender.learn(train)
    engine.memory.save(artifact_dir / "memory.json")
    print(f"learned symptom counts and disease weights -> {artifact_dir.name}/memory.json")

    # 4. α, by leave-one-out HitRate@5 on the hide-one-symptom test
    rates = tune_alpha(recommender, train, hide_one_cases)
    alpha = best_alpha(rates)
    table = {}
    for a, rate in rates.items():
        table[str(a)] = round(rate, 4)
    content = {"alpha": alpha, "test": "hide one symptom, leave-one-out, HitRate@5", "hit_rate": table}
    (artifact_dir / "alpha.json").write_text(json.dumps(content, indent=1), encoding="utf-8")
    shown = []
    for a, rate in rates.items():
        shown.append(f"{a}: {rate:.0%}")
    print("HitRate@5 by α: " + ", ".join(shown))
    print(f"α = {alpha} -> {artifact_dir.name}/alpha.json")


def load_reference_aliases(fallback):
    from symptom_rec.normalize.learn import load_aliases
    path = BUILD_DIR / "aliases.json"
    return load_aliases(path) if path.exists() else fallback


if __name__ == "__main__":
    main()
