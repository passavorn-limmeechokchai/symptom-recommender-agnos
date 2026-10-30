"""Freeze the rules-model protocol now; evaluate once a new local CSV is available."""
import argparse
import json
from pathlib import Path

from symptom_rec.api.service import load_service
from symptom_rec.baselines.counting import Cooccurrence
from symptom_rec.data.prospective import digest_file, freeze_protocol, new_cohort, summarize_comparison
from symptom_rec.data.protocol import evaluation_lock, fixed_split, locked_cases
from symptom_rec.data.sessions import find_csv, load_sessions
from symptom_rec.engine.adapter import EngineRecommender
from symptom_rec.evaluation import evaluate
from symptom_rec.paths import BUILD_DIR, KB_DIR, REPO_ROOT
from symptom_rec.reporting import write_report_section


def read_private_csv(path):
    try:
        sessions = load_sessions(path)
        if any(s.sex not in {"male", "female"} or s.age < 0 for s in sessions):
            raise ValueError("Invalid sex or age")
        return sessions
    except (OSError, KeyError, TypeError, ValueError):
        raise SystemExit("Cannot parse private CSV; check its schema, JSON, gender and age locally") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--original", type=Path, help="Original private CSV; defaults to data/raw/")
    parser.add_argument("--data", type=Path, help="New cohort CSV, required for evaluate")
    parser.add_argument("--directory", type=Path, default=BUILD_DIR / "prospective")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(BUILD_DIR.resolve()):
        raise SystemExit("Private protocols and evaluation locks must stay under ignored build/")
    if args.action == "evaluate" and not (directory / "protocol.json").exists():
        raise SystemExit("Freeze the protocol before collecting/evaluating a new cohort")
    if args.action == "evaluate" and args.data is None:
        raise SystemExit("--data is required; the original test set is not a new cohort")
    original = read_private_csv(args.original or find_csv())
    train_rows, _ = fixed_split(original, BUILD_DIR / "split.json")
    protocol = freeze_protocol(directory, REPO_ROOT, original)
    if args.action == "freeze":
        print(json.dumps({"status": "awaiting_new_data", "profile": protocol["profile"],
                          "protocol_sha256": digest_file(directory / "protocol.json"),
                          "primary": protocol["primary"], "bootstrap": protocol["bootstrap"]}, indent=2))
        return
    fresh, counts = new_cohort(original, read_private_csv(args.data))
    service = load_service(KB_DIR, BUILD_DIR, profile="rules")
    if service.mode != "hybrid":
        raise SystemExit("The frozen comparison requires complete original hybrid artifacts")
    # Freeze raw cases and target mappings before any ranking. A rerun must use the same cohort.
    lock = evaluation_lock(fresh, [], [s.row for s in fresh], service.normalizer,
                           directory / "evaluation_lock.json")
    by_row = {s.row: s for s in original}
    models = {"co-occurrence": Cooccurrence([by_row[row] for row in train_rows]),
              "engine": EngineRecommender(service.engine, service.normalizer,
                                          reference_ids=lock["reference_ids"])}
    settings = protocol["bootstrap"]
    results = {}
    for mode in ["hide_one", "first_step"]:
        cases = locked_cases(lock, mode)
        rows = {name: evaluate(model, fresh, cases_by_row=cases) for name, model in models.items()}
        results[mode] = summarize_comparison(rows, settings["rounds"], settings["seed"])
    result = {"status": "evaluated", "counts": counts, "model_version": service.model_version,
              "protocol_sha256": digest_file(directory / "protocol.json"),
              "cohort_sha256": lock["dataset_sha256"], "scope": protocol["scope"],
              "independence_limit": protocol["independence_limit"], "results": results}
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Prospective comparison", "", f"Frozen protocol: `{result['protocol_sha256']}`.", "",
             "Targets are recorded symptoms mapped by frozen rules, not physician normalization labels.", "",
             f"New input: {counts['input_rows']} rows; {counts['unique_sessions']} unique sessions; "
             f"{counts['duplicates_excluded']} exact copies excluded; zero original-data overlap.", "",
             "| Mode | Eligible sessions / cases | Engine @5 | Baseline @5 | Paired difference 95% CI |",
             "|---|---|---|---|---|"]
    for mode, score in results.items():
        rates = score["hitrate_at5"]
        low, high = score["paired_difference_95_interval"]
        lines.append(f"| {mode} | {score['eligible_sessions']} / {score['hidden_cases']} | "
                     f"{rates['engine']:.2%} | {rates['co-occurrence']:.2%} | {low:+.2%} to {high:+.2%} |")
    lines += ["", "Primary endpoint: hide-one HitRate@5. First-step is secondary; "
              "a favourable secondary result does not replace the primary conclusion.", "",
              "Paired intervals use 10,000 resamples of unique sessions. "
              "No patient ids are available, so distinct sessions cannot establish distinct patients. "
              "Interpret the interval with that dependence limitation and the cohort's collection process."]
    write_report_section("prospective-evaluation", "\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
