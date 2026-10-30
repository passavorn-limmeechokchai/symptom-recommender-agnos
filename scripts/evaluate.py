"""Evaluate the recommenders, the normalizer and the engine (see src/symptom_rec/evaluation.py).

    python scripts/evaluate.py      # run python scripts/build.py first

Recommenders, two tests: hide one symptom, and first step (chief complaint and HPI only):
    training set: leave-one-out hit rates, the numbers tuning looks at;
    test set: hit rates with 95% bootstrap intervals, and each model compared with co-occurrence.
Normalizer, test set: how many of the data's symptom names it places, whether what patients typed
    maps to a symptom recorded in the same session, and how many HPI answers it reads; with the
    rules alone and with the names learned from the training set (build/aliases.json).
Clinical cases: the engine on the textbook vignettes (tests/cases/vignettes.yaml).
The tables are printed and update their section in REPORT.md (aggregate numbers only).
"""
import argparse
import json
from pathlib import Path

from symptom_rec.baselines.counting import Cooccurrence, Popularity
from symptom_rec.data.protocol import evaluation_lock, fixed_split, locked_cases
from symptom_rec.data.sessions import find_csv, load_sessions
from symptom_rec.engine.adapter import EngineRecommender
from symptom_rec.engine.engine import Engine
from symptom_rec.evaluation import (
    K_VALUES,
    TYPED_RESULTS,
    bootstrap,
    check_vignette,
    difference,
    evaluate,
    evaluate_normalizer,
    first_step_cases,
    hide_one_cases,
    hit_rate,
    middle_95,
    session_hit_rate,
    typed_terms,
)
from symptom_rec.kb.loader import load_kb, load_lexicon, load_vignettes
from symptom_rec.normalize.learn import load_aliases
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.paths import BUILD_DIR, REPO_ROOT
from symptom_rec.reporting import write_report_section

TESTS = {
    "hide one symptom": hide_one_cases,
    "first step: chief complaint and HPI only": first_step_cases,
}
REFERENCE = "co-occurrence"  # every other model is compared with this one
COMPARE_AT_K = 5
LAYERS = ["exact", "learned", "fillers", "body_part", "split", "fuzzy", "semantic", "not placed"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, help="Agnos CSV (default: the one in data/raw/)")
    parser.add_argument("--semantic", action="store_true", help="Compare the train-calibrated local fallback")
    args = parser.parse_args()

    split_file = BUILD_DIR / "split.json"
    if not split_file.exists():
        raise SystemExit("build/split.json not found: run python scripts/build.py first")
    sessions = load_sessions(args.data or find_csv())
    train_rows, test_rows = fixed_split(sessions, split_file)
    by_row = {}
    for s in sessions:
        by_row[s.row] = s
    train = []
    for row in train_rows:
        train.append(by_row[row])
    test = []
    for row in test_rows:
        test.append(by_row[row])

    kb = load_kb()
    lexicon = load_lexicon()
    learned = load_aliases(BUILD_DIR / "aliases.json")
    alpha = json.loads((BUILD_DIR / "alpha.json").read_text(encoding="utf-8"))["alpha"]

    # the engine learns from the training set; three ways of mixing book and data
    engine = Engine(kb, alpha=alpha)
    normalizer = Normalizer(kb, lexicon, learned)
    lock = evaluation_lock(sessions, train_rows, test_rows, normalizer, BUILD_DIR / "evaluation_lock.json")
    EngineRecommender(engine, normalizer).learn(train)
    models = {
        "popularity": Popularity(train),
        "co-occurrence": Cooccurrence(train),
        "engine, data part only (α = 0)": EngineRecommender(engine, normalizer, 0.0, lock["reference_ids"]),
        "engine, book part only (α = 1)": EngineRecommender(engine, normalizer, 1.0, lock["reference_ids"]),
        f"engine (α = {alpha}, tuned)": EngineRecommender(engine, normalizer, alpha, lock["reference_ids"]),
    }

    semantic_normalizer = None
    if args.semantic:
        import torch

        from symptom_rec.normalize.semantic import load_search
        torch.set_num_threads(4)
        directory = BUILD_DIR / "semantic"
        search = load_search(kb, directory / "config.json", train)
        search.prepare([name for s in train + test for name in s.confirmed()])
        semantic_normalizer = Normalizer(kb, lexicon, load_aliases(directory / "aliases.json"), search)
        semantic_alpha = json.loads((directory / "alpha.json").read_text())["alpha"]
        semantic_engine = Engine(kb, alpha=semantic_alpha)
        EngineRecommender(semantic_engine, semantic_normalizer).learn(train)
        models[f"engine + semantic fallback (α = {semantic_alpha})"] = EngineRecommender(
            semantic_engine, semantic_normalizer, semantic_alpha, lock["reference_ids"])

    measurements = {}
    lines = ["# Evaluation", "", "Test membership, raw cases and baseline target mappings are locked "
             "in private `build/evaluation_lock.json`. "
             "Changing a normalizer does not redefine the targets.", ""]
    for mode, (title, make_cases) in zip(["hide_one", "first_step"], TESTS.items(), strict=True):
        train_results = {}
        test_results = {}
        for name, model in models.items():
            train_results[name] = evaluate(model, train, leave_one_out=True, make_cases=make_cases)
            test_results[name] = evaluate(model, test, cases_by_row=locked_cases(lock, mode))
        measurements[mode] = {"train": train_results, "test": test_results}
        lines += recommender_report(title, train_results, test_results)

    seen = typed_terms(train)
    rules_only = evaluate_normalizer(Normalizer(kb, lexicon), test, seen, lock["reference_ids"])
    with_learned = evaluate_normalizer(normalizer, test, seen, lock["reference_ids"])
    lines += normalizer_report(rules_only, with_learned, len(learned))
    semantic_results = None
    if semantic_normalizer is not None:
        semantic_results = evaluate_normalizer(semantic_normalizer, test, seen, lock["reference_ids"])
        lines += semantic_report(with_learned, semantic_results)
    cases = load_vignettes()
    clinical = {case.id: check_vignette(engine, case, alpha) for case in cases}
    lines += vignette_report(engine, cases, alpha, clinical)
    report = "\n".join(lines)
    print(report)
    section = "semantic-evaluation" if args.semantic else "evaluation"
    detail = {"recommendations": measurements, "normalizer": with_learned,
              "rules_only_normalizer": rules_only, "semantic_normalizer": semantic_results,
              "clinical": clinical, "alpha": alpha}
    (BUILD_DIR / f"{section}.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
    brief = (concise_semantic(with_learned, semantic_results, measurements) if args.semantic
             else concise_evaluation(measurements, with_learned, clinical, alpha))
    report_file = write_report_section(section, brief)
    print(f"-> {report_file.relative_to(REPO_ROOT)}")


def recommender_report(title, train_results, test_results):
    lines = [f"## Recommendations, {title}", ""]

    first = list(train_results)[0]
    lines.append(f"Training set, leave-one-out: {count_line(train_results[first])}")
    lines.append("")
    lines.append(header("Model"))
    for name, results in train_results.items():
        row = name
        for k in K_VALUES:
            row += f" | {percent(hit_rate(results, k))}"
        lines.append("| " + row + " |")
    lines.append("")

    lines.append(f"Test set: {count_line(test_results[first])}. In brackets: 95% bootstrap interval.")
    lines.append("")
    lines.append(header("Model"))
    rates = {}
    for k in K_VALUES:
        rates[k] = bootstrap(test_results, k)
    for name, results in test_results.items():
        row = name
        for k in K_VALUES:
            low, high = middle_95(rates[k][name])
            row += f" | {percent(hit_rate(results, k))} ({percent(low)}–{percent(high)})"
        lines.append("| " + row + " |")
    lines.append("")

    lines.append("Per-session mean: average within each session, then give each session equal weight.")
    lines.append("")
    lines.append(header("Model (per-session mean)"))
    macro_rates = {k: bootstrap(test_results, k, metric=session_hit_rate) for k in K_VALUES}
    for name, results in test_results.items():
        row = name
        for k in K_VALUES:
            low, high = middle_95(macro_rates[k][name])
            row += f" | {percent(session_hit_rate(results, k))} ({percent(low)}–{percent(high)})"
        lines.append("| " + row + " |")
    lines.append("")

    lines.append(f"Compared with {REFERENCE} on the test set, HitRate@{COMPARE_AT_K}:")
    lines.append("")
    for name, results in test_results.items():
        if name == REFERENCE:
            continue
        gain = hit_rate(results, COMPARE_AT_K) - hit_rate(test_results[REFERENCE], COMPARE_AT_K)
        low, high = middle_95(difference(rates[COMPARE_AT_K][name], rates[COMPARE_AT_K][REFERENCE]))
        lines.append(f"- {name}: {points(gain)} points (95% interval {points(low)} to {points(high)})")
    lines.append("")

    lines.append("Missed in the top 5 but asked anyway, as a red-flag or HPI question (test set):")
    lines.append("")
    for name, results in test_results.items():
        asked = 0
        cases = 0
        for r in results:
            asked += r["asked"]
            cases += r["cases"]
        if asked:
            lines.append(f"- {name}: {percent(asked / cases)} of cases")
    lines.append("")
    return lines


def vignette_report(engine, cases, alpha, checks_by_case=None):
    lines = ["## Clinical cases (textbook vignettes)", ""]
    passed = {}
    total = {}
    book_ranks = []
    for case in cases:
        checks = (checks_by_case[case.id] if checks_by_case is not None
                  else check_vignette(engine, case, alpha))
        for name, ok in checks.items():
            if name == "final_diagnosis_rank":
                book_ranks.append((case.id, ok))
            else:
                total[name] = total.get(name, 0) + 1
                passed[name] = passed.get(name, 0) + (1 if ok else 0)
    lines.append("Simple cases: every expectation must hold.")
    lines.append("")
    lines.append("| Expectation | Passed |")
    lines.append("| --- | --- |")
    for name, n in total.items():
        lines.append(f"| {name} | {passed[name]} / {n} |")
    lines.append("")
    lines.append("Book cases: rank of the confirmed diagnosis in the engine's DDx (the book case gives "
                 "more history than the KB can hold, so this is reported, not passed or failed).")
    lines.append("")
    lines.append("| Case | Rank |")
    lines.append("| --- | --- |")
    for case_id, rank in book_ranks:
        lines.append(f"| {case_id} | {rank if rank else 'not in DDx'} |")
    lines.append("")
    return lines


def normalizer_report(rules_only, with_learned, n_learned):
    lines = ["## Normalizer (test set)", ""]

    names = with_learned["names"]
    total = sum(names.values())
    lines.append(f"Symptom names in the data, in the data's own wording: {total} mentions.")
    lines.append("")
    lines.append("| Placed by | Share |")
    lines.append("| --- | --- |")
    for layer in LAYERS:
        lines.append(f"| {layer} | {percent(names.get(layer, 0) / total)} |")
    lines.append("")

    before = rules_only["typed"]
    after = with_learned["typed"]
    lines.append(
        "What patients typed, against the symptoms recorded in the same session. Terms typed exactly "
        f"like a recorded name ({sum(after['same text'].values())}) are left out; "
        f"{n_learned} names were learned from the training set."
    )
    lines.append("")
    lines.append(
        f"| Result | Rules only, all ({sum(before['seen'].values()) + sum(before['unseen'].values())}) "
        f"| Rules + learned, typed in training too ({sum(after['seen'].values())}) "
        f"| Rules + learned, new terms ({sum(after['unseen'].values())}) |"
    )
    lines.append("| --- | --- | --- | --- |")
    rules_all = {}
    for group in ("seen", "unseen"):
        for result, n in before[group].items():
            rules_all[result] = rules_all.get(result, 0) + n
    for result in TYPED_RESULTS:
        lines.append(
            f"| {result} | {share(rules_all, result)} | {share(after['seen'], result)} "
            f"| {share(after['unseen'], result)} |"
        )
    lines.append("")

    answers = with_learned["answers"]
    answered = answers["read"] + answers["not read"]
    lines.append(
        f"HPI answers of placed symptoms with at least one KB answer read: {answers['read']} of "
        f"{answered} ({percent(answers['read'] / answered)}). Many answers have no KB question to "
        "go to (for example COVID-19 contact); whether an answer is read correctly is checked by "
        "the unit tests."
    )
    lines.append("")
    return lines


def semantic_report(before, after):
    lines = ["## Semantic fallback and polarity-aware proxies", "",
             "All comparisons use the frozen baseline reference ids. These are session-consistency "
             "proxies, not physician-labelled semantic accuracy. A negated match is an assignment "
             "even though the legacy positive-id measure calls it not placed.", "",
             "| Group | Before: same symptom | With semantic: same symptom | Text occurrences |",
             "| --- | --- | --- | --- |"]
    for group in ["same text", "seen", "unseen"]:
        a, b = before["typed"][group], after["typed"][group]
        lines.append(f"| {group} | {a.get('same symptom', 0)} | "
                     f"{b.get('same symptom', 0)} | {sum(b.values())} |")
    lines += ["", "| Signed assignments / deferrals | Before | With semantic |", "| --- | --- | --- |"]
    for name in ["positive_assignments", "negative_assignments", "unsure_assignments",
                 "historical_assignments",
                 "agrees_with_record", "opposes_record",
                 "not_in_record", "deferred", "deferred_candidate_agrees"]:
        lines.append(f"| {name} | {before['signed'].get(name, 0)} | {after['signed'].get(name, 0)} |")
    lines += ["", "Deferred queries with a polarity-compatible reference among candidates "
              "(at most three candidates per unmatched phrase):", "",
              "| Group | Deferred queries | Before: candidate agrees | With semantic: candidate agrees |",
              "| --- | --- | --- | --- |"]
    for group in ["same text", "seen", "unseen"]:
        a, b = before["signed_by_group"][group], after["signed_by_group"][group]
        lines.append(f"| {group} | {b.get('deferred', 0)} | "
                     f"{a.get('deferred_candidate_agrees', 0)} | {b.get('deferred_candidate_agrees', 0)} |")
    lines += ["", "Assignment counts can exceed query counts for multi-symptom text. An assignment absent "
              "from the session record is unverified, not necessarily wrong. Candidate agreement is "
              "reported separately from automatic mapping. The inspected test set is a development "
              "comparison; fresh independent labels are needed for a confirmatory accuracy claim.", ""]
    return lines


def header(first_column):
    titles = first_column
    line = "---"
    for k in K_VALUES:
        titles += f" | HitRate@{k}"
        line += " | ---"
    return "| " + titles + " |\n| " + line + " |"


def count_line(results):
    cases = 0
    for r in results:
        cases += r["cases"]
    return f"{len(results)} sessions with associated symptoms, {cases} hidden-symptom cases"


def share(counter, name):
    total = sum(counter.values())
    if total == 0:
        return "–"
    return percent(counter.get(name, 0) / total)


def percent(value):
    return f"{value * 100:.0f}%"


def points(value):
    return f"{value * 100:+.0f}"


def hits_at5(rows):
    return f"{sum(r['hits'][5] for r in rows)}/{sum(r['cases'] for r in rows)} ({hit_rate(rows, 5):.2%})"


def concise_evaluation(measurements, normalizer, clinical, alpha):
    hide = measurements["hide_one"]["test"]
    first = measurements["first_step"]["test"]
    lines = ["# Results", "", "| Method | Hide-one @3 | Hide-one @5 | Hide-one @8 | First-step @5 |",
             "|---|---:|---:|---:|---:|"]
    for name, rows in hide.items():
        label = {"popularity": "Popularity", "co-occurrence": "Co-occurrence",
                 "engine, data part only (α = 0)": "Engine data (α = 0)",
                 "engine, book part only (α = 1)": "Book only (α = 1)"}.get(name, f"Hybrid (α = {alpha})")
        lines.append(f"| {label} | {hit_rate(rows, 3):.2%} | {hits_at5(rows)} | "
                     f"{hit_rate(rows, 8):.2%} | {hits_at5(first[name])} |")
    hybrid = f"engine (α = {alpha}, tuned)"
    intervals = []
    primary_interval = None
    for mode in ["hide_one", "first_step"]:
        rates = bootstrap(measurements[mode]["test"], 5)
        low, high = middle_95(difference(rates[hybrid], rates[REFERENCE]))
        if mode == "hide_one":
            primary_interval = (low, high)
        intervals.append(f"{mode.replace('_', '-')}: {points(low)} to {points(high)} percentage points")
    gain = hit_rate(hide[hybrid], 5) - hit_rate(hide[REFERENCE], 5)
    low, high = primary_interval
    conclusion = ("is above zero" if low > 0 else "is below zero" if high < 0 else "includes zero")
    net = (sum(r["hits"][5] for r in hide[hybrid])
           - sum(r["hits"][5] for r in hide["engine, data part only (α = 0)"]))
    lines += ["", f"The hybrid differs from co-occurrence by {gain * 100:+.2f} percentage points "
              "on hide-one. "
              f"The paired bootstrap 95% CI for that difference {conclusion} ({'; '.join(intervals)}). "
              f"Compared with the same engine at α = 0, the net difference is {net:+d} case(s). "
              "The raw-label baseline comparison does not isolate the contribution of the KB.", "",
              "Giving each session equal weight yields hide-one @5 of "
              f"{session_hit_rate(hide[hybrid], 5):.2%} "
              f"versus {session_hit_rate(hide[REFERENCE], 5):.2%} for the baseline. "
              "This is reported separately from the primary metric, "
              "which weights each hidden case equally.", "",
              "## Text normalization", "",
              "Compare symptoms extracted from typed text with confirmed symptoms in the same session:", "",
              "| Test text group | Matches a recorded symptom |", "|---|---:|"]
    labels = {"same text": "Same as a recorded name after cleaning",
              "seen": "Different from recorded names; seen in training",
              "unseen": "Different from recorded names; unseen in training"}
    for group, label in labels.items():
        counts = normalizer["typed"][group]
        lines.append(f"| {label} | {counts.get('same symptom', 0)}/{sum(counts.values())} |")
    answers = normalizer["answers"]
    lines += ["", "These are session-consistency proxies, not physician-labelled accuracy: "
              "records may be incomplete, and this metric does not penalize all extra predictions. "
              f"At least one HPI answer was parsed in {answers['read']}/{sum(answers.values())} items. "
              "That measures coverage of supported questions, not parsing correctness.", "",
              "## Clinical regression", ""]
    top3 = [c["top3_includes"] for c in clinical.values() if "top3_includes" in c]
    other = [value for c in clinical.values() for name, value in c.items()
             if name not in {"top3_includes", "final_diagnosis_rank"}]
    ranks = [c["final_diagnosis_rank"] for c in clinical.values() if "final_diagnosis_rank" in c]
    failures = [cid for cid, checks in clinical.items() if checks.get("top3_includes") is False]
    lines += [f"{len(clinical)} vignettes: top-three expectations {sum(top3)}/{len(top3)}; "
              f"safety, flow and other expectations {sum(other)}/{len(other)}; "
              f"book diagnoses in the top three "
              f"{sum(r is not None and r <= 3 for r in ranks)}/{len(ranks)}", "",
              f"Unmet top-three expectation: {', '.join(failures) or 'none'}. "
              "The leiomyoma book case ranks fourth; examination and ultrasound "
              "are outside the current input. "
              "Vignettes are regression checks and have informed development, so they are not independent "
              "clinical validation."]
    return "\n".join(lines)


def concise_semantic(before, after, measurements):
    lines = ["# E5-small test", "",
             "E5-small proposes candidates when existing rules cannot assign a symptom:", "",
             "| Proxy | Rules | E5-small |", "|---|---:|---:|"]
    for group in ["seen", "unseen"]:
        a, b = before["typed"][group], after["typed"][group]
        lines.append(f"| {group}: matches record | {a.get('same symptom', 0)}/{sum(a.values())} | "
                     f"{b.get('same symptom', 0)}/{sum(b.values())} |")
    a, b = before["signed"], after["signed"]
    lines.append(f"| Reference in top-three candidates when deferred | "
                 f"{a.get('deferred_candidate_agrees', 0)}/{a.get('deferred', 0)} | "
                 f"{b.get('deferred_candidate_agrees', 0)}/{b.get('deferred', 0)} |")
    for mode in ["hide_one", "first_step"]:
        rows = measurements[mode]["test"]
        baseline = next(value for name, value in rows.items() if name.startswith("engine ("))
        semantic = next(value for name, value in rows.items() if name.startswith("engine + semantic"))
        lines.append(f"| {mode} HitRate@5 | {hits_at5(baseline)} | {hits_at5(semantic)} |")
    lines += ["", "Candidate agreement is measured separately from automatic mapping and HitRate. "
              "It measures consistency with records, not accuracy verified by physicians."]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
