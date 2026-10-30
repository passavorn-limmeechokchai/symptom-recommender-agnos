"""Compare the frozen retrieval profile once against fixed references; aggregates only."""
import copy
import hashlib
import json

import torch

from symptom_rec.api.service import load_service
from symptom_rec.data.protocol import locked_cases
from symptom_rec.data.sessions import find_csv, load_sessions
from symptom_rec.engine.adapter import EngineRecommender
from symptom_rec.evaluation import evaluate, evaluate_normalizer, typed_terms
from symptom_rec.kb.loader import load_kb, load_lexicon
from symptom_rec.normalize.learn import learn_aliases
from symptom_rec.normalize.normalizer import Normalizer
from symptom_rec.normalize.retrieval import load_retrieval
from symptom_rec.normalize.text import key
from symptom_rec.paths import BUILD_DIR, KB_DIR
from symptom_rec.reporting import write_report_section


def legacy_counts(result):
    return {group: {"total": sum(counts.values()), **counts} for group, counts in result["typed"].items()}


def main():
    torch.set_num_threads(4)
    directory = BUILD_DIR / "retrieval"
    config_path = directory / "config.json"
    selected_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lock = json.loads((BUILD_DIR / "evaluation_lock.json").read_text(encoding="utf-8"))
    by_row = {s.row: s for s in load_sessions(find_csv())}
    train = [by_row[row] for row in lock["train"]]
    test = [by_row[row] for row in lock["test"]]
    baseline = load_service(KB_DIR, BUILD_DIR, profile="rules")
    updated = load_service(KB_DIR, BUILD_DIR, profile="retrieval")
    report = {"selected_profile_sha256": selected_hash, "development_policy": config["policy"],
              "reference": "frozen session proxies, not physician accuracy", "models": {}}
    for label, service in [("rules", baseline), ("retrieval", updated)]:
        normalizer = evaluate_normalizer(service.normalizer, test, typed_terms(train), lock["reference_ids"])
        model = EngineRecommender(service.engine, service.normalizer, reference_ids=lock["reference_ids"])
        counts = {}
        for mode in ["hide_one", "first_step"]:
            results = evaluate(model, test, cases_by_row=locked_cases(lock, mode))
            counts[mode] = {"cases": sum(r["cases"] for r in results),
                            "hits_at5": sum(r["hits"][5] for r in results)}
        report["models"][label] = {"normalizer": legacy_counts(normalizer),
                                    "signed": normalizer["signed"], "recommendations": counts}

    manifest = json.loads((BUILD_DIR / "normalization_review" / "manifest.json").read_text())
    heldout_keys = {t["key"] for t in manifest["tasks"] if t["subset"] == "validation"}
    heldout = []
    for session in train:
        terms = [t for t in session.typed if key(t) in heldout_keys]
        if terms:
            case = copy.copy(session)
            case.typed = terms
            heldout.append(case)
    source_rows = set(manifest["alias_train_rows"])
    source = [s for s in train if s.row in source_rows]
    kb, lexicon = load_kb(), load_lexicon()
    aliases = learn_aliases(Normalizer(kb, lexicon), source)
    report["review_holdout_proxy"] = {}
    search = load_retrieval(kb, config_path)
    for label, normalizer in [("rules", Normalizer(kb, lexicon, aliases)),
                              ("retrieval", Normalizer(kb, lexicon, aliases, search))]:
        result = evaluate_normalizer(normalizer, heldout, typed_terms(source), lock["reference_ids"])
        report["review_holdout_proxy"][label] = {"normalizer": legacy_counts(result),
                                                "signed": result["signed"]}
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != selected_hash:
        raise RuntimeError("Selected profile changed during evaluation")
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    write_report_section("retrieval-evaluation", concise_report(report))


def concise_report(report):
    policy = report["development_policy"]
    lines = ["# E5-base + ridge", "",
             "Fit a ridge classifier to E5-base embeddings of KB names and aliases. "
             "Select its gate using five folds of 677 training sessions, excluding all sessions "
             "containing the 49 review-validation phrase groups. "
             f"Selection yields {policy['agrees']}/{policy['accepted']} proxy agreements "
             f"from {policy['unique']} distinct phrase groups. "
             "No policy meets the acceptance criteria on novel development queries alone.", "",
             "| Proxy | Rules | E5-base + ridge |", "|---|---:|---:|"]
    for group in ["seen", "unseen"]:
        a = report["models"]["rules"]["normalizer"][group]
        b = report["models"]["retrieval"]["normalizer"][group]
        lines.append(f"| {group}: matches record | {a.get('same symptom', 0)}/{a['total']} | "
                     f"{b.get('same symptom', 0)}/{b['total']} |")
    for mode in ["hide_one", "first_step"]:
        a = report["models"]["rules"]["recommendations"][mode]
        b = report["models"]["retrieval"]["recommendations"][mode]
        lines.append(f"| {mode} HitRate@5 | {a['hits_at5']}/{a['cases']} | {b['hits_at5']}/{b['cases']} |")
    a, b = (report["models"][name]["signed"] for name in ["rules", "retrieval"])
    extra = b.get("positive_assignments", 0) - a.get("positive_assignments", 0)
    agrees = b.get("agrees_with_record", 0) - a.get("agrees_with_record", 0)
    unverified = b.get("not_in_record", 0) - a.get("not_in_record", 0)
    holdout = [report["review_holdout_proxy"][name]["normalizer"]["unseen"]
               for name in ["rules", "retrieval"]]
    lines += ["", f"There are {extra} additional positive assignments: {agrees} match a recorded symptom "
              f"and {unverified} cannot be verified from the record. "
              f"On unseen review-holdout text, agreement is "
              f"{holdout[0].get('same symptom', 0)}/{holdout[0]['total']} versus "
              f"{holdout[1].get('same symptom', 0)}/{holdout[1]['total']}.", "",
              "Gains on previously seen text do not establish accuracy on new text. "
              "This remains an opt-in experiment; rules remain the default. "
              "Details are in `build/retrieval/evaluation.json`."]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
