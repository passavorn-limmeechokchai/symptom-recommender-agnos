"""Private, train-derived phrase review and independent-label scoring.

No session diagnosis or model prediction is used as a gold label. The manifest is immutable;
physicians fill only the decision, labels, reviewer and note columns of its TSV worksheets.
"""
import csv
import hashlib
import json
import random
from collections import Counter

from ..data.protocol import session_digest
from .text import key

STATES = {"present", "absent", "unsure", "historical"}
DECISIONS = {"map", "clarify", "out_of_kb", "not_symptom"}
COLUMNS = ["id", "texts", "occurrences", "decision", "labels", "reviewer", "note"]


def make_manifest(train, kb, seed=20261028):
    phrases = {}
    for session in train:
        for text in session.typed:
            cleaned = key(text)
            if cleaned:
                phrases.setdefault(cleaned, Counter())[text] += 1
    keys = sorted(phrases)
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    validation = set(shuffled[:max(1, round(len(keys) * 0.2))])
    tasks = []
    for cleaned in keys:
        variants = phrases[cleaned]
        tasks.append({"id": hashlib.sha256(cleaned.encode()).hexdigest()[:16], "key": cleaned,
                      "texts": sorted(variants), "counts": dict(sorted(variants.items())),
                      "occurrences": sum(variants.values()),
                      "subset": "validation" if cleaned in validation else "development"})
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Phrase id collision")
    # A validation phrase may never enter alias learning, including via another session.
    alias_rows = [s.row for s in train if not any(key(t) in validation for t in s.typed)]
    catalog = [(sid, s.th, s.en) for sid, s in sorted(kb.symptoms.items())]
    return {"version": 1, "seed": seed, "train_sha256": session_digest(train),
            "train_rows": [s.row for s in train], "alias_train_rows": alias_rows,
            "catalog_sha256": hashlib.sha256(json.dumps(catalog, ensure_ascii=False).encode()).hexdigest(),
            "tasks": tasks}


def task_row(task):
    return {"id": task["id"], "texts": json.dumps(task["texts"], ensure_ascii=False),
            "occurrences": str(task["occurrences"]),
            "decision": "", "labels": "", "reviewer": "", "note": ""}


def prepare_review(directory, manifest, kb):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Review manifest differs from training data, split or KB; use a new directory")
    else:
        if any((directory / f"{part}.tsv").exists() for part in ["development", "validation"]):
            raise ValueError("Review sheets exist without their manifest; restore it before preparing")
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for part in ["development", "validation"]:
        sheet = directory / f"{part}.tsv"
        if not sheet.exists():
            with sheet.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter="\t")
                writer.writeheader()
                writer.writerows(task_row(t) for t in manifest["tasks"] if t["subset"] == part)
    catalog = directory / "symptoms.tsv"
    if not catalog.exists():
        with catalog.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["id", "th", "en", "kind", "parent"])
            writer.writerows([sid, s.th, s.en, s.kind, s.parent or ""] for sid, s in kb.symptoms.items())
    return {"groups": len(manifest["tasks"]),
            "occurrences": sum(t["occurrences"] for t in manifest["tasks"]),
            "subsets": {p: sum(t["subset"] == p for t in manifest["tasks"])
                        for p in ["development", "validation"]},
            "alias_training_sessions": len(manifest["alias_train_rows"])}


def parse_annotation(row, kb):
    """Validate only physician-editable columns; no prediction is a default label."""
    tid = row["id"]
    decision = row["decision"].strip()
    if not decision:
        if row["labels"].strip() or row["reviewer"].strip():
            raise ValueError(f"Missing decision for id {tid}")
        return None
    if decision not in DECISIONS or not row["reviewer"].strip():
        raise ValueError(f"Invalid decision or missing reviewer for id {tid}")
    labels = set()
    for label in row["labels"].split(";"):
        if not label.strip():
            continue
        fields = label.strip().split(":")
        if len(fields) != 2 or fields[0] not in kb.symptoms or fields[1] not in STATES:
            raise ValueError(f"Invalid symptom:state label for id {tid}")
        labels.add(tuple(fields))
    if decision == "map" and not labels:
        raise ValueError(f"Mapped phrase needs at least one label for id {tid}")
    if decision in {"out_of_kb", "not_symptom"} and labels:
        raise ValueError(f"Non-mapped decision must have no KB labels for id {tid}")
    return decision, labels


def load_review(directory, manifest, kb, subset):
    tasks = {t["id"]: t for t in manifest["tasks"] if t["subset"] == subset}
    reviewed = []
    seen = set()
    with (directory / f"{subset}.tsv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError("Review column names/order differ from the original sheet")
        for row in reader:
            tid = row["id"]
            if tid not in tasks or tid in seen:
                raise ValueError("Unknown or duplicate review id")
            seen.add(tid)
            task = tasks[tid]
            if (row["texts"] != task_row(task)["texts"]
                    or row["occurrences"] != str(task["occurrences"])):
                raise ValueError(f"Review input edited for id {tid}; only annotation columns are editable")
            annotation = parse_annotation(row, kb)
            if annotation is not None:
                decision, labels = annotation
                reviewed.append((task, decision, labels))
    if seen != set(tasks):
        raise ValueError("Review sheet has missing rows; retain unreviewed rows instead of deleting them")
    return reviewed, len(tasks) - len(reviewed)


def score_review(normalizer, reviewed, pending):
    """Exact signed sets penalize extra and missing symptoms. Report deferrals separately.

    Phrase means give each cleaned group equal weight and average its raw variants equally.
    Occurrence-weighted exact match is also returned. Candidate recall uses root+state and
    at most three candidates per deferred fragment, only on fully deferred mapped queries.
    """
    counts = Counter()
    exact_group_sum = 0.0
    exact_occurrences = 0
    mapped_occurrences = 0
    for task, decision, gold in reviewed:
        group_hits = 0
        for text in task["texts"]:
            matches = normalizer.read_symptoms(text)
            predicted = {(m.symptom, m.state) for m in matches if m.symptom}
            target = gold if decision == "map" else set()
            counts["true_positive"] += len(predicted & target)
            counts["false_positive"] += len(predicted - target)
            counts["false_negative"] += len(target - predicted)
            counts["raw_variants"] += 1
            counts["auto_assigned_variants"] += bool(predicted)
            if decision == "map":
                hit = predicted == target
                group_hits += hit
                counts["mapped_variants"] += 1
                counts["exact_variants"] += hit
                mapped_occurrences += task["counts"][text]
                exact_occurrences += hit * task["counts"][text]
                if not predicted:
                    counts["deferred_mapped_variants"] += 1
                    options = {(normalizer.root(sid), m.state) for m in matches
                               for sid in m.candidates[:3]}
                    roots = {(normalizer.root(sid), state) for sid, state in target}
                    counts["candidate_targets"] += len(roots)
                    counts["candidate_hits"] += len(options & roots)
            else:
                counts["nonmap_variants"] += 1
                counts["correctly_deferred_nonmap"] += not predicted
        if decision == "map":
            counts["mapped_groups"] += 1
            exact_group_sum += group_hits / len(task["texts"])

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    tp, fp, fn = (counts[n] for n in ["true_positive", "false_positive", "false_negative"])
    return {"status": "awaiting_review" if not reviewed else "partial" if pending else "complete",
            "reviewed_groups": len(reviewed), "pending_groups": pending, "counts": dict(counts),
            "metrics": {"assignment_precision": ratio(tp, tp + fp),
                        "assignment_recall": ratio(tp, tp + fn),
                        "assignment_f1": ratio(2 * tp, 2 * tp + fp + fn),
                        "exact_match_per_phrase": ratio(exact_group_sum, counts["mapped_groups"]),
                        "exact_match_per_occurrence": ratio(exact_occurrences, mapped_occurrences),
                        "auto_assignment_coverage": ratio(counts["auto_assigned_variants"],
                                                          counts["raw_variants"]),
                        "candidate_recall_at3_when_deferred": ratio(counts["candidate_hits"],
                                                                   counts["candidate_targets"]),
                        "appropriate_deferral": ratio(counts["correctly_deferred_nonmap"],
                                                       counts["nonmap_variants"])}}
