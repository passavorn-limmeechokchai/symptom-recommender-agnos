"""Keep the test assignment, inputs and reference mappings fixed in private build/."""
import hashlib
import json

from .split import load_split, save_split, split_sessions


def session_digest(sessions):
    content = json.dumps([vars(s) for s in sessions], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


def fixed_split(sessions, path):
    """Reuse an existing split; only create one when no split exists."""
    if not path.exists():
        train, test = split_sessions(sessions)
        save_split(path, train, test)
    train, test = load_split(path)
    rows = {s.row for s in sessions}
    if (set(train) & set(test) or set(train) | set(test) != rows
            or len(train) + len(test) != len(sessions)):
        raise ValueError("Saved split does not partition this dataset; use a separate build directory")
    sides = {}
    test_rows = set(test)
    for s in sessions:
        side = s.row in test_rows
        if s.group() in sides and sides[s.group()] != side:
            raise ValueError("An exact-copy group crosses the saved split")
        sides[s.group()] = side
    return train, test


def evaluation_lock(sessions, train, test, normalizer, path):
    """Freeze raw cases and baseline reference ids; reject changed data or assignments.

    The manifest contains confidential text and belongs only in ignored build/.
    Target ids are read by the original rules, never by the model being evaluated.
    """
    from ..evaluation import first_step_cases, hide_one_cases

    identity = {"version": 1, "dataset_sha256": session_digest(sessions),
                "train": list(train), "test": list(test)}
    if path.exists():
        content = json.loads(path.read_text(encoding="utf-8"))
        if any(content.get(k) != value for k, value in identity.items()):
            raise ValueError("Evaluation lock differs from data/split; start a separate experiment")
        return content
    by_row = {s.row: s for s in sessions}
    cases = {}
    for mode, make in [("hide_one", hide_one_cases), ("first_step", first_step_cases)]:
        cases[mode] = [{"row": row, "shown": shown, "hidden": hidden}
                       for row in test for shown, hidden in make(by_row[row])]
    names = {name for s in sessions for name in s.confirmed() + s.absent}
    content = identity | {"cases": cases,
                          "reference_ids": {name: normalizer.symptom_ids(name) for name in sorted(names)}}
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return content


def locked_cases(lock, mode):
    cases = {row: [] for row in lock["test"]}
    for item in lock["cases"][mode]:
        cases[item["row"]].append((item["shown"], item["hidden"]))
    return cases
