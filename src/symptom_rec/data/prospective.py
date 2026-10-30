"""Freeze a comparison before collecting new sessions; keep all identities private."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import UTC, datetime

from .protocol import session_digest


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(root):
    files = [root / "pyproject.toml", root / "scripts" / "prospective_evaluation.py"]
    files += sorted((root / "src" / "symptom_rec").rglob("*.py"))
    files += sorted((root / "kb").rglob("*.yaml"))
    files += [root / "build" / name for name in
              ["split.json", "evaluation_lock.json", "memory.json", "aliases.json", "alpha.json"]]
    return {str(path.relative_to(root)): digest_file(path) for path in files}


def runtime():
    packages = ["numpy", "pandas", "pyyaml", "pythainlp", "pydantic"]
    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


def freeze_protocol(directory, root, original):
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"version": 1, "original_sha256": session_digest(original),
                "files": fingerprint(root), "runtime": runtime(),
                "profile": "rules", "baseline": "co-occurrence, original training split only",
                "primary": "hide_one_hitrate_at5", "secondary": "first_step_hitrate_at5",
                "bootstrap": {"rounds": 10000, "seed": 20261029, "unit": "unique session"},
                "overlap_policy": "reject any session matching the original dataset",
                "duplicate_policy": "one copy per structurally identical session",
                "collection_policy": "Define a consecutive collection window before inspecting its results; "
                "do not select rows or stop collection using observed scores",
                "scope": "recorded-symptom recovery; not physician-labelled normalization accuracy",
                "independence_limit": "No patient ids: distinct sessions do not establish distinct patients"}
    path = directory / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if any(saved.get(k) != value for k, value in identity.items()):
            raise ValueError("Frozen protocol differs from code, model, original data or runtime")
        return saved
    content = identity | {"created_utc": datetime.now(UTC).isoformat()}
    # Exclusive creation never overwrites a protocol from another process.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(content, indent=2) + "\n")
    return content


def session_identity(session):
    content = {name: value for name, value in vars(session).items() if name not in {"row", "copy_of"}}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def new_cohort(original, fresh):
    seen = {session_identity(session) for session in original}
    identities = [session_identity(session) for session in fresh]
    overlap = sum(identity in seen for identity in identities)
    if overlap:
        raise ValueError(f"New cohort overlaps the original dataset ({overlap} rows); evaluation refused")
    unique = []
    seen = set()
    for session, identity in zip(fresh, identities, strict=True):
        if identity not in seen:
            unique.append(session)
            seen.add(identity)
    if not any(session.associated for session in unique):
        raise ValueError("New cohort has no associated symptoms to hide")
    return unique, {"input_rows": len(fresh), "unique_sessions": len(unique),
                    "duplicates_excluded": len(fresh) - len(unique), "original_overlap": 0}


def summarize_comparison(results, rounds=10000, seed=20261029):
    from ..evaluation import bootstrap, difference, hit_rate, middle_95, session_hit_rate

    draws = bootstrap(results, 5, rounds=rounds, seed=seed)
    delta = difference(draws["engine"], draws["co-occurrence"])
    low, high = middle_95(delta)
    return {"eligible_sessions": len(results["engine"]),
            "hidden_cases": sum(row["cases"] for row in results["engine"]),
            "hitrate_at5": {name: hit_rate(rows, 5) for name, rows in results.items()},
            "per_session_hitrate_at5": {name: session_hit_rate(rows, 5) for name, rows in results.items()},
            "engine_minus_baseline": hit_rate(results["engine"], 5) - hit_rate(results["co-occurrence"], 5),
            "paired_difference_95_interval": [low, high],
            "interval_above_zero": low > 0}
