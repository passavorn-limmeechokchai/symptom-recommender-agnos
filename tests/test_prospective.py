"""Prospective protocol identity, overlap isolation and paired metrics on invented sessions."""
import json

import pytest

from symptom_rec.data.prospective import freeze_protocol, new_cohort, session_identity, summarize_comparison
from symptom_rec.data.sessions import Session


def session(row=0, cc="ไอ", associated=None, copy_of=None):
    return Session(row, "female", 30, [cc], cc, {}, associated or [], [], [], None, copy_of)


@pytest.fixture
def root(tmp_path):
    files = ["pyproject.toml", "src/symptom_rec/example.py", "kb/symptoms.yaml",
             "scripts/prospective_evaluation.py", "build/split.json", "build/evaluation_lock.json",
             "build/memory.json", "build/aliases.json", "build/alpha.json"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("invented fixture")
    return tmp_path


def test_protocol_is_idempotent_and_rejects_model_change(root):
    directory = root / "build/prospective"
    original = [session()]
    saved = freeze_protocol(directory, root, original)
    before = (directory / "protocol.json").read_bytes()
    assert saved["primary"] == "hide_one_hitrate_at5"
    assert freeze_protocol(directory, root, original) == saved
    assert before == (directory / "protocol.json").read_bytes()
    (root / "build/alpha.json").write_text('{"alpha": 0.9}')
    with pytest.raises(ValueError, match="Frozen protocol differs"):
        freeze_protocol(directory, root, original)
    assert before == (directory / "protocol.json").read_bytes()


def test_protocol_rejects_original_data_change(root):
    directory = root / "build/prospective"
    freeze_protocol(directory, root, [session()])
    with pytest.raises(ValueError, match="differs"):
        freeze_protocol(directory, root, [session(cc="ไข้")])


def test_cohort_rejects_original_even_with_new_row_numbers():
    original = [session(row=41)]
    assert session_identity(original[0]) == session_identity(session(row=10, copy_of=2))
    with pytest.raises(ValueError, match="overlaps.*1 rows"):
        new_cohort(original, [session(row=0), session(row=1, cc="ไข้", associated=["ปวดหัว"])])


def test_cohort_deduplicates_without_training_or_resplitting():
    original = [session()]
    fresh = [session(row=10, cc="ไข้", associated=["ปวดหัว"]),
             session(row=11, cc="ไข้", associated=["ปวดหัว"], copy_of=10),
             session(row=12, cc="ผื่น")]
    unique, counts = new_cohort(original, fresh)
    assert [s.row for s in unique] == [10, 12]
    assert counts == {"input_rows": 3, "unique_sessions": 2, "duplicates_excluded": 1,
                      "original_overlap": 0}


def test_cohort_with_no_hidden_targets_cannot_report_accuracy():
    with pytest.raises(ValueError, match="no associated symptoms"):
        new_cohort([session()], [session(cc="ไข้")])


def test_primary_counts_cases_while_bootstrap_keeps_sessions_paired():
    rows = [{"row": 1, "cases": 1, "hits": {5: 1}}, {"row": 2, "cases": 3, "hits": {5: 0}}]
    results = {"engine": rows, "co-occurrence": rows}
    score = summarize_comparison(results, rounds=200)
    assert score == summarize_comparison(results, rounds=200)
    assert score["hitrate_at5"]["engine"] == 0.25
    assert score["per_session_hitrate_at5"]["engine"] == 0.5
    assert score["paired_difference_95_interval"] == [0, 0]
    assert not score["interval_above_zero"]
    assert score["eligible_sessions"] == 2 and score["hidden_cases"] == 4
    json.dumps(score, allow_nan=False)
    results["co-occurrence"] = list(reversed(rows))
    with pytest.raises(ValueError, match="same sessions"):
        summarize_comparison(results, rounds=200)


def test_cli_freezes_new_targets_and_reuses_frozen_training_only(root, monkeypatch, capsys):
    from types import SimpleNamespace

    from scripts import prospective_evaluation as cli

    from symptom_rec.engine.engine import Engine
    from symptom_rec.kb.loader import load_kb, load_lexicon
    from symptom_rec.normalize.normalizer import Normalizer

    original = [session(0, "ไอ", ["ไข้"]), session(1, "ผื่น", ["ปวดหัว"])]
    fresh = [session(10, "ไอ", ["ปวดหัว"]), session(11, "ไข้", ["ผื่น"])]
    new_path = root / "new.csv"
    directory = root / "build/prospective"
    kb = load_kb()
    engine = Engine(kb)
    normalizer = Normalizer(kb, load_lexicon())
    memory_before = dict(engine.memory.counts.sessions_with)
    report_sections = []
    monkeypatch.setattr(cli, "BUILD_DIR", root / "build")
    monkeypatch.setattr(cli, "REPO_ROOT", root)
    monkeypatch.setattr(cli, "find_csv", lambda: root / "original.csv")
    monkeypatch.setattr(cli, "load_sessions", lambda path: fresh if path == new_path else original)
    monkeypatch.setattr(cli, "fixed_split", lambda sessions, path: ([0], [1]))
    monkeypatch.setattr(cli, "load_service", lambda *args, **kwargs: SimpleNamespace(
        mode="hybrid", engine=engine, normalizer=normalizer, model_version="invented"))
    monkeypatch.setattr(cli, "write_report_section",
                        lambda name, content: report_sections.append((name, content)))
    monkeypatch.setattr("sys.argv", ["prospective_evaluation.py", "freeze", "--directory", str(directory)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_new_data"
    monkeypatch.setattr("sys.argv", ["prospective_evaluation.py", "evaluate", "--directory", str(directory),
                                    "--data", str(new_path)])
    cli.main()
    score = json.loads(capsys.readouterr().out)
    assert score["counts"]["unique_sessions"] == 2
    assert score["results"]["hide_one"]["hidden_cases"] == 2
    assert report_sections[0][0] == "prospective-evaluation"
    lock = json.loads((directory / "evaluation_lock.json").read_text())
    assert lock["train"] == [] and lock["test"] == [10, 11]
    assert set(lock["reference_ids"]) == {"ไอ", "ปวดหัว", "ไข้", "ผื่น"}
    assert engine.memory.counts.sessions_with == memory_before  # never learn from the fresh cohort
    fresh.append(session(12, "ปวดท้อง", ["คลื่นไส้"]))
    with pytest.raises(ValueError, match="lock differs"):
        cli.main()
