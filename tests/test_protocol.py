import json

import pytest

from symptom_rec.data.protocol import evaluation_lock, fixed_split, locked_cases
from symptom_rec.data.sessions import Session
from symptom_rec.data.split import save_split
from symptom_rec.data.validation import training_folds


def sessions():
    rows = [Session(i, "female", 30, ["ไอ"], "ไอ", {}, ["ไข้"] if i % 2 else [],
                    [], [], None) for i in range(20)]
    rows.append(Session(20, "female", 30, ["ไอ"], "ไอ", {}, ["ไข้"], [], [], None, copy_of=1))
    return rows


class Reference:
    def symptom_ids(self, name):
        return {"ไอ": ["cough"], "ไข้": ["fever"]}[name]


def test_saved_split_is_reused_without_rewriting(tmp_path):
    path = tmp_path / "split.json"
    save_split(path, list(range(19)) + [20], [19])
    before = path.read_bytes()
    assert fixed_split(sessions(), path) == (list(range(19)) + [20], [19])
    assert path.read_bytes() == before


def test_invalid_or_leaking_saved_splits_are_rejected(tmp_path):
    path = tmp_path / "split.json"
    save_split(path, list(range(20)), [20])
    with pytest.raises(ValueError, match="copy group"):
        fixed_split(sessions(), path)
    save_split(path, [0], [1])
    with pytest.raises(ValueError, match="partition"):
        fixed_split(sessions(), path)


def test_lock_keeps_inputs_and_reference_ids_and_rejects_data_change(tmp_path):
    rows = sessions()
    path = tmp_path / "lock.json"
    train, test = list(range(19)) + [20], [19]
    lock = evaluation_lock(rows, train, test, Reference(), path)
    assert locked_cases(lock, "hide_one")[19][0][1] == "ไข้"
    assert lock["reference_ids"]["ไข้"] == ["fever"]
    before = path.read_bytes()
    assert evaluation_lock(rows, train, test, None, path) == lock
    assert path.read_bytes() == before
    rows[0].age += 1
    with pytest.raises(ValueError, match="lock differs"):
        evaluation_lock(rows, train, test, Reference(), path)


def test_training_folds_keep_groups_together_and_cover_each_row_once():
    rows = sessions()
    folds = training_folds(rows)
    validated = []
    for train, validation in folds:
        assert {s.group() for s in train}.isdisjoint(s.group() for s in validation)
        validated += [s.row for s in validation]
        assert any(s.associated for s in validation)
        assert any(not s.associated for s in validation)
    assert sorted(validated) == list(range(21))
    assert [[s.row for s in v] for _, v in folds] == [
        [s.row for s in v] for _, v in training_folds(rows)]


def test_lock_does_not_accept_another_test_assignment(tmp_path):
    path = tmp_path / "lock.json"
    rows = sessions()
    evaluation_lock(rows, list(range(19)) + [20], [19], Reference(), path)
    with pytest.raises(ValueError, match="lock differs"):
        evaluation_lock(rows, list(range(18)) + [19, 20], [18], Reference(), path)
    assert json.loads(path.read_text())["test"] == [19]
