"""Review edits use invented text and temporary worksheets, never the private real sheets."""
import json

import pytest
from fastapi.testclient import TestClient

from symptom_rec.data.sessions import Session
from symptom_rec.kb.loader import load_kb
from symptom_rec.normalize.review import load_review, make_manifest, prepare_review
from symptom_rec.normalize.review_web import create_review_app


@pytest.fixture
def review(tmp_path):
    kb = load_kb()
    sessions = [Session(i, "female", 30, [text], "ไอ", {}, [], [], [], None)
                for i, text in enumerate(["ไอ", "ไม่มีไข้", "เคยปวดหัว", "ไม่แน่ใจว่าคลื่นไส้", "ผื่น"])]
    manifest = make_manifest(sessions, kb)
    prepare_review(tmp_path, manifest, kb)
    with TestClient(create_review_app(tmp_path, kb)) as client:
        yield client, tmp_path, manifest, kb


def payload(sheet, **changes):
    return {"revision": sheet["revision"], "decision": "map", "labels": "cough:present;fever:unsure",
            "reviewer": "invented physician", "note": "invented annotation", **changes}


def test_review_page_is_local_and_blind(review):
    client, _, _, _ = review
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert "textContent" in page.text and "innerHTML" not in page.text
    assert "https://" not in page.text  # no external assets or APIs
    assert any(s["id"] == "cough" for s in client.get("/api/catalog").json())
    validation = client.get("/api/tasks/validation").json()
    assert validation["reviewed"] == 0 and validation["pending"] == 1
    assert not any("prediction" in key or "candidate" in key for task in validation["tasks"] for key in task)
    assert client.get("/", headers={"host": "remote.example"}).status_code == 400


def test_save_survives_reload_and_rejects_stale_edit(review):
    client, directory, manifest, kb = review
    sheet = client.get("/api/tasks/development").json()
    tid = sheet["tasks"][0]["id"]
    path = directory / "development.tsv"
    manifest_before = (directory / "manifest.json").read_bytes()
    response = client.post(f"/api/tasks/development/{tid}", json=payload(sheet),
                           headers={"X-Review-Action": "save"})
    assert response.status_code == 200
    assert response.json()["reviewed"] == 1
    reviewed, pending = load_review(directory, manifest, kb, "development")
    assert len(reviewed) == 1 and pending == 3
    assert reviewed[0][2] == {("cough", "present"), ("fever", "unsure")}
    assert manifest_before == (directory / "manifest.json").read_bytes()
    before = path.read_bytes()
    response = client.post(f"/api/tasks/development/{tid}", json=payload(sheet, note="stale"),
                           headers={"X-Review-Action": "save"})
    assert response.status_code == 409 and path.read_bytes() == before


@pytest.mark.parametrize("change", [
    {"reviewer": ""}, {"decision": "map", "labels": ""}, {"labels": "cough:maybe"},
    {"labels": "unknown_symptom:present"}, {"decision": "not_symptom"}, {"texts": '["edited input"]'},
])
def test_invalid_annotation_never_modifies_sheet(review, change):
    client, directory, _, _ = review
    sheet = client.get("/api/tasks/development").json()
    tid = sheet["tasks"][0]["id"]
    path = directory / "development.tsv"
    before = path.read_bytes()
    response = client.post(f"/api/tasks/development/{tid}", json=payload(sheet, **change),
                           headers={"X-Review-Action": "save"})
    assert response.status_code == 422 and path.read_bytes() == before


def test_cross_origin_and_form_save_are_rejected(review):
    client, directory, _, _ = review
    sheet = client.get("/api/tasks/development").json()
    tid = sheet["tasks"][0]["id"]
    before = (directory / "development.tsv").read_bytes()
    for headers in [{}, {"X-Review-Action": "save", "Origin": "https://remote.example"}]:
        assert client.post(f"/api/tasks/development/{tid}", json=payload(sheet),
                           headers=headers).status_code == 403
    assert (directory / "development.tsv").read_bytes() == before


def test_catalog_drift_refuses_to_open_review(review):
    _, directory, _, kb = review
    saved = json.loads((directory / "manifest.json").read_text())
    saved["catalog_sha256"] = "changed"
    (directory / "manifest.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="catalog changed"):
        create_review_app(directory, kb)
