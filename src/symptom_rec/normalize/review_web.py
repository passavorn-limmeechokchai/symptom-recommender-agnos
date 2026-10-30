"""Loopback-only physician annotation UI; it never suggests or invents gold labels."""
import csv
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .review import COLUMNS, load_review, parse_annotation

Subset = Literal["development", "validation"]


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: str
    decision: Literal["map", "clarify", "out_of_kb", "not_symptom"]
    labels: str = ""
    reviewer: str
    note: str = ""


def snapshot(directory, manifest, kb, subset):
    # Validate the entire sheet, including immutable inputs, before displaying or editing it.
    reviewed, pending = load_review(directory, manifest, kb, subset)
    path = directory / f"{subset}.tsv"
    content = path.read_bytes()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {"revision": hashlib.sha256(content).hexdigest(), "tasks": rows,
            "reviewed": len(reviewed), "pending": pending, "subset": subset}


def save_annotation(directory, manifest, kb, subset, tid, payload):
    path = directory / f"{subset}.tsv"
    with (directory / ".review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = snapshot(directory, manifest, kb, subset)
        if current["revision"] != payload.revision:
            raise HTTPException(409, "Sheet changed; reload before saving")
        row = next((row for row in current["tasks"] if row["id"] == tid), None)
        if row is None:
            raise HTTPException(404, "Unknown review id")
        row.update(payload.model_dump(exclude={"revision"}))
        try:
            parse_annotation(row, kb)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                             dir=directory, prefix=".review-", delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, COLUMNS, delimiter="\t")
                writer.writeheader()
                writer.writerows(current["tasks"])
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return snapshot(directory, manifest, kb, subset)


def create_review_app(directory, kb):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    catalog = [(sid, s.th, s.en) for sid, s in sorted(kb.symptoms.items())]
    digest = hashlib.sha256(json.dumps(catalog, ensure_ascii=False).encode()).hexdigest()
    if digest != manifest["catalog_sha256"]:
        raise ValueError("Review catalog changed; restore the KB for this manifest")
    app = FastAPI(title="Local normalization review", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @app.middleware("http")
    async def local_only(request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            expected = f"{request.url.scheme}://{request.url.netloc}"
            if (origin is not None and origin != expected
                    or request.headers.get("x-review-action") != "save"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Save from the local review page only"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        return response

    @app.get("/", response_class=HTMLResponse)
    def page():
        return Path(__file__).with_name("review.html").read_text(encoding="utf-8")

    @app.get("/api/catalog")
    def symptoms():
        return [{"id": sid, "th": s.th, "en": s.en} for sid, s in sorted(kb.symptoms.items())]

    @app.get("/api/tasks/{subset}")
    def tasks(subset: Subset):
        with (directory / ".review.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            return snapshot(directory, manifest, kb, subset)

    @app.post("/api/tasks/{subset}/{tid}")
    def save(subset: Subset, tid: str, payload: Annotation, request: Request):
        return save_annotation(directory, manifest, kb, subset, tid, payload)

    return app
