"""Stateless HTTP interface. Configure paths before starting Uvicorn."""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request

from .. import __version__
from ..normalize.text import key
from ..paths import BUILD_DIR, KB_DIR
from .models import (
    CatalogResponse,
    HealthResponse,
    QuestionsResponse,
    RecommendRequest,
    RecommendResponse,
)
from .service import InputError, load_service


def create_app(service=None, kb_dir=None, build_dir=None, semantic=None, profile=None):
    @asynccontextmanager
    async def lifespan(app):
        enabled = semantic if semantic is not None else os.getenv("SYMPTOM_REC_SEMANTIC", "0") == "1"
        build_path = Path(build_dir or os.getenv("SYMPTOM_REC_BUILD_DIR", BUILD_DIR))
        selected = profile or os.getenv("SYMPTOM_REC_PROFILE")
        app.state.service = service or load_service(
            Path(kb_dir or os.getenv("SYMPTOM_REC_KB_DIR", KB_DIR)),
            build_path, enabled, selected)
        yield

    app = FastAPI(title="Symptom Recommender", version=__version__, lifespan=lifespan)

    def recommend(payload, request):
        try:
            return request.app.state.service.recommend(payload)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/recommend", response_model=RecommendResponse)
    def recommendations(payload: RecommendRequest, request: Request):
        """Send the complete history each time; returns all ranked candidates and safety gates."""
        return recommend(payload, request)

    @app.post("/v1/questions", response_model=QuestionsResponse)
    def questions(payload: RecommendRequest, request: Request):
        """HPI lookup only. Use /v1/recommend for urgency, red flags and ranking."""
        result = recommend(payload, request)
        return QuestionsResponse(**result.model_dump(include=set(QuestionsResponse.model_fields)))

    @app.get("/v1/symptoms", response_model=CatalogResponse)
    def symptoms(request: Request, q: str = "", sex: Literal["male", "female"] | None = None,
                 offset: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=2000)):
        service = request.app.state.service
        items = []
        needle = key(q)
        for sid, item in service.catalog.items():
            if sex and item.sex not in (None, sex):
                continue
            symptom = service.kb.symptoms.get(sid)
            names = [item.id, item.label, item.th or "", item.en or ""]
            if symptom:
                names.extend(symptom.aliases)
            if needle and not any(needle in key(name) for name in names):
                continue
            items.append(item)
        return CatalogResponse(total=len(items), offset=offset, items=items[offset:offset + limit])

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        service = request.app.state.service
        return HealthResponse(status="ok", model_version=service.model_version, model_mode=service.mode,
                              alpha=service.engine.alpha,
                              semantic_enabled=service.normalizer.semantic is not None,
                              normalizer_profile=service.profile,
                              chapters=len(service.kb.chapters), symptoms=len(service.catalog))

    return app


app = create_app()
