"""The public request/response contract; clinical rules remain in the KB and engine."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HpiValue = str | float | list[str] | None


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True, allow_inf_nan=False)


class PatientHistory(Model):
    present: list[str] = Field(default_factory=list)
    absent: list[str] = Field(default_factory=list)
    unsure: list[str] = Field(default_factory=list)
    historical: list[str] = Field(default_factory=list)
    hpi: dict[str, HpiValue] = Field(default_factory=dict)


class RecommendRequest(Model):
    sex: Literal["male", "female"]
    age: float = Field(ge=0)
    chief_complaint: str = Field(min_length=1, max_length=1000,
                                 description="KB symptom id or free text; the first mapped symptom is the CC")
    history: PatientHistory = Field(default_factory=PatientHistory)
    answers: dict[str, list[str]] = Field(
        default_factory=dict, description="Free-text HPI answers keyed by symptom id or symptom text")


class SymptomView(Model):
    id: str
    label: str
    th: str | None = None
    en: str | None = None
    kind: str
    sex: str | None = None
    parent: str | None = None
    red_flag: bool = False
    urgent: bool = False


class MatchView(Model):
    input: str
    symptom: SymptomView | None
    state: Literal["present", "absent", "unsure", "historical"]
    layer: str
    score: float
    applicable: bool
    candidates: list[SymptomView]


class QuestionView(Model):
    id: str
    symptom: str
    ask: str
    options: list[str] | None
    unit: str | None
    categories: dict | None
    grey_zones: list[dict]
    multiple: bool
    allow_unknown: bool


class RedFlagView(Model):
    symptom: SymptomView
    source: str
    answered: bool
    answer: Literal["yes", "no", "unknown"] | None


class DiseaseView(Model):
    id: str
    th: str
    en: str


class DifferentialView(DiseaseView):
    weight: float


class RecommendationView(Model):
    symptom: SymptomView
    score: float
    book_score: float
    data_score: float
    sources: list[str]
    book_reasons: list[DiseaseView]


class QuestionsResponse(Model):
    model_version: str
    chief_complaint: SymptomView | None
    needs_clarification: bool
    normalization: list[MatchView]
    normalized_history: PatientHistory
    unread_answers: dict[str, list[str]]
    hpi_questions: list[QuestionView]


class RecommendResponse(QuestionsResponse):
    normalizer_profile: str
    model_mode: Literal["hybrid", "book_only"]
    alpha: float
    chapters: list[str]
    urgent: bool
    urgent_reasons: list[str]
    red_flags: list[RedFlagView]
    recommendations: list[RecommendationView]
    ddx: list[DifferentialView]
    switch_chief_complaint: SymptomView | None


class CatalogResponse(Model):
    total: int
    offset: int
    items: list[SymptomView]


class HealthResponse(Model):
    status: str
    model_version: str
    model_mode: str
    alpha: float
    semantic_enabled: bool
    normalizer_profile: str
    chapters: int
    symptoms: int
