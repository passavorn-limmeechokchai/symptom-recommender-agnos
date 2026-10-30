"""Load once at startup; each request creates its own patient facts and result."""
import hashlib
import json
import math

from .. import __version__
from ..engine.engine import Engine
from ..engine.facts import Facts, ancestors, root
from ..kb.loader import load_kb, load_lexicon
from ..normalize.learn import load_aliases
from ..normalize.normalizer import Match, Normalizer
from .models import (
    DifferentialView,
    DiseaseView,
    MatchView,
    PatientHistory,
    QuestionView,
    RecommendationView,
    RecommendResponse,
    RedFlagView,
    SymptomView,
)


class InputError(ValueError):
    pass


class RecommenderService:
    def __init__(self, kb, normalizer, engine, mode="book_only", model_version=__version__, profile="rules"):
        self.kb = kb
        self.normalizer = normalizer
        self.engine = engine
        self.mode = mode
        self.model_version = model_version
        self.profile = profile
        self.catalog = {}
        for symptom in kb.symptoms.values():
            self.catalog[symptom.id] = SymptomView(
                id=symptom.id, label=symptom.th, th=symptom.th, en=symptom.en, kind=symptom.kind,
                sex=symptom.sex, parent=symptom.parent, red_flag=symptom.red_flag, urgent=symptom.urgent)
        for sid, count in engine.memory.counts.sessions_with.items():
            if sid.startswith("label:") and count > 0:
                self.catalog[sid] = SymptomView(id=sid, label=sid[6:], kind="unmapped")

    def question(self, question_id):
        spec = self.kb.questions[question_id]
        return QuestionView(id=question_id, symptom=question_id.split(".")[0],
                            **spec.model_dump(exclude={"sex"}))

    def disease(self, disease_id):
        disease = self.kb.diseases[disease_id]
        return DiseaseView(id=disease.id, th=disease.th, en=disease.en)

    def read(self, text):
        if not text.strip():
            raise InputError("Symptom text must not be blank")
        if text in self.catalog:
            return [Match(text, text, "id", 1.0)]
        if text.startswith("label:"):
            raise InputError("Unknown data-label symptom id")
        matches = self.normalizer.read_symptoms(text)
        # Exact recorded labels remain usable by the data model where the KB has no mapping.
        if (not any(m.symptom or m.state != "present" or m.layer == "ambiguous" for m in matches)
                and "label:" + text in self.catalog):
            return [Match(text, "label:" + text, "data_label", 1.0)]
        return matches

    def recommend(self, request):
        states = {"present": [], "absent": [], "unsure": [], "historical": []}
        hpi = {}
        matches = []
        chief = None
        needs_clarification = False

        def read_item(text, state, is_chief=False):
            nonlocal chief, needs_clarification
            for match in self.read(text):
                sid = match.symptom
                actual_state = match.state if match.state != "present" else state
                if state != "present" and match.state not in ("present", state):
                    raise InputError("Symptom text contradicts its history state; use an id or correct text")
                applicable = sid is None or self.catalog[sid].sex in (None, request.sex)
                candidates = [self.catalog[c] for c in match.candidates if c in self.catalog
                              and self.catalog[c].sex in (None, request.sex)]
                matches.append(MatchView(input=text, symptom=self.catalog.get(sid), state=actual_state,
                                         layer=match.layer, score=match.score, applicable=applicable,
                                         candidates=candidates))
                if sid is None:
                    needs_clarification = True
                    continue
                if sid not in states[actual_state]:
                    states[actual_state].append(sid)
                if is_chief and actual_state == "present" and chief is None:
                    chief = sid
                if not applicable:
                    needs_clarification = True

        read_item(request.chief_complaint, "present", True)
        for state in states:
            for text in getattr(request.history, state):
                read_item(text, state)
        unread_answers = {}
        for text, answers in request.answers.items():
            ids = [m.symptom for m in self.read(text) if m.symptom and m.state == "present"]
            for answer in answers:
                parsed = (self.normalizer.read_answer(ids[0], answer)
                          if ids and ids[0] in self.kb.symptoms else {})
                if not parsed:
                    unread_answers.setdefault(text, []).append(answer)
                for question, values in parsed.items():
                    hpi.setdefault(question, [])
                    for value in values:
                        if value not in hpi[question]:
                            hpi[question].append(value)
        # Explicit structured answers replace the parsed text for that question.
        for question, value in request.history.hpi.items():
            hpi[question] = value
        for question, value in hpi.items():
            self.validate_answer(question, value)

        present = [s for s in states["present"] if s in self.kb.symptoms]
        absent = [s for s in states["absent"] if s in self.kb.symptoms]
        labels = [s for s in states["present"] if s.startswith("label:")]
        facts = Facts(self.kb, request.sex, request.age,
                      chief if chief in self.kb.symptoms else None, present, absent, hpi, labels)
        for sid in facts.present:
            family = {sid, *ancestors(self.kb, sid)}
            if family & set(states["absent"]) or family & set(states["unsure"]):
                raise InputError("The same symptom cannot be present and absent/unsure")
        if set(states["absent"]) & set(states["unsure"]):
            raise InputError("The same symptom cannot be absent and unsure")
        if set(labels) & (set(states["absent"]) | set(states["unsure"])):
            raise InputError("The same symptom cannot have conflicting answers")

        result = self.engine.run(facts)
        # Unknown is not a negative, but the user has already answered this picker item.
        for sid in states["unsure"] + states["absent"]:
            result.skip.add(root(self.kb, sid))
        normalized_hpi = dict(facts.hpi) | facts.numbers
        # Keep stated symptoms here; inferred forms are recomputed from HPI every round.
        # Otherwise changing an HPI answer would retain a form derived from the old answer.
        history = PatientHistory(present=states["present"], absent=states["absent"],
                                 unsure=states["unsure"], historical=states["historical"], hpi=normalized_hpi)
        red_flags = []
        for flag in result.red_flags:
            sid = flag["symptom"]
            answer = ("yes" if sid in facts.known else "no" if sid in facts.absent
                      else "unknown" if sid in states["unsure"] else None)
            red_flags.append(RedFlagView(symptom=self.catalog[sid], source=flag["source"],
                                         answered=answer is not None, answer=answer))
        recommendations = []
        book_max = max(result.book.values(), default=0) or 1
        data_max = max(result.data.values(), default=0) or 1
        for sid, score in result.ranked(self.engine.alpha):
            sources = []
            if result.book.get(sid, 0) > 0 and self.engine.alpha > 0:
                sources.append("book")
            if result.data.get(sid, 0) > 0 and self.engine.alpha < 1:
                sources.append("cooccurrence")
            recommendations.append(RecommendationView(
                symptom=self.catalog[sid], score=score, book_score=result.book.get(sid, 0) / book_max,
                data_score=result.data.get(sid, 0) / data_max, sources=sources or ["popularity"],
                book_reasons=[self.disease(d) for d in result.book_reasons.get(sid, [])]))
        return RecommendResponse(
            normalizer_profile=self.profile,
            model_version=self.model_version, model_mode=self.mode, alpha=self.engine.alpha,
            chief_complaint=self.catalog.get(chief), needs_clarification=needs_clarification or chief is None,
            normalization=matches, normalized_history=history, unread_answers=unread_answers,
            hpi_questions=[self.question(q) for q in result.hpi_questions], chapters=result.chapters,
            urgent=bool(result.urgent), urgent_reasons=list(dict.fromkeys(result.urgent)),
            red_flags=red_flags, recommendations=recommendations,
            ddx=[DifferentialView(**self.disease(d).model_dump(), weight=w) for d, w in result.ddx],
            switch_chief_complaint=self.catalog.get(result.switch_to))

    def validate_answer(self, question_id, value):
        spec = self.kb.questions.get(question_id)
        if spec is None:
            raise InputError(f"Unknown HPI question: {question_id}")
        if value is None or value == "unknown":
            return
        if isinstance(value, (float, int)):
            if spec.unit is None or value < 0:
                raise InputError(f"Expected an option for HPI question: {question_id}")
            return
        values = value if isinstance(value, list) else [value]
        allowed = set(spec.options or spec.categories or {}) | {"unknown"}
        if not values or any(v not in allowed for v in values):
            raise InputError(f"Invalid option for HPI question: {question_id}")
        if spec.options and not spec.multiple and len(values) > 1:
            raise InputError(f"Choose one option for HPI question: {question_id}")


def load_service(kb_dir, build_dir, semantic=False, profile=None):
    profile = profile or ("semantic" if semantic else "rules")
    if profile not in {"rules", "semantic", "retrieval"}:
        raise RuntimeError("Unknown normalizer profile; choose rules, semantic or retrieval")
    semantic = profile == "semantic"
    kb = load_kb(kb_dir)
    artifact_dir = build_dir / "semantic" if semantic else build_dir
    paths = [artifact_dir / name for name in ("memory.json", "aliases.json", "alpha.json")]
    present = [p.exists() for p in paths]
    if any(present) and not all(present):
        raise RuntimeError("Incomplete runtime artifacts: run scripts/build.py or mount all three JSON files")
    if semantic and not all(present):
        raise RuntimeError("Semantic mode requires artifacts from scripts/build.py --semantic")
    alpha = 1.0
    if all(present):
        alpha = json.loads(paths[2].read_text(encoding="utf-8"))["alpha"]
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise RuntimeError("Artifact alpha must be a finite number between zero and one")
    engine = Engine(kb, alpha=alpha)
    if all(present):
        engine.memory.load(paths[0])
        for sid in engine.memory.counts.sessions_with:
            if sid not in kb.symptoms and not sid.startswith("label:"):
                raise RuntimeError("Memory uses unknown symptom ids; rebuild against the current KB")
    learned = load_aliases(paths[1]) if all(present) else {}
    if any(sid not in kb.symptoms for sid in learned.values()):
        raise RuntimeError("Aliases use unknown symptom ids; rebuild against the current KB")
    search = None
    if semantic:
        from ..normalize.semantic import load_search
        search = load_search(kb, build_dir / "semantic" / "config.json")
    if profile == "retrieval":
        from ..normalize.retrieval import load_retrieval
        search = load_retrieval(kb, build_dir / "retrieval" / "config.json")
    digest = hashlib.sha256()
    for path in sorted(kb_dir.rglob("*.yaml")) + ([*paths] if all(present) else []):
        digest.update(path.read_bytes())
    if semantic:
        digest.update((build_dir / "semantic" / "config.json").read_bytes())
    digest.update(profile.encode())
    if profile == "retrieval":
        digest.update((build_dir / "retrieval" / "config.json").read_bytes())
        digest.update((build_dir / "retrieval" / "classifier.npz").read_bytes())
    return RecommenderService(kb, Normalizer(kb, load_lexicon(kb_dir), learned, search), engine,
                              "hybrid" if all(present) else "book_only",
                              __version__ + "+" + digest.hexdigest()[:12], profile)
