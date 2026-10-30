PYTHON ?= python3
IMAGE ?= symptom-recommender:local
PROFILE ?= $(if $(filter 1,$(SYMPTOM_REC_SEMANTIC)),semantic,rules)
INCLUDE_SEMANTIC ?= $(if $(filter retrieval semantic,$(PROFILE)),1,0)
export PYTHONPATH := src

.PHONY: build test evaluate validate lint serve docker-build docker-run normalization-review normalization-eval retrieval-build retrieval-eval review prospective-freeze prospective-eval
build:
	$(PYTHON) scripts/build.py
test:
	$(PYTHON) -m pytest -q
evaluate:
	$(PYTHON) scripts/evaluate.py
validate:
	$(PYTHON) -m symptom_rec.kb.validate
lint:
	ruff check src scripts tests
serve:
	SYMPTOM_REC_PROFILE=$(PROFILE) $(PYTHON) -m uvicorn symptom_rec.api.app:app --host 127.0.0.1 --port 8000 --no-access-log
docker-build:
	docker build --build-arg INCLUDE_SEMANTIC=$(INCLUDE_SEMANTIC) -t $(IMAGE) .
docker-run:
	docker run --rm -p 127.0.0.1:8000:8000 -e SYMPTOM_REC_PROFILE=$(PROFILE) --mount type=bind,src="$(CURDIR)/build",dst=/app/build,readonly $(IMAGE)
normalization-review:
	$(PYTHON) scripts/normalization_review.py prepare
normalization-eval:
	$(PYTHON) scripts/normalization_review.py evaluate
retrieval-build:
	$(PYTHON) scripts/prepare_retrieval.py
retrieval-eval:
	$(PYTHON) scripts/evaluate_retrieval.py
review:
	$(PYTHON) scripts/review_normalization.py
prospective-freeze:
	$(PYTHON) scripts/prospective_evaluation.py freeze
prospective-eval:
	$(if $(strip $(DATA)),,$(error Set DATA to the new private CSV path))
	$(PYTHON) scripts/prospective_evaluation.py evaluate --data "$(DATA)"
