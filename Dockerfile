FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SYMPTOM_REC_KB_DIR=/app/kb \
    SYMPTOM_REC_BUILD_DIR=/app/build \
    HF_HUB_OFFLINE=1
WORKDIR /app
ARG INCLUDE_SEMANTIC=0
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN if [ "$INCLUDE_SEMANTIC" = "1" ]; then \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir '.[semantic]'; \
    else pip install --no-cache-dir .; fi && \
    useradd --uid 10001 --create-home appuser && mkdir /app/build
COPY kb/ ./kb/
ENV PYTHAINLP_DATA=/tmp/pythainlp-data SYMPTOM_REC_PROFILE=rules
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "symptom_rec.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
