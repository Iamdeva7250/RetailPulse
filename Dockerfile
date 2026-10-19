
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/processed/ ./data/processed/
COPY config.yaml .
COPY README.md .

RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
