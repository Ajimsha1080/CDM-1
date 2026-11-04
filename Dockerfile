# CDM-1 — Cognitive Decision Model Production Container
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY cdm/ ./cdm/
COPY sdk/ ./sdk/
COPY api/ ./api/
COPY experiments/release/cdm-1-general/ ./experiments/release/cdm-1-general/

EXPOSE 8000

# Run FastAPI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
