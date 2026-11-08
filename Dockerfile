# CDM-1 — Production Multi-Stage Container

# Stage 1: Build stage with compiler toolchain
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt ./
RUN pip install --no-cache-dir --user -r requirements-lock.txt

# Stage 2: Minimal hardened runtime image (no build tools, non-root user)
FROM python:3.11-slim AS runtime

# Create dedicated non-root application user and group
RUN groupadd -r -g 10001 cdmuser && \
    useradd -r -u 10001 -g cdmuser -d /app -s /sbin/nologin cdmuser

WORKDIR /app

# Copy installed python dependencies from builder
COPY --from=builder /root/.local /home/cdmuser/.local

# Ensure path includes user site-packages
ENV PATH=/home/cdmuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CDM_ENV=production \
    PORT=8000

# Copy application source code and verified release artifacts
COPY cdm/ ./cdm/
COPY sdk/ ./sdk/
COPY api/ ./api/
COPY experiments/release/cdm-1-general/ ./experiments/release/cdm-1-general/

# Prepare directories and assign permissions to non-root user
RUN mkdir -p /app/experiments && chown -R cdmuser:cdmuser /app /home/cdmuser

USER cdmuser

EXPOSE 8000

# Python-native healthcheck without external curl or build tools
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
