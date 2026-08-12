# --- build stage ---
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps for pyarrow / numpy wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY config.yaml ./
COPY src ./src
COPY dashboard ./dashboard

# Install the package itself (editable keeps the source tree mounted-friendly).
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# By default the data/model are expected to already exist (mounted volume or
# baked in). The compose file mounts ./data and ./mlruns.
CMD ["python", "-m", "credit_risk.serving.app"]
