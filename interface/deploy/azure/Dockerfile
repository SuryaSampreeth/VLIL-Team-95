# syntax=docker/dockerfile:1
# ReXTrust backend image for Azure Container Apps.
# Build context = a folder holding pipeline_core.py, radcheck.py and interface/{backend,assets} (see deploy.ps1).
# Weights are NOT baked in: the Azure Files share "models" is mounted at /models and linked to <app>/Models,
# which is where pipeline_core.py and config.yaml expect them.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Slow/flaky networks: long timeouts, retries, and a BuildKit cache so a failed build doesn't re-download wheels
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
# CPU wheels from the PyTorch index; PyPI stays available for their pure-Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
COPY interface/backend/requirements.txt interface/backend/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r interface/backend/requirements.txt

COPY pipeline_core.py radcheck.py ./
COPY interface/backend interface/backend
COPY interface/assets interface/assets
RUN ln -s /models /app/Models

WORKDIR /app/interface/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
