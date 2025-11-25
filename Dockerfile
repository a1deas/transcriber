FROM nvidia/cuda:12.1.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir "poetry>=1.8.0"

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main

COPY transcriber ./transcriber

RUN poetry install --no-interaction --no-ansi --only main

ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models

ENTRYPOINT ["transcriber"]
CMD ["--help"]
