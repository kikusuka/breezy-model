FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MODEL_REPO=Qwen/Qwen2.5-0.5B-Instruct-GGUF \
    MODEL_FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf \
    MODEL_NAME=breezy-default \
    N_CTX=2048 \
    N_THREADS=4 \
    CMAKE_ARGS="-DGGML_NATIVE=OFF" \
    FORCE_CMAKE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
       fastapi==0.115.12 \
       uvicorn==0.34.2 \
       huggingface-hub==0.31.2 \
       llama-cpp-python==0.3.38

COPY server.py /app/server.py

EXPOSE 7860
CMD ["python", "-u", "/app/server.py"]
