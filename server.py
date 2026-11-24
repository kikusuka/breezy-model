"""Breezy's lightweight OpenAI-compatible local-model HTTP server."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

PORT = int(os.getenv("PORT", "7860"))
MODEL_REPO = os.getenv("MODEL_REPO", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
MODEL_NAME = os.getenv("MODEL_NAME", "breezy-default")
API_KEY = os.getenv("API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN") or None
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", str(max(1, min(4, os.cpu_count() or 2)))))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
MAX_ALLOWED_TOKENS = int(os.getenv("MAX_ALLOWED_TOKENS", "1024"))

app = FastAPI(title="Breezy Model API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

state: dict[str, Any] = {
    "status": "starting",
    "error": None,
    "model": MODEL_NAME,
    "started_at": time.time(),
}
_model: Llama | None = None
_model_lock = threading.Lock()


def _load_model() -> None:
    """Download and load the GGUF model outside the web-server startup path."""
    global _model
    try:
        state["status"] = "downloading"
        print(f"[breezy] Downloading {MODEL_REPO}/{MODEL_FILE}", flush=True)
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            token=HF_TOKEN,
        )
        print(f"[breezy] Download complete: {model_path}", flush=True)

        state["status"] = "loading"
        print(
            f"[breezy] Loading GGUF (ctx={N_CTX}, threads={N_THREADS})",
            flush=True,
        )
        _model = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=128,
            verbose=False,
        )
        state["status"] = "ready"
        print("[breezy] Model is ready to accept requests", flush=True)
    except Exception as exc:
        state["status"] = "error"
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[breezy] Model startup failed: {state['error']}", flush=True)


@app.on_event("startup")
def on_startup() -> None:
    print(f"[breezy] HTTP server starting on 0.0.0.0:{PORT}", flush=True)
    threading.Thread(target=_load_model, name="breezy-model-loader", daemon=True).start()


def _authorize(authorization: str | None) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _get_model() -> Llama:
    if state["status"] == "error":
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "error": state["error"]},
        )
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail={"status": state["status"], "message": "Model is not ready yet"},
        )
    return _model


def _normalise_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")

    normalised: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Each message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"Unsupported message role: {role}")
        if not isinstance(content, str):
            raise HTTPException(status_code=400, detail="Message content must be a string")
        normalised.append({"role": role, "content": content})
    return normalised


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Breezy Model API",
        "status": str(state["status"]),
        "health": "/health",
        "api": "/v1/chat/completions",
    }


@app.get("/health")
@app.get("/healthz")
def health():
    status_code = 200 if state["status"] == "ready" else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": state["status"],
            "model": MODEL_NAME,
            "error": state["error"],
            "uptime_seconds": round(time.time() - state["started_at"], 1),
        },
    )


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)):
    _authorize(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(state["started_at"]),
                "owned_by": "breezy",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(
    body: dict[str, Any],
    authorization: str | None = Header(default=None),
):
    _authorize(authorization)
    model = _get_model()
    messages = _normalise_messages(body.get("messages"))

    try:
        requested_tokens = int(body.get("max_tokens") or DEFAULT_MAX_TOKENS)
        max_tokens = max(1, min(requested_tokens, MAX_ALLOWED_TOKENS))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.9))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid generation parameters")

    if not 0 <= temperature <= 2:
        raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
    if not 0 < top_p <= 1:
        raise HTTPException(status_code=400, detail="top_p must be greater than 0 and at most 1")

    request_id = "chatcmpl-" + uuid.uuid4().hex
    stream = bool(body.get("stream", False))

    if stream:
        def event_stream():
            try:
                with _model_lock:
                    for chunk in model.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True,
                    ):
                        chunk["id"] = request_id
                        chunk["model"] = MODEL_NAME
                        yield "data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                error = {"error": {"message": str(exc), "type": "inference_error"}}
                yield "data: " + json.dumps(error, ensure_ascii=False) + "\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        with _model_lock:
            result = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        result["id"] = request_id
        result["model"] = MODEL_NAME
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    print(f"[breezy] Launching Uvicorn on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
