# Breezy Model API

A small OpenAI-compatible chat-completions server for a CPU-hosted GGUF model. The intended deployment target is a Docker-based ModelScope Studio exposing port **7860**.

## Model

The default model is Qwen2.5 0.5B Instruct, quantized to GGUF Q4_K_M:

- Repository: `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
- File: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- License: Apache-2.0 (check the model repository for its current license and terms)

The model is downloaded from Hugging Face when the container starts. The first start therefore takes longer than later starts if the platform preserves its cache.

## Deploy on ModelScope Studio

1. Create or open a **Docker** Studio connected to this GitHub repository.
2. Use the repository root as the build context.
3. Expose port **7860**.
4. Keep the Dockerfile's default command. It launches `python -u /app/server.py`.
5. Build and start the Studio.
6. Wait for `GET /health` to return HTTP 200 with `"status": "ready"`.

The Docker image compiles `llama-cpp-python` from source. This worked in the earlier build attempt, but a successful image build does not prove that the hosting platform will successfully start the container. The runtime logs must show `[breezy] HTTP server starting` and then the model download/load messages.

## Environment variables

| Name | Default | Description |
|---|---|---|
| `PORT` | `7860` | HTTP listening port |
| `API_KEY` | empty | Optional API bearer token |
| `MODEL_REPO` | Qwen GGUF repository above | Hugging Face repository |
| `MODEL_FILE` | Q4_K_M file above | GGUF filename |
| `MODEL_NAME` | `breezy-default` | API model identifier |
| `HF_TOKEN` | empty | Optional Hugging Face token for gated/private repositories |
| `N_CTX` | `2048` | Context window |
| `N_THREADS` | up to 4 | CPU inference threads |
| `MAX_TOKENS` | `512` | Default output token limit |
| `MAX_ALLOWED_TOKENS` | `1024` | Server-side maximum output token limit |
| `CORS_ORIGINS` | `*` | Comma-separated allowed browser origins |

Set a long random `API_KEY` before exposing the endpoint publicly. If it is set, send `Authorization: Bearer YOUR_KEY`. If it is empty, the API is unauthenticated.

## API

- `GET /` — service information
- `GET /health` or `GET /healthz` — readiness and startup status
- `GET /v1/models` — OpenAI-style model list
- `POST /v1/chat/completions` — chat completion; supports SSE streaming

Example:

```bash
curl https://YOUR-STUDIO-HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "breezy-default",
    "messages": [
      {"role": "system", "content": "You are Breezy, a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 128
  }'
```

If you leave `API_KEY` empty, remove the Authorization header from the request.

## Important limitations

This is a **0.5B-parameter CPU model**, intended for lightweight chat, basic Q&A, and integration testing. It is not comparable to larger hosted models for complex reasoning, coding, or long conversations. CPU generation may be slow, and the server serializes inference requests to avoid overloading the small CPU instance.

ModelScope's free-tier availability, account verification, runtime limits, and container startup behavior are controlled by ModelScope. This repository cannot bypass those platform requirements or guarantee always-on hosting.
