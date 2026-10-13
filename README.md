# LLM Gateway

OpenAI-compatible API gateway for multiple LLM providers.

Project provides a single `/v1/chat/completions` endpoint and routes requests by model name to the correct upstream provider. It also adds rate limiting, usage/cost accounting in Redis, health checks, and unified error handling.

## Features

- OpenAI-compatible endpoint: `POST /v1/chat/completions`
- Model-based routing to providers: OpenAI, Anthropic, Gemini, DeepSeek, Perplexity, xAI
- Per-API-key rate limit (requests per minute, sliding window, Redis-backed)
- Usage and cost aggregation in Redis (daily and monthly buckets)
- Stats endpoint: `GET /stats/{api_key}`
- Health endpoint: `GET /health` with Redis + provider checks
- Structured error responses in OpenAI-like format
- Request tracing via `X-Request-ID`

## Tech Stack

- Python 3.11+
- FastAPI + Uvicorn
- httpx (async upstream clients)
- Redis (rate limit + stats)
- Docker / Docker Compose

## Quick Start (Docker)

1. Create local environment file:

```bash
cp .env.example .env
```

2. Fill provider API keys in `.env` (at least one provider key is required).

3. Run services:

```bash
docker compose up --build
```

4. Open API:

- App: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

All application and development commands run inside Docker; host Python is not
required. The autonomous mock-provider demo is tracked in the next infrastructure
PR. This first limiter PR retains the existing real-provider configuration.

## Configuration

Environment variables are loaded from `.env` (`.env.example` as template).

### Provider keys

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `PERPLEXITY_API_KEY`
- `XAI_API_KEY`

### Infrastructure and gateway

- `REDIS_URL` (default: `redis://localhost:6379/0`)
- `RATE_LIMIT_RPM` (default: `60`)
- `DEFAULT_API_KEY` (optional shared gateway token for `Authorization: Bearer ...`)

If `DEFAULT_API_KEY` is set, every request must use exactly this token.

## API

### 1) Chat Completions

`POST /v1/chat/completions`

The request body follows OpenAI chat completions schema used in this project.

Example:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_GATEWAY_KEY_OR_ANY_IF_NOT_SET>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

Notes:

- `stream=true` is currently not supported.
- Unknown `model` returns `400` with `model_not_found`.
- Upstream provider failures are normalized to gateway errors.

### 2) Stats

`GET /stats/{api_key}`

Returns usage for:

- current day (`today`)
- current month (`current_month`)

Security rule: bearer token must match path `api_key`.

Example:

```bash
curl -X GET http://localhost:8000/stats/my-api-key \
  -H "Authorization: Bearer my-api-key"
```

### 3) Health

`GET /health`

Returns:

- overall gateway status
- Redis status
- per-provider status and latency

If all providers are unhealthy, endpoint responds with `503`.

## Supported Models

Routing is configured in `app/config.py` (`Settings.provider_by_model`).

Current examples include:

- OpenAI: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-4.1-mini`, etc.
- Anthropic: `claude-opus-4-6`, `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001`
- Gemini: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`, etc.
- DeepSeek: `deepseek-chat`, `deepseek-reasoner`
- Perplexity: `sonar`, `sonar-pro`, `sonar-reasoning-pro`, `sonar-deep-research`
- xAI: `grok-4-1-fast-non-reasoning`, `grok-4-1-fast-reasoning`

## Project Structure

```text
.
├── app/
│   ├── main.py                  # FastAPI app, lifecycle, error handlers
│   ├── config.py                # settings + model-to-provider routing
│   ├── dependencies.py          # auth dependency
│   ├── middleware/ratelimit.py  # Redis sliding-window limiter
│   ├── routers/
│   │   ├── chat.py              # /v1/chat/completions
│   │   ├── stats.py             # /stats/{api_key}
│   │   └── health.py            # /health
│   ├── providers/               # provider adapters + registry
│   └── services/                # Redis + cost accounting
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## Development

Build and run the limiter quality checks and the complete test suite:

```bash
docker compose --profile checks build checks
docker compose --profile checks run --rm checks
```

The profile starts an isolated ephemeral `redis-test`, with no published port.
Real Redis integration tests are mandatory: missing or unreachable
`TEST_REDIS_URL` fails the run. Tests create UUID-prefixed keys and delete only
their own keys. They never flush the database or use application Redis.

The first PR checks lint, formatting and types for the limiter and its tests;
repository-wide quality checks and CI belong to the infrastructure PR. All
existing tests run. Dependency installation uses uv and the committed lockfile,
with release artifacts restricted to those available by 7 April 2026. See
[release evidence](docs/specs/dependencies.md).

Run just the real Redis suite:

```bash
docker compose --profile checks run --rm checks pytest -q tests/integration
```

The limiter atomically admits requests in `(now - window_seconds, now]`; an entry
at the left boundary expires. `remaining` reports slots after the decision, and
`reset_ts` is the oldest admission's expiry rounded up to Unix seconds on both
accepted and denied responses. Concurrent calls use UUID members even when
timestamps match. An accepted request refreshes TTL to `2 * window_seconds + 1`;
a denied request leaves TTL unchanged. Gateway clocks must be synchronized.

See [the delivery plan](docs/specs/README.md) for subsequent reviewable PRs.

## Limitations

- No streaming responses yet (`stream=true` is rejected)
- Redis is required for full rate-limit and usage accounting behavior
- Model catalog is static and configured in code (`provider_by_model`)

## License

Add your license here (for example: MIT).
