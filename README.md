# LLM Gateway

OpenAI-compatible API gateway for multiple LLM providers.

Project provides a single `/v1/chat/completions` endpoint and routes requests by model name to the correct upstream provider. It also adds rate limiting, usage/cost accounting in Redis, health checks, and unified error handling.

## Features

- OpenAI-compatible endpoint: `POST /v1/chat/completions`
- Model-based routing to providers: OpenAI, Anthropic, Gemini, DeepSeek, Perplexity, xAI
- Per-client rate limit (requests per minute, sliding window, Redis-backed)
- Usage and cost aggregation in Redis (daily and monthly buckets)
- Authenticated usage endpoint: `GET /me/usage`
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

   Generate a client key in Docker (after building the checks image):

```bash
docker compose --profile checks build checks
docker compose --profile checks run --rm --no-deps checks python scripts/create_client_key.py --client-id team-a
```

   Store the returned `api_key` securely. Put only its `registration` object into
   the `GATEWAY_CLIENTS` JSON array in `.env`. Empty configuration accepts no
   clients. Provider API keys and gateway client keys have separate purposes.

3. Run services:

```bash
docker compose up --build
```

4. Open API:

- App: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

All application and development commands run inside Docker; host Python is not
required. The autonomous mock-provider demo is tracked in the next infrastructure
PR. The current runtime uses the existing real-provider configuration.

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
- `GATEWAY_CLIENTS` (JSON array of `client_id`, `key_sha256`, `enabled` registrations)

Send the generated client key using `Authorization: Bearer ...`. Missing,
malformed, unknown and disabled keys return `401`. Store only hashes in client
registration configuration. Rotate a key by adding another hash for the same
stable client ID, then disable/remove the previous registration; quota and usage
remain attached to the client ID. Different clients have separate quota/usage.
Registrations reload when the process restarts; online administration comes later.

`DEFAULT_API_KEY` and arbitrary-token access are removed. Configure client
registrations explicitly. Existing aggregates indexed by old raw tokens are not
automatically migrated; retain/export them privately if needed and start the new
client-ID aggregates. Do not send historical secrets in URLs to access them.

## API

### 1) Chat Completions

`POST /v1/chat/completions`

The request body follows OpenAI chat completions schema used in this project.

Example:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_GENERATED_CLIENT_KEY>" \
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

### 2) Usage

`GET /me/usage`

Returns usage for:

- current day (`today`)
- current month (`current_month`)

The authenticated client is selected from its bearer key. The response contains
the stable `client_id`; no secret, hash or key suffix is returned. The former
`/stats/{api_key}` endpoint is removed.

Example:

```bash
curl http://localhost:8000/me/usage \
  -H "Authorization: Bearer <YOUR_GENERATED_CLIENT_KEY>"
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
│   ├── dependencies.py          # client identity dependency
│   ├── middleware/ratelimit.py  # Redis sliding-window limiter
│   ├── routers/
│   │   ├── chat.py              # /v1/chat/completions
│   │   ├── stats.py             # /me/usage
│   │   └── health.py            # /health
│   ├── providers/               # provider adapters + registry
│   └── services/                # Redis + cost accounting
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

## Development

Build and run admission/authentication quality checks and the complete test suite:

```bash
docker compose --profile checks build checks
docker compose --profile checks run --rm checks
```

The profile starts an isolated ephemeral `redis-test`, with no published port.
Real Redis integration tests are mandatory: missing or unreachable
`TEST_REDIS_URL` fails the run. Tests create UUID-prefixed keys and delete only
their own keys. They never flush the database or use application Redis.

The checks cover lint, formatting and types for admission, authentication and
usage modules, and lint/formatting for all tests and scripts. Repository-wide
provider quality checks and CI belong to the infrastructure PR. All
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
