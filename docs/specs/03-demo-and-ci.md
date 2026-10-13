# Reproducible demo and CI

Use pinned uv, committed uv.lock and frozen Docker installation. Dependency
releases must be available by 2026-04-07; record primary-source release metadata.
Run installation, lint, types, all tests and application exclusively in Docker.

`docker compose up --build` from a clean checkout starts an autonomous mock
provider demo with Redis health ordering; later `docker compose up` also works.
Real provider keys are optional environment configuration. Mock output is clearly
identified and deterministic. Preserve existing provider adapters.

A checks profile runs lint, formatting, types, unit tests and mandatory real
Redis tests against its own ephemeral service. CI runs the same commands plus a
build/start/readiness/chat smoke test. README quickstart requires no host Python.
Test both initial startup and restart; use bounded readiness waits.
