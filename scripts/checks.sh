#!/bin/sh
set -eu

# Repository-wide provider quality cleanup belongs to the demo/CI PR.
# Changed admission/authentication/usage modules and the full suite run here.
uv lock --check --offline
ruff check app/auth.py app/config.py app/dependencies.py app/middleware/ratelimit.py app/routers/chat.py app/routers/stats.py app/services/redis.py tests scripts
ruff format --check app/auth.py app/config.py app/dependencies.py app/middleware/ratelimit.py app/routers/chat.py app/routers/stats.py app/services/redis.py tests scripts
mypy --follow-imports=silent app/auth.py app/config.py app/dependencies.py app/middleware/ratelimit.py app/routers/chat.py app/routers/stats.py app/services/redis.py
pytest -q
