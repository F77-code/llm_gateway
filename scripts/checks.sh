#!/bin/sh
set -eu

# The first PR checks the limiter and its tests; repository-wide quality cleanup
# belongs to the demo/CI PR. The full existing test suite runs here already.
uv lock --check --offline
ruff check app/services/redis.py tests/test_redis.py tests/integration scripts/verify_dependency_dates.py
ruff format --check app/services/redis.py tests/test_redis.py tests/integration scripts/verify_dependency_dates.py
mypy --follow-imports=silent app/services/redis.py
pytest -q
