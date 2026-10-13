FROM ghcr.io/astral-sh/uv:0.10.0 AS uv

FROM python:3.12.13-slim-bookworm AS base
COPY --from=uv /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml uv.lock ./

FROM base AS checks
RUN uv sync --frozen --extra dev --no-install-project
COPY app ./app
COPY tests ./tests
COPY scripts ./scripts
CMD ["sh", "scripts/checks.sh"]

FROM base AS runtime
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
