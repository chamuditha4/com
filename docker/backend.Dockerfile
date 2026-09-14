# API and MCP server image (same code, different command).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/srv/.venv

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /srv

# Dependencies first for layer caching; locked, production group only.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend ./backend
COPY data ./data

RUN useradd --system --uid 10001 app \
    && mkdir -p /srv/data/artifacts \
    && chown -R app /srv/data/artifacts
USER app

ENV PATH=/srv/.venv/bin:$PATH \
    PYTHONPATH=/srv/backend \
    WEB_CONCURRENCY=2

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live')"

# uvicorn reads WEB_CONCURRENCY for the worker count. Workers are stateless when REDIS_URL is set.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend", "--proxy-headers"]
