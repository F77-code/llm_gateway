# Dependency release evidence

All locked artifacts were verified against PyPI JSON metadata
(upload timestamp and SHA-256) with a cutoff of 2026-04-08 00:00 UTC,
including all of 2026-04-07. Run the audit inside Docker:

```bash
docker compose --profile checks run --rm --no-deps checks python scripts/verify_dependency_dates.py
```

- [uv 0.10.0](https://github.com/astral-sh/uv/releases/tag/0.10.0): 2026-02-05.
- [Python 3.12.13](https://www.python.org/downloads/release/python-31213/): 2026-03-03.
- [Redis 7.4.7](https://github.com/redis/redis/releases/tag/7.4.7): 2025-11-02.

- [annotated-doc 0.0.4](https://pypi.org/pypi/annotated-doc/0.0.4/json): latest locked artifact uploaded 2025-11-10T22:07:42.062214+00:00.
- [annotated-types 0.7.0](https://pypi.org/pypi/annotated-types/0.7.0/json): latest locked artifact uploaded 2024-05-20T21:33:25.928805+00:00.
- [anyio 4.13.0](https://pypi.org/pypi/anyio/4.13.0/json): latest locked artifact uploaded 2026-03-24T12:59:09.671977+00:00.
- [asgi-lifespan 2.1.0](https://pypi.org/pypi/asgi-lifespan/2.1.0/json): latest locked artifact uploaded 2023-03-28T17:35:49.126445+00:00.
- [async-timeout 5.0.1](https://pypi.org/pypi/async-timeout/5.0.1/json): latest locked artifact uploaded 2024-11-06T16:41:39.600310+00:00.
- [black 26.3.1](https://pypi.org/pypi/black/26.3.1/json): latest locked artifact uploaded 2026-03-12T03:40:36.832434+00:00.
- [certifi 2026.2.25](https://pypi.org/pypi/certifi/2026.2.25/json): latest locked artifact uploaded 2026-02-25T02:54:17.342729+00:00.
- [click 8.3.2](https://pypi.org/pypi/click/8.3.2/json): latest locked artifact uploaded 2026-04-03T19:14:45.118083+00:00.
- [colorama 0.4.6](https://pypi.org/pypi/colorama/0.4.6/json): latest locked artifact uploaded 2022-10-25T02:36:22.414799+00:00.
- [fakeredis 2.34.1](https://pypi.org/pypi/fakeredis/2.34.1/json): latest locked artifact uploaded 2026-02-25T13:17:51.315606+00:00.
- [fastapi 0.135.3](https://pypi.org/pypi/fastapi/0.135.3/json): latest locked artifact uploaded 2026-04-01T16:23:59.328683+00:00.
- [h11 0.16.0](https://pypi.org/pypi/h11/0.16.0/json): latest locked artifact uploaded 2025-04-24T03:35:25.427659+00:00.
- [hiredis 3.3.1](https://pypi.org/pypi/hiredis/3.3.1/json): latest locked artifact uploaded 2026-03-16T15:21:08.092372+00:00.
- [httpcore 1.0.9](https://pypi.org/pypi/httpcore/1.0.9/json): latest locked artifact uploaded 2025-04-24T22:06:22.219726+00:00.
- [httpx 0.28.1](https://pypi.org/pypi/httpx/0.28.1/json): latest locked artifact uploaded 2024-12-06T15:37:23.222462+00:00.
- [idna 3.11](https://pypi.org/pypi/idna/3.11/json): latest locked artifact uploaded 2025-10-12T14:55:20.501377+00:00.
- [iniconfig 2.3.0](https://pypi.org/pypi/iniconfig/2.3.0/json): latest locked artifact uploaded 2025-10-18T21:55:43.219048+00:00.
- [isort 8.0.1](https://pypi.org/pypi/isort/8.0.1/json): latest locked artifact uploaded 2026-02-28T10:08:20.685520+00:00.
- [librt 0.8.1](https://pypi.org/pypi/librt/0.8.1/json): latest locked artifact uploaded 2026-02-17T16:13:06.101663+00:00.
- [lupa 2.7](https://pypi.org/pypi/lupa/2.7/json): latest locked artifact uploaded 2026-04-07T09:06:54.760637+00:00.
- [mypy 1.20.0](https://pypi.org/pypi/mypy/1.20.0/json): latest locked artifact uploaded 2026-03-31T16:55:14.959774+00:00.
- [mypy-extensions 1.1.0](https://pypi.org/pypi/mypy-extensions/1.1.0/json): latest locked artifact uploaded 2025-04-22T14:54:24.164338+00:00.
- [packaging 26.0](https://pypi.org/pypi/packaging/26.0/json): latest locked artifact uploaded 2026-01-21T20:50:39.064191+00:00.
- [pathspec 1.0.4](https://pypi.org/pypi/pathspec/1.0.4/json): latest locked artifact uploaded 2026-01-27T03:59:46.938903+00:00.
- [platformdirs 4.9.4](https://pypi.org/pypi/platformdirs/4.9.4/json): latest locked artifact uploaded 2026-03-05T18:34:13.271375+00:00.
- [pluggy 1.6.0](https://pypi.org/pypi/pluggy/1.6.0/json): latest locked artifact uploaded 2025-05-15T12:30:07.975376+00:00.
- [pydantic 2.12.5](https://pypi.org/pypi/pydantic/2.12.5/json): latest locked artifact uploaded 2025-11-26T15:11:46.471492+00:00.
- [pydantic-core 2.41.5](https://pypi.org/pypi/pydantic-core/2.41.5/json): latest locked artifact uploaded 2025-11-04T13:43:49.098415+00:00.
- [pydantic-settings 2.13.1](https://pypi.org/pypi/pydantic-settings/2.13.1/json): latest locked artifact uploaded 2026-02-19T13:45:08.055663+00:00.
- [pygments 2.20.0](https://pypi.org/pypi/pygments/2.20.0/json): latest locked artifact uploaded 2026-03-29T13:29:33.898791+00:00.
- [pytest 9.0.3](https://pypi.org/pypi/pytest/9.0.3/json): latest locked artifact uploaded 2026-04-07T17:16:18.027317+00:00.
- [pytest-asyncio 1.3.0](https://pypi.org/pypi/pytest-asyncio/1.3.0/json): latest locked artifact uploaded 2025-11-10T16:07:47.256373+00:00.
- [pytest-mock 3.15.1](https://pypi.org/pypi/pytest-mock/3.15.1/json): latest locked artifact uploaded 2025-09-16T16:37:27.081129+00:00.
- [python-dotenv 1.2.2](https://pypi.org/pypi/python-dotenv/1.2.2/json): latest locked artifact uploaded 2026-03-01T16:00:26.196056+00:00.
- [pytokens 0.4.1](https://pypi.org/pypi/pytokens/0.4.1/json): latest locked artifact uploaded 2026-01-30T01:03:45.924909+00:00.
- [redis 7.4.0](https://pypi.org/pypi/redis/7.4.0/json): latest locked artifact uploaded 2026-03-24T09:14:37.530329+00:00.
- [ruff 0.15.9](https://pypi.org/pypi/ruff/0.15.9/json): latest locked artifact uploaded 2026-04-02T18:17:20.829485+00:00.
- [sniffio 1.3.1](https://pypi.org/pypi/sniffio/1.3.1/json): latest locked artifact uploaded 2024-02-25T23:20:04.057878+00:00.
- [sortedcontainers 2.4.0](https://pypi.org/pypi/sortedcontainers/2.4.0/json): latest locked artifact uploaded 2021-05-16T22:03:42.897871+00:00.
- [starlette 1.0.0](https://pypi.org/pypi/starlette/1.0.0/json): latest locked artifact uploaded 2026-03-22T18:29:46.779636+00:00.
- [typing-extensions 4.15.0](https://pypi.org/pypi/typing-extensions/4.15.0/json): latest locked artifact uploaded 2025-08-25T13:49:26.313895+00:00.
- [typing-inspection 0.4.2](https://pypi.org/pypi/typing-inspection/0.4.2/json): latest locked artifact uploaded 2025-10-01T02:14:41.687923+00:00.
- [uvicorn 0.44.0](https://pypi.org/pypi/uvicorn/0.44.0/json): latest locked artifact uploaded 2026-04-06T09:23:22.826948+00:00.

Docker base tags pin component versions. OS layers/tags are not a
historical snapshot and may be rebuilt upstream; dependency artifact
hashes are frozen in uv.lock. The PyPI audit requires network access.
