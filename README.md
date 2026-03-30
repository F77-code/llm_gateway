# LLM Gateway

Единый OpenAI-совместимый API поверх нескольких LLM-провайдеров: маршрутизация по модели, rate limiting, учёт токенов и стоимости, health-check и fallback между провайдерами.

## Стек

**FastAPI**, **Redis**, **asyncio**, **httpx**, **Docker Compose**

## MVP

| Возможность | Описание |
|-------------|-----------|
| `POST /v1/chat/completions` | Формат запроса как у OpenAI Chat Completions |
| Маршрутизация | По полю `model`: например `gpt-4o` → OpenAI, `claude-3-5-sonnet` → Anthropic, `gemini-pro` → Google |
| Rate limiting | X запросов в минуту на API-ключ (Redis, sliding window) |
| Учёт | Токены и стоимость запроса, агрегаты в Redis |
| `GET /stats/{api_key}` | Токены и деньги за день и за месяц |
| `GET /health` | Ping доступности каждого провайдера |
| Fallback | При недоступности провайдера — следующий из настроенного списка |

## Вне MVP

- Стриминг ответов
- Полноценная БД (пока только Redis)
- Веб-UI

## Структура

```
gateway/
├── app/
│   ├── routers/chat.py          # основной эндпоинт
│   ├── providers/               # openai.py, anthropic.py, gemini.py
│   ├── middleware/ratelimit.py
│   ├── services/cost.py         # подсчёт стоимости
│   └── config.py                # провайдеры из .env
├── docker-compose.yml           # app + redis
└── .env.example
```

## Запуск

```bash
docker compose up --build
```

Переменные окружения — в `.env` по образцу `.env.example`.

## Зачем это

Тот же класс решений, что **LiteLLM** / **OpenRouter**: инфраструктурный прокси с единым контрактом, а не очередной CRUD.
