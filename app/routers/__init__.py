from fastapi import APIRouter

from app.routers import chat, health, stats

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(stats.router)
api_router.include_router(health.router)

__all__ = ["api_router"]
