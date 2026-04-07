from fastapi import APIRouter

from app.routers import chat, stats

api_router = APIRouter()
api_router.include_router(chat.router)
api_router.include_router(stats.router)

__all__ = ["api_router"]
