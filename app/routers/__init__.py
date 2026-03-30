from fastapi import APIRouter

from app.routers import chat

api_router = APIRouter()
api_router.include_router(chat.router)

__all__ = ["api_router"]
