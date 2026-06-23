from contextlib import asynccontextmanager
from fastapi import FastAPI
import redis.asyncio as aioredis

from app.core import redis_db
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    redis_db.redis_client = aioredis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )
    print("✅ Connected to Redis")
    
    yield  # server runs here
    
    # SHUTDOWN
    if redis_db.redis_client:
        await redis_db.redis_client.aclose()
        print("🔴 Redis connection closed")

app = FastAPI(lifespan=lifespan)

# Register the endpoints
app.include_router(router)