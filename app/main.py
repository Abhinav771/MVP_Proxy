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

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the endpoints
app.include_router(router)